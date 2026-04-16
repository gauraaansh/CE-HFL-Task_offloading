import numpy as np
import torch

from agent.ddqn_agent import DDQNAgent
from agent.agg_scheduler import AggScheduler
from env.mec_env import MECEnvironment
from utils.metrics import MetricsLogger
from utils.comm_cost import model_size_bytes
from utils.comm_energy import WirelessModel
from network.topology import Topology
from network.hierarchy_optimizer import HierarchyOptimizer
from config.config import Config


class AASHFLTrainer:
    """
    AAS-HFL: Adaptive Aggregation Scheduling for Hierarchical Federated Learning.

    Extends HierarchicalFLTrainer (PATGA-based 2-tier HFL) by replacing
    fixed aggregation intervals with a DDQN scheduler that decides each episode:
        0 → wait (no aggregation)
        1 → trigger device→edge aggregation only
        2 → trigger device→edge then edge→cloud aggregation

    Scheduler state:  [Δr_smooth, E_norm, avg_channel, τ_e, τ_c]
    Scheduler reward: -E_norm(action) - λ·max(0, -Δr_smooth)
    """

    def __init__(self, num_devices, num_edges):
        self.num_devices = num_devices
        self.num_edges   = num_edges
        cfg = Config()

        self.wireless  = WirelessModel()
        self.topology  = Topology(num_devices, num_edges)
        self.optimizer = HierarchyOptimizer(
            num_devices, num_edges,
            device_positions=self.topology.device_pos,
            edge_positions=self.topology.edge_pos,
        )
        self.edge_groups = self.optimizer.build_groups()

        # model size (same network as per-device agents)
        _tmp_agent      = DDQNAgent(state_dim=4, action_dim=2, cfg=cfg)
        self.model_bytes = model_size_bytes(_tmp_agent.online_net)
        self.model_bits  = self.model_bytes * 8

        # per-device FL upload cost (amortized over agg interval)
        fl_shares = self._compute_fl_shares(cfg)

        self.envs    = [
            MECEnvironment(fl_energy_share=fl_shares[i], alpha_fl=cfg.ALPHA_FL)
            for i in range(num_devices)
        ]
        self.agents  = [DDQNAgent(state_dim=4, action_dim=2, cfg=cfg) for _ in range(num_devices)]
        self.loggers = [MetricsLogger() for _ in range(num_devices)]

        # PATGA trees
        self.edge_trees = self._build_patga_trees(cfg)
        for i, tree in enumerate(self.edge_trees):
            print(f"[AAS-HFL PATGA] Edge {i}  tree_energy={tree['tree_energy']:.4e} J  "
                  f"parent_map={tree['parent']}")

        # worst-case E_comm: all devices at COMM_RANGE + all edges to farthest cloud point
        self.e_comm_max = self._compute_e_comm_max(cfg)

        # max cumulative task energy since last edge agg (for E_norm in state)
        # worst case: all devices, all steps, all local (task*1.0, task up to 1.0)
        self.e_task_max = num_devices * cfg.STEPS_PER_EPISODE * 1.0

        # scheduler
        self.scheduler = AggScheduler(cfg)

        # scheduler state tracking
        self.ema_reward          = 0.0
        self.prev_ema_reward     = 0.0
        self.delta_r_smooth      = 0.0
        self.cum_task_energy     = 0.0   # accumulated since last edge agg
        self.prev_mean_channel   = 0.5
        self.episodes_since_edge = 0
        self.episodes_since_cloud = 0

        # edge models from most recent edge aggregation (needed for cloud agg)
        self.edge_models = [{} for _ in range(num_edges)]

        # logging
        self.total_comm_energy = 0.0
        self.reward_history    = []
        self.comm_history      = []
        self.agg_log           = []   # per-episode scheduler action

    # =========================================================
    # Init helpers
    # =========================================================

    def _compute_fl_shares(self, cfg):
        """Amortized FL upload energy per device per episode."""
        shares = [0.0] * self.num_devices
        for edge_id, device_ids in enumerate(self.edge_groups):
            for d_id in device_ids:
                dist = np.linalg.norm(
                    self.topology.device_pos[d_id] - self.topology.edge_pos[edge_id]
                )
                e_upload = self.wireless.energy(self.model_bits, dist, is_edge=False)
                shares[d_id] = e_upload / max(cfg.EDGE_AGG_INTERVAL, 1)
        return shares

    def _compute_e_comm_max(self, cfg):
        """Worst-case communication energy for one cloud aggregation event."""
        e_edge = self.num_devices * self.wireless.energy(
            self.model_bits, cfg.COMM_RANGE, is_edge=False
        )
        max_edge_cloud_dist = max(
            np.linalg.norm(ep - self.topology.cloud_pos)
            for ep in self.topology.edge_pos
        )
        e_cloud = self.num_edges * self.wireless.energy(
            self.model_bits, max_edge_cloud_dist, is_edge=True
        )
        return e_edge + e_cloud

    def _build_patga_trees(self, cfg):
        trees = []
        for edge_id, device_ids in enumerate(self.edge_groups):
            d_max = cfg.D_MAX
            for attempt in range(40):
                try:
                    tree = self.optimizer.build_tree(
                        edge_id, device_ids, self.model_bits,
                        D_max=d_max, comm_range=cfg.COMM_RANGE,
                    )
                    if attempt > 0:
                        print(f"[AAS-HFL PATGA] Edge {edge_id} feasible at D_max={d_max:.4e}s "
                              f"(relaxed {attempt}x)")
                    trees.append(tree)
                    break
                except ValueError:
                    d_max *= 2.0
            else:
                raise RuntimeError(
                    f"[AAS-HFL PATGA] Edge {edge_id}: no feasible D_max after 40 doublings."
                )
        return trees

    # =========================================================
    # Scheduler state construction
    # =========================================================

    def _build_scheduler_state(self, cfg):
        delta  = float(np.clip(self.delta_r_smooth, -1.0, 1.0))
        e_norm = float(np.clip(self.cum_task_energy / (self.e_task_max + 1e-9), 0.0, 1.0))
        c_t    = float(self.prev_mean_channel)
        tau_e  = float(min(self.episodes_since_edge  / max(cfg.EDGE_AGG_INTERVAL, 1),  2.0))
        tau_c  = float(min(self.episodes_since_cloud / max(cfg.CLOUD_AGG_INTERVAL, 1), 2.0))
        return [delta, e_norm, c_t, tau_e, tau_c]

    # =========================================================
    # PATGA aggregation (reused from HierarchicalFLTrainer)
    # =========================================================

    def _aggregate_subtree(self, node, tree):
        own_sd   = self.agents[node].online_net.state_dict()
        children = tree['children'].get(node, [])
        if not children:
            return own_sd
        child_weights = [self._aggregate_subtree(c, tree) for c in children]
        all_weights   = [own_sd] + child_weights
        result = {}
        for key in own_sd.keys():
            result[key] = torch.stack([w[key] for w in all_weights]).mean(0)
        return result

    def _edge_aggregate_tree(self, edge_id):
        tree          = self.edge_trees[edge_id]
        edge_children = tree['children'].get('edge', [])
        if not edge_children:
            return {}
        subtree_weights = [self._aggregate_subtree(c, tree) for c in edge_children]
        if len(subtree_weights) == 1:
            return subtree_weights[0]
        result = {}
        for key in subtree_weights[0].keys():
            result[key] = torch.stack([w[key] for w in subtree_weights]).mean(0)
        return result

    def _tree_comm_energy(self, edge_id):
        tree   = self.edge_trees[edge_id]
        energy = 0.0
        for d_id in self.edge_groups[edge_id]:
            par = tree['parent'].get(d_id)
            if par is None:
                continue
            pos_d = self.topology.device_pos[d_id]
            pos_p = (self.topology.edge_pos[edge_id]
                     if par == 'edge' else self.topology.device_pos[par])
            dist  = np.linalg.norm(pos_d - pos_p)
            energy += self.wireless.energy(self.model_bits, dist, is_edge=False)
        return energy

    # =========================================================
    # Aggregation actions
    # =========================================================

    def _do_edge_aggregation(self):
        """Execute device→edge PATGA aggregation. Returns energy cost (J)."""
        energy = 0.0
        for edge_id in range(self.num_edges):
            energy += self._tree_comm_energy(edge_id)
            self.edge_models[edge_id] = self._edge_aggregate_tree(edge_id)
        return energy

    def _do_cloud_aggregation(self):
        """Execute edge→cloud aggregation. Returns energy cost (J)."""
        valid = [em for em in self.edge_models if em]
        if not valid:
            return 0.0

        energy = 0.0
        for edge_pos in self.topology.edge_pos:
            dist = np.linalg.norm(edge_pos - self.topology.cloud_pos)
            energy += self.wireless.energy(self.model_bits, dist, is_edge=True)

        global_weights = {}
        for key in valid[0].keys():
            global_weights[key] = torch.stack([em[key] for em in valid]).mean(0)

        for agent in self.agents:
            agent.online_net.load_state_dict(global_weights)
            agent.target_net.load_state_dict(global_weights)

        return energy

    # =========================================================
    # Training loop
    # =========================================================

    def train(self):
        cfg = Config()

        prev_sched_state = self._build_scheduler_state(cfg)

        for episode in range(cfg.EPISODES):

            # ---- per-device local training ----
            episode_rewards  = []
            episode_channels = []
            episode_task_energy = 0.0

            for device_id in range(self.num_devices):
                env    = self.envs[device_id]
                agent  = self.agents[device_id]
                logger = self.loggers[device_id]

                state        = env.reset()
                total_reward = 0.0
                total_delay  = 0.0
                total_energy = 0.0

                for _ in range(cfg.STEPS_PER_EPISODE):
                    action = agent.select_action(state)
                    next_state, reward, delay, energy, done = env.step(state, action)

                    agent.replay_buffer.push(state, action, reward, next_state, done)
                    agent.train_step(cfg.BATCH_SIZE)

                    episode_channels.append(state[2])
                    episode_task_energy += energy
                    state         = next_state
                    total_reward += reward
                    total_delay  += delay
                    total_energy += energy

                    if done:
                        break

                logger.log_episode(total_reward, total_delay, total_energy)
                episode_rewards.append(total_reward)

                if episode % cfg.TARGET_UPDATE == 0:
                    agent.update_target()

            # ---- update EMA reward and compute Δr_smooth ----
            mean_reward = float(np.mean(episode_rewards))
            self.prev_ema_reward = self.ema_reward
            self.ema_reward      = (cfg.EMA_BETA * self.ema_reward
                                    + (1.0 - cfg.EMA_BETA) * mean_reward)
            self.delta_r_smooth  = self.ema_reward - self.prev_ema_reward

            self.cum_task_energy   += episode_task_energy
            self.prev_mean_channel  = float(np.mean(episode_channels))

            # ---- scheduler decides ----
            sched_state = self._build_scheduler_state(cfg)
            action      = self.scheduler.select_action(sched_state)

            # ---- execute aggregation ----
            e_comm_this_step = 0.0

            if action >= 1:
                e_edge = self._do_edge_aggregation()
                e_comm_this_step += e_edge
                self.total_comm_energy += e_edge
                self.cum_task_energy    = 0.0          # reset accumulator
                self.episodes_since_edge = 0

            if action == 2:
                e_cloud = self._do_cloud_aggregation()
                e_comm_this_step += e_cloud
                self.total_comm_energy += e_cloud
                self.episodes_since_cloud = 0
                print(f"[AAS-HFL] ep {episode:4d} | cloud-agg | "
                      f"E_comm={self.total_comm_energy:.3e} J | "
                      f"Δr_smooth={self.delta_r_smooth:+.3f}")
            elif action == 1:
                print(f"[AAS-HFL] ep {episode:4d} | edge-agg  | "
                      f"E_comm={self.total_comm_energy:.3e} J | "
                      f"Δr_smooth={self.delta_r_smooth:+.3f}")

            self.episodes_since_edge  += 1
            self.episodes_since_cloud += 1

            # ---- scheduler reward ----
            e_norm_action = e_comm_this_step / (self.e_comm_max + 1e-12)
            penalty       = max(0.0, -self.delta_r_smooth)
            r_sched       = -e_norm_action - cfg.LAMBDA * penalty

            # ---- build next state, push to scheduler buffer, train ----
            next_sched_state = self._build_scheduler_state(cfg)
            done_sched       = (episode == cfg.EPISODES - 1)

            self.scheduler.replay_buffer.push(
                prev_sched_state, action, r_sched, next_sched_state, done_sched
            )
            self.scheduler.train_step(cfg.BATCH_SIZE)
            self.scheduler.soft_update_target()

            prev_sched_state = next_sched_state

            # ---- logging ----
            self.reward_history.append(mean_reward)
            self.comm_history.append(self.total_comm_energy)
            self.agg_log.append(action)

    # =========================================================
    # Summary
    # =========================================================

    def summarize(self):
        all_rewards, all_delays, all_energies = [], [], []
        for logger in self.loggers:
            stats = logger.get_averages()
            all_rewards.append(stats["avg_reward"])
            all_delays.append(stats["avg_delay"])
            all_energies.append(stats["avg_energy"])

        wait_count  = self.agg_log.count(0)
        edge_count  = self.agg_log.count(1)
        cloud_count = self.agg_log.count(2)

        print("\n=== AAS-HFL Summary ===")
        print(f"Avg Reward:          {sum(all_rewards)/len(all_rewards):.2f}")
        print(f"Avg Delay:           {sum(all_delays)/len(all_delays):.2f}")
        print(f"Avg Energy:          {sum(all_energies)/len(all_energies):.2f}")
        print(f"Total Comm Energy:   {self.total_comm_energy:.4e} J")
        print(f"Scheduler actions:   wait={wait_count}  edge={edge_count}  cloud={cloud_count}")
