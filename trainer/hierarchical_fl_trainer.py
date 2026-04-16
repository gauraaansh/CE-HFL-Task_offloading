import torch
import numpy as np

from agent.ddqn_agent import DDQNAgent
from env.mec_env import MECEnvironment
from utils.metrics import MetricsLogger
from config.config import Config

from network.hierarchy_optimizer import HierarchyOptimizer
from utils.comm_cost import model_size_bytes
from utils.comm_energy import WirelessModel
from network.topology import Topology


class HierarchicalFLTrainer:
    def __init__(self, num_devices, num_edges):
        self.num_devices = num_devices
        self.num_edges   = num_edges

        cfg = Config()

        # ---------------- Devices ----------------
        self.envs    = [MECEnvironment()                          for _ in range(num_devices)]
        self.agents  = [DDQNAgent(state_dim=4, action_dim=2, cfg=cfg) for _ in range(num_devices)]
        self.loggers = [MetricsLogger()                           for _ in range(num_devices)]

        # ---------------- Model size ----------------
        self.model_bytes = model_size_bytes(self.agents[0].online_net)
        self.model_bits  = self.model_bytes * 8

        # ---------------- Communication tracking ----------------
        self.total_comm_energy = 0.0
        self.reward_history    = []
        self.comm_history      = []

        # ---------------- Wireless physics model ----------------
        self.wireless = WirelessModel()

        # ---------------- Shared topology ----------------
        self.topology = Topology(self.num_devices, self.num_edges)

        # ---------------- Greedy device-to-edge assignment ----------------
        self.optimizer   = HierarchyOptimizer(
            num_devices, num_edges,
            device_positions=self.topology.device_pos,
            edge_positions=self.topology.edge_pos
        )
        self.edge_groups = self.optimizer.build_groups()
        print("Edge groups:", self.edge_groups)

        # ---------------- PATGA aggregation trees ----------------
        self.edge_trees = self._build_patga_trees(cfg)
        for i, tree in enumerate(self.edge_trees):
            print(f"[PATGA] Edge {i} tree_energy={tree['tree_energy']:.4e} J  "
                  f"parent_map={tree['parent']}")

    # =========================================================
    # PATGA tree construction
    # =========================================================
    def _build_patga_trees(self, cfg):
        trees = []
        for edge_id, device_ids in enumerate(self.edge_groups):
            tree = self.optimizer.build_tree(
                edge_id, device_ids, self.model_bits,
                D_max=cfg.D_MAX, comm_range=cfg.COMM_RANGE
            )
            trees.append(tree)
        return trees

    # =========================================================
    # Bottom-up tree aggregation (PATGA)
    # =========================================================
    def _aggregate_subtree(self, node, tree):
        """
        Recursively collect aggregated weights bottom-up through the
        PATGA relay tree.  Leaf nodes contribute their own weights;
        intermediate nodes average their own weights with their
        children's already-aggregated weights.
        """
        own_sd   = self.agents[node].online_net.state_dict()
        children = tree['children'].get(node, [])

        if not children:            # leaf node
            return own_sd

        child_weights = [self._aggregate_subtree(c, tree) for c in children]
        all_weights   = [own_sd] + child_weights

        result = {}
        for key in own_sd.keys():
            result[key] = torch.stack([w[key] for w in all_weights]).mean(0)
        return result

    def edge_aggregate_tree(self, edge_id):
        """
        Bottom-up PATGA aggregation for one edge server.
        Returns the aggregated weight dict for that edge server's group.
        """
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

    # =========================================================
    # Cloud aggregation (unchanged)
    # =========================================================
    def cloud_aggregate(self, edge_weights_list):
        cloud_weights = {}
        for key in edge_weights_list[0].keys():
            cloud_weights[key] = torch.stack(
                [ew[key] for ew in edge_weights_list], dim=0
            ).mean(0)
        return cloud_weights

    # =========================================================
    # Tree-based energy for device→edge tier
    # =========================================================
    def _tree_comm_energy(self, edge_id):
        """
        Sum energy over every link in the PATGA tree for edge_id.
        Each device pays energy for its uplink to its parent (relay or edge).
        """
        tree   = self.edge_trees[edge_id]
        energy = 0.0

        for d_id in self.edge_groups[edge_id]:
            par = tree['parent'].get(d_id)
            if par is None:
                continue
            pos_d = self.topology.device_pos[d_id]
            pos_p = (self.topology.edge_pos[edge_id]
                     if par == 'edge'
                     else self.topology.device_pos[par])
            dist  = np.linalg.norm(pos_d - pos_p)
            energy += self.wireless.energy(self.model_bits, dist, is_edge=False)

        return energy

    # =========================================================
    # Training loop
    # =========================================================
    def train(self):
        cfg = Config()

        for episode in range(cfg.EPISODES):

            # -------------------------------------------------
            # Local training — every device, every episode
            # -------------------------------------------------
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

                    state         = next_state
                    total_reward += reward
                    total_delay  += delay
                    total_energy += energy

                    if done:
                        break

                logger.log_episode(total_reward, total_delay, total_energy)

                if episode % cfg.TARGET_UPDATE == 0:
                    agent.update_target()

            # -------------------------------------------------
            # 1. Device → Edge aggregation (PATGA tree)
            # -------------------------------------------------
            if episode % cfg.EDGE_AGG_INTERVAL == 0 and episode > 0:
                self.edge_models = []

                for edge_id, device_ids in enumerate(self.edge_groups):
                    # energy: sum over all uplinks in the PATGA tree
                    self.total_comm_energy += self._tree_comm_energy(edge_id)

                    # bottom-up weighted aggregation through the tree
                    edge_weights = self.edge_aggregate_tree(edge_id)
                    self.edge_models.append(edge_weights)

            # -------------------------------------------------
            # 2. Edge → Cloud aggregation
            # -------------------------------------------------
            if episode % cfg.CLOUD_AGG_INTERVAL == 0 and episode > 0:
                # filter out empty edge models (safety)
                valid_edge_models = [ew for ew in self.edge_models if ew]

                if valid_edge_models:
                    # energy: each edge server → cloud
                    for edge_pos in self.topology.edge_pos:
                        dist = np.linalg.norm(edge_pos - self.topology.cloud_pos)
                        self.total_comm_energy += self.wireless.energy(
                            self.model_bits, dist, is_edge=True
                        )

                    global_weights = self.cloud_aggregate(valid_edge_models)

                    for agent in self.agents:
                        agent.online_net.load_state_dict(global_weights)
                        agent.target_net.load_state_dict(global_weights)

                    print(f"[HFL] Cloud agg ep {episode} | "
                          f"Cumulative comm energy: {self.total_comm_energy:.2e} J")

            # -------------------------------------------------
            # Logging
            # -------------------------------------------------
            avg_reward = (
                sum(l.episode_rewards[-1] for l in self.loggers) / self.num_devices
            )
            self.reward_history.append(avg_reward)
            self.comm_history.append(self.total_comm_energy)

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

        print("\n=== Hierarchical FL DDQN Summary ===")
        print(f"Avg Reward: {sum(all_rewards)/len(all_rewards):.2f}")
        print(f"Avg Delay:  {sum(all_delays)/len(all_delays):.2f}")
        print(f"Avg Energy: {sum(all_energies)/len(all_energies):.2f}")
