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
        self.num_edges = num_edges

        cfg = Config()

        # ---------------- Devices ----------------
        self.envs = [MECEnvironment() for _ in range(num_devices)]

        self.agents = [
            DDQNAgent(state_dim=4, action_dim=2, cfg=cfg)
            for _ in range(num_devices)
        ]

        self.loggers = [MetricsLogger() for _ in range(num_devices)]

        # ---------------- Model size ----------------
        self.model_bytes = model_size_bytes(self.agents[0].online_net)

        # ---------------- Communication tracking ----------------
        self.total_comm_energy = 0
        self.reward_history = []
        self.comm_history = []

        # ---------------- Wireless physics model ----------------
        self.wireless = WirelessModel()

        # ---------------- Shared topology (VERY IMPORTANT) ----------------
        self.topology = Topology(self.num_devices, self.num_edges)

        # ---------------- Hierarchical grouping ----------------
        self.optimizer = HierarchyOptimizer(num_devices, num_edges)
        self.edge_groups = self.optimizer.build_groups()

        print("Optimized groups:", self.edge_groups)

    # =========================================================
    # Edge aggregation
    # =========================================================
    def edge_aggregate(self, edge_device_ids):
        edge_weights = {}

        for key in self.agents[0].online_net.state_dict().keys():
            edge_weights[key] = torch.stack(
                [self.agents[i].online_net.state_dict()[key]
                 for i in edge_device_ids],
                dim=0
            ).mean(dim=0)

        return edge_weights

    # =========================================================
    # Cloud aggregation
    # =========================================================
    def cloud_aggregate(self, edge_weights_list):
        cloud_weights = {}

        for key in edge_weights_list[0].keys():
            cloud_weights[key] = torch.stack(
                [edge_weights[key] for edge_weights in edge_weights_list],
                dim=0
            ).mean(dim=0)

        return cloud_weights

    # =========================================================
    # Training loop
    # =========================================================
    def train(self):
        cfg = Config()

        for episode in range(cfg.EPISODES):

            # =================================================
            # Local training
            # =================================================
            for device_id in range(self.num_devices):
                env = self.envs[device_id]
                agent = self.agents[device_id]
                logger = self.loggers[device_id]

                state = env.reset()

                total_reward = 0
                total_delay = 0
                total_energy = 0

                for _ in range(cfg.STEPS_PER_EPISODE):

                    action = agent.select_action(state)

                    next_state, reward, delay, energy, done = env.step(state, action)

                    agent.replay_buffer.push(
                        state, action, reward, next_state, done
                    )

                    agent.train_step(cfg.BATCH_SIZE)

                    state = next_state

                    total_reward += reward
                    total_delay += delay
                    total_energy += energy

                    if done:
                        break

                logger.log_episode(total_reward, total_delay, total_energy)

                if episode % cfg.TARGET_UPDATE == 0:
                    agent.update_target()
            
            
            bits = self.model_bytes * 8
            
            # =================================================
            # 1️⃣ Device → Edge aggregation (cheap, frequent)
            # =================================================
            if episode % cfg.EDGE_AGG_INTERVAL == 0 and episode > 0:
            
                self.edge_models = []
            
                for edge_id, device_ids in enumerate(self.edge_groups):
            
                    edge_pos = self.topology.edge_pos[edge_id]
            
                    # energy: device → edge
                    for d_id in device_ids:
                        device_pos = self.topology.device_pos[d_id]
                        distance = np.linalg.norm(device_pos - edge_pos)
            
                        self.total_comm_energy += self.wireless.energy(
                            bits, distance, is_edge=False
                        )
            
                    edge_weights = self.edge_aggregate(device_ids)
                    self.edge_models.append(edge_weights)
            
            
            # =================================================
            # 2️⃣ Edge → Cloud aggregation (expensive, rare)
            # =================================================
            if episode % cfg.CLOUD_AGG_INTERVAL == 0 and episode > 0:
            
                edge_weights_list = self.edge_models
            
                # energy: edge → cloud
                for edge_pos in self.topology.edge_pos:
                    distance = np.linalg.norm(edge_pos - self.topology.cloud_pos)
            
                    self.total_comm_energy += self.wireless.energy(
                        bits, distance, is_edge=True
                    )
            
                global_weights = self.cloud_aggregate(edge_weights_list)
            
                for agent in self.agents:
                    agent.online_net.load_state_dict(global_weights)
                    agent.target_net.load_state_dict(global_weights)
            
                print(
                    f"[Hierarchical FL] Cloud aggregation | Energy so far: {self.total_comm_energy:.2e} J"
                )
            

            # =================================================
            # Logging
            # =================================================
            avg_reward = (
                sum(l.episode_rewards[-1] for l in self.loggers)
                / self.num_devices
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
