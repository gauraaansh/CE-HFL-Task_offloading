import torch
from agent.ddqn_agent import DDQNAgent
from env.mec_env import MECEnvironment
from utils.metrics import MetricsLogger
from config.config import Config

class FlatFLTrainer:
    def __init__(self, num_devices):
        self.num_devices = num_devices
        self.envs = [MECEnvironment() for _ in range(num_devices)]
        self.agents = [
            DDQNAgent(state_dim=4, action_dim=2, cfg=Config())
            for _ in range(num_devices)
        ]
        self.loggers = [MetricsLogger() for _ in range(num_devices)]

    def fedavg(self):
        global_weights = {}
        for key in self.agents[0].online_net.state_dict().keys():
            global_weights[key] = torch.stack(
                [agent.online_net.state_dict()[key]
                 for agent in self.agents],
                dim=0
            ).mean(dim=0)

        for agent in self.agents:
            agent.online_net.load_state_dict(global_weights)
            agent.target_net.load_state_dict(global_weights)

    def train(self):
        cfg = Config()

        for episode in range(cfg.EPISODES):
            for device_id in range(self.num_devices):
                env = self.envs[device_id]
                agent = self.agents[device_id]
                logger = self.loggers[device_id]

                state = env.reset()
                total_reward = 0
                total_delay = 0
                total_energy = 0

                for step in range(cfg.STEPS_PER_EPISODE):
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

                logger.log_episode(
                    total_reward, total_delay, total_energy
                )

                if episode % cfg.TARGET_UPDATE == 0:
                    agent.update_target()

            # 🔹 FL aggregation step
            if episode % cfg.FL_AGG_INTERVAL == 0 and episode > 0:
                self.fedavg()
                print(f"[Flat FL] Aggregation at episode {episode}")

    def summarize(self):
        all_rewards, all_delays, all_energies = [], [], []

        for logger in self.loggers:
            stats = logger.get_averages()
            all_rewards.append(stats["avg_reward"])
            all_delays.append(stats["avg_delay"])
            all_energies.append(stats["avg_energy"])

        print("\n=== Flat FL DDQN Summary ===")
        print(f"Avg Reward: {sum(all_rewards)/len(all_rewards):.2f}")
        print(f"Avg Delay:  {sum(all_delays)/len(all_delays):.2f}")
        print(f"Avg Energy: {sum(all_energies)/len(all_energies):.2f}")

