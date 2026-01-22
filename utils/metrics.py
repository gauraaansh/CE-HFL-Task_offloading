class MetricsLogger:
    def __init__(self):
        self.episode_rewards = []
        self.episode_delays = []
        self.episode_energies = []

    def log_episode(self, reward, delay, energy):
        self.episode_rewards.append(reward)
        self.episode_delays.append(delay)
        self.episode_energies.append(energy)

    def get_averages(self):
        return {
            "avg_reward": sum(self.episode_rewards) / len(self.episode_rewards),
            "avg_delay": sum(self.episode_delays) / len(self.episode_delays),
            "avg_energy": sum(self.episode_energies) / len(self.episode_energies),
        }

