import torch
import torch.nn as nn
import torch.optim as optim
import random

from agent.q_network import QNetwork
from agent.replay_buffer import ReplayBuffer

class DDQNAgent:
    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim = action_dim
        self.gamma = cfg.GAMMA
        self.epsilon = cfg.EPSILON_START
        self.epsilon_min = cfg.EPSILON_MIN
        self.epsilon_decay = cfg.EPSILON_DECAY

        self.online_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.online_net.parameters(), lr=cfg.LR
        )

        self.replay_buffer = ReplayBuffer(cfg.BUFFER_SIZE)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state)
        return q_values.argmax(dim=1).item()

    def train_step(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(batch_size)

        q_values = self.online_net(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_actions = self.online_net(next_states).argmax(dim=1)
        next_q_values = self.target_net(next_states)
        next_q_value = next_q_values.gather(
            1, next_actions.unsqueeze(1)
        ).squeeze(1)

        target = rewards + self.gamma * next_q_value * (1 - dones)

        loss = nn.MSELoss()(q_value, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(
            self.epsilon * self.epsilon_decay, self.epsilon_min
        )

    def update_target(self):
        self.target_net.load_state_dict(
            self.online_net.state_dict()
        )

