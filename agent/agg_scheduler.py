import torch
from agent.ddqn_agent import DDQNAgent


class AggScheduler(DDQNAgent):
    """
    DDQN-based aggregation scheduler for AAS-HFL.

    Inherits DDQNAgent (state_dim=5, action_dim=3) but differs in two ways:
      1. Soft target updates instead of hard copy (sparse sample rate: 1 decision/episode).
      2. Slower epsilon decay (SCHED_EPSILON_DECAY < EPSILON_DECAY).

    Actions: {0: wait, 1: edge-agg, 2: edge+cloud-agg}

    State: [Δr_smooth, E_norm, avg_channel, τ_e, τ_c]

    Reward (computed externally in AASHFLTrainer):
        r = -E_norm(action) - λ · max(0, -Δr_smooth)
    """

    def __init__(self, cfg):
        super().__init__(
            state_dim=cfg.SCHED_STATE_DIM,
            action_dim=cfg.SCHED_ACTION_DIM,
            cfg=cfg,
        )
        self.epsilon_decay = cfg.SCHED_EPSILON_DECAY
        self.tau = cfg.SCHED_TAU

    def soft_update_target(self):
        for t_param, o_param in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            t_param.data.copy_(
                self.tau * o_param.data + (1.0 - self.tau) * t_param.data
            )
