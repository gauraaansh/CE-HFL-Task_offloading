import numpy as np

class MECEnvironment:
    def __init__(self, max_queue=10, fl_energy_share=0.0, alpha_fl=0.0):
        self.max_queue = max_queue
        # amortized FL upload cost per episode for this device (J)
        self.fl_energy_share = fl_energy_share
        self.alpha_fl = alpha_fl

    def reset(self):
        return [
            np.random.rand(),                 # task size
            np.random.randint(0, self.max_queue),
            np.random.rand(),                 # channel quality
            1.0                               # battery
        ]

    def step(self, state, action):
        task, queue, channel, battery = state

        if action == 0:  # local execution
            delay = task * 0.8
            energy = task * 1.0
        else:            # offload
            delay = task / (channel + 1e-5)
            energy = task * 0.3

        reward = -(delay + energy + self.alpha_fl * self.fl_energy_share)

        next_state = [
            np.random.rand(),
            max(queue - 1, 0),
            np.random.rand(),
            battery - energy * 0.01
        ]

        done = next_state[3] <= 0
        return next_state, reward, delay, energy, done

