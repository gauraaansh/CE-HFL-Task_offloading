import numpy as np

class MECEnvironment:
    def __init__(self, max_queue=10):
        self.max_queue = max_queue

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

        reward = -(delay + energy)

        next_state = [
            np.random.rand(),
            max(queue - 1, 0),
            np.random.rand(),
            battery - energy * 0.01
        ]

        done = next_state[3] <= 0
        return next_state, reward, delay, energy, done

