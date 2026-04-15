import numpy as np

class Topology:
    def __init__(self, num_devices, num_edges):
        self.device_pos = np.random.rand(num_devices, 2) * 100
        self.edge_pos = np.random.rand(num_edges, 2) * 100
        self.cloud_pos = np.array([50, 300])  # far away
