import numpy as np

class HierarchyOptimizer:
    def __init__(self, num_devices, num_edges, device_positions=None, edge_positions=None, area_size=100):
        self.num_devices = num_devices
        self.num_edges = num_edges

        # use positions from Topology if provided, otherwise generate independently
        self.device_positions = device_positions if device_positions is not None \
            else np.random.rand(num_devices, 2) * area_size

        self.edge_positions = edge_positions if edge_positions is not None \
            else np.random.rand(num_edges, 2) * area_size

        # random channel quality per device
        self.channel_quality = np.random.rand(num_devices) + 0.1

    def transmission_cost(self, d, e):
        """
        cost = energy + delay proxy
        similar to Paper 2 objective
        """
        distance = np.linalg.norm(
            self.device_positions[d] - self.edge_positions[e]
        )

        energy_cost = distance ** 2
        delay_cost = 1.0 / self.channel_quality[d]

        return energy_cost + delay_cost

    def build_groups(self):
        """
        assign each device to lowest cost edge
        (PATG-style greedy assignment)
        """
        groups = [[] for _ in range(self.num_edges)]

        for d in range(self.num_devices):
            costs = [
                self.transmission_cost(d, e)
                for e in range(self.num_edges)
            ]

            best_edge = int(np.argmin(costs))
            groups[best_edge].append(d)

        return groups
