import numpy as np
import heapq


class HierarchyOptimizer:
    def __init__(self, num_devices, num_edges,
                 device_positions=None, edge_positions=None, area_size=100):

        self.num_devices = num_devices
        self.num_edges   = num_edges

        self.device_positions = device_positions if device_positions is not None \
            else np.random.rand(num_devices, 2) * area_size
        self.edge_positions = edge_positions if edge_positions is not None \
            else np.random.rand(num_edges, 2) * area_size

        self.channel_quality = np.random.rand(num_devices) + 0.1

        # wireless constants — must match WirelessModel in utils/comm_energy.py
        self.B        = 10e6
        self.N0       = 1e-13
        self.alpha    = 3.5
        self.P_device = 0.1

    # ------------------------------------------------------------------ #
    # Wireless helpers                                                     #
    # ------------------------------------------------------------------ #
    def _pos(self, node, edge_id):
        return self.edge_positions[edge_id] if node == 'edge' \
            else self.device_positions[node]

    def _dist(self, a, b):
        return float(np.linalg.norm(a - b))

    def _rate(self, distance):
        h = 1.0 / (distance ** self.alpha + 1e-9)
        return self.B * np.log2(1.0 + self.P_device * h / self.N0)

    def _link_delay(self, model_bits, distance):
        return model_bits / self._rate(distance)

    def _link_energy(self, model_bits, distance):
        R = self._rate(distance)
        return self.P_device * model_bits / R

    # ------------------------------------------------------------------ #
    # Tree helpers                                                         #
    # ------------------------------------------------------------------ #
    def _get_subtree(self, node, children):
        """All nodes in the subtree rooted at node (node itself included)."""
        result = [node]
        for c in children.get(node, []):
            result.extend(self._get_subtree(c, children))
        return result

    # ------------------------------------------------------------------ #
    # Algorithm 1 — PATGA (Paper 2)                                       #
    # ------------------------------------------------------------------ #
    def build_tree(self, edge_id, device_ids, model_bits,
                   D_max=2.0, comm_range=70.0):
        """
        Parameter Aggregation Tree Generation with Energy Cost Optimized
        Under Delay Constraint.  Implements Algorithm 1 from Paper 2.

        Parameters
        ----------
        edge_id     : int   — index into self.edge_positions
        device_ids  : list  — device indices assigned to this edge server
        model_bits  : float — model size in bits (Z in the paper)
        D_max       : float — maximum allowed upload delay (seconds)
        comm_range  : float — max distance for a direct wireless link (metres)

        Returns
        -------
        dict with keys:
            parent      {node: parent_node}    device -> who it sends to
            children    {node: [child_nodes]}
            path_delay  {node: float}          total delay to edge server (s)
            tree_energy float                  total energy for one aggregation (J)
        """
        EDGE = 'edge'

        if not device_ids:
            return {'parent': {}, 'children': {EDGE: []},
                    'path_delay': {}, 'tree_energy': 0.0}

        all_nodes = list(device_ids) + [EDGE]

        # ---- Lines 1-2: build comm graph within comm_range ----
        delay_g  = {n: {} for n in all_nodes}
        energy_g = {n: {} for n in all_nodes}

        for m in all_nodes:
            for n in all_nodes:
                if m == n:
                    continue
                d = self._dist(self._pos(m, edge_id), self._pos(n, edge_id))
                if d <= comm_range:
                    delay_g[m][n]  = self._link_delay(model_bits, d)
                    energy_g[m][n] = self._link_energy(model_bits, d)

        # ---- Lines 3-4: Dijkstra min-delay spanning tree from edge ----
        path_delay = {n: float('inf') for n in all_nodes}
        parent     = {n: None         for n in all_nodes}
        path_delay[EDGE] = 0.0
        pq = [(0.0, EDGE)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > path_delay[u]:
                continue
            for v, w in delay_g[u].items():
                nd = path_delay[u] + w
                if nd < path_delay[v]:
                    path_delay[v] = nd
                    parent[v]     = u
                    heapq.heappush(pq, (nd, v))

        # ---- fallback: device unreachable within comm_range ----
        # force a direct link so the tree is always complete
        for d_id in device_ids:
            if path_delay[d_id] == float('inf'):
                print(f"[PATGA] Device {d_id} unreachable via comm_range "
                      f"{comm_range}m — forced direct link to edge {edge_id}")
                raw_d = self._dist(self.device_positions[d_id],
                                   self.edge_positions[edge_id])
                path_delay[d_id] = self._link_delay(model_bits, raw_d)
                parent[d_id]     = EDGE
                delay_g[d_id][EDGE]  = path_delay[d_id]
                energy_g[d_id][EDGE] = self._link_energy(model_bits, raw_d)
                delay_g[EDGE][d_id]  = path_delay[d_id]
                energy_g[EDGE][d_id] = energy_g[d_id][EDGE]

        # ---- Lines 5-6: feasibility warning ----
        if any(path_delay[d] > D_max for d in device_ids):
            print(f"[PATGA] Warning: min-delay tree exceeds D_max={D_max}s "
                  f"for edge {edge_id} — proceeding anyway")

        # build children dict from parent pointers
        children = {n: [] for n in all_nodes}
        for n in all_nodes:
            if n != EDGE and parent[n] is not None:
                children[parent[n]].append(n)

        # ---- Lines 7-10: initialise improvement phase ----
        marked   = set()   # replaced links — cannot be replaced again
        improved = True

        # ---- Lines 11-24: iterative link replacement ----
        while improved:
            improved  = False
            best_gain = 0.0
            best_old  = None   # (m, old_parent)
            best_new  = None   # (m, new_parent)

            for m in device_ids:
                cur_par = parent[m]
                if (m, cur_par) in marked:
                    continue

                subtree_m         = self._get_subtree(m, children)
                max_subtree_delay = max(path_delay[s] for s in subtree_m)
                old_energy        = energy_g[m].get(cur_par, float('inf'))

                for v in all_nodes:
                    if v == m or v == cur_par:
                        continue
                    # comm range check (link must exist in graph)
                    if v not in delay_g[m]:
                        continue
                    # loop check: v must not be inside m's subtree
                    if v in subtree_m:
                        continue

                    new_delay_m = delay_g[m][v] + path_delay[v]
                    delta       = new_delay_m - path_delay[m]

                    # D_max must hold for every node in m's subtree
                    if max_subtree_delay + delta > D_max:
                        continue

                    gain = old_energy - energy_g[m][v]
                    if gain > best_gain:
                        best_gain = gain
                        best_old  = (m, cur_par)
                        best_new  = (m, v)

            # Lines 20-24: apply the single best replacement this iteration
            if best_old is not None:
                m, old_par = best_old
                _, new_par = best_new

                # shift path delays for m and all its descendants
                delta = (delay_g[m][new_par] + path_delay[new_par]) - path_delay[m]
                for s in self._get_subtree(m, children):
                    path_delay[s] += delta

                # rewire
                children[old_par].remove(m)
                parent[m] = new_par
                children[new_par].append(m)

                marked.add((m, new_par))
                improved = True

        # ---- compute total tree energy (sum of all uplinks) ----
        tree_energy = sum(
            energy_g[m].get(parent[m], 0.0)
            for m in device_ids if parent[m] is not None
        )

        return {
            'parent':      {k: v for k, v in parent.items()     if k != EDGE},
            'children':    children,
            'path_delay':  {k: v for k, v in path_delay.items() if k != EDGE},
            'tree_energy': tree_energy,
        }

    # ------------------------------------------------------------------ #
    # Legacy greedy grouping (kept as comparison baseline)                #
    # ------------------------------------------------------------------ #
    def transmission_cost(self, d, e):
        distance = np.linalg.norm(
            self.device_positions[d] - self.edge_positions[e]
        )
        return distance ** 2 + 1.0 / self.channel_quality[d]

    def build_groups(self):
        """Greedy nearest-edge assignment (PATGA-lite baseline)."""
        groups = [[] for _ in range(self.num_edges)]
        for d in range(self.num_devices):
            costs = [self.transmission_cost(d, e) for e in range(self.num_edges)]
            groups[int(np.argmin(costs))].append(d)
        return groups
