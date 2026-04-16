"""
Tests for PATGA implementation in network/hierarchy_optimizer.py

Run from project root:
    python -m pytest tests/test_patga.py -v
or
    python tests/test_patga.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from network.hierarchy_optimizer import HierarchyOptimizer
from config.config import Config

# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def make_optimizer(device_pos, edge_pos):
    num_devices = len(device_pos)
    num_edges   = len(edge_pos)
    opt = HierarchyOptimizer(num_devices, num_edges,
                              device_positions=np.array(device_pos, dtype=float),
                              edge_positions=np.array(edge_pos,   dtype=float))
    return opt

def model_bits():
    """Use same model size as real code (~70KB Q-network)."""
    return 70_000 * 8      # ~560_000 bits


# ------------------------------------------------------------------ #
# Test 1: Basic tree structure — all devices appear, valid parents    #
# ------------------------------------------------------------------ #
def test_tree_structure():
    print("\n--- Test 1: Basic tree structure ---")

    # 4 devices clustered near edge server at (50,50)
    device_pos = [[45,45], [55,45], [45,55], [55,55]]
    edge_pos   = [[50,50]]

    opt  = make_optimizer(device_pos, edge_pos)
    tree = opt.build_tree(0, [0,1,2,3], model_bits(), D_max=0.05, comm_range=70.0)

    parent   = tree['parent']
    children = tree['children']

    print(f"  parent map:   {parent}")
    print(f"  children map: {dict(children)}")
    print(f"  path_delay:   {tree['path_delay']}")
    print(f"  tree_energy:  {tree['tree_energy']:.6e} J")

    # every device must have a parent
    for d in [0,1,2,3]:
        assert d in parent, f"Device {d} has no parent"
        assert parent[d] is not None, f"Device {d} parent is None"

    # parents must be either 'edge' or another device
    for d, p in parent.items():
        assert p == 'edge' or p in [0,1,2,3], f"Device {d} has invalid parent {p}"

    # no device can be its own parent
    for d, p in parent.items():
        assert d != p, f"Device {d} is its own parent"

    # tree_energy must be positive
    assert tree['tree_energy'] > 0, "tree_energy should be positive"

    print("  PASSED")


# ------------------------------------------------------------------ #
# Test 2: No cycles                                                   #
# ------------------------------------------------------------------ #
def test_no_cycles():
    print("\n--- Test 2: No cycles in tree, depth bounded by D_max ---")

    device_pos = [[10,10], [20,20], [30,30], [40,40], [50,50]]
    edge_pos   = [[60,60]]

    # tight D_max: ~2 hops max — prevents deep chains
    opt  = make_optimizer(device_pos, edge_pos)
    bits = model_bits()
    # one hop at ~14m ≈ 0.002s, set D_max to allow at most 2 hops
    D_max = 0.006
    tree = opt.build_tree(0, [0,1,2,3,4], bits, D_max=D_max, comm_range=70.0)

    parent = tree['parent']
    print(f"  parent map: {parent}")

    def walk_to_root(start, parent, max_steps=20):
        node    = start
        steps   = 0
        visited = set()
        while node != 'edge':
            assert node not in visited, f"Cycle detected at node {node}"
            visited.add(node)
            node  = parent[node]
            steps += 1
            assert steps <= max_steps, f"Walk from {start} exceeded {max_steps} steps"
        return steps

    max_depth = 0
    for d in [0,1,2,3,4]:
        depth = walk_to_root(d, parent)
        max_depth = max(max_depth, depth)
        print(f"  Device {d} depth = {depth}")

    # with tight D_max, no device should be in a chain longer than ~3 hops
    assert max_depth <= 3, f"Max depth {max_depth} too large — D_max not constraining tree"

    print("  PASSED")


# ------------------------------------------------------------------ #
# Test 3: Delay constraint is respected                               #
# ------------------------------------------------------------------ #
def test_delay_constraint():
    print("\n--- Test 3: All path delays ≤ D_max ---")

    device_pos = [[10,10], [20,20], [50,50], [80,80]]
    edge_pos   = [[60,60]]
    D_max      = 0.025

    opt  = make_optimizer(device_pos, edge_pos)
    tree = opt.build_tree(0, [0,1,2,3], model_bits(), D_max=D_max, comm_range=70.0)

    for d, delay in tree['path_delay'].items():
        print(f"  Device {d}: path_delay = {delay:.6f}s  (D_max={D_max})")
        assert delay <= D_max + 1e-9, \
            f"Device {d} path_delay {delay:.6f}s exceeds D_max {D_max}s"

    print("  PASSED")


# ------------------------------------------------------------------ #
# Test 4: PATGA tree energy ≤ direct (star) energy                   #
# ------------------------------------------------------------------ #
def test_patga_saves_energy():
    print("\n--- Test 4: PATGA tree energy ≤ direct star energy ---")

    # devices spread out, so relay paths should help
    np.random.seed(42)
    device_pos = np.random.rand(8, 2) * 80
    edge_pos   = np.array([[40.0, 40.0]])

    opt = make_optimizer(device_pos, edge_pos)

    # Direct (star): every device sends straight to edge
    cfg = Config()
    bits = model_bits()

    direct_energy = sum(
        opt._link_energy(bits, opt._dist(device_pos[d], edge_pos[0]))
        for d in range(8)
    )

    # PATGA tree
    tree = opt.build_tree(0, list(range(8)), bits, D_max=cfg.D_MAX, comm_range=cfg.COMM_RANGE)
    patga_energy = tree['tree_energy']

    print(f"  Direct (star) energy: {direct_energy:.6e} J")
    print(f"  PATGA tree energy:    {patga_energy:.6e} J")
    print(f"  Saving:               {(1 - patga_energy/direct_energy)*100:.1f}%")

    assert patga_energy <= direct_energy + 1e-12, \
        "PATGA tree energy should not exceed direct star energy"

    print("  PASSED")


# ------------------------------------------------------------------ #
# Test 5: Hard stop raises ValueError when D_max infeasible           #
# ------------------------------------------------------------------ #
def test_infeasible_dmax_raises():
    print("\n--- Test 5: Infeasible D_max raises ValueError ---")

    device_pos = [[0, 0]]   # device far from edge
    edge_pos   = [[99, 99]] # edge server far away
    D_max      = 1e-9       # impossibly tight

    opt = make_optimizer(device_pos, edge_pos)

    try:
        opt.build_tree(0, [0], model_bits(), D_max=D_max, comm_range=200.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Raised ValueError as expected: {e}")

    print("  PASSED")


# ------------------------------------------------------------------ #
# Test 6: Known controlled topology — verify relay is chosen          #
# ------------------------------------------------------------------ #
def test_controlled_relay():
    print("\n--- Test 6: Controlled topology — verify relay chosen when beneficial ---")

    # Edge at (0,0), Device A at (10,0), Device B at (60,0)
    # B→Edge = 60m (expensive), B→A→Edge = 50m + 10m (should be cheaper)
    device_pos = [[10.0, 0.0],   # device 0 (relay candidate A)
                  [60.0, 0.0]]   # device 1 (B, far from edge)
    edge_pos   = [[0.0,  0.0]]

    opt  = make_optimizer(device_pos, edge_pos)
    bits = model_bits()

    # D_max: must allow B→A→Edge path
    # delay(B→A) + delay(A→Edge): both short distances, should be fine
    opt_tree = opt.build_tree(0, [0, 1], bits, D_max=0.05, comm_range=70.0)

    print(f"  parent map:  {opt_tree['parent']}")
    print(f"  path_delay:  {opt_tree['path_delay']}")
    print(f"  tree_energy: {opt_tree['tree_energy']:.6e} J")

    # compute direct energies for comparison
    e_direct_0 = opt._link_energy(bits, 10.0)   # device 0 direct to edge
    e_direct_1 = opt._link_energy(bits, 60.0)   # device 1 direct to edge
    e_relay_1  = opt._link_energy(bits, 50.0)   # device 1 via device 0

    print(f"  Direct energy device 0 (10m): {e_direct_0:.6e} J")
    print(f"  Direct energy device 1 (60m): {e_direct_1:.6e} J")
    print(f"  Relay  energy device 1 (50m): {e_relay_1:.6e} J")

    if e_relay_1 < e_direct_1:
        # relay is cheaper, PATGA should have moved device 1 under device 0
        assert opt_tree['parent'][1] == 0, \
            f"Expected device 1 to relay via device 0, got parent={opt_tree['parent'][1]}"
        print("  Device 1 correctly relays through device 0")
    else:
        print("  Direct cheaper at this distance, direct connection expected")

    print("  PASSED")


# ------------------------------------------------------------------ #
# Test 7: Trainer builds trees without crashing                       #
# ------------------------------------------------------------------ #
def test_trainer_builds():
    print("\n--- Test 7: HierarchicalFLTrainer initialises with PATGA trees ---")
    from trainer.hierarchical_fl_trainer import HierarchicalFLTrainer

    trainer = HierarchicalFLTrainer(num_devices=10, num_edges=2)

    assert len(trainer.edge_trees) == 2, "Should have 2 edge trees"

    for i, tree in enumerate(trainer.edge_trees):
        assert 'parent'      in tree, f"Tree {i} missing 'parent'"
        assert 'children'    in tree, f"Tree {i} missing 'children'"
        assert 'path_delay'  in tree, f"Tree {i} missing 'path_delay'"
        assert 'tree_energy' in tree, f"Tree {i} missing 'tree_energy'"
        print(f"  Edge {i}: {len(trainer.edge_groups[i])} devices, "
              f"tree_energy={tree['tree_energy']:.4e} J, "
              f"parent_map={tree['parent']}")

    print("  PASSED")


# ------------------------------------------------------------------ #
# Test 8: D_max retry logic in trainer                                #
# ------------------------------------------------------------------ #
def test_trainer_dmax_retry():
    print("\n--- Test 8: Trainer retries with doubled D_max on infeasible topology ---")

    # temporarily patch Config.D_MAX to something very tight
    original_dmax = Config.D_MAX
    Config.D_MAX = 1e-9   # impossibly tight — must trigger retry

    try:
        from trainer.hierarchical_fl_trainer import HierarchicalFLTrainer
        trainer = HierarchicalFLTrainer(num_devices=6, num_edges=2)
        # if we get here without exception, retry logic worked
        print(f"  Trainer initialised successfully after D_max retry")
        for i, tree in enumerate(trainer.edge_trees):
            print(f"  Edge {i}: tree_energy={tree['tree_energy']:.4e} J")
        print("  PASSED")
    finally:
        Config.D_MAX = original_dmax   # restore


# ------------------------------------------------------------------ #
# Runner                                                              #
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    tests = [
        test_tree_structure,
        test_no_cycles,
        test_delay_constraint,
        test_patga_saves_energy,
        test_infeasible_dmax_raises,
        test_controlled_relay,
        test_trainer_builds,
        test_trainer_dmax_retry,
    ]

    passed = 0
    failed = 0

    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print('='*40)
