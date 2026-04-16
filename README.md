# CE-HFL-Offloading: Project Research Document

> **Type:** Workshop Paper
> 
> **Status:** Phase 1 complete — energy & reward comparison done. Phase 2 (novel contribution) in progress.

---

## Table of Contents

1. [What This Project Is](#1-what-this-project-is)
2. [Research Timeline](#2-research-timeline)
3. [The Two Base Papers](#3-the-two-base-papers)
4. [Phase 1: Completed Results](#4-phase-1-completed-results)
5. [The Gap We Are Filling](#5-the-gap-we-are-filling)
6. [Our Novel Contribution](#6-our-novel-contribution)
7. [Why It Is Novel — Literature Evidence](#7-why-it-is-novel--literature-evidence)
8. [System Architecture](#8-system-architecture)
9. [Mathematical Model](#9-mathematical-model)
10. [What Has Been Implemented](#10-what-has-been-implemented)
11. [What Still Needs To Be Done](#11-what-still-needs-to-be-done)
12. [Experiment Plan & Baselines](#12-experiment-plan--baselines)
13. [Expected Results](#13-expected-results)
14. [Paper Outline](#14-paper-outline)
15. [File Structure](#15-file-structure)
16. [References](#16-references)

---

## 1. What This Project Is

This project combines two research papers on energy optimization in Federated Learning (FL) systems deployed over Mobile Edge Computing (MEC) infrastructure. The goal is to produce an independent, novel research contribution suitable for a **workshop paper submission**.

The core question we address:

> *Both prior papers treat FL aggregation timing as a fixed, pre-scheduled interval. Can a reinforcement learning agent learn WHEN to trigger each tier of aggregation — dynamically trading off convergence quality against communication energy cost?*

We answer this by building a **DDQN-based adaptive aggregation scheduler** for two-tier Hierarchical Federated Learning (HFL) in a MEC task offloading system, and showing it reduces total communication energy compared to fixed-interval baselines while maintaining task performance.

---

## 2. Research Timeline

This is the chronological build-up of the project, reflected in git history.

| Date | Commit | What Was Built |
|---|---|---|
| Jan 22, 2026 | `9cab18f` | Everything from scratch — DDQN agent, Q-network, replay buffer, MEC environment, config, all 3 trainers (Independent, Flat FL, HFL). No energy model yet. |
| Feb 5, 2026 | `7bda2fe` | Added PATG-style grouping — `hierarchy_optimizer.py` with greedy min-cost device→edge assignment (energy proxy = distance², delay proxy = 1/channel). |
| Feb 10, 2026 | `fe0c931` | First Flat FL vs. HFL comparison — added `plot_results.py`, `comm_cost.py`, updated `main.py` to run both and generate reward + communication curves side by side. |
| Feb 11, 2026 | `dea16cf` | **Energy comparison between Flat FL and HFL** — created `comm_energy.py` (full `WirelessModel` using Shannon capacity and pathloss), wired it into both trainers to compute and track real physical communication energy per aggregation event. Generated updated curves showing energy difference. |
| Apr 16, 2026 | `ae79117` | Project documentation push — added both PDFs, `network/topology.py`, updated config with `EDGE_AGG_INTERVAL`/`CLOUD_AGG_INTERVAL`, created this README. |

**Key insight from timeline:** The Feb 11 work (energy comparison) was the natural precursor to Option A (FL-aware reward) — the energy model was built specifically to measure the gap between Flat FL and HFL aggregation costs. That infrastructure now forms the foundation of the novel contribution.

---

## 3. The Two Base Papers

### Paper 1 — Task Offloading via Federated RL in MEC

> **Title:** "Task offloading mechanism based on federated reinforcement learning in mobile edge computing"
> **Authors:** Jie Li, Zhiping Yang, Xingwei Wang, Yichao Xia, Shijian Ni (Northeastern University, China)
> **Published:** Digital Communications and Networks, Vol. 9, 2023, pp. 492–504

**What it does:**
- Each mobile device runs a local DDQN that decides per-task: execute locally or offload to MEC server
- All devices periodically sync their DDQN policies via FedAvg (Federated Averaging)
- Reward: `-(α·delay + β·energy)` — joint cost minimization over task execution only
- FL aggregation interval is **fixed** (every N episodes)
- Results: ~19% energy reduction vs. baselines, 10–30% delay reduction

**Key components taken from this paper:**
- DDQN architecture for binary offloading decisions
- MEC environment model (state: task size, queue, channel quality, battery)
- FedAvg aggregation protocol
- Reward formulation structure

**Key limitation:** FL is used only to share the offloading policy. The energy cost of FL communication itself is never fed back into the RL reward. Aggregation timing is fixed, not adaptive.

---

### Paper 2 — Energy Optimization for HFL with Delay Constraint via Node Cooperation

> **Title:** "On Energy Optimization for Hierarchical Federated Learning With Delay Constraint Through Node Cooperation"
> **Authors:** Zhun Li, Manbae Jee, Sadam Mehrabi, Song Guo (IEEE)
> **Published:** IEEE Internet of Things Journal, Vol. 11, No. 9, May 2024

**What it does:**
- Proposes a 3-tier hierarchy: devices → edge servers → cloud
- **PATGA** (Parameter Aggregation Tree Generation with Energy Cost): builds an energy-minimizing aggregation tree under a hard delay constraint `D_max`
  - Phase 1: Build minimum-delay tree via Dijkstra
  - Phase 2: Iteratively replace high-energy links with cheaper ones, staying within `D_max`
- **CE-HFL** (Cost-Efficient HFL): operates the full training loop using the PATGA-constructed topology
- Proves the underlying decision problem is NP-hard (reduction to weighted Steiner tree)
- Results: 22–29% energy reduction vs. HierAVG, CTL, RSTDC baselines

**Key components taken from this paper:**
- Two-tier hierarchical aggregation structure (device→edge, edge→cloud)
- Shannon-capacity based wireless energy model: `E = P * bits / (B * log2(1 + SNR))`
- Distance-based pathloss model: `h = 1 / (d^α)`
- PATGA-inspired greedy device-to-edge grouping (energy + delay cost)
- Separate aggregation intervals for each tier (frequent edge, rare cloud)

**Key limitation:** PATGA optimizes *which path* model updates travel (the aggregation tree topology — which nodes relay, which links are used) but not *when* aggregation happens. The total number of training rounds T is a fixed input to PATGA, and aggregation runs at every round t ∈ T. Energy savings come from path selection, not from reducing aggregation frequency. There is no mechanism to decide whether a given round's aggregation is worth its communication cost, and no adaptation to convergence state or changing channel conditions during training.

---

## 4. Phase 1: Completed Results

### What Was Run

Three training setups were compared over 500 episodes with 10 devices:

| Setup | Aggregation | Energy tracking |
|---|---|---|
| Flat FL | FedAvg, all devices → cloud, every 10 eps | Device→cloud distance via `WirelessModel` |
| Hierarchical FL | Device→edge every 10 eps, edge→cloud every 100 eps | Two-tier energy via `WirelessModel` |

### Key Finding 1 — Communication Energy

HFL accumulates significantly **less** communication energy than Flat FL over the same number of episodes. The reason is structural:
- Flat FL: every device transmits directly to the cloud (long distance, high pathloss) at every aggregation
- HFL: devices only transmit to their nearest edge server (short distance, low energy). The expensive edge→cloud step only happens every 100 episodes.

This empirically validates Paper 2's core claim and establishes the baseline that the Phase 2 novel contribution will improve upon further.

**Output:** `communication_curve.png`

### Key Finding 2 — Reward Convergence

Both Flat FL and HFL converge to similar average task reward, confirming that the hierarchical aggregation does not hurt the quality of the learned offloading policy. The energy saving comes without a convergence penalty.

**Output:** `reward_curve.png`

### What This Motivates

The Phase 1 energy comparison raises a natural follow-up question: **if aggregation timing drives energy cost, why is it fixed?** Both curves show that most of the energy in Flat FL is wasted during episodes where the model had already converged and aggregation added no value. This directly motivates the Phase 2 contribution — an RL agent that learns when aggregation is worth triggering.

---

## 5. The Gap We Are Filling

Both papers share a common assumption that is never questioned:

| | Paper 1 (Li et al., 2023) | Paper 2 (Li et al., 2024) |
|---|---|---|
| **Aggregation timing** | Fixed every N episodes | Fixed — aggregation runs every round T (given as input) |
| **Aggregation path** | Flat (all→cloud) | PATGA optimizes relay tree topology per round |
| **FL cost in RL reward** | Not included | Not applicable (no RL) |
| **Adaptation to channel** | Via offloading decision only | No runtime adaptation |

**Gap 1:** Neither paper treats aggregation timing as a decision variable. Paper 1 uses a hardcoded episode interval. Paper 2's PATGA optimizes which path updates travel but still runs at every training round — it never decides to skip a round. There is no mechanism for the system to say "the model hasn't diverged enough to justify the cost right now, defer this aggregation."

**Gap 2:** In Paper 1, the DDQN reward optimizes only task execution cost. The energy spent uploading model parameters during FL rounds is completely invisible to the RL agent — even though it drains the same device battery and uses the same radio channel.

**Our contribution fills both gaps simultaneously.**

---

## 6. Our Novel Contribution

### Contribution 1 (Main): DDQN-Based Adaptive Aggregation Scheduler

We introduce a lightweight DDQN agent — the **Aggregation Scheduler** — that operates at the episode level and decides at each episode:

```
Action space: { 0: wait (no aggregation),
                1: trigger device→edge aggregation,
                2: trigger device→edge AND edge→cloud aggregation }
```

**Scheduler state:**
```
[
  avg_reward_delta,          # model performance trend (proxy for divergence)
  cumulative_energy_this_window,   # energy spent since last aggregation
  avg_channel_quality,       # current wireless conditions
  episodes_since_last_edge_agg,    # time since last cheap aggregation
  episodes_since_last_cloud_agg    # time since last expensive aggregation
]
```

**Scheduler reward:**
```
r = -comm_energy_this_step - λ * max(0, convergence_drop)
```

Where `comm_energy_this_step` is computed via the same `WirelessModel` used throughout, and `convergence_drop` penalizes the agent if skipping aggregation causes reward regression.

The agent learns: aggregate when it's worth the energy cost; defer when channel conditions make it expensive.

---

### Contribution 2 (Supporting): FL-Aware Task Offloading Reward

We extend the per-device DDQN reward to include the device's amortized share of FL communication energy:

**Original (Paper 1):**
```
r = -(delay + task_energy)
```

**Ours:**
```
r = -(delay + task_energy + α * fl_energy_per_device_per_round)
```

Where `fl_energy_per_device_per_round` is the wireless cost of uploading model parameters to the assigned edge server, computed from `WirelessModel` using the device's position in `Topology`. This makes the device agent aware that aggressive task offloading may deplete battery needed for FL uploads.

---

### The RL Analogy: Same Principle, Different Decision

This contribution deliberately mirrors the structure of Paper 1:

| | Paper 1 DDQN (Task Offloading) | Our DDQN (Aggregation Scheduling) |
|---|---|---|
| **Question answered** | Is it worth sending this task to the edge? | Is it worth aggregating models right now? |
| **State** | [task_size, queue, channel, battery] | [reward_delta, cum_energy, channel, t_edge, t_cloud] |
| **Action** | {local, offload} | {wait, edge-agg, cloud-agg} |
| **Cost vs benefit** | Computation saved vs transmission energy | Convergence improvement vs FL comm energy |

Paper 1's DDQN optimizes *task routing*. Our DDQN optimizes *model synchronization timing*. The same RL framework that learns where to compute now also learns when to synchronize — extending the adaptive decision-making from the task plane to the FL plane.

---

### Combined System Name: **AAS-HFL**
**Adaptive Aggregation Scheduling for Hierarchical Federated Learning in MEC**

---

## 7. Why It Is Novel — Literature Evidence

### What exists and why it is different from ours

#### 1. FedAA (AAAI 2025) — Closest Near-Miss
> He, Chen, Zhang. "FedAA: A Reinforcement Learning Perspective on Adaptive Aggregation for Fair and Robust Federated Learning." AAAI 2025, pp. 17085–17093.
> [arxiv.org/abs/2402.05541](https://arxiv.org/abs/2402.05541)

FedAA uses DDPG to control **aggregation weights** (how much each client contributes). Our work controls **aggregation timing** (when to trigger each tier). FedAA is flat FL, targets fairness/robustness against malicious clients, and has no energy objective, no MEC context, and no hierarchical structure. These are different problems.

#### 2. DDQN / DRL Task Offloading in MEC — Saturated Area
> Multiple papers 2023–2025 including:
> - [Federated DRL for Task Offloading in Vehicular MEC (Journal of Network and Computer Applications, 2024)](https://www.sciencedirect.com/science/article/abs/pii/S1084804524001188)
> - [Federated DRL for Smart Cities, MDPI Sensors 2022](https://www.mdpi.com/1424-8220/22/13/4738)
> - [Federated DRL for Industrial IoT, MDPI Applied Sciences 2023](https://www.mdpi.com/2076-3417/13/11/6708)
> - [Double DQN RL-Based Computational Offloading and Resource Allocation for MEC (MONAMI 2023, Springer)](https://link.springer.com/chapter/10.1007/978-3-031-55471-1_18)

All of these use RL for task offloading decisions. None of them use RL to schedule FL aggregation rounds. None operate in a two-tier hierarchy with the energy of the aggregation tier itself in the reward.

#### 3. Adaptive Aggregation Interval — Analytical, Not RL
> Shiqiang Wang et al. "Adaptive Federated Learning in Resource Constrained Edge Computing Systems." IEEE JSAC, 2019.
> [arxiv.org/abs/1804.05271](https://arxiv.org/abs/1804.05271)

These approaches use analytical control-theory bounds to set aggregation frequency. They are not learned policies, do not operate in HFL, and are not energy-driven.

#### 4. Hierarchical FL Scheduling Papers
> [Optimal Resource Management for HFL over HetNets with Wireless Energy Transfer (IEEE IoT Journal, 2023)](https://arxiv.org/abs/2305.01953)
> [Task Scheduling in Edge Computing with HFL-Cluster (UCC '25, 2025)](https://eprints.whiterose.ac.uk/id/eprint/234705/)

These use RL for **task scheduling** inside clusters (which computational task runs where), not for **FL aggregation scheduling** (when the FL round triggers). The level of decision-making is different.

#### 5. Hierarchical DRL for Task Offloading (Joint)
> [Hierarchical DRL for Joint Task Offloading, MDPI 2025](https://www.mdpi.com/2079-9292/14/24/4816)
> [Latency-Aware Energy-Efficient Task Offloading with DQN, MDPI 2025](https://www.mdpi.com/2079-9292/14/15/3090)

These use hierarchical DRL for offloading in edge-cloud but the "hierarchy" refers to the decision hierarchy of the RL agent (macro/micro actions), not the FL aggregation hierarchy. No FL training is involved.

### The Unique Combination We Claim

No existing paper combines all of:
1. Two-tier HFL (device→edge, edge→cloud with separate intervals)
2. RL agent deciding WHEN each tier aggregates
3. Energy minimization as the optimization objective
4. MEC task offloading context with DDQN per device
5. FL aggregation cost included in per-device RL reward

---

## 8. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLOUD SERVER                         │
│              (Global Model Aggregation)                 │
│          Cloud Agg triggered by Scheduler               │
└──────────────────┬──────────────────┬───────────────────┘
                   │ edge→cloud       │ edge→cloud
           (expensive, rare)    (expensive, rare)
                   │                  │
        ┌──────────┴───┐    ┌─────────┴────┐
        │  EDGE SERVER 0│    │  EDGE SERVER 1│
        │  (Edge Agg)   │    │  (Edge Agg)   │
        │  triggered by │    │  triggered by │
        │  Scheduler    │    │  Scheduler    │
        └──┬──┬──┬──────┘    └──┬──┬──┬─────┘
           │  │  │              │  │  │
      dev→edge (cheap, more frequent)
           │  │  │              │  │  │
        [D0][D1][D2]          [D3][D4][D5...]
     Each device runs:
     - MECEnvironment (local/offload task decisions)
     - DDQNAgent (task offloading policy)
     - Battery model (shared resource: tasks + FL uploads)

                    ┌────────────────────┐
                    │  AGGREGATION       │
                    │  SCHEDULER AGENT   │  ← NEW (our contribution)
                    │  (DDQN)            │
                    │                    │
                    │  State: [reward_Δ, │
                    │  energy, channel,  │
                    │  t_edge, t_cloud]  │
                    │                    │
                    │  Action: {wait,    │
                    │  edge-agg,         │
                    │  cloud-agg}        │
                    └────────────────────┘
```

### Key Design Decisions

- **Shared `WirelessModel`:** The same physics model (Shannon capacity, pathloss exponent α=3.5) is used for computing task offloading channel rates AND FL communication energy. This ensures energy is measured consistently across both planes.
- **Shared `Topology`:** Device positions, edge positions, and cloud position are fixed at init so that the Scheduler learns based on stable spatial structure. Channel quality varies per step (fading).
- **Decoupled agents:** Per-device DDQN learns task decisions. Scheduler DDQN learns aggregation timing. They interact only through the shared reward signal (FL energy flows into device reward via Contribution 2).

---

## 9. Mathematical Model

### Wireless Energy Model (from Paper 2)

Channel gain with pathloss:
```
h(d) = 1 / (d^α + ε)          where α = 3.5 (pathloss exponent)
```

Shannon capacity:
```
R = B * log2(1 + P*h(d) / N0)  where B = 10 MHz, N0 = 1e-13
```

Transmission delay and energy:
```
T_tx = bits / R
E_tx = P * T_tx
```

Device→edge: `P = P_device = 0.1 W`
Edge→cloud: `P = P_edge = 1.0 W`

### Task Execution Model (from Paper 1)

Local execution:
```
delay_local = task_size * 0.8
energy_local = task_size * 1.0
```

Offload to MEC:
```
delay_offload = task_size / (channel + ε)
energy_offload = task_size * 0.3
```

### Original Reward (Paper 1):
```
r = -(delay + energy)
```

### Extended Reward (Our Contribution 2):
```
fl_cost_share = E_tx(bits=model_bytes*8, d=device_to_edge_dist, P=P_device) / agg_interval
r = -(delay + task_energy + α * fl_cost_share)
```

### Scheduler Reward (Our Contribution 1):
```
r_sched = -comm_energy_triggered - λ * max(0, avg_reward_prev - avg_reward_now)
```

Where `λ` balances energy savings against convergence quality. If the agent defers aggregation and reward degrades, it is penalized.

### Total System Energy (what we minimize, Paper 2's objective):
```
E_total = Σ_devices E_tx(device→edge) * num_edge_aggs
        + Σ_edges E_tx(edge→cloud)   * num_cloud_aggs
```

---

## 10. What Has Been Implemented

### Fully Done ✅

| Component | File | Description |
|---|---|---|
| DDQN Agent | `agent/ddqn_agent.py` | Double DQN, epsilon-greedy, replay buffer |
| Q-Network | `agent/q_network.py` | 4→128→128→action_dim MLP |
| Replay Buffer | `agent/replay_buffer.py` | Fixed-capacity deque, batch sampling |
| MEC Environment | `env/mec_env.py` | Local/offload step, reward, battery drain |
| Wireless Energy Model | `utils/comm_energy.py` | Shannon capacity, pathloss, E=P*T |
| Model Size Calculator | `utils/comm_cost.py` | Counts params, converts to bytes |
| Metrics Logger | `utils/metrics.py` | Per-episode reward/delay/energy logging |
| Plot Utilities | `utils/plot_results.py` | Reward curve + comm energy curve |
| Topology | `network/topology.py` | Random 2D placement of devices/edges/cloud |
| Hierarchy Optimizer | `network/hierarchy_optimizer.py` | Greedy device→edge grouping (PATGA-lite) |
| Config | `config/config.py` | All hyperparameters centralized |
| Independent Trainer | `trainer/independent_trainer.py` | No FL baseline (commented out in main) |
| Flat FL Trainer | `trainer/flat_fl_trainer.py` | FedAvg, fixed interval, device→cloud energy |
| Hierarchical FL Trainer | `trainer/hierarchical_fl_trainer.py` | 2-tier agg, edge→cloud energy, fixed intervals |
| Main Entry Point | `main.py` | Runs Flat FL vs HFL, plots results |

### Phase 1 Results Generated ✅
- `reward_curve.png` — Flat FL vs HFL reward comparison
- `communication_curve.png` — Flat FL vs HFL cumulative communication energy

### Partially Implemented ⚠️

| Component | Status | What's Missing |
|---|---|---|
| PATGA | Simplified only | Full delay-constrained iterative link replacement |
| FL energy in reward | Tracked in trainer, not in reward | Need to pass back to `MECEnvironment` |
| Independent baseline | Implemented, commented out | Just needs uncomment in `main.py` |

---

## 11. What Still Needs To Be Done

### Phase 2 — Main Novel Contribution

#### 2a. Aggregation Scheduler Agent
- [ ] Create `agent/agg_scheduler.py` — new DDQN with 5-dim state, 3-action output
- [ ] Define state construction in `HierarchicalFLTrainer`: collect `[reward_delta, cum_energy, avg_channel, t_since_edge, t_since_cloud]`
- [ ] Replace `if episode % EDGE_AGG_INTERVAL == 0` with scheduler decision in `hierarchical_fl_trainer.py`
- [ ] Define scheduler reward: `-comm_energy - λ*convergence_penalty`
- [ ] Train scheduler jointly with per-device DDQNs
- [ ] Add `AAS_HFL_Trainer` to `trainer/` that wraps this

#### 2b. FL-Aware Reward (Supporting Contribution)
- [ ] Pass device-to-edge distance into `MECEnvironment` at init
- [ ] Compute `fl_cost_share` per device using `WirelessModel`
- [ ] Add `α * fl_cost_share` term to `MECEnvironment.step()` reward
- [ ] Add `α` as a tunable hyperparameter in `Config`

#### 2c. Full PATGA Path Optimization (Layer 1 — strengthens paper)
Paper 2's PATGA optimizes *which path* model updates travel. Currently we use a greedy nearest-edge assignment. Full PATGA replaces this with a delay-constrained relay tree that lets intermediate devices cooperate to reduce transmission distance.

- [ ] Implement Phase 1 of PATGA in `HierarchyOptimizer`: build minimum-delay spanning tree via Dijkstra
- [ ] Implement Phase 2: iteratively replace high-energy links with cheaper cooperative relay paths, subject to `D_max`
- [ ] Add `D_max` as a constraint parameter in `Config`
- [ ] Wire PATGA tree structure into energy calculation in `hierarchical_fl_trainer.py`
- [ ] Compare energy: greedy grouping vs. PATGA grouping as an ablation

**Relationship to contribution 1 (scheduler):** These are orthogonal. PATGA optimizes the path taken when aggregation is triggered. The scheduler DDQN optimizes when aggregation is triggered. The full AAS-HFL system uses both: PATGA picks the best path, the scheduler decides whether the round is worth doing at all.

### Phase 3 — Experiments & Ablation

- [ ] Run all 4 baselines: Independent, Flat FL, HFL-fixed, AAS-HFL
- [ ] Ablation: AAS-HFL without FL-aware reward vs. with
- [ ] Sensitivity: vary `λ` (convergence vs. energy tradeoff)
- [ ] Sensitivity: vary number of devices (5, 10, 20)
- [ ] Sensitivity: vary channel variability
- [ ] Multiple seeds (3–5) for statistical validity

### Phase 4 — Paper Writing
- [ ] Introduction: motivate adaptive aggregation
- [ ] Related Work: cite all papers in Section 16
- [ ] System Model: formalize equations from Section 9
- [ ] Algorithm description: pseudocode for AAS-HFL
- [ ] Experiments: plots, tables
- [ ] Conclusion + future work

---

## 12. Experiment Plan & Baselines

### Baselines

| Name | Description | Source |
|---|---|---|
| **Independent DDQN** | Each device trains alone, no FL | Ablation |
| **Flat FL (Fixed)** | FedAvg, all→cloud, every 10 eps | Paper 1 style |
| **HFL-Fixed** | 2-tier, device→edge every 10, edge→cloud every 100 | Paper 2 style |
| **AAS-HFL (ours)** | 2-tier, RL decides when to aggregate each tier | Our contribution |
| **AAS-HFL + FL-reward (ours)** | Above + FL cost in device reward | Full system |

### Metrics

| Metric | Description |
|---|---|
| Average task reward | Learning quality (convergence speed and final value) |
| Cumulative comm energy (J) | Total FL aggregation energy over training |
| Energy per unit convergence | Efficiency metric (energy / reward gained) |
| Number of aggregation events | How often scheduler triggers each tier |

### Hardware / Setup
- 2 GPUs available
- Simulation environment (no real dataset needed — standard for MEC papers)
- MNIST can be added for model quality validation (Paper 2 uses it)
- ~500 episodes per run, 10 devices, 2 edge servers (matching current config)

---

## 13. Expected Results

Based on the system design and existing baseline results:

1. **AAS-HFL vs HFL-Fixed:** The scheduler should reduce total comm energy by deferring cloud aggregations when channel quality is poor or when reward hasn't diverged enough to warrant syncing. Expected: 15–25% comm energy reduction.

2. **AAS-HFL vs Flat FL:** Flat FL pays maximum comm energy (every device→cloud every round). HFL with adaptive scheduling should be significantly cheaper. Expected: 30–50% reduction.

3. **Reward quality:** Adaptive scheduling may slightly slow early convergence (fewer aggregations early = more local drift) but should match or exceed fixed-interval HFL at convergence.

4. **FL-aware reward:** Devices with FL-aware reward should be more conservative on task offloading when battery is low, preserving energy for FL uploads. This should show up as smoother battery curves and fewer episode terminations.

---

## 14. Paper Outline

**Title (draft):** "Adaptive Aggregation Scheduling for Energy-Efficient Hierarchical Federated Learning in Mobile Edge Computing"

**Target:** FL/MEC workshop at INFOCOM, ICC, or similar (4–6 pages)

```
I. Introduction
   - MEC + FL energy challenge
   - Fixed-interval limitation in existing work
   - Our contributions (bullet list)

II. Related Work
   - Federated RL for task offloading [Paper 1, MDPI papers]
   - HFL energy optimization [Paper 2]
   - Adaptive aggregation [FedAA, Wang et al.]
   - Why ours is different

III. System Model
   - Network topology (devices, edges, cloud)
   - Wireless energy model (Shannon capacity)
   - Task execution model
   - FL aggregation model (two-tier)

IV. Problem Formulation
   - Joint minimization: task energy + FL comm energy
   - Subject to: convergence quality, delay feasibility

V. Proposed AAS-HFL Framework
   - Per-device DDQN (task offloading)
   - FL-aware reward extension
   - Aggregation Scheduler DDQN
   - Training procedure (Algorithm 1 pseudocode)

VI. Experiments
   - Setup and baselines
   - Reward convergence comparison
   - Communication energy comparison
   - Ablation study
   - Sensitivity analysis

VII. Conclusion
```

---

## 15. File Structure

```
ce-hfl-offloading/
├── README.md                           ← this file (project doc + GitHub page)
├── main.py                             ← entry point (update for Phase 2)
├── FL TASK.pdf                         ← Paper 1 (Li et al., 2023)
├── On_Energy_Optimization_...pdf       ← Paper 2 (Li et al., 2024)
├── reward_curve.png                    ← Phase 1 result
├── communication_curve.png             ← Phase 1 result
│
├── agent/
│   ├── ddqn_agent.py                   ✅ done
│   ├── q_network.py                    ✅ done
│   ├── replay_buffer.py                ✅ done
│   └── agg_scheduler.py               ← TODO (Phase 2)
│
├── env/
│   └── mec_env.py                      ✅ done (extend reward in Phase 2)
│
├── trainer/
│   ├── independent_trainer.py          ✅ done
│   ├── flat_fl_trainer.py              ✅ done
│   ├── hierarchical_fl_trainer.py      ✅ done
│   ├── aas_hfl_trainer.py             ← TODO (Phase 2, main contribution)
│   └── multi_device_trainer.py         (unused — can delete or repurpose)
│
├── network/
│   ├── topology.py                     ✅ done
│   └── hierarchy_optimizer.py          ✅ done (extend to full PATGA — optional)
│
├── config/
│   └── config.py                       ✅ done (add α, λ, D_max in Phase 2)
│
└── utils/
    ├── comm_energy.py                  ✅ done
    ├── comm_cost.py                    ✅ done
    ├── metrics.py                      ✅ done
    └── plot_results.py                 ✅ done (extend for new baselines)
```

---

## 16. References

### Base Papers
1. Jie Li et al. "Task offloading mechanism based on federated reinforcement learning in mobile edge computing." *Digital Communications and Networks*, Vol. 9, 2023, pp. 492–504. **(Paper 1 — Base)**

2. Zhun Li et al. "On Energy Optimization for Hierarchical Federated Learning With Delay Constraint Through Node Cooperation." *IEEE Internet of Things Journal*, Vol. 11, No. 9, May 2024. **(Paper 2 — Base)** — *PDF verified locally in repo. Not found via online search indexing — verify DOI directly on IEEE Xplore if citing.*

### Closest Existing Work (Why We Are Novel)
3. Jialuo He, Wei Chen, Xiaojin Zhang. "FedAA: A Reinforcement Learning Perspective on Adaptive Aggregation for Fair and Robust Federated Learning." *AAAI 2025*, pp. 17085–17093. https://arxiv.org/abs/2402.05541 — *RL for aggregation weights (not timing), flat FL, fairness goal, no energy, no HFL. Different problem.*

4. Shiqiang Wang, Tiffany Tuor, Theodoros Salonidis, Kin K. Leung et al. "Adaptive Federated Learning in Resource Constrained Edge Computing Systems." *IEEE Journal on Selected Areas in Communications*, Vol. 37, No. 6, 2019. https://arxiv.org/abs/1804.05271 — *Analytical control-theory bound (not RL) for adaptive aggregation interval, flat FL, no energy cost model.*

5. Xu Zhao, Yichuan Wu, Tianhao Zhao, Feiyu Wang, Maozhen Li. "Federated deep reinforcement learning for task offloading and resource allocation in mobile edge computing-assisted vehicular networks." *Journal of Network and Computer Applications*, Vol. 229, 2024. https://www.sciencedirect.com/science/article/abs/pii/S1084804524001188 — *RL for task offloading, no aggregation scheduling.*

6. Chen, Xing, Guizhong Liu. "Federated Deep Reinforcement Learning-Based Task Offloading and Resource Allocation for Smart Cities in a Mobile Edge Network." *MDPI Sensors*, Vol. 22, No. 13, 2022. https://www.mdpi.com/1424-8220/22/13/4738 — *Flat FL + DRL, no HFL, no aggregation scheduling.*

7. "Federated Deep Reinforcement Learning for Energy-Efficient Edge Computing Offloading and Resource Allocation in Industrial Internet." *MDPI Applied Sciences*, Vol. 13, No. 11, 2023. https://www.mdpi.com/2076-3417/13/11/6708 — *Energy in reward but for task execution, not FL rounds.*

8. Hengzhou Ye, Jiaming Li, Junyao Gao, Haoxiang Wen. "A Hierarchical Deep Reinforcement Learning Approach for Joint Dependent Task Offloading and Service Placement in MEC." *MDPI Electronics*, Vol. 14, No. 24, 2025. https://www.mdpi.com/2079-9292/14/24/4816 — *Hierarchical RL for task decisions only, no FL aggregation scheduling.*

9. Amina Benaboura, Rachid Bechar, Walid Kadri, Tu Dac Ho, Zhenni Pan, Shaaban Sahmoud. "Latency-Aware and Energy-Efficient Task Offloading in IoT and Cloud Systems with DQN Learning." *MDPI Electronics*, Vol. 14, No. 15, 2025. https://www.mdpi.com/2079-9292/14/15/3090 — *DQN for offloading, no FL, no aggregation scheduling.*

10. L. Alsalem, K. Djemame. "Task Scheduling in Edge Computing Environments: a Hierarchical Cluster-based Federated Deep Reinforcement Learning Approach." *UCC '25: 18th IEEE/ACM Int. Conf. on Utility and Cloud Computing*, 2025. https://eprints.whiterose.ac.uk/id/eprint/234705/ — *RL for task scheduling inside clusters using FL, not FL aggregation round scheduling.*

11. Rami Hamdi, Ahmed Ben Said, Emna Baccour, Aiman Erbad, Amr Mohamed, Mounir Hamdi, Mohsen Guizani. "Optimal Resource Management for Hierarchical Federated Learning over HetNets with Wireless Energy Transfer." *IEEE Internet of Things Journal*, 2023. https://arxiv.org/abs/2305.01953 — *Resource allocation in HFL using heuristic algorithm (H2RMA), not RL-based aggregation timing.*

---

*Last updated: 2026-04-16 (Paper 2 limitation clarified; RL analogy section added; PATGA TODO expanded)*
*Author: Solo researcher*
