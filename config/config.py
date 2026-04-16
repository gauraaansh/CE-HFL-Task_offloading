class Config:
    GAMMA = 0.99
    LR = 1e-3

    EPSILON_START = 1.0
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 0.995

    BUFFER_SIZE = 100000
    BATCH_SIZE = 64

    TARGET_UPDATE = 10
    EPISODES = 500
    STEPS_PER_EPISODE = 50
    FL_AGG_INTERVAL = 10
    
    # ===============================
    # Hierarchical FL time scales
    # ===============================
    EDGE_AGG_INTERVAL  = FL_AGG_INTERVAL   # device → edge (cheap, frequent)
    CLOUD_AGG_INTERVAL = 100               # edge → cloud (expensive, rare)

    # ===============================
    # PATGA parameters (Paper 2)
    # ===============================
    # D_MAX is scaled to our model size (~70KB, ~0.003s per hop at 50m).
    # Paper 2 uses 5MB model; their D_max allows ~3-8 hops. We match that
    # ratio: 4 hops × 0.003s/hop = 0.012s. Without this, PATGA degenerates
    # to MST (deep chains) because the constraint is never binding.
    D_MAX      = 0.012  # maximum upload delay constraint (seconds)
    COMM_RANGE = 70.0   # max wireless link distance (metres), Paper 2 uses 70m

    # ===============================
    # AAS-HFL Scheduler (Phase 2)
    # ===============================
    LAMBDA              = 0.5    # convergence penalty weight in scheduler reward
    ALPHA_FL            = 0.1    # FL energy weight in per-device task reward
    EMA_BETA            = 0.9    # EMA smoothing factor for reward delta
    SCHED_EPSILON_DECAY = 0.997  # slower ε decay (scheduler sees only 1 decision/episode)
    SCHED_TAU           = 0.005  # soft target update coefficient
    SCHED_STATE_DIM     = 5
    SCHED_ACTION_DIM    = 3

