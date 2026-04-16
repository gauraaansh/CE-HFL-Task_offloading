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
    D_MAX      = 2.0    # maximum upload delay constraint (seconds)
    COMM_RANGE = 70.0   # max wireless link distance (metres), Paper 2 uses 70m
    
    
