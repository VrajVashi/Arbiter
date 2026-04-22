# ARBITER Configuration

# --- Curriculum ---
LEVEL_THRESHOLDS = {1: 20.0, 2: 18.0, 3: 15.0, 4: 15.0, 5: 15.0, 6: 15.0, 7: 12.0}
ADVANCE_WINDOW = 30  # episodes to average over

# --- Obfuscation budget per level ---
OBFUSCATION_BUDGET = {1: 0, 2: 0, 3: 0, 4: 3, 5: 5, 6: 5, 7: 5}

# --- Query budget per episode ---
QUERY_BUDGET = 20

# --- Reward constants ---
REWARD_CAUSAL_MAX       = 1.0
REWARD_COUNTERFACTUAL_MAX = 2.0
REWARD_HIGH_CONF_PENALTY  = -0.5
REWARD_TOM_BONUS          = 3.0
REWARD_CHAIN_MULTIPLIER   = 2.0
REWARD_CONSISTENCY_PENALTY = -1.0
REWARD_BUDGET_EFFICIENCY  = 0.3
REWARD_CORRECT_TYPE       = 10.0
REWARD_CORRECT_DEMOGRAPHIC = 5.0
REWARD_CORRECT_ACTION     = 3.0
REWARD_DECOYS_ELIMINATED  = 2.0

# --- Graph size ---
NUM_EXPLICIT_FEATURES = 6
NUM_PROXY_FEATURES    = 3
NUM_HIDDEN_FEATURES   = 2
NUM_DECISIONS         = 45

# --- Defender ---
DEFENDER_ADAPT_EVERY = 50   # episodes between frequency-table updates
