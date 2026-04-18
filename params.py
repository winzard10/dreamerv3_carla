# params.py
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
BATCH_SIZE  = 4
NUM_CLASSES = 28
H, W        = 128, 128

# Model dimensions — must match between train and test
DETER_DIM         = 512
EMBED_DIM         = 1024
STOCH_CATEGORICALS = 32
STOCH_CLASSES      = 32
FREE_NATS           = 0.5

# Phase A
PHASE_A_STEPS = 20000
PHASE_A_PATH  = "checkpoints/world_model/world_model_pretrained.pth"

# Phase B
PHASE_B_STEPS  = 2000
PART_B_EPISODE = 5000
TRAIN_EVERY    = 5
IMAG_HORIZON   = 8 # 15
GAMMA          = 0.97 # 0.99
LAMBDA         = 0.95

# Learning rates
WM_LR     = 8e-5
ACTOR_LR  = 3e-5
CRITIC_LR = 3e-5

# Loss scales
SEM_SCALE       = 10.0
REWARD_SCALE    = 1.0
CONT_SCALE      = 1.0
KL_SCALE        = 1.0   
ENT_SCALE       = 1e-4 # 1e-3
OVERSHOOT_K     = 3
OVERSHOOT_SCALE = 0.1
GOAL_SCALE = 1.0
VEC_SCALE  = 1.0        # velocity vector loss scale

# TwoHot reward distribution
BINS = 255
VMIN = -20.0
VMAX =  20.0

# EMA for target critic
TARGET_EMA = 0.99

# Checkpoints
LOAD_PRETRAINED = True
CKPT_DIR        = "checkpoints/dreamerv3"
CKPT_PATH       = "checkpoints/dreamerv3/dreamerv3_latest.pth"

# Logging
IMAG_LOG_EVERY            = 100
IMAG_LOG_HORIZON          = 10
IMAG_LOG_EXAMPLES         = 4
FIXED_VAL_ENABLED         = True

# =============================================================================
# Evaluation / Test
# =============================================================================
TEST_MODEL         = "dreamerv3_ep400.pth"
TEST_TOWN          = "Town10HD"
TEST_NUM_EPISODES  = 20 
SHOW_RECON         = True
SHOW_SPECTATOR     = True
SHOW_EVERY_N_STEPS = 3


# =============================================================================
# Data Collection
# =============================================================================
SEQ_LEN              = 10    # increase when collecting new data for better prior
COLLECT_SAVE_DIR     = "./data/expert_sequences"
COLLECT_TARGET_STEPS = 50000
COLLECT_LOOKAHEAD    = 3     # waypoints ahead for expert steering
COLLECT_STEER_GAIN   = 0.85
COLLECT_THROTTLE     = 0.4  # maps to 40% throttle in CarlaEnv wrapper
COLLECT_LOG_EVERY    = 100