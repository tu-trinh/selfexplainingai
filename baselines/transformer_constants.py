import sys
sys.path.append("/nas/ucb/tutrinh/selfexplainingai")
sys.path.append("/Users/tutrinh/Work/CHAI/selfexplainingai")

import torch
from package.infrastructure.env_constants import COLOR_NAMES, SKILL_PHRASES, MAX_ROOM_SIZE
from package.infrastructure.obj_constants import OBJ_NAME_MAPPING

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

### SPM Constants ###
SPM_OBS_TRAINING_FILE = "baselines/training_observations.txt"
SPM_OBS_PREFIX = "obs"
SPM_TRAJ_TRAINING_FILE = "baselines/training_trajectories.txt"
SPM_TRAJ_PREFIX = "traj"

### Tokenization Constants ###
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
INPUT_OBS_VOCAB_SIZE = len(COLOR_NAMES) + len(OBJ_NAME_MAPPING) + len(SKILL_PHRASES)*3 + 90  # some extra
OUTPUT_ACTION_SIZE = 6

### Model Hyperparameters ###
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 128

### Architecture Constants ###
NUM_HEADS = {"obs": 4, "traj": 4}
NUM_LAYERS = {"obs": 4, "traj": 4}
D_MODEL = {"obs": 256, "traj": 256}
D_QKV = {"obs": 64, "traj": 64}
D_FF = {"obs": 256, "traj": 256}
DROPOUT_RATE = {"obs": 0.1, "traj": 0.1}
MAX_OBS_LEN = MAX_ROOM_SIZE**2 + 100  # 100 extra
MAX_SEQ_LEN = 30 * MAX_OBS_LEN + 30  # 30 diff obs-act pairs
