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
SPM_SKILL_TRAINING_FILE = "baselines/training_skills.txt"
SPM_SKILL_PREFIX = "skill"

### Tokenization Constants ###
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
# For listener task (obs)
INPUT_OBS_VOCAB_SIZE = len(COLOR_NAMES) + len(OBJ_NAME_MAPPING) + len(SKILL_PHRASES)*3 + 90  # some extra
OUTPUT_ACTION_SIZE = 6
# For speaker task (traj)
INPUT_TRAJ_VOCAB_SIZE = len(COLOR_NAMES) + len(OBJ_NAME_MAPPING) + OUTPUT_ACTION_SIZE + 100  # some extra :)
OUTPUT_SKILL_VOCAB_SIZE = len(SKILL_PHRASES)*3 + len(COLOR_NAMES) + len(OBJ_NAME_MAPPING) + 100  # some extra :)

### Model Hyperparameters ###
LEARNING_RATE = 1e-3
BATCH_SIZE = {"obs": 64, "traj": 1}
NUM_EPOCHS = 100

### Architecture Constants ###
NUM_HEADS = {"obs": 4, "traj": 8}
NUM_LAYERS = {"obs": 4, "traj": 8}
D_MODEL = {"obs": 256, "traj": 512}
D_QKV = {"obs": 64, "traj": 64}
D_FF = {"obs": 256, "traj": 512}
DROPOUT_RATE = {"obs": 0.1, "traj": 0.1}
MAX_OBS_LEN = MAX_ROOM_SIZE**2 + 150  # some extra
MAX_TRAJ_LEN = 25 * (MAX_OBS_LEN + 2)  # 25 diff obs-act pairs, longest seen was 24 pairs
MAX_SKILL_LEN = 15  # longest seen
