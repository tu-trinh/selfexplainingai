import torch
from package.infrastructure.env_constants import COLOR_NAMES, SKILL_PHRASES, MAX_ROOM_SIZE
from package.infrastructure.obj_constants import OBJ_NAME_MAPPING

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

### SPM Constants ###
SPM_OBS_TRAINING_FILE = "baselines/training_observations.txt"
SPM_OBS_PREFIX = "obs"

### Tokenization Constants ###
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
INPUT_OBS_VOCAB_SIZE = len(COLOR_NAMES) + len(OBJ_NAME_MAPPING) + len(SKILL_PHRASES) + 50  # some extra
OUTPUT_ACTION_SIZE = 6

### Model Hyperparameters ###
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
NUM_EPOCHS = 20
TRAIN_SIZE = 0.7
VALID_SIZE = 0.15
TEST_SIZE = 0.15

### Architecture Constants ###
NUM_HEADS = 4
NUM_LAYERS = 4
D_MODEL = 256
D_QKV = 64
D_FF = 256
DROPOUT_RATE = 0.1
MAX_OBS_LEN = (MAX_ROOM_SIZE - 2)**2 + 100  # 100 extra
MAX_SEQ_LEN = 30 * MAX_OBS_LEN + 30  # 30 diff obs-act pairs
