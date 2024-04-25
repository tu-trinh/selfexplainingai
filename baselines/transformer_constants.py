import torch

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

### Tokenization Constants ###
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
INPUT_VOCAB_SIZE = 800
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
MAX_SEQ_LEN = 30
