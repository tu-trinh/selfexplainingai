import torch
from torch.utils.data import DataLoader, Dataset
from transformer_constants import *
from transformer_modules import *
import random
import sentencepiece as spm


class DataHolder(Dataset):
    """
    Custom dataset class to store zipped inputs and outputs
    """
    def __init__(self, func_tokens, deriv_tokens_in, deriv_tokens_out):
        super().__init__()
        self.func_data = torch.LongTensor(func_tokens)
        self.deriv_in_data = torch.LongTensor(deriv_tokens_in)
        self.deriv_out_data = torch.LongTensor(deriv_tokens_out)

    def __getitem__(self, idx):
        return self.func_data[idx], self.deriv_in_data[idx], self.deriv_out_data[idx]

    def __len__(self):
        return self.func_data.shape[0]


def process_data(data_file):
    """
    Processes a data file into different data sets for various training and testing purposes
    """
    # Split into train, valid, test
    with open(data_file) as f:
        lines = f.readlines()
    random.shuffle(lines)
    train_size = int(len(lines) * TRAIN_SIZE)
    val_size = int(len(lines) * VALID_SIZE)
    train_data = lines[:train_size]
    val_data = lines[train_size : train_size + val_size]
    test_data = lines[train_size + val_size:]
    with open("./data/train_processed.txt", "w") as f:
        f.writelines(train_data)
    with open("./data/valid_processed.txt", "w") as f:
        f.writelines(val_data)
    with open("./data/test_processed.txt", "w") as f:
        f.writelines(test_data)

    # Make training sets for SentencePiece
    train_funcs, train_derivs = load_file("./data/train_processed.txt")
    valid_funcs, valid_derivs = load_file("./data/valid_processed.txt")
    functions = list(train_funcs) + list(valid_funcs)
    derivatives = list(train_derivs) + list(valid_derivs)
    with open("./data/sp_training_input.txt", "w") as f:
        f.writelines("\n".join(functions))
    with open("./data/sp_training_output.txt", "w") as f:
        f.writelines("\n".join(derivatives))


def load_file(file_path):
    """
    Loads the test file and extracts all functions/derivatives
    """
    data = open(file_path, "r").readlines()
    functions, derivatives = zip(*[line.strip().split("=") for line in data])
    return functions, derivatives


def build_vocab():
    """
    Trains SentencePiece on the math vocabulary
    """
    spm.SentencePieceTrainer.train(
        input = "./data/sp_training_input.txt",
        model_prefix = "input",
        vocab_size = INPUT_VOCAB_SIZE,
        pad_id = PAD_ID,
        bos_id = BOS_ID,
        eos_id = EOS_ID,
        unk_id = UNK_ID,
        user_defined_symbols = [r'[\d.]+', r'[a-zA-Z]{2,}']
    )
    spm.SentencePieceTrainer.train(
        input = "./data/sp_training_output.txt",
        model_prefix = "output",
        vocab_size = OUTPUT_ACTION_SIZE,
        pad_id = PAD_ID,
        bos_id = BOS_ID,
        eos_id = EOS_ID,
        unk_id = UNK_ID,
        user_defined_symbols = [r'[\d.]+', r'[a-zA-Z]{2,}']
    )


def build_data_loader(file_path, func_spp, deriv_spp):
    """
    Returns an iterable over a set of data
    """
    funcs, derivs = load_file(file_path)
    func_tokens = tokenize_input(funcs, func_spp)
    deriv_tokens_in, deriv_tokens_out = tokenize_output(derivs, deriv_spp)
    dataholder = DataHolder(func_tokens, deriv_tokens_in, deriv_tokens_out)
    dataloader = DataLoader(dataholder, batch_size = BATCH_SIZE, shuffle = True)
    return dataloader


def tokenize_input(data, func_spp):
    """
    Tokenize functions with help from SentencePiece
    """
    tokens = []
    for elem in data:
        token = func_spp.EncodeAsIds(elem.strip()) + [EOS_ID]
        tokens.append(standardize_length(token))
    return tokens


def tokenize_output(data, deriv_spp):
    """
    Tokenize derivatives with help from SentencePiece
    """
    in_tokens = []
    out_tokens = []
    for elem in data:
        token = deriv_spp.EncodeAsIds(elem.strip())
        in_tokens.append(standardize_length([BOS_ID] + token))
        out_tokens.append(standardize_length(token + [EOS_ID]))
    return in_tokens, out_tokens


def standardize_length(tokens):
    """
    Ensure sequences are all of same length
    """
    if len(tokens) < MAX_SEQ_LEN:
        tokens += [PAD_ID for _ in range(MAX_SEQ_LEN - len(tokens))]
    else:
        tokens = tokens[:MAX_SEQ_LEN]
    return tokens