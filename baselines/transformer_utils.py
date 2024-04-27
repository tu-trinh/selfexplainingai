import sys
sys.path.append("/nas/ucb/tutrinh/selfexplainingai")
sys.path.append("/Users/tutrinh/Work/CHAI/selfexplainingai")

from package.infrastructure.basic_utils import flatten_list

import re
import torch
from torch.utils.data import DataLoader, Dataset
from transformer_constants import *
from transformer_modules import *
import sentencepiece as spm
from sentencepiece import SentencePieceProcessor
import pickle
from typing import List, Tuple, Dict


"""
TRAJECTORY TRANSFORMER UTILS
"""
class TrajectoryDataHolder(Dataset):
    pass  # TODO: add another for the other



"""
OBSERVATION TRANSFORMER UTILS
"""
class ObservationDataHolder(Dataset):
    """
    Custom dataset class to store input observations and output actions
    """
    def __init__(self, state_tokens: List[List[int]], actions: List[int]):
        super().__init__()
        self.state_data = torch.LongTensor(state_tokens)
        self.action_data = torch.LongTensor(actions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.state_data[idx], self.action_data[idx]

    def __len__(self) -> int:
        return self.state_data.shape[0]


def preprocess_obs(obs: str) -> str:
    """
    Takes in a SINGULAR observation and preprocesses it into compressed form
    """
    single_line_obs = " ".join(obs.strip().split("\n"))
    return single_line_obs


def preprocess_obs_list(data: Dict) -> List[str]:
    """
    From a data dictionary, get all observations present in all trajectories, compress them, add task, and flatten into a singular list
    This correctly excludes the final obs
    """
    all_obs = []
    obs_act_seqs = data["traj_fully_obs_text"]
    skills = data["skill"]
    for i, oas in enumerate(obs_act_seqs):
        matches = re.findall(r"Obs \d+: ([\w ():,'\n]*?)\nAct \d+: (\w+)", oas)
        for match in matches:
            all_obs.append(f"Current task: {skills[i]}. Current observation: {preprocess_obs(match[0])}")
    return all_obs


def preprocess_action_list(data: Dict) -> List[int]:
    """
    From a data dictionary, get all actions for each skill and flatten into a singular list
    This should correspond directly with the preprocessed obs list as (s, a) pairs
    """
    all_actions = flatten_list(data["actions"])
    return all_actions


"""
SENTENCEPIECE UTILS
"""
def process_data_for_spm(data_file: str, mode: str) -> None:
    """
    Processes a data file into SentencePiece-formatted datasets
    """
    with open(data_file, "rb") as f:
        dataset = pickle.load(f)
    training_data = dataset["train"]
    validation_data = dataset["val"]
    
    # Smush train and valid observations together for vocab-building
    if mode == "obs":
        with open(SPM_OBS_TRAINING_FILE, "w") as f:
            for ds in [training_data, validation_data]:
                obs_list = preprocess_obs_list(ds)
                for obs in obs_list:
                    f.write(obs + "\n")
    # TODO: add another for the other


def build_vocab(mode: str) -> None:
    """
    Trains SentencePiece on the input vocabularies
    """
    if mode == "obs":
        spm.SentencePieceTrainer.train(
            input = SPM_OBS_TRAINING_FILE,
            model_prefix = SPM_OBS_PREFIX,
            vocab_size = INPUT_OBS_VOCAB_SIZE,
            pad_id = PAD_ID,
            bos_id = BOS_ID,
            eos_id = EOS_ID,
            unk_id = UNK_ID,
            model_type = "bpe"
        )
    # TODO: add another for the other


def tokenize_observations(observations: List[str], obs_spp: SentencePieceProcessor) -> List[List[int]]:
    """
    Tokenize observations with help from SentencePiece
    """
    all_tokens = []
    for obs in observations:
        tokens = obs_spp.EncodeAsIds(obs.strip()) + [EOS_ID]
        all_tokens.append(standardize_length(tokens, "obs"))
    return all_tokens
# TODO: add another for the other


# def tokenize_output(data, deriv_spp):
#     """
#     Tokenize derivatives with help from SentencePiece
#     """
#     in_tokens = []
#     out_tokens = []
#     for elem in data:
#         token = deriv_spp.EncodeAsIds(elem.strip())
#         in_tokens.append(standardize_length([BOS_ID] + token))
#         out_tokens.append(standardize_length(token + [EOS_ID]))
#     return in_tokens, out_tokens


"""
COMMON UTILS
"""
def standardize_length(tokens: List[int], mode: str) -> List[int]:
    """
    Ensure sequences are all of same length
    """
    assert mode in ["obs", "traj"]
    if mode == "obs":
        max_length = MAX_OBS_LEN
    else:
        max_length = MAX_SEQ_LEN
    if len(tokens) < max_length:
        tokens += [PAD_ID for _ in range(max_length - len(tokens))]
    else:
        tokens = tokens[:max_length]
    return tokens


def build_data_loaders(file_path: str, spp: SentencePieceProcessor, mode: str, training: bool) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns iterables over train, val, and test sets
    """
    assert mode in ["obs", "traj"]

    with open(file_path, "rb") as f:
        dataset = pickle.load(f)
    dataloaders = []
    if mode == "obs":
        for key in ["train", "val"] if training else ["test"]:
            obs_list = preprocess_obs_list(dataset[key])
            all_action_ints = preprocess_action_list(dataset[key])
            assert len(obs_list) == len(all_action_ints), f"(s, a) pairs don't match up for {key} set: {len(obs_list)} observations and {len(all_action_ints)} actions"
            all_obs_tokens = tokenize_observations(obs_list, spp)
            dataholder = ObservationDataHolder(all_obs_tokens, all_action_ints)
            dataloader = DataLoader(dataholder, batch_size = BATCH_SIZE, shuffle = True)
            dataloaders.append(dataloader)
    else:
        pass  # TODO: add another for other
    return tuple(dataloaders)
