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
    """
    Custom dataset class to store input trajectories and output skill names (two sets)
    """
    def __init__(self, traj_tokens: List[List[int]], expected_skill_tokens: List[List[int]], model_skill_tokens: List[List[int]]):
        super().__init__()
        self.trajectory_data = torch.LongTensor(traj_tokens)
        self.skill_data_in = torch.LongTensor(expected_skill_tokens)
        self.skill_data_out = torch.LongTensor(model_skill_tokens)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.trajectory_data[idx], self.skill_data_in[idx], self.skill_data_out[idx]
    
    def __len__(self) -> int:
        return self.skill_data_in.shape[0]


def preprocess_trajectory_list(data: Dict) -> List[str]:
    """
    From a data dictionary, get complete trajectories (including final obs and all actions) and compress into one string
    """
    all_trajectories = []
    obs_act_seqs = data["traj_fully_obs_text"]
    for i, oas in enumerate(obs_act_seqs):
        matches = re.findall(r"Obs \d+: ([\w ():,'\n]*?)\nAct \d+: (\w+)", oas)
        curr_pair_idx = 0
        compressed_traj = []
        for match in matches:
            compressed_traj.append(f"OBS {curr_pair_idx}: {preprocess_obs(match[0])}. ACT {curr_pair_idx}: {match[1].strip()}.")
            curr_pair_idx += 1
        final_obs = re.search(r"Final obs: ([\w ():,'\n]*)", oas).group(1).strip()
        compressed_traj.append(f"FINAL OBS: {preprocess_obs(final_obs)}")
        all_trajectories.append(" ".join(compressed_traj))
    return all_trajectories


def preprocess_skill_list(data: Dict) -> List[int]:
    """
    From a data dictionary, get all skills and flatten into a singular list (should already be flat)
    This should correspond directly with the preprocessed trajectory list
    """
    all_skills = data["skill"]
    return all_skills



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
    single_line_obs = "; ".join(obs.strip().split("\n"))
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
    else:
        # Smush train and valid trajectories together for vocab-building
        with open(SPM_TRAJ_TRAINING_FILE, "w") as f:
            for ds in [training_data, validation_data]:
                traj_list = preprocess_trajectory_list(ds)
                for traj in traj_list:
                    f.write(traj + "\n")
        # Smush train and valid skill names together for vocab-building
        with open(SPM_SKILL_TRAINING_FILE, "w") as f:
            for ds in [training_data, validation_data]:
                skill_list = preprocess_skill_list(ds)
                for skill in skill_list:
                    f.write(skill + "\n")


def build_vocab(mode: str) -> None:
    """
    Trains SentencePiece on the input vocabularies
    """
    spm.SentencePieceTrainer.train(
        input = SPM_OBS_TRAINING_FILE if mode == "obs" else SPM_TRAJ_TRAINING_FILE,
        model_prefix = SPM_OBS_PREFIX if mode == "obs" else SPM_TRAJ_PREFIX,
        vocab_size = INPUT_OBS_VOCAB_SIZE if mode == "obs" else INPUT_TRAJ_VOCAB_SIZE,
        max_sentence_length = 4192 if mode == "obs" else 19000,  # default vs upper bound on longest seen ¯\_(ツ)_/¯
        pad_id = PAD_ID,
        bos_id = BOS_ID,
        eos_id = EOS_ID,
        unk_id = UNK_ID,
        model_type = "bpe"
    )
    if mode == "traj":
        spm.SentencePieceTrainer.train(
            input = SPM_SKILL_TRAINING_FILE,
            model_prefix = SPM_SKILL_PREFIX,
            vocab_size = OUTPUT_SKILL_VOCAB_SIZE,
            pad_id = PAD_ID,
            bos_id = BOS_ID,
            eos_id = EOS_ID,
            unk_id = UNK_ID,
            model_type = "bpe"
        )


def tokenize_observations(observations: List[str], obs_spp: SentencePieceProcessor) -> List[List[int]]:
    """
    Tokenize observations with help from SentencePiece
    """
    all_tokens = []
    for obs in observations:
        tokens = [BOS_ID] + obs_spp.EncodeAsIds(obs.strip()) + [EOS_ID]
        all_tokens.append(standardize_length(tokens, "obs"))
    return all_tokens


def tokenize_trajectories(trajectories: List[str], traj_spp: SentencePieceProcessor) -> List[List[int]]:
    """
    Tokenize trajectories with help from SentencePiece
    """
    all_tokens = []
    for traj in trajectories:
        tokens = [BOS_ID] + traj_spp.EncodeAsIds(traj.strip()) + [EOS_ID]
        all_tokens.append(standardize_length(tokens, "traj"))
    return all_tokens


def tokenize_skills(skills: List[str], skill_spp: SentencePieceProcessor) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Tokenize skill names with help from SentencePiece
    """
    skills_in = []
    skills_out = []
    for skill in skills:
        tokens = skill_spp.EncodeAsIds(skill.strip())
        skills_in.append(standardize_length([BOS_ID] + tokens, "skill"))
        skills_out.append(standardize_length(tokens + [EOS_ID], "skill"))
    return skills_in, skills_out


"""
COMMON UTILS
"""
def standardize_length(tokens: List[int], mode: str) -> List[int]:
    """
    Ensure sequences are all of same length
    """
    assert mode in ["obs", "traj", "skill"]
    if mode == "obs":
        max_length = MAX_OBS_LEN
    elif mode == "traj":
        max_length = MAX_TRAJ_LEN
    else:
        max_length = MAX_SKILL_LEN
    if len(tokens) < max_length:
        tokens += [PAD_ID for _ in range(max_length - len(tokens))]
    else:
        tokens = tokens[:max_length]
    return tokens


def build_data_loaders(file_path: str, spps: List[SentencePieceProcessor], mode: str, training: bool) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns iterables over train, val, and test sets
    """
    assert mode in ["obs", "traj"]

    with open(file_path, "rb") as f:
        dataset = pickle.load(f)
    dataloaders = []
    if mode == "obs":
        spp = spps[0]
        for key in ["train", "val"] if training else ["test"]:
            obs_list = preprocess_obs_list(dataset[key])
            all_action_ints = preprocess_action_list(dataset[key])
            assert len(obs_list) == len(all_action_ints), f"(s, a) pairs don't match up for {key} set: {len(obs_list)} observations and {len(all_action_ints)} actions"

            all_obs_tokens = tokenize_observations(obs_list, spp)
            dataholder = ObservationDataHolder(all_obs_tokens, all_action_ints)
            dataloader = DataLoader(dataholder, batch_size = BATCH_SIZE[mode], shuffle = True)
            dataloaders.append(dataloader)
    else:
        traj_spp = spps[0]
        skill_spp = spps[1]
        for key in ["train", "val"] if training else ["test"]:
            traj_list = preprocess_trajectory_list(dataset[key])
            skill_list = preprocess_skill_list(dataset[key])
            assert len(traj_list) == len(skill_list), f"Trajectories and skills don't match up for {key} set: {len(traj_list)} trajectories and {len(skill_list)} skills"

            all_traj_tokens = tokenize_trajectories(traj_list, traj_spp)
            in_skill_tokens, out_skill_tokens = tokenize_skills(skill_list, skill_spp)
            dataholder = TrajectoryDataHolder(all_traj_tokens, in_skill_tokens, out_skill_tokens)
            dataloader = DataLoader(dataholder, batch_size = BATCH_SIZE[mode], shuffle = True)
            dataloaders.append(dataloader)
    
    return tuple(dataloaders)
