import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mindgrid.envs.edits import Edits
from mindgrid.builder import make_env
from mindgrid.infrastructure.config_utils import make_config
from mindgrid.infrastructure.basic_utils import format_seconds
from mindgrid.infrastructure.env_constants import ACTION_TO_IDX, COLOR_NAMES, MAX_ROOM_SIZE
from mindgrid.infrastructure.obj_constants import OBJ_NAME_MAPPING

import argparse
import sentencepiece as spm
from sentencepiece import SentencePieceProcessor
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import pickle
import time



"""
Given a trajectory and environment edits, can the agent tell how the observations will edit?
Inputs:
- Initial state (grid + integer direction)
- edits (text)
- Sliding window of observations (grid + integer direction) and actions (integer)
- Queries (text)
Outputs:
- Answers (text)
"""
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



"""
Transformer Modules
"""
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, need_weights = False)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.tensor):
        attn_output = self.self_attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, need_weights = False)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, need_weights = False)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, target: torch.tensor, memory: torch.tensor):
        self_attn_output = self.self_attention(target, target, target)
        target = self.norm1(target + self_attn_output)
        cross_attn_output = self.cross_attention(target, memory, memory)
        target = self.norm2(target + cross_attn_output)
        ff_output = self.feed_forward(target)
        target = self.norm3(target + ff_output)
        return target


class StateEncoder(nn.Module):
    def __init__(self, tensor_channels: int, dir_vocab_size: int, d_model: int):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels = 3, out_channels = d_model, kernel_size = 3, stride = 1, padding = 1)
        self.dir_embedder = nn.Embedding(num_embeddings = dir_vocab_size, embedding_dim = d_model)
        self.fc = nn.Linear(d_model * tensor_channels**2 + d_model, d_model)

    def forward(self, state_tensor: torch.tensor, directions: torch.tensor):
        assert state_tensor.dim() == 4, f"Expected 4D tensor (B, N, N, 3); got {state_tensor.shape}"
        assert directions.dim() == 1, f"Expected 1D tensor (B); got {directions.shape}"
        batch_size = state_tensor.size(0)
        state_features = self.cnn(state_tensor).view(batch_size, -1)
        dir_embedding = self.dir_embedder(directions)
        combined_state = torch.cat((state_features, dir_embedding), dim = -1)
        state_embedding = self.fc(combined_state)
        return state_embedding


class EditEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int):
        super().__init__()
        self.embedder = nn.Embedding(vocab_size, d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads) for _ in range(num_layers)])

    def forward(self, edits: torch.tensor):
        assert edits.dim() == 3, f"Expected 3D tensor (B, C, edit_length); got {edits.shape}"
        edit_embeddings = self.embedder(edits)
        for layer in self.encoder_layers:
            edit_embeddings = layer(edit_embeddings)
        return edit_embeddings


class ObservationEncoder(nn.Module):
    def __init__(self, tensor_channels: int, d_model: int):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels = 3, out_channels = d_model, kernel_size = 3, stride = 1, padding = 1)
        self.fc = nn.Linear(d_model * tensor_channels**2 * 2, d_model)

    def forward(self, observations: torch.tensor):
        assert observations.dim() == 4, f"Expected 4D tensor (B, 2M, 2M, 3); got {observations.shape}"
        batch_size = observations.size(0)
        obs_features = self.cnn(observations).view(batch_size, -1)
        obs_embeddings = self.fc(obs_features)
        return obs_embeddings


class ActionEncoder(nn.Module):
    def __init__(self, action_vocab_size: int, d_model: int):
        super().__init__()
        self.embedder = nn.Embedding(action_vocab_size, d_model)

    def forward(self, actions: torch.tensor):
        assert actions.dim() == 2; f"Expected 2D tensor (B, 2); got {actions.shape}"
        action_embeddings = self.embedder(actions)
        return action_embeddings


class QueryEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int):
        super().__init__()
        self.embedder = nn.Embedding(vocab_size, d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads) for _ in range(num_layers)])

    def forward(self, queries: torch.tensor):
        assert queries.dim() == 3; f"Expected 3D tensor (B, Q, query_length); got {queries.shape}"
        query_embeddings = self.embedder(queries)
        for layer in self.encoder_layers:
            query_embeddings = layer(query_embeddings)
        return query_embeddings
    

class InputTransformer(nn.Module):
    def __init__(self, d_model: int, num_layers: int, num_heads: int):
        super().__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads) for _ in range(num_layers)])

    def forward(self, combined_inputs: torch.tensor):
        assert combined_inputs.dim() == 3, f"Expected 3D tensor (B, something, d_model); got {combined_inputs.shape}"
        for layer in self.encoder_layers:
            combined_inputs = layer(combined_inputs)
        return combined_inputs


class AnswerDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int):
        super().__init__()
        self.embedder = nn.Embedding(vocab_size, d_model)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, memory: torch.tensor, queries: torch.tensor):
        assert memory.dim() == 3, f"Expected 3D tensor (B, something, d_model); got {memory.shape}"
        assert queries.dim() == 3, f"Expected 3D tensor (B, Q, query_length); got {queries.shape}"
        query_embeddings = self.embedder(queries)
        for layer in self.decoder_layers:
            query_embeddings = layer(query_embeddings, memory)
        answers = self.fc(query_embeddings)
        return answers


class BLModel(nn.Module):
    def __init__(self, state_encoder, edit_encoder, obs_encoder, act_encoder, query_encoder, input_trans, ans_decoder):
        super().__init__()
        self.state_encoder = state_encoder
        self.edit_encoder = edit_encoder
        self.obs_encoder = obs_encoder
        self.act_encoder = act_encoder
        self.query_encoder = query_encoder
        self.input_trans = input_trans
        self.ans_decoder = ans_decoder
    
    def forward(self, initial_state, initial_dir, edits, observations, actions, queries):
        state_embedding = self.state_encoder(initial_state, initial_dir)
        edit_embedding = self.edit_encoder(edits)
        obs_embedding = self.obs_encoder(observations)
        act_embedding = self.act_encoder(actions)
        query_embedding = self.query_encoder(queries)
        combined_embeddings = torch.cat((
            state_embedding.unsqueeze(1),  # (B, 1, d_model)
            edit_embedding,  # (B, C, d_model)
            obs_embedding.view(obs_embedding.size(0), -1, obs_embedding.size(-1)),  # (B, 2, d_model)
            act_embedding.view(act_embedding.size(0), -1, act_embedding.size(-1)),  # (B, 2, d_model)
            query_embedding  # (B, Q, d_model)
        ), dim = 1)
        combined_output = self.input_trans(combined_embeddings)
        final_output = self.ans_decoder(combined_output, query_embedding)
        return final_output



"""
Data Processing
"""
class BLDatapoint:
    def __init__(
            self,
            init_state: torch.tensor = None,
            init_dir: int = -1,
            edits: torch.tensor = None,
            observations: torch.tensor = None,
            actions: torch.tensor = None,
            queries: torch.tensor = None,
            answers: torch.tensor = None
        ):
        self.init_state = init_state
        self.init_dir = init_dir
        self.edits = edits
        self.observations = observations
        self.actions = actions
        self.queries = queries
        self.answers = answers


class BLDataset(Dataset):
    def __init__(self):
        self.data = []
    
    def add(self, datapoint: BLDatapoint):
        self.data.append(datapoint)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return item.init_state, item.init_dir, item.edits, item.observations, item.actions, item.queries, item.answers


def load_data(split):
    with open("datasets/worldmodel_listen_data_5000_v2.pickle", "rb") as f:
        data = pickle.load(f)[split]
    with open("datasets/worldmodel_listen_games_5000_v2.pickle", "rb") as f:
        games = pickle.load(f)[split]
    return data, games


def train_spm(file: str, model_name: str, vocab_size: int):
    spm.SentencePieceTrainer.train(input = file, model_prefix = model_name, vocab_size = vocab_size,
                                   pad_id = 0, bos_id = 1, eos_id = 2, unk_id = 3)


def create_vocabularies():
    edit_filename = os.path.join(THIS_DIR, "edits_for_spm.txt")
    query_filename = os.path.join(THIS_DIR, "queries_for_spm.txt")
    answer_filename = os.path.join(THIS_DIR, "answers_for_spm.txt")
    edit_file = open(edit_filename, "w")
    query_file = open(query_filename, "w")
    answer_file = open(answer_filename, "w")

    for split in ["train", "val_in", "val_out"]:
        data, _ = load_data(split)
        for datapoint in data:
            edit_file.write("\n".join(datapoint["edit_descriptions"]) + "\n")
            for query_set in datapoint["queries"]:
                for query in query_set:
                    query_file.write(query["question"].strip() + "\n")
                    answer_file.write(str(query["answer"]).strip() + "\n")
    edit_file.close()
    query_file.close()
    answer_file.close()
    
    train_spm(edit_filename, os.path.join(THIS_DIR, "edits"), min(181, len(COLOR_NAMES) + len(OBJ_NAME_MAPPING) + len(Edits) * 50))
    train_spm(query_filename, os.path.join(THIS_DIR, "queries"), min(115, len(COLOR_NAMES) + len(OBJ_NAME_MAPPING) + 200))
    train_spm(answer_filename, os.path.join(THIS_DIR, "answers"), len(COLOR_NAMES) + len(OBJ_NAME_MAPPING) + MAX_ROOM_SIZE + 20)


def expand_datapoint(datapoint: Dict, gamepoint: Dict, edit_spp: SentencePieceProcessor, query_spp: SentencePieceProcessor, answer_spp: SentencePieceProcessor):
    # Extract out all datapoint information first and encode textual data
    game_config = make_config(config_str = gamepoint["config"])
    observer = game_config.roles.observer
    nonobserver = "human" if observer == "ai" else "ai"
    init_env = make_env(getattr(game_config, nonobserver).world_model)
    init_env_state = init_env.get_state()
    init_state = (torch.tensor(init_env_state.full_obs), init_env_state.agent_dir)
    edits = [torch.tensor(edit_spp.encode_as_ids(edit)) for edit in datapoint["edit_descriptions"]]
    observations = [torch.tensor(obs) for obs in datapoint["partial_obs"]]
    actions = [ACTION_TO_IDX[action] for action in datapoint["actions"]]
    queries, answers = [], []
    for query in datapoint["queries"]:
        this_queries, this_answers = [], []
        for question in query:
            this_queries.append(torch.tensor(query_spp.encode_as_ids(question["question"])))
            this_answers.append(torch.tensor(answer_spp.encode_as_ids(question["answer"])))
        queries.append(torch.stack(this_queries))
        answers.append(torch.stack(this_answers))
    queries = torch.stack(queries)
    answers = torch.stack(answers)
    # Then make sliding window training datapoints
    ret_dicts = []
    for i in range(len(queries)):
        if i == 0:
            ret_obs, ret_acts = torch.tensor([]), torch.tensor([])
        elif i == 1:
            ret_obs, ret_acts = [observations[0]], [actions[0]]
        else:
            ret_obs = [observations[i - 2], observations[i - 1]]
            ret_acts = [actions[i - 2], actions[i - 1]]
        ret_dicts.append({
            "initial_state": init_state,
            "edits": edits,
            "observations": ret_obs,
            "actions": ret_acts,
            "queries": queries[i],
            "answers": answers[i]
        })
    return ret_dicts


def pad_truncate_tensor(tensors: torch.tensor, max_shape: Tuple):
    reshaped_tensors = []
    for t in tensors:
        pad_dims = [(0, max(0, max_shape[i] - t.shape[i])) for i in range(len(t.shape) - 1, -1, -1)]
        padded_t = F.pad(t, pad_dims)
        truncated_t = padded_t[tuple(slice(0, max_shape[i]) for i in range(len(padded_t.shape)))]
        reshaped_tensors.append(truncated_t)
    return torch.stack(reshaped_tensors)


def preprocess_data(dataset: BLDataset, split: str,
                    max_initial_state_shape: int = None, max_observation_shape: int = None,
                    max_num_edits: int = None, max_num_queries: int = None,
                    edit_length: int = None, query_length: int = None, answer_length: int = None):
    # Extract all data
    initial_states = []  # (N, N, 3) + int
    edits = []  # text ids
    observations = []  # (M, M, 3) + int
    actions = []  # int
    queries = []  # text ids
    answers = []  # text ids
    if split == "train":
        max_num_edits = 0
        max_num_queries = 0
    data_logs, edit_logs, obs_logs, query_logs = [], [], [], []

    data, games = load_data(split)
    edit_spp = spm.SentencePieceProcessor()
    edit_spp.load(os.path.join(THIS_DIR, "edits.model"))
    query_spp = spm.SentencePieceProcessor()
    query_spp.load(os.path.join(THIS_DIR, "queries.model"))
    answer_spp = spm.SentencePieceProcessor()
    answer_spp.load(os.path.join(THIS_DIR, "answers.model"))
    for i, (datapoint, gamepoint) in enumerate(zip(data, games)):
        expanded_datapoint = expand_datapoint(datapoint, gamepoint, edit_spp, query_spp, answer_spp)
        data_logs.append(len(expanded_datapoint))
        obs_logs[i] = {}
        query_logs[i] = {}
        for j, training_dict in enumerate(expanded_datapoint):
            if j == 0:
                initial_states.append(training_dict["initial_state"])
                edits_length = len(edits)
                edits.extend(training_dict["edits"])
                edit_logs[i] = list(range(edits_length, edits_length + len(training_dict["edits"])))
                if split == "train":
                    max_num_edits = max(max_num_edits, len(training_dict["edits"]))
            else:
                observations_length = len(observations)
                observations.append(training_dict["observations"][-1])
                actions.append(training_dict["actions"][-1])
                if j == 1:
                    obs_logs[i][j] = [observations_length]
                else:
                    obs_logs[i][j] = [observations_length - 1, observations_length]
                queries_length = len(queries)
                queries.extend(training_dict["queries"])
                answers.extend(training_dict["answers"])
                query_logs[i][j] = list(range(queries_length, queries_length + len(training_dict["queries"])))
                if split == "train":
                    max_num_queries = max(max_num_queries, len(training_dict["queries"]))
    
    # Pad/truncate necessary data objects
    if split == "train":
        max_initial_state_shape = tuple(max([state[0].shape[i] for state in initial_states]) for i in range(3))
    initial_state_grids = pad_truncate_tensor([state[0] for state in initial_states], max_initial_state_shape)
    initial_state_directions = torch.stack([state[1] for state in initial_states])
    assert len(initial_state_grids) == len(data), f"`initial_state_grids` has length {len(initial_state_grids)}, data has length {len(data)}"
    assert len(initial_state_directions) == len(data), f"`initial_state_directions` has length {len(initial_state_directions)}, data has length {len(data)}"

    if split == "train":
        max_observation_shape = tuple(max([obs.shape[i] for obs in observations]) for i in range(3))
    prev_obs_length = len(observations)
    observations = pad_truncate_tensor(observations, max_observation_shape)
    assert len(observations) == prev_obs_length, f"messed up padding observations: {prev_obs_length} vs. {observations.shape}"

    prev_edits_length = len(edits)
    edits = pad_sequence(edits, batch_first = True, padding_value = 0)
    assert len(edits) == prev_edits_length, f"messed up padding edits: {prev_edits_length} vs. {edits.shape}"
    prev_queries_length = len(queries)
    queries = pad_sequence(queries, batch_first = True, padding_value = 0)
    assert len(queries) == prev_queries_length, f"messed up padding queries: {prev_queries_length} vs. {queries.shape}"
    prev_answers_length = len(answers)
    answers = pad_sequence(answers, batch_first = True, padding_value = 0)
    assert len(answers) == prev_answers_length, f"messed up padding answers: {prev_answers_length} vs. {answers.shape}"

    # Regroup data back into their original Datapoints and put into the Dataset
    null_observations = pad_truncate_tensor([torch.zeros((1, 1, 3)), torch.zeros((1, 1, 3))], max_observation_shape)
    assert null_observations[0].shape == max_observation_shape, f"messed up padding null observations: {null_observations[0].shape} vs. {max_observation_shape}"
    cat_obs_shape = None
    for i in range(len(data)):
        for j in range(len(data_logs[i])):
            bl_datapoint = BLDatapoint()
            bl_datapoint.init_state = initial_state_grids[i]
            bl_datapoint.init_dir = initial_state_directions[i]
            bl_datapoint.edits = torch.stack(edits[k] for k in edit_logs[i])
            bl_datapoint.edits = F.pad(bl_datapoint.edits, (0, 0, 0, max_num_edits - bl_datapoint.edits.size(0)))
            if j == 0:
                bl_datapoint.observations = torch.cat([null_observations[0], null_observations[1]])
                if not cat_obs_shape:
                    cat_obs_shape = bl_datapoint.observations.shape
                bl_datapoint.actions = torch.tensor([-1, -1])
            elif j == 1:
                bl_datapoint.observations = torch.cat([null_observations[0], observations[obs_logs[i][j][0]]])
                bl_datapoint.actions = torch.tensor([-1, actions[obs_logs[i][j][0]]])
            else:
                bl_datapoint.observations = torch.cat([observations[obs_logs[i][j][0]], observations[obs_logs[i][j][1]]])
                bl_datapoint.actions = torch.tensor([actions[obs_logs[i][j][0]], actions[obs_logs[i][j][0]]])
            assert bl_datapoint.observations.shape == (cat_obs_shape[0], *max_observation_shape[1:]), f"oops! datapoint obs is shaped {bl_datapoint.observations.shape} and we want {(cat_obs_shape[0], *max_observation_shape[1:])}"
            assert bl_datapoint.actions.shape == (2,), f"oops! datapoint act is shaped {bl_datapoint.actions.shape}"
            bl_datapoint.queries = torch.stack(queries[k] for k in query_logs[i][j])
            bl_datapoint.queries = F.pad(bl_datapoint.queries, (0, 0, 0, max_num_queries - bl_datapoint.queries.size(0)))
            bl_datapoint.answers = torch.stack(answers[k] for k in query_logs[i][j])
            bl_datapoint.answers = F.pad(bl_datapoint.answers, (0, 0, 0, max_num_queries - bl_datapoint.answers.size(0)))
            dataset.add(bl_datapoint)
    
    if split == "train":
        edit_length = len(edits[0])
        query_length = len(queries[0])
        answer_length = len(answers[0])
    
    return (
        max_initial_state_shape, cat_obs_shape,
        max_num_edits, max_num_queries,
        edit_length, query_length, answer_length,
        edit_spp.get_piece_size(), query_spp.get_piece_size(), answer_spp.get_piece_size()
    )



"""
Training
"""
class Wrapper:
    def __init__(self, d_model: int, num_layers: int, num_heads: int, num_epochs: int = None, save_every: int = None, train_mode: bool = True):
        self.train_dataset = BLDataset()
        (
            self.state_shape, self.obs_shape,
            self.max_edits, self.max_queries,
            self.edit_length, self.query_length, self.ans_length,
            self.edit_vocab_size, self.query_vocab_size, self.answer_vocab_size
        ) = preprocess_data(self.train_dataset, "train")
        assert len(self.train_dataset) != 0, "Dataset is empty!!!"
        self.train_dataloader = DataLoader(self.train_dataset, batch_size = 4, shuffle = True)
        
        if train_mode:
            self.val_in_dataset = BLDataset()
            preprocess_data(self.val_in_dataset, "test_in", self.state_shape, self.obs_shape, self.max_edits, self.max_queries, self.edit_length, self.query_length, self.ans_length)
            self.val_out_dataset = BLDataset()
            preprocess_data(self.val_out_dataset, "test_out", self.state_shape, self.obs_shape, self.max_edits, self.max_queries, self.edit_length, self.query_length, self.ans_length)
            self.val_dataloaders = {
                "in": DataLoader(self.val_in_dataset, batch_size = 1, shuffle = True),
                "out": DataLoader(self.val_out_dataset, batch_size = 1, shuffle = True)
            }
        else:
            self.test_in_dataset = BLDataset()
            preprocess_data(self.test_in_dataset, "test_in", self.state_shape, self.obs_shape, self.max_edits, self.max_queries, self.edit_length, self.query_length, self.ans_length)
            self.test_out_dataset = BLDataset()
            preprocess_data(self.test_out_dataset, "test_out", self.state_shape, self.obs_shape, self.max_edits, self.max_queries, self.edit_length, self.query_length, self.ans_length)
            self.test_dataloaders = {
                "in": DataLoader(self.test_in_dataset, batch_size = 1, shuffle = True),
                "out": DataLoader(self.test_out_dataset, batch_size = 1, shuffle = True)
            }

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.model = BLModel(
            StateEncoder(self.state_shape, 4, self.d_model),
            EditEncoder(self.edit_vocab_size, self.d_model, self.num_layers, self.num_heads),
            ObservationEncoder(self.obs_shape, self.d_model),
            ActionEncoder(7, self.d_model),
            QueryEncoder(self.query_vocab_size, self.d_model, self.num_layers, self.num_heads),
            InputTransformer(self.d_model, self.num_layers, self.num_heads),
            AnswerDecoder(self.answer_vocab_size, self.d_model, self.num_layers, self.num_heads)
        ).to(DEVICE)
        
        if train_mode:
            self.lr = 1e-4
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
            self.num_epochs = num_epochs
            self.save_interval = save_every
            self.val_losses = {"in": float("inf"), "out": float("inf")}
        self.log_file = os.path.join(THIS_DIR, "log.txt")
        self.model_file = os.path.join(THIS_DIR, "model.pt")
    

    def train(self):
        self.model.train()
        for i in range(self.num_epochs):
            losses = []
            start = time.time()
            for batch in self.train_dataloader:
                init_states, init_dirs, edits, observations, actions, queries, answers = [item.to(DEVICE) for item in batch]
                output = self.model(init_states, init_dirs, edits, observations, actions, queries)
                self.optimizer.zero_grad()
                loss = F.cross_entropy(output.view(-1, self.answer_vocab_size), answers.view(-1))
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            end = time.time()
            epoch_loss = np.mean(losses)
            with open(self.log_file, "a") as f:
                f.write(f"Epoch {i} training time: {format_seconds(end - start)}\n")
                f.write(f"Epoch {i} loss: {epoch_loss}\n")
            continue_training = self.validate()
            if continue_training:
                if i % self.save_interval == 0:
                    torch.save(self.model.state_dict(), self.model_file)
            else:
                return
    

    def validate(self):
        self.model.eval()
        losses_plateaued = {"in": False, "out": False}
        with torch.no_grad():
            for split in ["in", "out"]:
                losses = []
                start = time.time()
                for batch in self.val_dataloaders[split]:
                    init_states, init_dirs, edits, observations, actions, queries, answers = [item.to(DEVICE) for item in batch]
                    output = self.model(init_states, init_dirs, edits, observations, actions, queries)
                    loss = F.cross_entropy(output.view(-1, self.answer_vocab_size), answers.view(-1))
                    losses.append(loss.item())
                end = time.time()
                split_loss = np.mean(losses)
                with open(self.log_file, "a") as f:
                    f.write(f"val_{split} evaluation time: {format_seconds(end - start)}\n")
                    f.write(f"val_{split} loss: {split_loss}\n")
                if split_loss < self.val_losses[split]:
                    self.val_losses[split] = split_loss
                else:
                    losses_plateaued[split] = True
        if all(losses_plateaued.values()):
            with open(self.log_file, "a") as f:
                f.write(f"Both val losses have plateaued, stopping training\n")
            return False
        else:
            with open(self.log_file, "a") as f:
                f.write(f"At least one val loss improved, continuing to train\n")
            return True
    

    def evaluate(self):
        self.model.load_state_dict(torch.load(self.model_file))
        self.model.eval()
        with torch.no_grad():
            for split in ["in", "out"]:
                score = []  # +x where x is proportion of questions in a query correct
                start = time.time()
                for batch in self.test_dataloaders[split]:
                    init_states, init_dirs, edits, observations, actions, queries, answers = [item.to(DEVICE) for item in batch]
                    output = self.model(init_states, init_dirs, edits, observations, actions, queries)
                    preds = output.argmax(dim = -1)
                    prop_correct = (preds == answers).sum().item() / len(answers)
                    score.append(prop_correct)
                end = time.time()
                avg_score = np.mean(score)
                with open(self.log_file, "a") as f:
                    f.write(f"test_{split} evaluation time: {format_seconds(end - start)}\n")
                    f.write(f"test_{split} score: {avg_score}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", "-s", action = "store_true", default = False)
    parser.add_argument("--train", "-t", action = "store_true", default = False)
    parser.add_argument("--eval", "-e", action = "store_true", default = False)
    args = parser.parse_args()

    if args.setup:
        create_vocabularies()
    elif args.train:
        wrapper = Wrapper(d_model = 256, num_layers = 4, num_heads = 4, num_epochs = 10, save_every = 1)
        wrapper.train()
    elif args.eval:
        wrapper = Wrapper(d_model = 256, num_layers = 4, num_heads = 4, train_mode = False)
        wrapper.evaluate()
