import torch
from torch import nn
from transformer_constants import *
import numpy as np
from transformer_modules import *
from tqdm import tqdm
from transformer_utils import *
import argparse
# from torchinfo import summary
import sentencepiece as spm
from typing import Any, List, Tuple, Union


class ModelWrapper:
    def __init__(self, mode: str, training: bool):
        assert mode in ["obs", "traj"]
        self.mode = mode

        # Data initialization
        if mode == "obs":
            self.obs_spp = spm.SentencePieceProcessor()
            self.obs_spp.load(f"baselines/{SPM_OBS_PREFIX}.model")
            dataloaders = build_data_loaders("datasets/intention_datasets.pkl", [self.obs_spp], "obs", training)
            self.train_data_loader = dataloaders[0]
            self.valid_data_loader = dataloaders[1]
        else:
            self.traj_spp = spm.SentencePieceProcessor()
            self.traj_spp.load(f"baselines/{SPM_TRAJ_PREFIX}.model")
            self.skill_spp = spm.SentencePieceProcessor()
            self.skill_spp.load(f"baselines/{SPM_SKILL_PREFIX}.model")
            dataloaders = build_data_loaders("datasets/intention_datasets.pkl", [self.traj_spp, self.skill_spp], "traj", training)
            self.train_data_loader = dataloaders[0]
            self.valid_data_loader = dataloaders[1]

        # Model initialization
        if mode == "obs":
            self.model = ObservationTransformer(INPUT_OBS_VOCAB_SIZE, OUTPUT_ACTION_SIZE).to(DEVICE)
        else:
            self.model = TrajectoryTransformer(INPUT_TRAJ_VOCAB_SIZE, OUTPUT_SKILL_VOCAB_SIZE).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = LEARNING_RATE)
        self.loss_func = nn.CrossEntropyLoss()
        self.best_loss = float("inf")
    
    
    def mask(self, input_data: Any, desired_output: Any = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates masks on model inputs and expected outputs, if needed
        """
        enc_mask, dec_mask = None, None
        enc_mask = (input_data != PAD_ID).unsqueeze(1).to(DEVICE)
        if self.mode == "traj":
            dec_mask = (desired_output != PAD_ID).unsqueeze(1).to(DEVICE)
            helper_mask = torch.ones([1, MAX_SKILL_LEN, MAX_SKILL_LEN], dtype = torch.bool)
            helper_mask = torch.tril(helper_mask).to(DEVICE)
            dec_mask = dec_mask & helper_mask
        return enc_mask, dec_mask

    
    def train_model(self) -> None:
        """
        Runs training process and save model that performs the best on a the validation set
        """
        for epoch in range(NUM_EPOCHS):
            self.model.train()
            losses = []
            for batch in self.train_data_loader:  # NOTE: removed tqdm for now to not clog outputs since no wandb yet
                if self.mode == "obs":
                    states, actions = batch
                    states = states.to(DEVICE)
                    actions = actions.to(DEVICE)
                    enc_mask, _ = self.mask(states)
                    action_dist = self.model(states, enc_mask)
                else:
                    trajs, in_skills, out_skills = batch
                    trajs = trajs.to(DEVICE)
                    in_skills = in_skills.to(DEVICE)
                    out_skills = out_skills.to(DEVICE)
                    enc_mask, dec_mask = self.mask(trajs, in_skills)
                    skill_dist = self.model(trajs, in_skills, enc_mask, dec_mask)

                self.optimizer.zero_grad()
                if self.mode == "obs":
                    loss = self.loss_func(action_dist, actions)
                else:
                    loss = self.loss_func(
                        skill_dist.view(-1, OUTPUT_SKILL_VOCAB_SIZE),
                        out_skills.view(out_skills.shape[0] * out_skills.shape[1])
                    )
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

                if self.mode == "obs":
                    del states, actions, enc_mask, action_dist
                else:
                    del trajs, in_skills, out_skills, enc_mask, dec_mask, skill_dist
                if DEVICE == torch.device("cuda"):
                    torch.cuda.empty_cache()
            
            print("Finished training epoch {} with avg. loss {}".format(epoch + 1, round(np.mean(losses), 3)))
            valid_loss = self.validate_model()
            print("Validation loss:", valid_loss)

            # Check to see if we can save a new best model
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                state_dict = {
                    "model_dict": self.model.state_dict(),
                    "optimizer_dict": self.optimizer.state_dict(),
                    "validation_loss": valid_loss
                }
                chkpt_name = "intention_listener" if self.mode == "obs" else "intention_speaker"
                torch.save(state_dict, f"baselines/{chkpt_name}_checkpoint.tar")
                print("New best validation accuracy achieved, saved to checkpoint")

    
    def validate_model(self) -> float:
        """
        Tests the current model on a validation set
        """
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in self.valid_data_loader:  # NOTE: removed tqdm for now to not clog outputs since no wandb yet
                if self.mode == "obs":
                    states, actions = batch
                    states = states.to(DEVICE)
                    actions = actions.to(DEVICE)
                    enc_mask, _ = self.mask(states)
                    action_dist = self.model(states, enc_mask)
                else:
                    trajs, in_skills, out_skills = batch
                    trajs = trajs.to(DEVICE)
                    in_skills = in_skills.to(DEVICE)
                    out_skills = out_skills.to(DEVICE)
                    enc_mask, dec_mask = self.mask(trajs, in_skills)
                    skill_dist = self.model(trajs, in_skills, enc_mask, dec_mask)

                if self.mode == "obs":
                    loss = self.loss_func(action_dist, actions)
                else:
                    loss = self.loss_func(
                        skill_dist.view(-1, OUTPUT_SKILL_VOCAB_SIZE),
                        out_skills.view(out_skills.shape[0] * out_skills.shape[1])
                    )
                losses.append(loss.item())

                if self.mode == "obs":
                    del states, actions, enc_mask, action_dist
                else:
                    del trajs, in_skills, out_skills, enc_mask, dec_mask, skill_dist
                if DEVICE == torch.device("cuda"):
                    torch.cuda.empty_cache()
        
        valid_loss = np.mean(losses)
        return valid_loss
    
    
    def infer(self, input_data: Union[List[str]]) -> Union[List[int]]:
        """
        Run inference with the model (test time)
        """
        self.model.eval()

        # Process input observation and run through encoder side
        if self.mode == "obs":
            if isinstance(input_data, str):
                input_data = [input_data]

            observations = tokenize_observations(input_data, self.obs_spp)
            observations = torch.LongTensor(observations).to(DEVICE)
            masks = (observations != PAD_ID).unsqueeze(1).to(DEVICE)
            with torch.no_grad():
                action_dists = self.model(observations, masks)
                action_preds = action_dists.argmax(dim = -1)
            return action_preds.tolist()
        
        # Process input trajectory and run through model
        else:
            pass  # TODO: add another for the other
    

    def load_state(self, checkpoint_file: str) -> None:
        """
        Sets model and optimizer training state to be ones saved prior
        """
        if DEVICE == torch.device("cuda"):
            state_dict = torch.load(checkpoint_file)
        else:
            state_dict = torch.load(checkpoint_file, map_location = torch.device("cpu"))
        self.model.load_state_dict(state_dict["model_dict"])
        self.optimizer.load_state_dict(state_dict["optimizer_dict"])
        self.best_loss = state_dict["validation_loss"]
    
    
    def load_model(self, model_file: str) -> None:
        """
        Sets only model weights to be the one saved prior
        """
        if DEVICE == torch.device("cuda"):
            state_dict = torch.load(model_file)
        else:
            state_dict = torch.load(model_file, map_location = torch.device("cpu"))
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", type = str, required = True)
    parser.add_argument("--setup", "-s", action = "store_true")
    parser.add_argument("--train", "-t", action = "store_true")
    args = parser.parse_args()

    if args.setup:
        process_data_for_spm("datasets/intention_datasets.pkl", args.mode)
        build_vocab(args.mode)
    elif args.train:
        wrapper = ModelWrapper(args.mode, True)
        # print(summary(wrapper.model, [(BATCH_SIZE, MAX_SEQ_LEN), (BATCH_SIZE, MAX_SEQ_LEN), (BATCH_SIZE, 1, MAX_SEQ_LEN),
        #                               (BATCH_SIZE, MAX_SEQ_LEN, MAX_SEQ_LEN)],
        #                               dtypes = [torch.long, torch.long, torch.long, torch.long]))
        wrapper.train_model()