import torch
from torch import nn
from transformer_constants import *
import numpy as np
from transformer_modules import *
from tqdm import tqdm
from utils import *
import argparse
from torchinfo import summary


class ModelWrapper:
    def __init__(self, training):
        # Data initialization
        self.func_spp = spm.SentencePieceProcessor()
        self.func_spp.load("./tokenization/input.model")
        self.deriv_spp = spm.SentencePieceProcessor()
        self.deriv_spp.load("./tokenization/output.model")
        if training:
            self.train_data_loader = build_data_loader("./data/train_processed.txt", self.func_spp, self.deriv_spp)
            self.valid_data_loader = build_data_loader("./data/valid_processed.txt", self.func_spp, self.deriv_spp)

        # Model initialization
        self.model = Transformer(INPUT_VOCAB_SIZE, OUTPUT_ACTION_SIZE).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = LEARNING_RATE)
        self.loss_func = nn.CrossEntropyLoss()
        self.best_loss = float("inf")
    
    
    def mask(self, func_data, deriv_data):
        """
        Creates masks on inputs
        """
        enc_mask = (func_data != PAD_ID).unsqueeze(1).to(DEVICE)
        dec_mask = (deriv_data != PAD_ID).unsqueeze(1).to(DEVICE)
        helper_mask = torch.ones([1, MAX_SEQ_LEN, MAX_SEQ_LEN], dtype = torch.bool)
        helper_mask = torch.tril(helper_mask).to(DEVICE)
        dec_mask = dec_mask & helper_mask
        return enc_mask, dec_mask
    
    
    def train_model(self):
        """
        Runs training process and save model that performs the best on a the validation set
        """
        for epoch in range(NUM_EPOCHS):
            self.model.train()
            losses = []
            for batch in tqdm(self.train_data_loader):
                func, deriv_in, deriv_out = batch
                func = func.to(DEVICE)
                deriv_in = deriv_in.to(DEVICE)
                deriv_out = deriv_out.to(DEVICE)

                enc_mask, dec_mask = self.mask(func, deriv_in)
                output = self.model(func, deriv_in, enc_mask, dec_mask)

                self.optimizer.zero_grad()
                loss = self.loss_func(
                    output.view(-1, OUTPUT_ACTION_SIZE),
                    deriv_out.view(deriv_out.shape[0] * deriv_out.shape[1])
                )
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
                del func, deriv_in, deriv_out, enc_mask, dec_mask, output
                if DEVICE == torch.device("cuda"):
                    torch.cuda.empty_cache()
            
            print("Finished training epoch {}".format(epoch + 1))
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
                torch.save(state_dict, "checkpoint.tar")
                print("New best validation accuracy achieved, saved to checkpoint")

    
    def validate_model(self):
        """
        Tests the current model on a validation set
        """
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in tqdm(self.valid_data_loader):
                func, deriv_in, deriv_out = batch
                func = func.to(DEVICE)
                deriv_in = deriv_in.to(DEVICE)
                deriv_out = deriv_out.to(DEVICE)

                enc_mask, dec_mask = self.mask(func, deriv_in)
                output = self.model(func, deriv_in, enc_mask, dec_mask)

                loss = self.loss_func(
                    output.view(-1, OUTPUT_ACTION_SIZE),
                    deriv_out.view(deriv_out.shape[0] * deriv_out.shape[1])
                )
                losses.append(loss.item())
                del func, deriv_in, deriv_out, enc_mask, dec_mask, output
                if DEVICE == torch.device("cuda"):
                    torch.cuda.empty_cache()
        
        valid_loss = np.mean(losses)
        return valid_loss
    
    
    def infer(self, function):
        """
        Run inference with the model; outputs the derivative prediction given a function
        """
        self.model.eval()

        # Process input and run through encoder side
        func_tokens = standardize_length(self.func_spp.EncodeAsIds(function))
        func_tokens = torch.LongTensor(func_tokens).unsqueeze(0).to(DEVICE)
        func_mask = (func_tokens != PAD_ID).unsqueeze(1).to(DEVICE)
        func_embedding = self.model.input_embedding(func_tokens)
        func_embedding = self.model.positional_encoding(func_embedding)
        enc_output = self.model.encoder(func_embedding, func_mask)

        # Conduct greedy search to find best decoder output
        sequence = torch.LongTensor([PAD_ID for _ in range(MAX_SEQ_LEN)]).unsqueeze(0).to(DEVICE)
        sequence[0] = BOS_ID
        curr_len = 1
        for i in range(MAX_SEQ_LEN):
            deriv_mask = (sequence != PAD_ID).unsqueeze(1).to(DEVICE)
            helper_mask = torch.ones([1, MAX_SEQ_LEN, MAX_SEQ_LEN], dtype = torch.bool)
            helper_mask = torch.tril(helper_mask).to(DEVICE)
            deriv_mask = deriv_mask & helper_mask

            deriv_embedding = self.model.output_embedding(sequence)
            deriv_embedding = self.model.positional_encoding(deriv_embedding)
            
            dec_output = self.model.decoder(deriv_embedding, enc_output, func_mask, deriv_mask)
            dec_output = self.model.linear(dec_output)
            dec_output = self.model.softmax(dec_output)
            dec_output = dec_output.argmax(dim = -1)
            id = dec_output[0][i].item()

            if i + 1 < MAX_SEQ_LEN:
                sequence[0][i + 1] = id
                curr_len += 1
            if id == EOS_ID:
                break
        
        # Decode output into understandable math
        if sequence[0][-1].item() == PAD_ID:
            final_output = sequence[0][1:curr_len].tolist()
        else:
            final_output = sequence[0][1:].tolist()
        final_output = self.deriv_spp.decode_ids(final_output)
        return final_output
    

    def load_state(self, checkpoint_file = "checkpoint.tar"):
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
    
    def load_model(self, model_file = "model.pt"):
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
    parser.add_argument("--setup", "-s", action = "store_true")
    parser.add_argument("--train", "-t", action = "store_true")
    args = parser.parse_args()

    if args.setup:
        process_data("./data/train.txt")
        build_vocab()
    elif args.train:
        wrapper = ModelWrapper(True)
        # print(summary(wrapper.model, [(BATCH_SIZE, MAX_SEQ_LEN), (BATCH_SIZE, MAX_SEQ_LEN), (BATCH_SIZE, 1, MAX_SEQ_LEN),
        #                               (BATCH_SIZE, MAX_SEQ_LEN, MAX_SEQ_LEN)],
        #                               dtypes = [torch.long, torch.long, torch.long, torch.long]))
        wrapper.train_model()