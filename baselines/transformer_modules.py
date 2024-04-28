import torch
from torch import nn
from transformer_constants import *
import numpy as np


"""
Building Blocks
"""
class MultiheadAttention(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode

        # Initialize weights
        self.Wq = nn.Parameter(torch.Tensor(NUM_HEADS[self.mode], D_MODEL[self.mode], D_QKV[self.mode]))
        self.Wk = nn.Parameter(torch.Tensor(NUM_HEADS[self.mode], D_MODEL[self.mode], D_QKV[self.mode]))
        self.Wv = nn.Parameter(torch.Tensor(NUM_HEADS[self.mode], D_MODEL[self.mode], D_QKV[self.mode]))
        nn.init.xavier_normal_(self.Wq)
        nn.init.xavier_normal_(self.Wk)
        nn.init.xavier_normal_(self.Wv)
        self.Wo = nn.Linear(D_MODEL[self.mode], D_MODEL[self.mode])

        # Initialize layers
        self.dropout = nn.Dropout(DROPOUT_RATE[self.mode])
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, q, k, v, mask):
        batch_size, length, _ = q.shape

        # Make different heads
        query = torch.einsum("blm, nmd -> bnld", [q, self.Wq])
        key = torch.einsum("blm, nmd -> bnld", [k, self.Wk])
        value = torch.einsum("blm, nmd -> bnld", [v, self.Wv])

        # Scaled dot-product attention
        attention = torch.matmul(query, key.permute(0, 1, 3, 2))
        attention /= np.sqrt(D_QKV[self.mode])
        mask = mask.unsqueeze(1)
        attention = attention.masked_fill_(mask == 0, -1e9)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        attention = torch.matmul(attention, value)

        # Concatenate heads and output
        attention = attention.permute(0, 2, 1, 3).contiguous().reshape(batch_size, length, D_MODEL[self.mode])
        all_attention = self.Wo(attention)
        output = self.dropout(all_attention)
        return output


class FeedForward(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode

        # Initialize layers
        self.layers = nn.Sequential(
            nn.Linear(D_MODEL[self.mode], D_FF[self.mode]),
            nn.ReLU(),
            nn.Linear(D_FF[self.mode], D_MODEL[self.mode]),
            nn.Dropout(DROPOUT_RATE[self.mode])
        )

    def forward(self, x):
        output = self.layers(x)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, max_length: int, mode: str):
        super().__init__()
        self.max_length = max_length
        self.mode = mode

        # Initialize constants
        self.power = torch.arange(0, D_MODEL[self.mode], step = 2, dtype = torch.float32)[:] / D_MODEL[self.mode]
        self.divisor = 10000 ** self.power

         # Create positional encodings
        self.seq_pos = torch.arange(1, self.max_length + 1, dtype = torch.float32)[None, :, None]
        self.indices = self.seq_pos.repeat(*[1, 1, D_MODEL[self.mode] // 2])
        self.sin_embedding = torch.sin(self.indices / self.divisor)
        self.cos_embedding = torch.cos(self.indices / self.divisor)
        self.pos_shape = (1, self.max_length, -1)
        self.final_embedding = torch.stack((self.sin_embedding, self.cos_embedding), dim = 3).view(self.pos_shape)
        self.final_embedding = self.final_embedding.to(DEVICE).requires_grad_(False)

    def forward(self, x):
        x *= np.sqrt(D_MODEL[self.mode])
        output = x + self.final_embedding
        return output


"""
Layers
"""
class EncoderLayer(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode

        # Initialize layers and sublayers
        self.attention = MultiheadAttention(self.mode)
        self.norm1 = nn.LayerNorm(D_MODEL[self.mode])
        self.feed_forward = FeedForward(self.mode)
        self.norm2 = nn.LayerNorm(D_MODEL[self.mode])

    def forward(self, x, mask):
        att_output = self.attention(x, x, x, mask)
        x = self.norm1(x + att_output)
        ff_output = self.feed_forward(x)
        final_output = self.norm2(x + ff_output)
        return final_output


class DecoderLayer(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode

        # Initialize layers and sublayers
        self.masked_attention = MultiheadAttention(self.mode)
        self.norm1 = nn.LayerNorm(D_MODEL[self.mode])
        self.attention = MultiheadAttention(self.mode)
        self.norm2 = nn.LayerNorm(D_MODEL[self.mode])
        self.feed_forward = FeedForward(self.mode)
        self.norm3 = nn.LayerNorm(D_MODEL[self.mode])

    def forward(self, x, enc_output, enc_mask, dec_mask):
        masked_att_output = self.masked_attention(x, x, x, dec_mask)
        x = self.norm1(x + masked_att_output)
        att_output = self.attention(x, enc_output, enc_output, enc_mask)
        x = self.norm2(x + att_output)
        ff_output = self.feed_forward(x)
        final_output = self.norm3(x + ff_output)
        return final_output


"""
Transformer
"""
class TransformerEncoder(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        self.layers = nn.ModuleList([EncoderLayer(self.mode) for _ in range(NUM_LAYERS[self.mode])])

    def forward(self, x, mask):
        for i in range(NUM_LAYERS[self.mode]):
            x = self.layers[i](x, mask)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        self.layers = nn.ModuleList([DecoderLayer(self.mode) for _ in range(NUM_LAYERS[self.mode])])

    def forward(self, x, enc_output, enc_mask, dec_mask):
        for i in range(NUM_LAYERS[self.mode]):
            x = self.layers[i](x, enc_output, enc_mask, dec_mask)
        return x


class TrajectoryTransformer(nn.Module):
    def __init__(self, input_vocab_size: int, output_vocab_size: int):
        super().__init__()
        self.mode = "traj"

        # Initialize layers
        self.input_embedding = nn.Embedding(input_vocab_size, D_MODEL[self.mode])
        self.output_embedding = nn.Embedding(output_vocab_size, D_MODEL[self.mode])
        self.input_positional_encoding = PositionalEncoding(MAX_TRAJ_LEN, self.mode)
        self.output_positional_encoding = PositionalEncoding(MAX_SKILL_LEN, self.mode)
        self.encoder = TransformerEncoder(self.mode)
        self.decoder = TransformerDecoder(self.mode)
        self.linear = nn.Linear(D_MODEL[self.mode], output_vocab_size)
        self.softmax = nn.LogSoftmax(dim = -1)

    def forward(self, enc_x, dec_x, enc_mask, dec_mask):
        # Preprocess
        enc_x = enc_x.to(DEVICE)
        dec_x = dec_x.to(DEVICE)
        enc_mask = enc_mask.to(DEVICE)
        dec_mask = dec_mask.to(DEVICE)
        enc_x = self.input_embedding(enc_x)
        enc_x = self.input_positional_encoding(enc_x)
        dec_x = self.output_embedding(dec_x)
        dec_x = self.output_positional_encoding(dec_x)

        # Feed into model
        enc_output = self.encoder(enc_x, enc_mask)
        dec_output = self.decoder(dec_x, enc_output, enc_mask, dec_mask)

        # Get output
        final_output = self.linear(dec_output)
        final_output = self.softmax(final_output)
        return final_output
    

class ObservationTransformer(nn.Module):
    def __init__(self, input_vocab_size: int, output_action_size: int):
        super().__init__()
        self.mode = "obs"

        # Initialize layers
        self.input_embedding = nn.Embedding(input_vocab_size, D_MODEL[self.mode])
        self.positional_encoding = PositionalEncoding(MAX_OBS_LEN, self.mode)
        self.encoder = TransformerEncoder(self.mode)
        self.linear = nn.Linear(D_MODEL[self.mode], output_action_size)
        self.softmax = nn.LogSoftmax(dim = -1)

    def forward(self, x, mask):
        # Preprocess
        x = x.to(DEVICE)
        mask = mask.to(DEVICE)
        x = self.input_embedding(x)
        x = self.positional_encoding(x)

        # Feed into model
        output = self.encoder(x, mask)

        # Get output
        final_output = self.linear(output[:, -1, :])
        final_output = self.softmax(final_output)
        return final_output
