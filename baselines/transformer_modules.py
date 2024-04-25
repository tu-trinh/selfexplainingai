import torch
from torch import nn
from transformer_constants import *
import numpy as np


"""
Building Blocks
"""
class MultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize weights
        self.Wq = nn.Parameter(torch.Tensor(NUM_HEADS, D_MODEL, D_QKV))
        self.Wk = nn.Parameter(torch.Tensor(NUM_HEADS, D_MODEL, D_QKV))
        self.Wv = nn.Parameter(torch.Tensor(NUM_HEADS, D_MODEL, D_QKV))
        nn.init.xavier_normal_(self.Wq)
        nn.init.xavier_normal_(self.Wk)
        nn.init.xavier_normal_(self.Wv)
        self.Wo = nn.Linear(D_MODEL, D_MODEL)

        # Initialize layers
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, q, k, v, mask):
        batch_size, length, _ = q.shape

        # Make different heads
        query = torch.einsum("blm, nmd -> bnld", [q, self.Wq])
        key = torch.einsum("blm, nmd -> bnld", [k, self.Wk])
        value = torch.einsum("blm, nmd -> bnld", [v, self.Wv])

        # Scaled dot-product attention
        attention = torch.matmul(query, key.permute(0, 1, 3, 2))
        attention /= np.sqrt(D_QKV)
        mask = mask.unsqueeze(1)
        attention = attention.masked_fill_(mask == 0, -1e9)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        attention = torch.matmul(attention, value)

        # Concatenate heads and output
        attention = attention.permute(0, 2, 1, 3).contiguous().reshape(batch_size, length, D_MODEL)
        all_attention = self.Wo(attention)
        output = self.dropout(all_attention)
        return output


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.layers = nn.Sequential(
            nn.Linear(D_MODEL, D_FF),
            nn.ReLU(),
            nn.Linear(D_FF, D_MODEL),
            nn.Dropout(DROPOUT_RATE)
        )

    def forward(self, x):
        output = self.layers(x)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize constants
        self.power = torch.arange(0, D_MODEL, step = 2, dtype = torch.float32)[:] / D_MODEL
        self.divisor = 10000 ** self.power

         # Create positional encodings
        self.seq_pos = torch.arange(1, MAX_SEQ_LEN + 1, dtype = torch.float32)[None, :, None]
        self.indices = self.seq_pos.repeat(*[1, 1, D_MODEL // 2])
        self.sin_embedding = torch.sin(self.indices / self.divisor)
        self.cos_embedding = torch.cos(self.indices / self.divisor)
        self.pos_shape = (1, MAX_SEQ_LEN, -1)
        self.final_embedding = torch.stack((self.sin_embedding, self.cos_embedding), dim = 3).view(self.pos_shape)
        self.final_embedding = self.final_embedding.to(DEVICE).requires_grad_(False)

    def forward(self, x):
        x *= np.sqrt(D_MODEL)
        output = x + self.final_embedding
        return output


"""
Layers
"""
class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers and sublayers
        self.attention = MultiheadAttention()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.feed_forward = FeedForward()
        self.norm2 = nn.LayerNorm(D_MODEL)

    def forward(self, x, mask):
        att_output = self.attention(x, x, x, mask)
        x = self.norm1(x + att_output)
        ff_output = self.feed_forward(x)
        final_output = self.norm2(x + ff_output)
        return final_output


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers and sublayers
        self.masked_attention = MultiheadAttention()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.attention = MultiheadAttention()
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.feed_forward = FeedForward()
        self.norm3 = nn.LayerNorm(D_MODEL)

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
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(NUM_LAYERS)])

    def forward(self, x, mask):
        for i in range(NUM_LAYERS):
            x = self.layers[i](x, mask)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(NUM_LAYERS)])

    def forward(self, x, enc_output, enc_mask, dec_mask):
        for i in range(NUM_LAYERS):
            x = self.layers[i](x, enc_output, enc_mask, dec_mask)
        return x


class TrajectoryTransformer(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size):
        super().__init__()

        # Initialize layers
        self.input_embedding = nn.Embedding(input_vocab_size, D_MODEL)
        self.output_embedding = nn.Embedding(output_vocab_size, D_MODEL)
        self.positional_encoding = PositionalEncoding()
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()
        self.linear = nn.Linear(D_MODEL, output_vocab_size)
        self.softmax = nn.LogSoftmax(dim = -1)

    def forward(self, enc_x, dec_x, enc_mask, dec_mask):
        # Preprocess
        enc_x = enc_x.to(DEVICE)
        dec_x = dec_x.to(DEVICE)
        enc_mask = enc_mask.to(DEVICE)
        dec_mask = dec_mask.to(DEVICE)
        enc_x = self.input_embedding(enc_x)
        enc_x = self.positional_encoding(enc_x)
        dec_x = self.output_embedding(dec_x)
        dec_x = self.positional_encoding(dec_x)

        # Feed into model
        enc_output = self.encoder(enc_x, enc_mask)
        dec_output = self.decoder(dec_x, enc_output, enc_mask, dec_mask)

        # Get output
        final_output = self.linear(dec_output)
        final_output = self.softmax(final_output)
        return final_output
    

class ObservationTransformer(nn.Module):
    def __init__(self, input_vocab_size, output_action_size):
        super().__init__()

        # Initialize layers
        self.input_embedding = nn.Embedding(input_vocab_size, D_MODEL)
        self.positional_encoding = PositionalEncoding()
        self.encoder = TransformerEncoder()
        self.linear = nn.Linear(D_MODEL, output_action_size)
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
