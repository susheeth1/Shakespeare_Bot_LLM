# File: modules/transformer_block.py
import torch.nn as nn
from modules.attention import MultiHeadSelfAttention
from modules.feedforward import FeedForward

class TransformerBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, d_model: int, n_head: int, block_size: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffwd = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
        return x