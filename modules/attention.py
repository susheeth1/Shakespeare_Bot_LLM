# File: modules/attention.py
import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiHeadSelfAttention(nn.Module):
    """ Multi-head self-attention module """

    def __init__(self, d_model: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        
        self.d_model = d_model
        self.n_head = n_head
        self.head_size = d_model // n_head
        
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # Batch size, sequence length, embedding dimensionality (d_model)

        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y = self.resid_dropout(self.c_proj(y))
        return y