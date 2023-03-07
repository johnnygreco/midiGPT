import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import ModelConfigure

__all__ = ["CasualMultiHeadAttention", "Block", "FeedForward"]


class GELU(nn.Module):
    """GELU activation function: https://arxiv.org/abs/1606.08415"""

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CasualMultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfigure) -> None:
        super().__init__()
        q_k_v_size = 3 * config.embedding_size
        self.attn = nn.Linear(config.embedding_size, q_k_v_size)
        self.output_projection = nn.Linear(config.embedding_size, config.embedding_size)
        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)
        self.embed_dropout = nn.Dropout(config.embed_dropout_prob)
        _tril_reshape = torch.tril(torch.ones(config.context_length, config.context_length)).view(
            1, 1, config.context_length, config.context_length
        )
        self.register_buffer("casual_mask", _tril_reshape)
        self.num_heads = config.num_heads
        self.embedding_size = config.embedding_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d_batch, d_context, d_embed = x.size()
        q, k, v = self.attn(x).split(self.embedding_size, dim=2)
        q = q.view(d_batch, d_context, self.num_heads, d_embed // self.num_heads).transpose(1, 2)
        k = k.view(d_batch, d_context, self.num_heads, d_embed // self.num_heads).transpose(1, 2)
        v = v.view(d_batch, d_context, self.num_heads, d_embed // self.num_heads).transpose(1, 2)

        attn = q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5)
        attn = attn.masked_fill(self.casual_mask[:, :, :d_context, :d_context] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        attended_values = attn @ v
        attended_values = attended_values.transpose(1, 2).contiguous().view(*x.size())

        output = self.output_projection(attended_values)
        output = self.embed_dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfigure) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embedding_size, 4 * config.embedding_size),
            GELU(),
            nn.Linear(4 * config.embedding_size, config.embedding_size),
            nn.Dropout(config.embed_dropout_prob),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: ModelConfigure):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.embedding_size)
        self.multi_head_self_attention = CasualMultiHeadAttention(config)
        self.layer_norm_2 = nn.LayerNorm(config.embedding_size)
        self.ff_net = FeedForward(config)

    def forward(self, x: torch.Tensor):
        x = x + self.multi_head_self_attention(self.layer_norm_1(x))
        x = x + self.ff_net(self.layer_norm_2(x))
        return x
