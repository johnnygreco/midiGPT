import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import Configure

device = "cuda" if torch.cuda.is_available() else "cpu"


class CasualAttentionHead(nn.Module):
    def __init__(self, config: Configure) -> None:
        super().__init__()
        q_k_v_size = 3 * config.embedding_size
        self.attn = nn.Linear(config.embedding_size, q_k_v_size)
        self.output_projection = nn.Linear(config.embedding_size, config.embedding_size)
        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)
        self.embed_dropout = nn.Dropout(config.embed_dropout_prob)
        _tril_reshape = (
            torch.tril(torch.ones(config.context_length, config.context_length)).view(
                1, 1, config.context_length, config.context_length
            ),
        )
        self.register_buffer("casual_mask", _tril_reshape)
        self.num_heads = config.num_heads
        self.embedding_size = config.embedding_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.attn(x).split(self.embedding_size, dim=2)
