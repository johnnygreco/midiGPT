from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from . import utils
from .components import Block
from .config import ModelConfigure, TrainConfigure

__all__ = ["GPT"]


class GPT(nn.Module):
    def __init__(self, config: Union[ModelConfigure, TrainConfigure]):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.embedding_size)
        self.position_embedding_table = nn.Embedding(config.context_length, config.embedding_size)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.num_blocks)])
        self.final_layer_norm = nn.LayerNorm(config.embedding_size)
        self.reproject_to_vocab = nn.Linear(config.embedding_size, config.vocab_size)
        self.apply(self._init_weights)
        self.context_length = config.context_length
        self.device = utils.get_auto_device() if config.device == "auto" else config.device

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = x.shape
        token_embedding = self.token_embedding_table(x)  # (B, T, C)
        position_embedding = self.position_embedding_table(torch.arange(T, device=self.device))  # (T, C)
        x = token_embedding + position_embedding  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.final_layer_norm(x)  # (B, T, C)
        logits = self.reproject_to_vocab(x)  # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.context_length :]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
