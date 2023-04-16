from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from . import utils
from .components import CasualAttentionBlock
from .config import ModelConfigure, TrainConfigure

__all__ = ["GPT"]


class GPT(nn.Module):
    def __init__(self, config: Union[ModelConfigure, TrainConfigure]):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.embedding_size)
        self.position_embedding_table = nn.Embedding(config.context_length, config.embedding_size)
        self.blocks = nn.Sequential(*[CasualAttentionBlock(config) for _ in range(config.num_blocks)])
        self.final_layer_norm = nn.LayerNorm(config.embedding_size)
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size)

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

    @classmethod
    def from_checkpoint(cls, path: Union[str, Path], **kwargs):
        checkpoint = torch.load(path, **kwargs)
        config = checkpoint["model_config"]
        model = cls(ModelConfigure(**config))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(utils.get_auto_device() if config["device"] == "auto" else config["device"])
        return model

    def forward(self, x: torch.LongTensor, targets: Optional[torch.LongTensor] = None):
        token_embedding = self.token_embedding_table(x)  # (B, T, C)
        position_embedding = self.position_embedding_table(torch.arange(x.shape[1], device=self.device))  # (T, C)
        x = token_embedding + position_embedding  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.final_layer_norm(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = (
            None
            if targets is None
            else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: Union[List[int], torch.LongTensor],
        num_generated_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        as_list: bool = False,
    ):
        idx = torch.as_tensor(idx, dtype=torch.long, device=self.device)[None, ...]
        changed_training_mode = False
        if self.training:
            self.eval()
            changed_training_mode = True
        for _ in range(num_generated_tokens):
            # if the sequence context is growing too long we must crop it at context_length
            idx_cond = idx if idx.size(1) <= self.context_length else idx[:, -self.context_length :]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / (temperature + 1e-8)
            # optionally crop the logits to only the top k options
            if top_k is not None:
                logits[logits < torch.topk(logits, top_k)[0][:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            idx_next = torch.multinomial(probs, num_samples=1) if do_sample else torch.topk(probs, k=1, dim=-1)[1]
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        if changed_training_mode:
            self.train()
        return idx.cpu().squeeze().tolist() if as_list else idx

    def save(self, path: Union[str, Path]):
        torch.save(self.state_dict(), path)
