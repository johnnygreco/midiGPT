import warnings
from typing import Dict

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.rich import tqdm
from tqdm.std import TqdmExperimentalWarning

from . import utils
from .config import TrainConfigure
from .datasets import DatasetType
from .gpt import GPT

__all__ = ["Trainer"]

warnings.simplefilter("ignore", TqdmExperimentalWarning)


class Trainer:
    def __init__(self, config: TrainConfigure):
        self.config = config
        self.model = GPT(config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.device = utils.get_auto_device() if config.device == "auto" else config.device
        self.model.to(self.device)

    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        mean_losses = {}
        self.model.eval()
        for split in ["train", "valid"]:
            loader_iter = iter(getattr(self, f"{split}_loader"))
            losses = torch.zeros(self.config.eval_iters)
            for i in range(self.config.eval_iters):
                batch = next(loader_iter)
                x, y = [t.to(self.device) for t in batch]
                losses[i] = self.model(x, y)[1].item()
            mean_losses[split] = losses.mean()
        self.model.train()
        return mean_losses

    def _print_current_loss(self, batch_num: int, epoch: int):
        if batch_num % self.config.eval_interval == 0:
            losses = self.estimate_loss()
            tqdm.write(f"Epoch {epoch} | train loss: {losses['train']:.4f}, valid loss: {losses['valid']:.4f}")

    def train(self, dataset: DatasetType, train_fraction: float = 0.9, shuffle: bool = True):
        indices = torch.arange(len(dataset))
        self.train_loader = DataLoader(
            Subset(dataset, indices[: int(len(dataset) * train_fraction)]),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
        )
        self.valid_loader = DataLoader(
            Subset(dataset, indices[int(len(dataset) * train_fraction) :]),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
        )
        batches_per_epoch = self.config.batches_per_epoch if self.config.batches_per_epoch else len(self.train_loader)
        with tqdm(total=self.config.num_epochs * batches_per_epoch, desc="Training") as progress_bar:
            for epoch in range(1, self.config.num_epochs + 1):
                for batch_num, (x, y) in enumerate(self.train_loader, start=1):
                    x, y = x.to(self.device), y.to(self.device)
                    _, self.loss = self.model(x, y)
                    self.optimizer.zero_grad(set_to_none=True)
                    self.loss.backward()
                    self.optimizer.step()
                    progress_bar.update(1)
                    self._print_current_loss(batch_num, epoch)
                    if self.config.batches_per_epoch and batch_num > self.config.batches_per_epoch:
                        break
