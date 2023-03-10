import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader
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
        self.loss_history = []
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.device = utils.get_auto_device() if config.device == "auto" else config.device
        self.model.to(self.device)

    def train(self, dataset: DatasetType, shuffle: bool = True):
        self.train_loader = DataLoader(dataset=dataset, batch_size=self.config.batch_size, shuffle=shuffle)
        batches_per_epoch = self.config.batches_per_epoch if self.config.batches_per_epoch else len(self.train_loader)
        total_iterations = self.config.num_epochs * batches_per_epoch
        with tqdm(total=total_iterations, desc=f"Training for {self.config.num_epochs} epochs:") as progress_bar:
            for epoch in range(1, self.config.num_epochs + 1):
                running_loss = 0.0
                for batch_num, (x, y) in enumerate(self.train_loader, start=1):
                    x, y = x.to(self.device), y.to(self.device)
                    _, self.loss = self.model(x, y)
                    self.optimizer.zero_grad(set_to_none=True)
                    self.loss.backward()
                    self.optimizer.step()
                    running_loss += self.loss.item()
                    self.loss_history.append(self.loss.item())
                    if batch_num % self.config.eval_interval == 0:
                        average_loss = running_loss / self.config.eval_interval
                        tqdm.write(
                            f"epoch: {epoch:<4.0f}  |  "
                            f"batch: {batch_num:<7.0f}  |  "
                            f"average batch loss: {average_loss:<.4f}"
                        )
                        running_loss = 0.0
                    progress_bar.update(1)
                    if self.config.batches_per_epoch and batch_num > self.config.batches_per_epoch:
                        break
                if self.config.save_per_epoch_checkpoints:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "loss": self.loss.item(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                        },
                        Path(self.config.checkpoint_path) / f"epoch_{epoch}.ckpt",
                    )
