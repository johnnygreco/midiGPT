import warnings

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
        with tqdm(total=self.config.num_epochs * batches_per_epoch, desc="Training:") as progress_bar:
            running_loss = 0.0
            for epoch in range(1, self.config.num_epochs + 1):
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
                        tqdm.write(f"epoch: {epoch:<4.0f}  |  batch: {batch_num:<7.0f}  |  loss: {average_loss:<.4f}")
                        running_loss = 0.0
                    progress_bar.update(1)
                    if self.config.batches_per_epoch and batch_num > self.config.batches_per_epoch:
                        break
