import warnings

from torch import optim
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
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.device = utils.get_auto_device() if config.device == "auto" else config.device
        self.model.to(self.device)

    def train(self, dataset: DatasetType):
        data_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        batches_per_epoch = self.config.batches_per_epoch if self.config.batches_per_epoch else len(data_loader)
        with tqdm(total=self.config.num_epochs * batches_per_epoch, desc="Training") as progress_bar:
            for epoch in range(self.config.num_epochs):
                for i, (x, y) in enumerate(data_loader):
                    x, y = x.to(self.device), y.to(self.device)
                    _, self.loss = self.model(x, y)
                    self.optimizer.zero_grad(set_to_none=True)
                    self.loss.backward()
                    self.optimizer.step()
                    progress_bar.update(1)
                    if i % self.config.eval_interval == 0:
                        tqdm.write(f"Epoch {epoch + 1} loss: {self.loss.item():.4f}")
                    if self.config.batches_per_epoch and i + 1 > self.config.batches_per_epoch:
                        break
