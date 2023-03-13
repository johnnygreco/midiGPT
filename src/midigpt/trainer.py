import warnings
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import DataLoader
from tqdm.rich import tqdm
from tqdm.std import TqdmExperimentalWarning

from . import utils
from .config import ModelConfigure, TrainConfigure
from .datasets import DatasetType
from .gpt import GPT

__all__ = ["Trainer"]

warnings.simplefilter("ignore", TqdmExperimentalWarning)


class Trainer:
    def __init__(self, config: TrainConfigure):
        self.config = config
        self.model = GPT(config)
        self.checkpoint_path = Path(self.config.checkpoint_path)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.device = utils.get_auto_device() if config.device == "auto" else config.device
        self.model.to(self.device)
        self._loss_history = []

    def _save_loss_history(self):
        with open(self.checkpoint_path / "loss_history.txt", "a") as f:
            f.write("\n".join(map(str, self._loss_history)) + "\n")
        self._loss_history = []

    def _save_model_if_best(self, epoch: int, batch_num: int, loss: float, lowest_loss: float):
        if loss < lowest_loss or self.config.save_all_checkpoints:
            lowest_loss = loss
            self.checkpoint_path.mkdir(parents=True, exist_ok=True)
            file_name = (
                "best_model.ckpt" if self.config.overwrite_checkpoints else f"epoch_{epoch}_batch_{batch_num}.ckpt"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "batch_num": batch_num,
                    "loss": loss,
                    "train_config": self.config.dict(),
                    "model_config": self.config.model_config.dict(),
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                self.checkpoint_path / file_name,
            )
        return lowest_loss

    @classmethod
    def from_checkpoint(cls, path: Union[str, Path]):
        checkpoint = torch.load(path)
        model_config = checkpoint["model_config"]
        model = GPT(ModelConfigure(**model_config))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(utils.get_auto_device() if model_config["device"] == "auto" else model_config["device"])
        trainer = cls(TrainConfigure(**checkpoint["train_config"]))
        trainer.model = model
        trainer.loss = checkpoint["loss"]
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return trainer

    def train(self, dataset: DatasetType, shuffle: bool = True):

        self.train_loader = DataLoader(dataset=dataset, batch_size=self.config.batch_size, shuffle=shuffle)
        total_iterations = self.config.num_epochs * len(self.train_loader)

        with tqdm(total=total_iterations, desc=f"Training for {self.config.num_epochs} epochs:") as progress_bar:

            self.model.train()
            lowest_loss = float("inf")

            for epoch in range(1, self.config.num_epochs + 1):
                running_loss = {"epoch": 0.0, "batch_interval": 0.0}
                for batch_num, (x, y) in enumerate(self.train_loader, start=1):
                    x, y = x.to(self.device), y.to(self.device)

                    # perform forward pass
                    _, self.loss = self.model(x, y)

                    # perform backpropagation
                    self.optimizer.zero_grad(set_to_none=True)
                    self.loss.backward()
                    self.optimizer.step()

                    # update loss logging variables
                    running_loss["epoch"] += self.loss.item()
                    running_loss["batch_interval"] += self.loss.item()
                    self._loss_history.append(self.loss.item())

                    # log average batch loss and save checkpoint if at eval interval
                    if batch_num % self.config.eval_interval == 0:
                        average_loss = running_loss["batch_interval"] / self.config.eval_interval
                        running_loss["batch_interval"] = 0.0
                        tqdm.write(
                            f"epoch: {epoch:<4.0f}  |  "
                            f"batch: {batch_num:<7.0f}  |  "
                            f"average batch loss: {average_loss:<.4f}"
                        )
                        lowest_loss = self._save_model_if_best(epoch, batch_num, average_loss, lowest_loss)
                        self._save_loss_history()

                    progress_bar.update(1)

                # log epoch loss and save checkpoint
                average_loss = running_loss["epoch"] / len(self.train_loader)
                tqdm.write(f"epoch: {epoch:<4.0f} complete  ->  average epoch loss: {average_loss:<.4f}")
                lowest_loss = self._save_model_if_best(epoch, batch_num, average_loss, lowest_loss)
                self._save_loss_history()
