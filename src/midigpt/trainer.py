import warnings
from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader, Subset
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
        self._validation_loader = None

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

    @torch.no_grad()
    def _print_epoch_loss(self, epoch: int, average_train_loss: float):
        if self._validation_loader is None:
            tqdm.write(f"***** epoch: {epoch} complete  ->  average train loss: {average_train_loss:<.4f} *****")
        else:
            self.model.eval()
            average_valid_loss = 0.0
            for _ in range(self.config.batches_per_eval):
                x, y = next(iter(self._validation_loader))
                x, y = x.to(self.device), y.to(self.device)
                _, loss = self.model(x, y)
                average_valid_loss += loss.item() / self.config.batches_per_eval
            self.model.train()
            tqdm.write(
                f"***** epoch: {epoch}  complete  ->  "
                f"average train loss: {average_train_loss:<.4f}  |  "
                f"average validation loss: {average_valid_loss:<.4f} *****"
            )

    def _check_context_length(self, dataset: DatasetType, name: str):
        if isinstance(dataset, Subset):
            context_length = dataset.dataset.context_length
        else:
            context_length = dataset.context_length
        if context_length != self.config.context_length:
            raise ValueError(f"{name} {context_length=} does not match {self.config.context_length=}")

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

    def train(self, dataset: DatasetType, validation_dataset: Optional[DatasetType] = None, shuffle: bool = True):
        self._check_context_length(dataset, "training dataset")
        self.train_loader = DataLoader(dataset=dataset, batch_size=self.config.batch_size, shuffle=shuffle)
        total_iterations = self.config.num_epochs * len(self.train_loader)
        if validation_dataset is not None:
            self._check_context_length(validation_dataset, "validation dataset")
            self._validation_loader = DataLoader(
                dataset=validation_dataset, batch_size=self.config.batch_size, shuffle=True
            )

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
                self._print_epoch_loss(epoch, average_loss)
                lowest_loss = self._save_model_if_best(epoch, batch_num, average_loss, lowest_loss)
                self._save_loss_history()
