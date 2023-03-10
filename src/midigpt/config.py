from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, root_validator

__all__ = ["ModelConfigure", "TrainConfigure"]


class ModelConfigure(BaseModel):
    vocab_size: int
    context_length: int = 64
    embedding_size: int = 32
    num_heads: int = 4
    num_blocks: int = 4
    attn_dropout_prob: float = 0.1
    embed_dropout_prob: float = 0.1
    device: str = "auto"

    @root_validator
    def check_embedding_size_num_heads_ratio(cls, values):
        num_heads = values.get("num_heads")
        embedding_size = values.get("embedding_size")
        if embedding_size % num_heads != 0:
            raise ValueError(f"{embedding_size=} is not divisible by {num_heads=}")
        return values


class TrainConfigure(ModelConfigure):
    batch_size: int = 32
    learning_rate: float = 5e-4
    eval_iters: int = 200
    eval_interval: int = 500
    num_epochs: int = 3
    batches_per_epoch: Optional[int] = None
    checkpoint_path: Union[str, Path] = Path("checkpoints")
    save_all_checkpoints: bool = False
    overwrite_checkpoints: bool = True
