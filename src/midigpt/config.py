from pydantic import BaseModel, root_validator

__all__ = ["Configure"]


class Configure(BaseModel):
    batch_size: int = 32
    context_length: int = 64
    embedding_size: int = 32
    num_heads: int = 4
    num_layers: int = 4
    attn_dropout_prob: float = 0.1
    embed_dropout_prob: float = 0.1

    @root_validator
    def check_embedding_size_num_heads_ratio(cls, values):
        num_heads = values.get("num_heads")
        embedding_size = values.get("embedding_size")
        if embedding_size % num_heads != 0:
            raise ValueError(f"{embedding_size=} is not divisible by {num_heads=}")
        return values
