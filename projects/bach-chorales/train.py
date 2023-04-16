from pathlib import Path

import pandas as pd

from midigpt import TrainConfigure, Trainer
from midigpt.datasets import BachChoraleDataset


def load_chorales(file_paths):
    return [pd.read_csv(p).values.tolist() for p in file_paths]


jsb_chorales_path = Path("jsb_chorales")
train_chorales = load_chorales(sorted(jsb_chorales_path.glob("train/chorale_*.csv")))
test_chorales = load_chorales(sorted(jsb_chorales_path.glob("test/chorale_*.csv")))
train_chorales = train_chorales + test_chorales
valid_chorales = load_chorales(sorted(jsb_chorales_path.glob("valid/chorale_*.csv")))

context_length = 256
train_dataset = BachChoraleDataset(train_chorales, context_length=context_length)
validation_dataset = BachChoraleDataset(valid_chorales, context_length=context_length)

assert train_dataset.vocab_size == validation_dataset.vocab_size

config = TrainConfigure(
    vocab_size=train_dataset.vocab_size,
    context_length=train_dataset.context_length,
    embedding_size=64,
    num_heads=8,
    num_blocks=12,
    num_epochs=4,
    batch_size=96,
    attn_dropout_prob=0.1,
    embed_dropout_prob=0.1,
    learning_rate=5e-4,
)

trainer = Trainer(config)

print(f"{trainer.model.num_params} model parameters")

trainer.train(train_dataset, validation_dataset=validation_dataset)
