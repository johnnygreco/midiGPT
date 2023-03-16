from pathlib import Path

import pandas as pd

from midigpt import TrainConfigure, Trainer
from midigpt.datasets import TetradNoteDataset

REPO_PATH = Path(__file__).parent.parent.parent


def load_chorales(file_paths):
    return [pd.read_csv(p).values.tolist() for p in file_paths]


jsb_chorales_path = Path(REPO_PATH / "local_data/midi/jsb_chorales")
train_chorales = load_chorales(sorted(jsb_chorales_path.glob("train/chorale_*.csv")))
valid_chorales = load_chorales(sorted(jsb_chorales_path.glob("valid/chorale_*.csv")))
test_chorales = load_chorales(sorted(jsb_chorales_path.glob("test/chorale_*.csv")))

context_length = 128

dataset = TetradNoteDataset(train_chorales, context_length=context_length)

config = TrainConfigure(
    vocab_size=dataset.vocab_size,
    context_length=dataset.context_length,
    embedding_size=128,
    num_epochs=2,
    batch_size=96,
)

trainer = Trainer(config)

trainer.train(dataset)
