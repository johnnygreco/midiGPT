"""
Following Karpathy's minGPT, train on the Tiny Shakespeare dataset for development.
- minGPT repo: https://github.com/karpathy/minGPT
- Tiny Shakespeare dataset: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
"""
from pathlib import Path

from midigpt import TrainConfigure, Trainer
from midigpt.datasets import TextCharacterDataset

REPO_PATH = Path(__file__).parent.parent.parent

with open(REPO_PATH / "local_data/tiny-shakespeare.txt", "r") as f:
    corpus = f.read()

vocabulary = sorted(list(set(corpus)))
config = TrainConfigure(vocab_size=len(vocabulary), context_length=128, embedding_size=128, num_epochs=2, batch_size=96)

dataset = TextCharacterDataset(corpus, vocabulary, config)
trainer = Trainer(config)

trainer.train(dataset)
