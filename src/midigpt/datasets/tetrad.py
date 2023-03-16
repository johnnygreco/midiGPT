from itertools import chain

import torch
from torch.utils.data import Dataset

__all__ = ["TetradNoteDataset"]


class TetradNoteDataset(Dataset):
    def __init__(self, tetrad_corpus, context_length):
        assert len(tetrad_corpus[0][0]) == 4, "tetrad_corpus must be a list of lists of 4 integers"
        assert context_length % 4 == 0, "context_length must be a multiple of 4"

        corpus = list(chain.from_iterable(chain.from_iterable(tetrad_corpus)))
        assert min(set(corpus) - {0}) > 35 and max(corpus) < 82, "notes must be in range [36, 81] except for 0."

        self.corpus = corpus
        self.vocab = [0] + torch.arange(36, 82).tolist()
        self.vocab_size = len(self.vocab)
        self.context_length = context_length

    def __len__(self):
        return len(self.corpus) - self.context_length

    def __getitem__(self, idx):
        context = self.corpus[idx : idx + self.context_length + 1]
        x = torch.tensor(context[:-1], dtype=torch.long)
        y = torch.tensor(context[1:], dtype=torch.long)
        return x, y
