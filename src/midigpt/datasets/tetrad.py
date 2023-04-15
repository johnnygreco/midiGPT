from itertools import chain

import torch
from torch.utils.data import Dataset

__all__ = ["BachChoraleDataset", "BachChoralesEncoder"]


class BachChoralesEncoder:
    _C1 = 36
    _A5 = 81

    def __init__(self, c1_encoded=1):
        self.c1_encoded = c1_encoded
        self.shift = self._C1 - c1_encoded
        self.vocab = [0] + (torch.arange(self._C1, self._A5 + 1) - self.shift).tolist()
        self.vocab_size = len(self.vocab)

    def encode(self, chorale):
        chorale = torch.as_tensor(chorale)
        chorale[chorale > 0] = chorale[chorale > 0] - self.shift
        return chorale

    def decode(self, chorale):
        chorale = torch.as_tensor(chorale)
        chorale[chorale > 0] = chorale[chorale > 0] + self.shift
        return chorale


class BachChoraleDataset(Dataset):
    _C1 = 36
    _A5 = 81

    def __init__(self, chorales, context_length, c1_encoded=1):
        assert len(chorales[0][0]) == 4, "chorales must be a list of lists of 4 integers"
        assert context_length % 4 == 0, "context_length must be a multiple of 4"
        self.encoder = BachChoralesEncoder(c1_encoded=c1_encoded)
        corpus = torch.tensor(list(chain.from_iterable(chain.from_iterable(chorales))), dtype=torch.long)
        corpus = self.encoder.encode(corpus)
        self.corpus = corpus.tolist()
        self.shift = self._C1 - c1_encoded
        self.min_note = self._C1 - self.shift
        self.max_note = self._A5 - self.shift
        self.vocab = [0] + (torch.arange(self._C1, self._A5 + 1) - self.shift).tolist()
        self.vocab_size = len(self.vocab)
        self.context_length = context_length
        assert min(set(self.corpus) - {0}) >= self.min_note and max(self.corpus) <= self.max_note, "invalid note range"

    def __len__(self):
        return len(self.corpus) - self.context_length

    def __getitem__(self, idx):
        context = self.corpus[idx : idx + self.context_length + 1]
        x = torch.tensor(context[:-1], dtype=torch.long)
        y = torch.tensor(context[1:], dtype=torch.long)
        return x, y
