from itertools import chain

import torch
from torch.utils.data import Dataset

__all__ = ["BachChoraleDataset"]


class BachChoraleDataset(Dataset):
    _C1 = 36
    _A5 = 81

    def __init__(self, chorales, context_length, shift_c1_to_one=True):
        assert len(chorales[0][0]) == 4, "tetrad_corpus must be a list of lists of 4 integers"
        assert context_length % 4 == 0, "context_length must be a multiple of 4"

        self.corpus = list(chain.from_iterable(chain.from_iterable(chorales)))
        assert min(set(self.corpus) - {0}) >= self._C1 and max(self.corpus) <= self._A5, "notes not in 0 + [36, 81]"
        self.vocab = [0] + torch.arange(self._C1, self._A5 + 1).tolist()
        self.vocab_size = len(self.vocab)
        self.context_length = context_length
        self.shift_c1_to_one = shift_c1_to_one

    def __len__(self):
        return len(self.corpus) - self.context_length

    def __getitem__(self, idx):
        context = self.corpus[idx : idx + self.context_length + 1]
        shift = self._C1 - 1 if self.shift_c1_to_one else 0
        x = torch.tensor(context[:-1], dtype=torch.long) - shift
        y = torch.tensor(context[1:], dtype=torch.long) - shift
        return x, y
