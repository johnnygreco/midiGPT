from pathlib import Path
from typing import List, Union

import torch
from torch.utils.data import Dataset

__all__ = ["TextCharacterDataset", "TextCharacterTokenizer"]


class TextCharacterTokenizer:
    def __init__(self, vocabulary: List[str]):
        self.vocabulary = vocabulary
        self.vocab_size = len(self.vocabulary)
        self.str_to_int = {s: i for i, s in enumerate(self.vocabulary)}
        self.int_to_str = {i: s for i, s in enumerate(self.vocabulary)}

    @classmethod
    def from_corpus(cls, text_corpus: str):
        vocabulary = sorted(set(text_corpus))
        return cls(vocabulary)

    @classmethod
    def from_file(cls, file_path: Union[str, Path]):
        with open(file_path, "r") as f:
            text_corpus = f.read()
        return cls.from_corpus(text_corpus)

    def encode(self, text: str):
        return [self.str_to_int[s] for s in text]

    def decode(self, tokens: List[int]):
        return "".join([self.int_to_str[t] for t in tokens])


class TextCharacterDataset(Dataset):
    def __init__(self, text_corpus: str, vocabulary: List[str], context_length: int):
        self.text_corpus = text_corpus
        self.vocabulary = vocabulary
        self.vocab_size = len(self.vocabulary)
        self.context_length = context_length
        self.tokenizer = TextCharacterTokenizer(vocabulary)

    def __len__(self):
        return len(self.text_corpus) - self.context_length

    def __getitem__(self, idx):
        context = self.text_corpus[idx : idx + self.context_length + 1]
        context_encoded = self.tokenizer.encode(context)
        x = torch.tensor(context_encoded[:-1], dtype=torch.long)
        y = torch.tensor(context_encoded[1:], dtype=torch.long)
        return x, y
