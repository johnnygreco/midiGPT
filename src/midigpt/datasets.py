import torch
from torch.utils.data import Dataset

__all__ = ["DatasetType", "TextCharacterDataset"]


class TextCharacterDataset(Dataset):
    def __init__(self, text_corpus: str, context_length: int):
        self.text_corpus = text_corpus
        self.vocabulary = sorted(list(set(text_corpus)))
        self.vocab_size = len(self.vocabulary)
        self.str_to_int = {s: i for i, s in enumerate(self.vocabulary)}
        self.int_to_str = {i: s for i, s in enumerate(self.vocabulary)}
        self.context_length = context_length

    def __len__(self):
        return len(self.text_corpus) - self.context_length

    def __getitem__(self, idx):
        context = self.text_corpus[idx : idx + self.context_length + 1]
        context_encoded = [self.str_to_int[s] for s in context]
        x = torch.tensor(context_encoded[:-1], dtype=torch.long)
        y = torch.tensor(context_encoded[1:], dtype=torch.long)
        return x, y


DatasetType = TextCharacterDataset
