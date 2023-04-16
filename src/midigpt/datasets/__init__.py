from typing import Type, Union

from .tetrad import BachChoraleDataset, BachChoralesEncoder
from .text_character import TextCharacterDataset, TextCharacterTokenizer

DatasetType = Union[Type[BachChoraleDataset], Type[TextCharacterDataset]]
