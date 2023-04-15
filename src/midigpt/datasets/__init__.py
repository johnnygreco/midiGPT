from typing import Type, Union

from .tetrad import BachChoraleDataset
from .text_character import TextCharacterDataset, TextCharacterTokenizer

DatasetType = Union[Type[BachChoraleDataset], Type[TextCharacterDataset]]
