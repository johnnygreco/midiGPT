from typing import Type, Union

from .tetrad import TetradNoteDataset
from .text_character import TextCharacterDataset, TextCharacterTokenizer

DatasetType = Union[Type[TetradNoteDataset], Type[TextCharacterDataset]]
