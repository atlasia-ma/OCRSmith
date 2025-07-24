# src/ocrsmith/datasets/loaders

from ..TextLoaderInterface import TextLoaderInterface
from abc import abstractmethod

class BaseTextDataLoader(TextLoaderInterface):
    @abstractmethod
    def load_texts():
        pass
    
    @abstractmethod
    def __iter__(self):
        pass
