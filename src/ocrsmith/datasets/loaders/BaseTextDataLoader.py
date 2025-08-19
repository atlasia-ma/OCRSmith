# src/ocrsmith/datasets/loaders

from ..TextLoaderInterface import TextLoaderInterface
from abc import abstractmethod

class BaseTextDataLoader(TextLoaderInterface):
    def __init__(self, text_column='text', title_column=None):
        self.text_column = text_column
        self.title_column = title_column
        self.texts = []    
    @abstractmethod
    def load_texts():
        pass
    
    @abstractmethod
    def __iter__(self):
        pass
