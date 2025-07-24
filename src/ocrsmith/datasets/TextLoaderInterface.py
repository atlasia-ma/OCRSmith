# src/ocrsmith/datasets/TextLoaderInterface.py

from abc import ABC, abstractmethod

class TextLoaderInterface(ABC):
    @abstractmethod
    def load_texts():
        pass
    
    @abstractmethod
    def __iter__(self):
        pass
    