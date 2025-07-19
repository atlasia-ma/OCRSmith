# src/ocrsmith/core/backgrounds/BackgroundInterface.py

from abc import ABC, abstractmethod

class BackgroundInterface(ABC):
    @abstractmethod
    def render(self):
        """Render the background"""
        pass
