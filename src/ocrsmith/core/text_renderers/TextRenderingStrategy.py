# src/ocrsmith/core/text_renderers/TextRenderingStrategy.py
from abc import ABC, abstractmethod

class TextRenderingStrategy(ABC):
    @abstractmethod
    def render_text():
        pass
    