# src/ocrsmith/core/text_renderers/TextRenderingStrategy.py
from abc import ABC, abstractmethod

class TextRenderingStrategy(ABC):
    @abstractmethod
    def render_text(self, font, text, **kwargs):
        """Render text to an image, returning (image, mask, (w, h))."""
        raise NotImplementedError
    