# src/ocrsmith/core/text_renderers/TextRendererInterface.py

from abc import ABC, abstractmethod

class TextRendererInterface:
    @abstractmethod
    def generate_text_image():
        pass
    