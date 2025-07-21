# src/ocrsmith/core/text_renderers/TextRendererInterface.py

from abc import ABC, abstractmethod

class BaseTextRenderer:
    @abstractmethod
    def generate_text_image():
        pass