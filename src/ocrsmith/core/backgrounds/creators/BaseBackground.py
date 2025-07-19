# src/ocrsmith/core/backgrounds/creators/BaseBackground.py

from ..BackgroundInterface import BackgroundInterface
from ocrsmith.config import load_config, AppConfig

class BaseBackground(BackgroundInterface):
    def __init__(self):
        self.config = load_config() or AppConfig()

    def render(self):
        raise NotImplementedError("Subclasses must implement the render() method.")
