# src/ocrsmith/core/backgrounds/strategies/BaseBackground.py
from ocrsmith.config import AppConfig, load_config

from ..BackgroundStrategy import BackgroundStrategy


class BaseBackground(BackgroundStrategy):
    def __init__(self):
        self.config = load_config() or AppConfig()

    def render(self, width: int, height: int, **kwargs):
        raise NotImplementedError("Subclasses must implement the render() method.")
