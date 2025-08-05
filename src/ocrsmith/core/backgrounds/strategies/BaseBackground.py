# src/ocrsmith/core/backgrounds/strategies/BaseBackground.py
from ..BackgroundStrategy import BackgroundStrategy
from ocrsmith.config import load_config, AppConfig

class BaseBackground(BackgroundStrategy):
    def __init__(self):
        self.config = load_config() or AppConfig()
    
    def render(self, width: int, height: int, **kwargs):
        raise NotImplementedError("Subclasses must implement the render() method.")
