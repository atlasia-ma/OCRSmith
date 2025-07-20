# src/ocrsmith/core/background_manager.py

from ocrsmith.config import AppConfig
from .backgrounds import BackgroundFactory
import random

class BackgroundManager:
    def __init__(self, config: AppConfig, factory: BackgroundFactory):
        self.backgrounds = [
            factory.create(background.type, **vars(background))
            for background in config.backgrounds
        ]
        if not self.backgrounds:
            raise ValueError("No backgrounds were created. Please check your configuration.")

    def get_random_background(self):
        return random.choice(self.backgrounds)
    