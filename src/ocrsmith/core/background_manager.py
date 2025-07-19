# src/ocrsmith/core/background_manager.py

import random

class BackgroundManager:
    def __init__(self, background_configs, factory):
        self.backgrounds = [
            factory.create(cfg['type'], **cfg)
            for cfg in background_configs
        ]
        if not self.backgrounds:
            raise ValueError("No backgrounds were created. Please check your configuration.")

    def get_random_background(self):
        return random.choice(self.backgrounds)
    