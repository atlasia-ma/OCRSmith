# src/ocrsmith/core/backgrounds/BackgroundFactory.py

class BackgroundFactory:
    def __init__(self):
        self._builders = {}

    def register_builder(self, key, builder):
        self._builders[key] = builder

    def create(self, key, **kwargs):
        builder = self._builders.get(key)
        if not builder:
            available_keys = ", ".join(self._builders.keys())
            raise ValueError(
                f"Unknown builder key '{key}'. "
                f"Available builders are: [{available_keys}]"
            )
        print(kwargs)
        return builder(**kwargs)
    