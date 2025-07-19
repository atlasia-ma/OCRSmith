# src/ocrsmith/core/backgrounds/BackgroundFactory.py

class BackgroundFactory:
    def __init__(self):
        self._creators = {}

    def register_creator(self, key, creator):
        self._creators[key] = creator

    def create(self, key, **kwargs):
        creator = self._creators.get(key)
        if not creator:
            available_keys = ", ".join(self._creators.keys())
            raise ValueError(
                f"Unknown creator key '{key}'. "
                f"Available creators are: [{available_keys}]"
            )
        return creator(**kwargs)
    