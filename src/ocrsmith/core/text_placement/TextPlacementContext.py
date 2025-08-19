# src/ocrsmith/core/text_placement/TextPlacementContext.py
class TextPlacementContext:
    
    def __init__(self, strategy=None):
        self._strategy = strategy
    
    def set_strategy(self, strategy):
        self._strategy = strategy
    
    def place_text(self, text_image, background_image, **kwargs):
        if not self._strategy:
            raise ValueError("No text placement strategy set")
        return self._strategy.place_text(text_image, background_image, **kwargs)
    