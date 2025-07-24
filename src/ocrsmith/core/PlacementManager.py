# src/ocrsmith/core/text_placement/placers/CenterPlacementStrategy.py

from .text_placement import TextPlacementStrategy, PlacementResult

class PlacementManager:
    """Manages text placement strategies and handles composition"""
    def __init__(self):
        self.strategies = {}
        self.default_strategy = 'random'
    
    def register_strategy(self, name: str, strategy: TextPlacementStrategy):
        self.strategies[name] = strategy
    
    def get_strategy(self, name: str = None) -> TextPlacementStrategy:
        strategy_name = name or self.default_strategy
        return self.strategies.get(strategy_name, self.strategies[self.default_strategy])
    
    def place_text(self, text_image, background_image, strategy_name=None, **kwargs) -> PlacementResult:
        strategy = self.get_strategy(strategy_name)
        return strategy.place_text(text_image, background_image, **kwargs)
    