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
        
        # Return requested strategy if it exists
        if strategy_name in self.strategies:
            return self.strategies[strategy_name]
        
        # Return default strategy if it exists and requested strategy doesn't
        if self.default_strategy in self.strategies and strategy_name != self.default_strategy:
            return self.strategies[self.default_strategy]
        
        # If neither exists, raise a descriptive error
        available_strategies = list(self.strategies.keys())
        raise ValueError(
            f"Strategy '{strategy_name}' not found. "
            f"Available strategies: {available_strategies}. "
            f"Please register the strategy first."
        )
    
    def place_text(self, text_image, background_image, strategy_name=None, **kwargs) -> PlacementResult:
        strategy = self.get_strategy(strategy_name)
        return strategy.place_text(text_image, background_image, **kwargs)