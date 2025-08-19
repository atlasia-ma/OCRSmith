# src/ocrsmith/core/TextPlacementManager.py

from ocrsmith.config import AppConfig
from .text_placement.TextPlacementContext import TextPlacementContext
from .text_placement.strategies import CenterPlacementStrategy, RandomPlacementStrategy, GridPlacementStrategy, PageNumberPlacementStrategy, PageTitlePlacementStrategy

import random
from typing import Dict, Any, List

class TextPlacementManager:
    """Manages text placement strategies and provides random selection."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.context = TextPlacementContext()
        self.strategy_registry = self._build_strategy_registry()
        self.placement_configs = self._parse_placement_configs()
        
        if not self.placement_configs:
            raise ValueError("No text placements were configured. Please check your configuration.")
    
    def _build_strategy_registry(self) -> Dict[str, type]:
        """Build registry mapping text placement types to strategy classes."""
        return {
            'center': CenterPlacementStrategy,
            'random': RandomPlacementStrategy,
            'grid': GridPlacementStrategy,
            'page_number': PageNumberPlacementStrategy,
            'page_title': PageTitlePlacementStrategy
        }
    
    def _parse_placement_configs(self) -> List[Dict[str, Any]]:
        """Parse config text placements into strategy configurations."""
        configs = []
        
        for placement in self.config.text_placements:
            placement_config = placement.model_dump(exclude_none=True)
            placement_type = placement_config.pop('type', None)
            weight = float(placement_config.pop('weight', 1.0))
            
            if placement_type not in self.strategy_registry:
                available = ", ".join(self.strategy_registry.keys())
                raise ValueError(f"Unknown text placement type '{placement_type}'. Available: [{available}]")
            
            configs.append({
                'type': placement_type,
                'strategy_class': self.strategy_registry[placement_type],
                'params': placement_config,
                'weight': max(0.0, weight)
            })
        
        return configs
    
    def get_random_placement(self):
        """Get a random text placement strategy with its configuration."""
        if not self.placement_configs:
            raise ValueError("No text placement configurations available")
        
        weights = [cfg.get('weight', 1.0) for cfg in self.placement_configs]
        config = random.choices(self.placement_configs, weights=weights, k=1)[0]
        strategy = config['strategy_class'](**config['params'])
        self.context.set_strategy(strategy)
        return self.context
    
    def get_placement_by_type(self, placement_type: str):
        """Get a specific text placement type (first match if multiple exist)."""
        matching_configs = [cfg for cfg in self.placement_configs if cfg['type'] == placement_type]
        
        if not matching_configs:
            available = list(set(cfg['type'] for cfg in self.placement_configs))
            raise ValueError(f"No text placement of type '{placement_type}' found. Available: {available}")
        
        config = matching_configs[0]
        strategy = config['strategy_class'](**config['params'])
        self.context.set_strategy(strategy)
        return self.context
    
    def place_random_text(self, text_image, background_image, **kwargs):
        """Convenience method to place text with a random strategy."""
        context = self.get_random_placement()
        return context.place_text(text_image, background_image, **kwargs)