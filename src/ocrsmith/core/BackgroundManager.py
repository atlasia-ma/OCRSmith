# src/ocrsmith/core/BackgroundManager.py

from ocrsmith.config import AppConfig
from .backgrounds.BackgroundContext import BackgroundContext
from .backgrounds.strategies.ImageBackground import ImageBackground
from .backgrounds.strategies.SolidColorBackground import SolidColorBackground
from .backgrounds.strategies.GradientBackground import GradientBackground
from .backgrounds.strategies.TextureBackground import TextureBackground
import random
from typing import Dict, Any, List

class BackgroundManager:
    """Manages background strategies and provides random selection."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.context = BackgroundContext()
        self.strategy_registry = self._build_strategy_registry()
        self.background_configs = self._parse_background_configs()
        
        if not self.background_configs:
            raise ValueError("No backgrounds were configured. Please check your configuration.")
    
    def _build_strategy_registry(self) -> Dict[str, type]:
        """Build registry mapping background types to strategy classes."""
        return {
            'image': ImageBackground,
            'solid': SolidColorBackground,
            'gradient': GradientBackground,
            'texture': TextureBackground
        }
    
    def _parse_background_configs(self) -> List[Dict[str, Any]]:
        """Parse config backgrounds into strategy configurations."""
        configs = []
        
        for background in self.config.backgrounds:
            bg_config = background.model_dump(exclude_none=True)
            bg_type = bg_config.pop('type', None)
            weight = float(bg_config.pop('weight', 1.0))
            
            if bg_type not in self.strategy_registry:
                available = ", ".join(self.strategy_registry.keys())
                raise ValueError(f"Unknown background type '{bg_type}'. Available: [{available}]")
            
            configs.append({
                'type': bg_type,
                'strategy_class': self.strategy_registry[bg_type],
                'params': bg_config,
                'weight': max(0.0, weight)
            })
        
        return configs
    
    def get_random_background(self):
        """Get a random background strategy with its configuration."""
        if not self.background_configs:
            raise ValueError("No background configurations available")
        
        # Select random background config
        weights = [cfg.get('weight', 1.0) for cfg in self.background_configs]
        config = random.choices(self.background_configs, weights=weights, k=1)[0]
        
        # Create strategy instance
        strategy = config['strategy_class'](**config['params'])
        
        # Set strategy in context
        self.context.set_strategy(strategy)
        
        return self.context
    
    def get_background_by_type(self, bg_type: str):
        """Get a specific background type (first match if multiple exist)."""
        matching_configs = [cfg for cfg in self.background_configs if cfg['type'] == bg_type]
        
        if not matching_configs:
            available = list(set(cfg['type'] for cfg in self.background_configs))
            raise ValueError(f"No background of type '{bg_type}' found. Available: {available}")
        
        config = matching_configs[0]
        strategy = config['strategy_class'](**config['params'])
        self.context.set_strategy(strategy)
        
        return self.context
    
    def get_all_backgrounds(self) -> List[BackgroundContext]:
        """Get all configured backgrounds as separate context instances."""
        contexts = []
        
        for config in self.background_configs:
            strategy = config['strategy_class'](**config['params'])
            context = BackgroundContext(strategy)
            contexts.append(context)
        
        return contexts
    
    def render_random_background(self, width: int, height: int, **kwargs):
        """Convenience method to render a random background directly."""
        context = self.get_random_background()
        return context.render(width, height, **kwargs)
    
    def get_available_types(self) -> List[str]:
        """Get list of available background types."""
        return list(self.strategy_registry.keys())
    
    def get_configured_types(self) -> List[str]:
        """Get list of configured background types."""
        return [cfg['type'] for cfg in self.background_configs]


# Usage examples and migration guide:

def example_basic_usage():
    """Basic usage example."""
    from ocrsmith.config import load_config
    
    # Load your existing config
    config = load_config()
    
    # Create manager (replaces factory)
    manager = BackgroundManager(config)
    
    # Get random background (same as before, but returns context)
    context = manager.get_random_background()
    img = context.render(800, 600)
    
    # Or render directly
    img = manager.render_random_background(800, 600)
    
    return img


def example_specific_background():
    """Get specific background type."""
    from ocrsmith.config import load_config
    
    config = load_config()
    manager = BackgroundManager(config)
    
    # Get specific background type
    image_context = manager.get_background_by_type('image')
    img = image_context.render(800, 600, image_path='custom.jpg', mode='crop')
    
    gradient_context = manager.get_background_by_type('gradient')
    img2 = gradient_context.render(800, 600, direction='diagonal')
    
    return img, img2


def example_advanced_usage():
    """Advanced usage with parameter overrides."""
    from ocrsmith.config import load_config
    
    config = load_config()
    manager = BackgroundManager(config)
    
    # Get random background but override parameters
    context = manager.get_random_background()
    
    # Override parameters at render time
    img1 = context.render(800, 600, color=(255, 0, 0))  # Override solid color
    img2 = context.render(800, 600, image_path='new.jpg', mode='tile')  # Override image
    
    return img1, img2
