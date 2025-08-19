# src/ocrsmith/core/TextRenderingManager.py

from ocrsmith.config import AppConfig
from .text_renderers.TextRenderingContext import TextRenderingContext
from .text_renderers.strategies import HorizontalRenderingStrategy
from .text_renderers.strategies.SimpleTextRenderingStrategy import SimpleTextRenderingStrategy
from .text_renderers.strategies.VerticalRenderingStrategy import VerticalRenderingStrategy
import random
from typing import Dict, Any, List

class TextRenderingManager:
    """Manages text rendering strategies and provides random selection."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.context = TextRenderingContext()
        self.strategy_registry = self._build_strategy_registry()
        self.text_configs = self._parse_text_rendering_configs()
        
        if not self.text_configs:
            raise ValueError("No text renderers were configured. Please check your configuration.")
    
    def _build_strategy_registry(self) -> Dict[str, type]:
        """Build registry mapping text renderer types to strategy classes."""
        return {
            'horizontal': HorizontalRenderingStrategy,
            'simple': SimpleTextRenderingStrategy,
            'vertical': VerticalRenderingStrategy,
        }
    
    def _parse_text_rendering_configs(self) -> List[Dict[str, Any]]:
        """Parse config text renderers into strategy configurations."""
        configs = []
        
        for text_renderer in self.config.text_renderers:
            text_config = text_renderer.model_dump(exclude_none=True)
            text_type = text_config.pop('type', None)
            weight = float(text_config.pop('weight', 1.0))
            
            if text_type not in self.strategy_registry:
                available = ", ".join(self.strategy_registry.keys())
                raise ValueError(f"Unknown text renderer type '{text_type}'. Available: [{available}]")
            
            configs.append({
                'type': text_type,
                'strategy_class': self.strategy_registry[text_type],
                'params': text_config,
                'weight': max(0.0, weight)
            })
        
        return configs
    
    def get_random_text_renderer(self):
        """Get a random text rendering strategy with its configuration."""
        if not self.text_configs:
            raise ValueError("No text renderer configurations available")
        
        weights = [cfg.get('weight', 1.0) for cfg in self.text_configs]
        config = random.choices(self.text_configs, weights=weights, k=1)[0]
        strategy = config['strategy_class'](**config['params'])
        self.context.set_strategy(strategy)
        return self.context
    
    def get_text_renderer_by_type(self, renderer_type: str):
        """Get a specific text renderer type (first match if multiple exist)."""
        matching_configs = [cfg for cfg in self.text_configs if cfg['type'] == renderer_type]
        
        if not matching_configs:
            available = list(set(cfg['type'] for cfg in self.text_configs))
            raise ValueError(f"No text renderer of type '{renderer_type}' found. Available: {available}")
        
        config = matching_configs[0]
        strategy = config['strategy_class'](**config['params'])
        self.context.set_strategy(strategy)
        return self.context
    
    def render_random_text(self, font, text: str, **kwargs):
        """Convenience method to render text with a random strategy."""
        context = self.get_random_text_renderer()
        return context.render_text(font, text, **kwargs)