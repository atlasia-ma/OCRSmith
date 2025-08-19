# src/ocrsmith/core/AugmentationManager.py

from .augmentation import AugmentationContext, BlurAugmentation, NoiseAugmentation, RotationAugmentation, BrightnessAugmentation
from typing import Tuple, Union


from ocrsmith.config import AppConfig

import random
from typing import Dict, Any, List

class AugmentationManager:
    """Manages augmentation strategies and provides random selection."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.context = AugmentationContext()
        self.strategy_registry = self._build_strategy_registry()
        self.augmentation_configs = self._parse_augmentation_configs()

    def _build_strategy_registry(self) -> Dict[str, type]:
        """Build registry mapping augmentation types to strategy classes."""
        return {
            'blur': BlurAugmentation,
            'noise': NoiseAugmentation,
            'rotation': RotationAugmentation,
            'brightness': BrightnessAugmentation
        }
    
    def _parse_augmentation_configs(self) -> List[Dict[str, Any]]:
        """Parse config augmentations into strategy configurations."""
        configs = []
        
        for augmentation in self.config.augmentations:
            aug_config = augmentation.model_dump(exclude_none=True)
            aug_type = aug_config.pop('type', None)
            probability = float(aug_config.pop('probability', 1.0))
            enabled = bool(aug_config.pop('enabled', True))
            weight = float(aug_config.pop('weight', 1.0))
            
            if aug_type not in self.strategy_registry:
                available = ", ".join(self.strategy_registry.keys())
                raise ValueError(f"Unknown augmentation type '{aug_type}'. Available: [{available}]")
            
            configs.append({
                'type': aug_type,
                'strategy_class': self.strategy_registry[aug_type],
                'params': aug_config,
                'probability': max(0.0, min(1.0, probability)),
                'enabled': enabled,
                'weight': max(0.0, weight)
            })
        
        return configs
    
    def get_random_augmentation(self):
        """Get a random augmentation strategy with its configuration."""
        if not self.augmentation_configs:
            raise ValueError("No augmentation configurations available")
        
        config = random.choice(self.augmentation_configs)
        strategy = config['strategy_class'](**config['params'])
        self.context.set_strategy(strategy)
        return self.context
    
    def get_augmentation_by_type(self, aug_type: str):
        """Get a specific augmentation type (first match if multiple exist)."""
        matching_configs = [cfg for cfg in self.augmentation_configs if cfg['type'] == aug_type]
        
        if not matching_configs:
            available = list(set(cfg['type'] for cfg in self.augmentation_configs))
            raise ValueError(f"No augmentation of type '{aug_type}' found. Available: {available}")
        
        config = matching_configs[0]
        strategy = config['strategy_class'](**config['params'])
        self.context.set_strategy(strategy)
        return self.context
    
    def _sample_value(self, value: Union[float, Tuple[float, float]]) -> float:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            lo, hi = float(value[0]), float(value[1])
            if hi < lo:
                lo, hi = hi, lo
            return random.uniform(lo, hi)
        return float(value)

    def apply_pipeline(self, image):
        """Apply all configured augmentations in random order honoring per-augmentation probability.
        Returns the possibly augmented image."""
        if not self.augmentation_configs:
            return image
        # Shuffle a shallow copy
        pipeline = [cfg for cfg in self.augmentation_configs if cfg.get('enabled', True)]
        if self.config.augmentation_order == 'random':
            random.shuffle(pipeline)
        out = image
        for cfg in pipeline:
            if random.random() > cfg.get('probability', 1.0):
                continue
            params = dict(cfg['params'])
            # Sample ranges if present
            for key, val in list(params.items()):
                if isinstance(val, (list, tuple)):
                    params[key] = self._sample_value(val)  # type: ignore
            strategy = cfg['strategy_class'](**params)
            self.context.set_strategy(strategy)
            out = self.context.apply(out)
        return out
    