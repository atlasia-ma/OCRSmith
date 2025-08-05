# src/ocrsmith/core/OCRSmithEngine.py

from .AugmentationPipeline import AugmentationPipeline
from .PlacementManager import PlacementManager
from .TextRenderer import TextRenderer
from ..datasets import TextDataManager

from .text_placement import CenterPlacementStrategy, RandomPlacementStrategy, GridPlacementStrategy, PageNumberPlacementStrategy, PageTitlePlacementStrategy
from .augmentation import BlurAugmentation, NoiseAugmentation
from .text_renderers import HorizontalRenderingStrategy

class OCRSmithEngine:
    """Main engine that orchestrates all components"""
    def __init__(self, config):
        self.config = config
        
        # Initialize managers
        self.background_manager = None
        self.font_manager = None
        self.placement_manager = PlacementManager()
        self.text_data_manager = TextDataManager()
        self.augmentation_pipeline = AugmentationPipeline()
        
        # Setup default strategies
        self._setup_default_strategies()
    
    def _setup_default_strategies(self):
        """Setup default placement strategies"""
        self.placement_manager.register_strategy('random', RandomPlacementStrategy())
        self.placement_manager.register_strategy('center', CenterPlacementStrategy())
        self.placement_manager.register_strategy('grid', GridPlacementStrategy())
        self.placement_manager.register_strategy('title', PageTitlePlacementStrategy())
        self.placement_manager.register_strategy('page_number', PageNumberPlacementStrategy())
    
    def setup_augmentations(self, augmentation_config):
        """Setup augmentation pipeline from config"""
        if augmentation_config.get('noise', {}).get('enabled', False):
            noise_aug = NoiseAugmentation(
                noise_factor=augmentation_config['noise'].get('factor', 0.1)
            )
            self.augmentation_pipeline.add_augmentation(
                noise_aug, 
                augmentation_config['noise'].get('probability', 0.5)
            )
        
        if augmentation_config.get('blur', {}).get('enabled', False):
            blur_aug = BlurAugmentation(
                blur_radius=augmentation_config['blur'].get('radius', 1.0)
            )
            self.augmentation_pipeline.add_augmentation(
                blur_aug,
                augmentation_config['blur'].get('probability', 0.3)
            )
        
        # Add more augmentations as needed
    
    def generate_sample(self, text=None, placement_strategy='random'):
        """Generate a single OCR training sample"""
        # Get text
        if text is None:
            text = self.text_data_manager.get_random_text()
        
        if not text:
            raise ValueError("No text provided and no text data loaded")
        
        # Generate text image
        font = self.font_manager.load_font()
        text_renderer = TextRenderer(HorizontalRenderingStrategy())
        text_image, mask, (width, height) = text_renderer.generate_text_image(font, text)
        
        # Generate background
        background_creator = self.background_manager.get_random_background()
        background_image = background_creator.render(width + 100, height + 100).convert("RGBA")
        
        # Place text on background using PlacementManager
        placement_result = self.placement_manager.place_text(
            text_image, 
            background_image, 
            strategy_name=placement_strategy
        )
        
        # Apply augmentations to the composed image
        final_image = self.augmentation_pipeline.apply_all(placement_result.composed_image)
        
        return final_image, text, placement_result.bbox, placement_result.metadata
    
    def generate_dataset(self, num_samples, output_dir="output"):
        """Generate multiple samples for training"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        annotations = []
        
        for i in range(num_samples):
            try:
                image, text, bbox, metadata = self.generate_sample()
                
                # Save image
                image_path = os.path.join(output_dir, f"sample_{i:06d}.png")
                image.save(image_path)
                
                # Store annotation with metadata
                annotations.append({
                    'image_path': image_path,
                    'text': text,
                    'bbox': bbox,
                    'placement_metadata': metadata
                })
                
                if i % 100 == 0:
                    print(f"Generated {i} samples...")
                    
            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                continue
        
        # Save annotations
        import json
        with open(os.path.join(output_dir, 'annotations.json'), 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
        
        return annotations
