# src/ocrsmith/core/app.py

from ocrsmith.config import load_config
from ocrsmith.core import OCRSmithEngine, BackgroundManager, FontManager
from ocrsmith.core.backgrounds import BackgroundFactory, SolidColorBackground, NoiseBackground, ImageBackground, GradientBackground

def main():
    # Load configuration
    configs = load_config()
    
    # Initialize engine
    engine = OCRSmithEngine(configs)
    
    factory = BackgroundFactory()
    factory.register_creator('solid', SolidColorBackground)
    factory.register_creator('image', ImageBackground)
    factory.register_creator('noise', NoiseBackground)
    factory.register_creator('gradient', GradientBackground)
    
    engine.background_manager = BackgroundManager(configs, factory)
    engine.font_manager = FontManager(font_paths=["assets/fonts"], default_size=24)
    
    engine.text_data_manager.load_from_source(
        'csv', 
        'assets/text_data/sentences.csv', 
        text_column='darija_ar'
    )
    
    # Setup augmentations
    augmentation_config = {
        'noise': {'enabled': True, 'factor': 0.05, 'probability': 0.3},
        'blur': {'enabled': True, 'radius': 0.5, 'probability': 0.2}
    }
    engine.setup_augmentations(augmentation_config)
    
    # Generate samples
    engine.generate_dataset(num_samples=100, output_dir="training_data")

if __name__ == "__main__":
    main()