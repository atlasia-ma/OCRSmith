# OCRSmith

**OCRSmith** is a powerful Python library for generating **synthetic OCR datasets** with comprehensive support for **Arabic and Latin text**. It provides a flexible, modular architecture for creating high-quality training data for OCR models from various text sources.

---

## 🚀 Features

### Core Functionality

- **Synthetic text image generation** with configurable fonts and backgrounds
- **Multi-language support**: Arabic and Latin text rendering with proper font handling
- **Flexible text placement strategies**: random, centered, grid-based, and contextual positioning
- **Rich augmentation pipeline**: noise, blur, brightness, rotation, and custom effects
- **Comprehensive background generation**: solid colors, gradients, noise patterns, and custom images

### Text Placement & Layout

- **Contextual placement strategies**:
  - **Page titles**: Top-centered positioning with proper margins
  - **Page numbers**: Bottom-right corner placement
  - **Random placement**: Within configurable margins
  - **Grid-based placement**: Structured positioning
  - **Center placement**: Perfect centering on backgrounds
- **Smart composition**: Each placement strategy handles image composition internally
- **Rich metadata**: Detailed placement information for training optimization

### Data Sources

- **Multiple input formats**:
  - CSV files with configurable text columns
  - Hugging Face datasets with automatic loading
  - Parquet files for efficient data handling
  - Direct text input
- **Batch processing**: Generate thousands of samples efficiently
- **Memory-optimized**: Iterator-based text loading for large datasets

### Augmentation System

- **Pipeline-based augmentation**: Chain multiple effects with probability control
- **Built-in augmentations**:
  - Gaussian noise injection
  - Blur effects
  - Brightness adjustment
  - Rotation transforms
- **Extensible**: Easy to add custom augmentation strategies
- **Configurable probabilities**: Fine-tune augmentation frequency

---

## 🏗️ Architecture

OCRSmith follows a modular, strategy-pattern architecture:

```
OCRSmith/
├── core/
│   ├── BackgroundManager.py       # Background generation orchestration
│   ├── FontManager.py            # Font loading and caching
│   ├── TextRenderer.py           # Text-to-image rendering
│   ├── placement/                # Text placement strategies
│   │   ├── RandomPlacement       # Random positioning
│   │   ├── CenterPlacement       # Centered positioning
│   │   ├── GridPlacement         # Grid-based positioning
│   │   ├── PageTitlePlacement    # Title positioning
│   │   └── PageNumberPlacement   # Page number positioning
│   ├── backgrounds/              # Background generation
│   │   ├── SolidColorBackground  # Solid color backgrounds
│   │   ├── GradientBackground    # Gradient backgrounds
│   │   ├── NoiseBackground       # Noise pattern backgrounds
│   │   └── ImageBackground       # Custom image backgrounds
│   ├── augmentation/             # Image augmentation pipeline
│   │   ├── NoiseAugmentation     # Noise injection
│   │   ├── BlurAugmentation      # Blur effects
│   │   ├── BrightnessAugmentation # Brightness adjustment
│   │   └── RotationAugmentation  # Rotation transforms
│   └── fonts/                    # Font management system
├── datasets/                     # Data loading utilities
│   ├── CSVTextLoader            # CSV file support
│   ├── ParquetTextLoader        # Parquet file support
│   └── HuggingFaceTextLoader    # HuggingFace dataset support
└── config/                      # Configuration management
```

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/OCRSmith.git
cd OCRSmith

# Install in production mode
pip install .

# Install in development mode
pip install -e .
```

---

## 🎯 Quick Start

### Basic Usage

```python
from ocrsmith.core.BackgroundManager import BackgroundManager
from ocrsmith.core.backgrounds.BackgroundFactory import BackgroundFactory
from ocrsmith.core.backgrounds.creators import *
from ocrsmith.config import load_config
from ocrsmith.core.FontManager import FontManager
from ocrsmith.core.TextRenderer import TextRenderer
from ocrsmith.core.text_renderers.strategies.HorizontalRenderingStrategy import HorizontalRenderingStrategy
from ocrsmith.core.placement import PlacementManager, RandomPlacementStrategy

# Setup background factory
factory = BackgroundFactory()
factory.register_creator('solid', SolidColorBackground)
factory.register_creator('gradient', GradientBackground)
factory.register_creator('noise', NoiseBackground)
factory.register_creator('image', ImageBackground)

# Load configuration and initialize managers
configs = load_config()
background_manager = BackgroundManager(configs, factory)
font_manager = FontManager(font_paths=["assets/fonts"], default_size=24)

# Setup placement
placement_manager = PlacementManager()
placement_manager.register_strategy('random', RandomPlacementStrategy())

# Generate text image
font = font_manager.load_font(font_size=18)
text = "Sample text for OCR training"
text_renderer = TextRenderer(HorizontalRenderingStrategy())
text_image, mask, (width, height) = text_renderer.generate_text_image(font, text)

# Generate background
background_creator = background_manager.get_random_background()
background_image = background_creator.render(width + 100, height + 100)

# Place text and get composed image
placement_result = placement_manager.place_text(text_image, background_image, 'random')
final_image = placement_result.composed_image

# Save result
final_image.save("output.png")
```

### CLI Usage

```bash
python -m ocrsmith.core.app --config path/to/config.yaml --num-samples 100 \
  --output-dir outputs \
  --set text_data.source_path=assets/text_data/sentences.csv \
  --set text_data.text_column=darija_ar \
  --set seed=123 \
  --workers 4
```

Supported overrides (minimal): `output.images_dir`, `output.metadata_file`, `text_data.source_type`, `text_data.source_path`, `text_data.text_column`, `seed`.

Max image constraints:
- Set in YAML under `layout.max_width` and `layout.max_height` (e.g., 600 and 200). The renderer will trim text words until the rendered image fits.
- Example snippet in `config/default_config.yaml` shows defaults.

### Advanced Usage with OCRSmith Engine

```python
from ocrsmith.core import OCRSmithEngine

# Initialize engine
engine = OCRSmithEngine(configs)

# Load text data from CSV
engine.text_data_manager.load_from_source(
    'csv', 
    'data/texts.csv', 
    text_column='text'
)

"""
Augmentations are configured in YAML. Each augmentation supports `probability` and parameter ranges, e.g.:

augmentations:
  - type: blur
    blur_radius: [0.5, 2.0]
    probability: 0.3
  - type: rotation
    max_angle: [0, 5]
    probability: 0.2
"""

# Generate dataset
annotations = engine.generate_dataset(
    num_samples=1000, 
    output_dir="training_data"
)
```

---

## 🔧 Configuration

OCRSmith uses YAML configuration files for easy customization:

```yaml
# config/default_config.yaml
backgrounds:
  solid:
    enabled: true
    colors: ["#FFFFFF", "#F0F0F0", "#E0E0E0"]
  
  gradient:
    enabled: true
    directions: ["horizontal", "vertical", "diagonal"]
  
  noise:
    enabled: true
    intensity: [0.1, 0.3]

fonts:
  default_size: 24
  size_range: [16, 32]
  paths: ["assets/fonts"]

placement:
  default_strategy: "random"
  margins:
    x: 20
    y: 20

augmentation:
  noise:
    enabled: true
    factor: 0.05
    probability: 0.3
  
  blur:
    enabled: true
    radius: 0.5
    probability: 0.2
```

---

## 📊 Supported Font Collections

OCRSmith includes extensive font support:

### Arabic Fonts

- **Amiri**: Traditional Arabic typography (Regular, Bold, Italic, BoldItalic)
- **Fustat**: Modern Arabic font family (7 weights)
- **IBM Plex Sans Arabic**: Professional Arabic fonts (7 weights)
- **Kufam**: Versatile Arabic/Latin dual-script font (10 styles)
- **Mada**: Clean, modern Arabic font (8 weights)
- **Mirza**: Elegant Arabic display font (4 weights)
- **Noto Sans Arabic**: Google's comprehensive Arabic font family
- **Noto Kufi Arabic**: Kufi-style Arabic fonts
- **Noto Naskh Arabic**: Traditional Naskh Arabic fonts
- **Vazirmatn**: High-quality Persian/Arabic font (9 weights)

### Latin Fonts

- **IBM Plex Sans**: Modern, professional Latin fonts
- **Noto Sans Mono**: Monospace fonts for technical text

---

## 📈 Dataset Generation

### Output Format

OCRSmith generates datasets with rich annotations:

```json
{
  "image_path": "sample_000001.png",
  "text": "النص العربي للاختبار",
  "bbox": [45, 67, 234, 98],
  "placement_metadata": {
    "placement_type": "random",
    "position": [45, 67],
    "margins": [20, 20],
    "content_type": "body_text"
  }
}
```

### Batch Generation

```python
# Generate large datasets efficiently
engine.generate_dataset(
    num_samples=10000,
    output_dir="large_dataset",
    placement_strategies=['random', 'center', 'title']
)
```

---

## 🎨 Customization

### Adding Custom Placement Strategies

```python
class CustomPlacementStrategy(TextPlacementStrategy):
    def place_text(self, text_image, background_image, **kwargs):
        # Custom placement logic
        x, y = self.calculate_position(text_image, background_image)
      
        # Compose image
        composed_image = background_image.copy()
        composed_image.paste(text_image, (x, y), text_image)
      
        # Return result with metadata
        bbox = (x, y, x + text_image.size[0], y + text_image.size[1])
        metadata = {'placement_type': 'custom', 'position': (x, y)}
      
        return PlacementResult(composed_image, bbox, metadata)

# Register custom strategy
placement_manager.register_strategy('custom', CustomPlacementStrategy())
```

### Adding Custom Augmentations

```python
class CustomAugmentation(AugmentationStrategy):
    def apply(self, image, **kwargs):
        # Custom augmentation logic
        return modified_image

# Add to pipeline
engine.augmentation_pipeline.add_augmentation(
    CustomAugmentation(), 
    probability=0.4
)
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/OCRSmith.git
cd OCRSmith

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Font providers for high-quality Arabic and Latin fonts
- The OCR community for inspiration and feedback
- Contributors who help improve OCRSmith

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/OCRSmith/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/OCRSmith/discussions)
- **Documentation**: [Full Documentation](https://ocrsmith.readthedocs.io)

---

*Made with ❤️ for the OCR community*
