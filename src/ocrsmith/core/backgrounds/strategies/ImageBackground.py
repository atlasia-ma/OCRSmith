# src/ocrsmith/core/backgrounds/strategies/ImageBackground.py
from .BaseBackground import BaseBackground
from PIL import Image
from pathlib import Path

class ImageBackground(BaseBackground):
    """Strategy for rendering image-based backgrounds with your exact logic."""
    
    def __init__(self, image_path: str = None, mode: str = 'stretch'):
        super().__init__()
        self.default_image_path = image_path
        self.default_mode = mode
        self._cached_images = {}  
    
    def render(self, width: int, height: int, **kwargs) -> Image.Image:

        image_path = kwargs.get('image_path', self.default_image_path)
        mode = kwargs.get('mode', self.default_mode)
        
        if not image_path:
            raise ValueError("No image_path provided")
        

        base_image = self._load_image(image_path)
        

        if mode == 'stretch':
            return base_image.resize((width, height), Image.Resampling.LANCZOS)
        
        elif mode == 'crop':

            img_ratio = base_image.width / base_image.height
            target_ratio = width / height
            
            if img_ratio > target_ratio:

                new_width = int(base_image.height * target_ratio)
                left = (base_image.width - new_width) // 2
                cropped = base_image.crop((left, 0, left + new_width, base_image.height))
            else:

                new_height = int(base_image.width / target_ratio)
                top = (base_image.height - new_height) // 2
                cropped = base_image.crop((0, top, base_image.width, top + new_height))
            
            return cropped.resize((width, height), Image.Resampling.LANCZOS)
        
        elif mode == 'tile':

            result = Image.new('RGB', (width, height))
            for y in range(0, height, base_image.height):
                for x in range(0, width, base_image.width):
                    result.paste(base_image, (x, y))
            return result
        
        elif mode == 'center':

            result = Image.new('RGB', (width, height), (255, 255, 255))
            x = (width - base_image.width) // 2
            y = (height - base_image.height) // 2
            result.paste(base_image, (x, y))
            return result
        
        else:
            raise ValueError(f"Unknown render mode '{mode}' for ImageBackground")
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load image with caching and path resolution."""

        path = Path(image_path)
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parents[5]
            path = project_root / path
        
        path_str = str(path)
        

        if path_str not in self._cached_images:
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            
            img = Image.open(path)

            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            self._cached_images[path_str] = img
        
        return self._cached_images[path_str].copy()