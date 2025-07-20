# src/ocrsmith/core/backgrounds/creators/ImageBackground.py
from .BaseBackground import BaseBackground
from PIL import Image

class ImageBackground(BaseBackground):
    def __init__(self, image_path: str, mode: str = 'stretch', **_ignored):
        self.image_path = image_path
        self.mode = mode
        self.base_image = Image.open(image_path)
    
    def render(self, width: int, height: int) -> Image.Image:
        if self.mode == 'stretch':
            return self.base_image.resize((width, height), Image.Resampling.LANCZOS)
        
        elif self.mode == 'crop':
            # Crop to fit while maintaining aspect ratio
            img_ratio = self.base_image.width / self.base_image.height
            target_ratio = width / height
            
            if img_ratio > target_ratio:
                # Image is wider, crop width
                new_width = int(self.base_image.height * target_ratio)
                left = (self.base_image.width - new_width) // 2
                cropped = self.base_image.crop((left, 0, left + new_width, self.base_image.height))
            else:
                # Image is taller, crop height
                new_height = int(self.base_image.width / target_ratio)
                top = (self.base_image.height - new_height) // 2
                cropped = self.base_image.crop((0, top, self.base_image.width, top + new_height))
            
            return cropped.resize((width, height), Image.Resampling.LANCZOS)
        
        elif self.mode == 'tile':
            # Tile the image to fill the canvas
            result = Image.new('RGB', (width, height))
            for y in range(0, height, self.base_image.height):
                for x in range(0, width, self.base_image.width):
                    result.paste(self.base_image, (x, y))
            return result
        
        elif self.mode == 'center':
            # Center the image on a colored background
            result = Image.new('RGB', (width, height), (255, 255, 255))
            x = (width - self.base_image.width) // 2
            y = (height - self.base_image.height) // 2
            result.paste(self.base_image, (x, y))
            return result
