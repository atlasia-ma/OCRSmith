# ocrsmith/core/fonts/FontLoader.py

from .FontCache import FontCache
from pathlib import Path
from typing import Union
from PIL import ImageFont
from PIL.ImageFont import FreeTypeFont

class FontLoader:
    
    @staticmethod
    def create_font(font_path: Union[str, Path], size: int, use_cache: bool = True) -> FreeTypeFont:
        font_path = Path(font_path).resolve()
        
        if use_cache:
            cache = FontCache()
            cache_key = (str(font_path), size)
            
            # Try to get from cache first
            cached_font = cache.get(cache_key)
            if cached_font:
                return cached_font
        
        try:
            font = ImageFont.truetype(str(font_path), size=size)
            
            if use_cache:
                cache = FontCache()
                cache.set(cache_key, font)
            
            return font
            
        except (OSError, IOError) as e:
            raise ValueError(f"Unable to load font from '{font_path}': {e}")
    
    @staticmethod
    def create_default_font(size: int = 32) -> FreeTypeFont:
        try:
            return ImageFont.load_default()
        except Exception:
            try:
                return ImageFont.load()
            except Exception:
                raise ValueError("Unable to load any default font")
            