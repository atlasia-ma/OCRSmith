# src/ocrsmith/core/FontManager.py

from .fonts import FontLoader, FontCache
import random
from pathlib import Path
from typing import List, Optional, Union, Tuple
from PIL.ImageFont import FreeTypeFont
import logging

class FontManager:
    def __init__(self, font_paths: Optional[List[Union[str, Path]]] = None, 
                 default_size: int = 32):
        self.logger = logging.getLogger(__name__)
        self.font_paths = self._validate_font_paths(font_paths or [])
        self.default_size = default_size
    
    def _validate_font_paths(self, font_paths: List[Union[str, Path]]) -> List[Path]:
        validated_paths = []
        
        for path in font_paths:
            path_obj = Path(path).expanduser().resolve()
            if path_obj.exists():
                if path_obj.is_file() and self._is_font_file(path_obj):
                    validated_paths.append(path_obj)
                elif path_obj.is_dir():
                    font_files = self._get_font_files_from_dir(path_obj)
                    validated_paths.extend(font_files)
            else:
                self.logger.warning(f"Font path does not exist: {path}")
        
        return validated_paths
    
    def _is_font_file(self, path: Path) -> bool:
        font_extensions = {'.ttf', '.otf', '.ttc', '.pfb', '.pfm'}
        return path.suffix.lower() in font_extensions
    
    def _get_font_files_from_dir(self, directory: Path) -> List[Path]:
        font_files = []
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file() and self._is_font_file(file_path):
                    font_files.append(file_path)
        except PermissionError:
            self.logger.warning(f"Permission denied accessing directory: {directory}")
        return font_files
    
    def add_font_path(self, path: Union[str, Path]) -> None:
        validated_paths = self._validate_font_paths([path])
        self.font_paths.extend(validated_paths)
    
    def get_random_font_path(self) -> Path:
        if not self.font_paths:
            raise ValueError("No valid font paths available. Add font paths.")
        
        return random.choice(self.font_paths)
    
    def load_font(self, font_path: Optional[Union[str, Path]] = None, 
                  font_size: Optional[int] = None, use_cache: bool = True) -> FreeTypeFont:
        if font_size is None:
            font_size = self.default_size
        
        if font_path is None:
            font_path = self.get_random_font_path()
        else:
            font_path = Path(font_path).expanduser().resolve()
            if not font_path.exists():
                self.logger.warning(f"Specified font path does not exist: {font_path}")
                font_path = self.get_random_font_path()
        
        return FontLoader.create_font(font_path, font_size, use_cache)
    
    def get_default_font(self, font_size: Optional[int] = None) -> FreeTypeFont:
        if font_size is None:
            font_size = self.default_size
        
        return FontLoader.create_default_font(font_size)
    
    @staticmethod
    def get_text_dimensions(font: FreeTypeFont, text: str) -> Tuple[int, int]:
        if not text:
            return (0, 0)
        
        bbox = font.getbbox(text)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return (width, height)
    
    @staticmethod
    def get_text_width(font: FreeTypeFont, text: str) -> int:
        if not text:
            return 0
        
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0]
    
    @staticmethod
    def get_text_height(font: FreeTypeFont, text: str) -> int:
        if not text:
            bbox = font.getbbox("Ag")
            return bbox[3] - bbox[1]
        
        bbox = font.getbbox(text)
        return bbox[3] - bbox[1]
    
    def clear_cache(self) -> None:
        """Clear the singleton font cache."""
        FontCache().clear()
    
    def get_cache_info(self) -> dict:
        """Get information about the font cache."""
        return FontCache().get_cache_info()
    
    def get_available_fonts(self) -> List[Path]:
        """Get list of all available font paths."""
        return self.font_paths.copy()
    
    def __len__(self) -> int:
        """Return number of available fonts."""
        return len(self.font_paths)
    
    def __repr__(self) -> str:
        cache_size = FontCache().size()
        return f"FontManager(fonts={len(self.font_paths)}, cached_fonts={cache_size})"
    