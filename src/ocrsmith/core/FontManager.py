# src/ocrsmith/core/FontManager.py

import random
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict, Any

from PIL.ImageFont import FreeTypeFont
import logging

from ocrsmith.config import AppConfig # Assuming AppConfig is defined here
from .fonts import FontLoader, FontCache

class FontManager:
    def __init__(self, config: AppConfig, default_size: int = 32):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.default_size = default_size
        self.font_configs = self._parse_font_configs()
        
        if not self.font_configs:
            self.logger.warning("No valid fonts were found or configured. Falling back to default PIL font.")

    def __len__(self) -> int:
         
        return len(self.font_configs)

    def __repr__(self) -> str:
         
        cache_size = FontCache().size()
        return f"FontManager(configs={len(self.font_configs)}, cached_fonts={cache_size})"

    def _is_font_file(self, path: Path) -> bool:
         
        font_extensions = {'.ttf', '.otf', '.ttc', '.pfb', '.pfm'}
        return path.suffix.lower() in font_extensions

    def _find_fonts_at_path(self, path: Union[str, Path]) -> List[Path]:
        validated_paths = []
        path_obj = Path(path).resolve()

        if not path_obj.exists():
            self.logger.warning(f"Font path does not exist: {path}")
            return []
        
        if path_obj.is_file() and self._is_font_file(path_obj):
            validated_paths.append(path_obj)
        elif path_obj.is_dir():
            try:
                for file_path in path_obj.rglob('*'):
                    if file_path.is_file() and self._is_font_file(file_path):
                        validated_paths.append(file_path)
            except PermissionError:
                self.logger.warning(f"Permission denied accessing directory: {path_obj}")
        
        return validated_paths

    def _parse_font_configs(self) -> List[Dict[str, Any]]:
         
        configs = []
        
        for font in self.config.fonts:
            font_config = font.model_dump(exclude_none=True)
            font_path_str = font_config.get('path')
            
            if not font_path_str:
                self.logger.warning(f"Skipping font config due to missing 'path': {font_config}")
                continue

            font_paths = self._find_fonts_at_path(font_path_str)

            for path in font_paths:
                # Use the name from config, otherwise fallback to filename stem
                font_name = font_config.get('name', path.stem)
                
                configs.append({
                    'font_name': font_name,
                    'font_path': path,
                    'size': font_config.get('size') 
                })
        
        return configs

    def get_random_font(self, font_size: Optional[int] = None) -> FreeTypeFont:
        if not self.font_configs:
            self.logger.warning("No fonts available, falling back to default PIL font.")
            return self.get_default_font(font_size or self.default_size)
        
        config = random.choice(self.font_configs)
        size_to_load = font_size or config.get('size') or self.default_size
        
        return FontLoader.create_font(config['font_path'], size_to_load, use_cache=True)

    def get_font_by_name(self, name: str, font_size: Optional[int] = None) -> FreeTypeFont:
        matching_configs = [cfg for cfg in self.font_configs if cfg['font_name'] == name]
        
        if not matching_configs:
            available_names = list(set(cfg['font_name'] for cfg in self.font_configs))
            raise ValueError(f"No font named '{name}' found. Available names: {available_names}")
            
        config = matching_configs[0]
        size_to_load = font_size or config.get('size') or self.default_size

        return FontLoader.create_font(config['font_path'], size_to_load, use_cache=True)

    def get_default_font(self, font_size: Optional[int] = None) -> FreeTypeFont:
         
        return FontLoader.create_default_font(font_size or self.default_size)

    def get_available_font_names(self) -> List[str]:
         
        return sorted(list(set(cfg['font_name'] for cfg in self.font_configs)))

    def clear_cache(self) -> None:
         
        FontCache().clear()

    def get_cache_info(self) -> dict:
         
        return FontCache().get_cache_info()

    @staticmethod
    def get_text_width(font: FreeTypeFont, text: str) -> int:
         
        return round(font.getlength(text))

    @staticmethod
    def get_text_height(font: FreeTypeFont, text: Optional[str] = None ) -> int:
        if not text:
            bbox = font.getbbox("هوما مخبّين شي حاجة, أنا متيقّن!هومAgp") # Use representative chars for empty string height
            return bbox[3] - bbox[1]
        
        bbox = font.getbbox(text)
        return bbox[3] - bbox[1]

    @staticmethod
    def get_text_dimensions(font: FreeTypeFont, text: str) -> Tuple[int, int]:
         
        return (FontManager.get_text_width(font, text), FontManager.get_text_height(font, text))