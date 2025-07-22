# src/ocrsmith/core/text_renderers/renderers/HorizontalRenderingStrategy.py

from .BaseTextRenderingStrategy import BaseTextRenderingStrategy

from PIL import ImageDraw
from PIL.ImageFont import FreeTypeFont

from ...FontManager import FontManager

class HorizontalRenderingStrategy(BaseTextRenderingStrategy):
    def render_text(self, font: FreeTypeFont, text: str, character_spacing: int = 0,
                   text_color: str ="#000000", stroke_width: int= 0, stroke_fill: str = "#282828") -> tuple:
        extra_padding = font.size # To make it as clean as possible, somtimes with italic we don't get the correct hight, it will be removed at the end
        text_width = FontManager.get_text_width(font, text)
        text_height = FontManager.get_text_height(font, text)
        
        width = text_width + (len(text) - 1) * character_spacing + extra_padding
        height = text_height + extra_padding
        
        # Use base class methods
        img, mask = self._prepare_canvas(width, height)
        fill, stroke_fill = self._generate_text_and_fill_colors(text_color, stroke_fill)
        
        draw = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)
        
        # Horizontal-specific positioning logic
        x_pos = 2
        _, top, _, _ = font.getbbox(text)
        y_offset = -top + extra_padding // 2
        
        for i, char in enumerate(text):
            self._draw_character(draw, draw_mask, font, char, x_pos, y_offset, 
                               fill, stroke_fill, stroke_width, i)
            
            char_width = FontManager.get_text_width(font, char)
            x_pos += char_width + character_spacing
        
        # Crop to content
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
            mask = mask.crop(bbox)
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            width, height = img.size  # Fallback if bbox is None

        return img, mask, (width, height)
