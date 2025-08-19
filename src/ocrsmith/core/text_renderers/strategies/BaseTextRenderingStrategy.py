# src/ocrsmith/core/text_renderers/strategies/BaseTextRenderingStrategy.py

from abc import abstractmethod

from ..TextRenderingStrategy import TextRenderingStrategy
from ...FontManager import FontManager
from ocrsmith.utils import generate_random_color

from PIL import Image
from PIL.ImageFont import FreeTypeFont

class BaseTextRenderingStrategy(TextRenderingStrategy):
    @abstractmethod
    def render_text(self, font: FreeTypeFont, text: str, **kwargs):
        raise NotImplementedError
    
    def _prepare_canvas(
        self, width: int, height: int
    ):
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        mask = Image.new("RGB", (width, height), (0, 0, 0))

        return img, mask
    
    def _generate_text_and_fill_colors(self, text_color: str, stroke_fill: str):
        fill = generate_random_color(text_color)
        stroke_fill = generate_random_color(stroke_fill)
        return fill, stroke_fill
    
    def _draw_character(self, draw, draw_mask, font, char, x, y, fill, stroke_fill, stroke_width, char_index):
        draw.text((x, y), char, fill=fill, font=font,
                 stroke_width=stroke_width, stroke_fill=stroke_fill)
        
        # Mask coloring logic
        mask_color = ((char_index + 1) // (255 * 255), 
                     (char_index + 1) // 255, 
                     (char_index + 1) % 255)
        draw_mask.text((x, y), char, fill=mask_color, font=font,
                      stroke_width=stroke_width, stroke_fill=stroke_fill)