# src/ocrsmith/core/text_renderers/renderers/HorizontalRenderingStrategy.py

from .BaseTextRenderingStrategy import BaseTextRenderingStrategy

from PIL import ImageDraw
from PIL.ImageFont import FreeTypeFont

from ...FontManager import FontManager

class HorizontalRenderingStrategy(BaseTextRenderingStrategy):
    def render_text(self, font: FreeTypeFont, text: str, spacing: int = 0,
                   text_color: str ="#000000", stroke_width: int= 0, stroke_fill: str = "#282828", render_one_line = False, align = "right") -> tuple:
        
        horizontal_extra_padding = font.size * 3 # To make it as clean as possible, somtimes with italic we don't get the correct hight, it will be removed at the end
        vertical_extra_padding = font.size
        
        if render_one_line:
            text = (" ").join(text.split('\n'))
        
        lines = text.split('\n')
        text_height = FontManager.get_text_height(font, text) * len(lines) + spacing * (len(lines) - 1)
        
        lines_width = []
        for line in lines:
            lines_width.append(FontManager.get_text_width(font, line))
        
        text_width = max(lines_width) if lines_width else 0 
            
        width = text_width + horizontal_extra_padding
        height = text_height + vertical_extra_padding 
        
        # Use base class methods
        img, mask = self._prepare_canvas(width, height)
        fill, stroke_fill = self._generate_text_and_fill_colors(text_color, stroke_fill)
        
        draw = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)
        
        # Horizontal-specific positioning logic
        x_pos = horizontal_extra_padding // 2
        _, top, _, _ = font.getbbox(text)
        y_offset = -top + vertical_extra_padding // 2
        
        """
        # we can do it our selfs if we need spacing between letters
        for i, char in enumerate(text):
            self._draw_character(draw, draw_mask, font, char, x_pos, y_offset, 
                               fill, stroke_fill, stroke_width, i)
            
            char_width = FontManager.get_text_width(font, char)
            x_pos += char_width + spacing
        """
        
        draw.text((x_pos, y_offset), text, fill=fill, font=font, spacing=spacing, align=align)
        draw_mask.text((x_pos, y_offset), text, fill=stroke_fill, font=font, spacing=spacing, align=align)
        # Crop to content
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
            mask = mask.crop(bbox)
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            width, height = img.size  # Fallback if bbox is None

        return img, mask, (width, height)
