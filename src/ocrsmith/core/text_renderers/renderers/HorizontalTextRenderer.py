# src/ocrsmith/core/text_renderers/renderers/HorizontalTextRenderer.py

from .BaseTextRenderer import BaseTextRenderer
from ...FontManager import FontManager
from ocrsmith.utils import generate_random_color

from PIL import Image, ImageDraw
from PIL.ImageFont import FreeTypeFont
import random as rnd

class HorizontalStrategy(BaseTextRenderer):
    
    def generate_text_image(self, font: FreeTypeFont , text: str, character_spacing: int,
                            text_color, stroke_fill: str = "#282828", stroke_width: int = 0):
        
        extra_padding = 10 #gonna be removed at the end
        
        character_spacing = rnd.randint(1,4)
        text_width = FontManager.get_text_width(font, text) + (len(text) - 1) * character_spacing + extra_padding
        text_height = FontManager.get_text_height(font, text) + extra_padding
        
        # Create images
        img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
        mask = Image.new("RGB", (text_width, text_height), (0, 0, 0))
        
        draw = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)
        
        # Generate colors
        fill = generate_random_color(text_color)
        stroke_fill = generate_random_color(stroke_fill)
        
        # Draw text character by character
        x_pos = 2
        _, top, _, _ = font.getbbox(text)
        y_offset = -top + extra_padding//2
        for i, char in enumerate(text):
            draw.text((x_pos, y_offset), char, fill=fill, font=font,
                     stroke_width=stroke_width, stroke_fill=stroke_fill)
            draw_mask.text((x_pos, y_offset), char, 
                          fill=((i + 1) // (255 * 255), (i + 1) // 255, (i + 1) % 255),
                          font=font, stroke_width=stroke_width, stroke_fill=stroke_fill)
            
            char_width = FontManager.get_text_width(font, char)
            x_pos += char_width + character_spacing
        
        
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
            mask = mask.crop(bbox)
        
        return img, mask

