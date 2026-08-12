# src/ocrsmith/core/text_renderers/renderers/VerticalRenderingStrategy.py

from PIL import ImageDraw
from PIL.ImageFont import FreeTypeFont

from ...FontManager import FontManager
from .BaseTextRenderingStrategy import BaseTextRenderingStrategy


class VerticalRenderingStrategy(BaseTextRenderingStrategy):
    def render_text(
        self,
        font: FreeTypeFont,
        text: str,
        character_spacing: int,
        text_color: str,
        stroke_width: int,
        stroke_fill: str = "#282828",
        **kwargs,
    ) -> tuple:

        extra_padding = font.size  # To ensure clean rendering, especially with italic fonts

        # Calculate dimensions similar to horizontal strategy
        max_char_width = self._get_vertical_text_width(font, text)

        # For height: use font height consistently
        _, font_top, _, font_bottom = font.getbbox(text)  # Get overall text dimensions
        font_height = font_bottom - font_top

        total_height = font_height * len(text) + (len(text) - 1) * character_spacing + extra_padding
        total_width = max_char_width + extra_padding

        # Create canvas
        img, mask = self._prepare_canvas(total_width, total_height)

        draw = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)

        # Generate colors
        fill, stroke_fill = self._generate_text_and_fill_colors(text_color, stroke_fill)

        # Position like horizontal but vertically
        x_offset = extra_padding // 2
        y_pos = extra_padding // 2 - font_top  # Compensate for font top offset

        for i, char in enumerate(text):
            self._draw_character(
                draw, draw_mask, font, char, x_offset, y_pos, fill, stroke_fill, stroke_width, i
            )

            # Move to next line
            y_pos += font_height + character_spacing

        # Crop to actual content
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
            mask = mask.crop(bbox)
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            width, height = img.size  # Fallback if bbox is None

        return img, mask, (width, height)

    def _get_vertical_text_width(self, font, text):
        unique_chars = set(text)  # Avoid duplicate calculations
        char_widths = [
            FontManager.get_text_width(font, c) if c != " " else FontManager.get_text_width(font, "A")
            for c in unique_chars
        ]

        return max(char_widths)
