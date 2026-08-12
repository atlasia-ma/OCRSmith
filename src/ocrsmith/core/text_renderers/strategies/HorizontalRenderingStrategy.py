# src/ocrsmith/core/text_renderers/strategies/HorizontalRenderingStrategy.py

import random as rnd

from PIL import ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont

from ...FontManager import FontManager
from .BaseTextRenderingStrategy import BaseTextRenderingStrategy


def wrap_text_by_pixels(paragraphs, font, max_width, max_height, spacing, line_height=None):
    lines = []
    if not line_height:
        line_height = FontManager.get_text_height(font)
    text_height = 0
    # Provide safe defaults if constraints are not provided
    eff_max_width = max_width if isinstance(max_width, int) and max_width > 0 else 10_000
    eff_max_height = max_height if isinstance(max_height, int) and max_height > 0 else 10_000

    for paragraph in paragraphs:
        words = paragraph.split()
        current_line = ""
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            w = FontManager.get_text_width(font, test_line)
            if w <= eff_max_width:
                current_line = test_line
            else:
                # Add current line if it fits in height
                if current_line:  # Don't add empty lines
                    if text_height + line_height + spacing <= eff_max_height:
                        lines.append(current_line)
                        text_height += line_height + spacing
                    else:
                        break  # Stop if adding this line would exceed max_height
                current_line = word

                # Check if single word exceeds max_width
                if FontManager.get_text_width(font, word) > eff_max_width:
                    # Handle very long words - you might want to break them or skip them
                    current_line = word  # Keep it anyway, but it will overflow

        # Add the last line of the paragraph
        if current_line:
            if text_height + line_height <= eff_max_height:
                lines.append(current_line)
                text_height += line_height + spacing
            else:
                break  # Stop if adding this line would exceed max_height

    return lines


class HorizontalRenderingStrategy(BaseTextRenderingStrategy):
    def render_text(
        self,
        font: FreeTypeFont,
        text: str,
        spacing: int = 1,
        text_color: str = "#000000",
        stroke_width: int = 0,
        stroke_fill: str = "#282828",
        render_one_line=False,
        align="right",
        fit: bool = True,
        max_width: int = None,
        max_height: int = None,
        min_width: int = None,
        min_height: int = None,
    ) -> tuple:
        horizontal_extra_padding = (
            font.size * 3
        )  # To make it as clean as possible, sometimes with italic we don't get the correct hight, It will be removed at the end
        vertical_extra_padding = font.size

        contains_title = False
        flag = False

        title = ""
        if isinstance(text, dict):
            title = text.get("title", "أطلسيا")
            text = text.get("content", "")

        paragraphs = text.split("\n")
        if title:
            rnd_num = rnd.random()
            if rnd_num < 0.5:
                paragraphs.insert(0, "")
            else:
                paragraphs.insert(0, "─")
            paragraphs.insert(0, title)
            contains_title = True

        line_height = FontManager.get_text_height(font)
        lines = wrap_text_by_pixels(paragraphs, font, max_width, max_height, spacing, line_height)
        # Guard against empty content (e.g., whitespace-only input)
        if not lines:
            lines = [" "]

        line_heights = [line_height] * len(lines)
        text_height = sum(line_heights) + (len(lines) - 1) * spacing
        text_width = max([FontManager.get_text_width(font, line) for line in lines])

        width = text_width + horizontal_extra_padding
        height = text_height + vertical_extra_padding

        img, mask = self._prepare_canvas(width, height)
        fill, stroke_fill = self._generate_text_and_fill_colors(text_color, stroke_fill)

        draw = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)
        draw_mask.fontmode = "1"

        current_y = 0
        original_font = font
        for i, line in enumerate(lines):
            if line == "":
                last_added_y = line_heights[i] + spacing
                current_y += last_added_y // 2

                continue

            if line == "─":
                font = ImageFont.truetype(font="./fonts/NotoSansMono-Light.ttf", size=font.size)
                flag = True

            line_w = FontManager.get_text_width(font, line)
            x = text_width - line_w
            if contains_title and i == 0:
                rnd_num = rnd.random()
                if rnd_num < 0.5:
                    x = max(0, (text_width - line_w) // 2)  # center

            if line == "─":
                line_w = FontManager.get_text_width(font, line)
                factor = 3 * (text_width // line_w) // 4
                line = line * factor
                line_w = line_w * factor
                x = max(0, (text_width - line_w) // 2)  # center

            draw.text(
                (x, current_y),
                line,
                fill=fill,
                font=font,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )
            draw_mask.text(
                (x, current_y),
                line,
                fill=((i + 1) // (255 * 255), (i + 1) // 255, (i + 1) % 255),
                font=font,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )
            last_added_y = line_heights[i] + spacing
            current_y += last_added_y
            if flag:
                font = original_font

        label = " ".join([line.replace("─", "") for line in lines if line.strip()])
        family, style = font.getname()
        if fit:
            bbox = img.getbbox()
            if bbox:  # In case bbox is None (completely transparent)
                img = img.crop(bbox)
                mask = mask.crop(bbox)
                width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            else:
                width, height = img.size

        self.last_rendered_text = label
        return img, mask, (width, height)
