from .BaseTextRenderingStrategy import BaseTextRenderingStrategy
from PIL import ImageDraw
from PIL.ImageFont import FreeTypeFont
from ...FontManager import FontManager


class SimpleTextRenderingStrategy(BaseTextRenderingStrategy):
    def __init__(self, **kwargs):
        pass

    def render_text(self, font: FreeTypeFont, text: str, text_color: str = "#000000", **kwargs) -> tuple:
        padding = font.size
        lines = text.split('\n')
        text_height = FontManager.get_text_height(font, text)
        text_width = max(FontManager.get_text_width(font, line) for line in lines) if lines else 0
        width = text_width + padding
        height = text_height + padding

        img, mask = self._prepare_canvas(width, height)
        draw = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)

        _, top, _, _ = font.getbbox(text)
        x = padding // 2
        y = -top + padding // 2
        draw.text((x, y), text, fill=text_color, font=font)
        draw_mask.text((x, y), text, fill=(255, 255, 255), font=font)

        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
            mask = mask.crop(bbox)
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            width, height = img.size

        return img, mask, (width, height)

