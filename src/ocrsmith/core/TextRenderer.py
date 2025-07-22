from PIL.ImageFont import FreeTypeFont
from ocrsmith.core.text_renderers.TextRenderingStrategy import TextRenderingStrategy

class TextRenderer:
    def __init__(self, strategy: TextRenderingStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: TextRenderingStrategy):
        self._strategy = strategy
    
    def generate_text_image(self, font: FreeTypeFont, text: str, **kwargs):
        return self._strategy.render_text(font, text, **kwargs)
    