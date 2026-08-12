# src/ocrsmith/core/text_renderers/TextRenderingContext.py


class TextRenderingContext:
    def __init__(self, strategy=None):
        self._strategy = strategy

    def set_strategy(self, strategy):
        self._strategy = strategy

    def render_text(self, font, text, **kwargs):
        if not self._strategy:
            raise ValueError("No text rendering strategy set")
        return self._strategy.render_text(font, text, **kwargs)
