import pytest
from PIL import Image

from ocrsmith.core.text_renderers.strategies.HorizontalRenderingStrategy import HorizontalRenderingStrategy
from PIL.ImageFont import ImageFont
from ocrsmith.core.FontManager import FontManager


def test_horizontal_trims_to_max_width_height(monkeypatch):
    # Use default PIL font via FontManager
    fm = FontManager.__new__(FontManager)
    # Stub methods that are used
    from PIL import ImageFont as PILImageFont
    try:
        font = PILImageFont.load_default()
    except Exception:
        # Fallback to a basic truetype shipped with PIL
        import os
        font = PILImageFont.truetype(font=os.environ.get('PIL_TEST_FONT', 'DejaVuSans.ttf'), size=12)

    # Monkeypatch FontManager static methods
    from ocrsmith.core.FontManager import FontManager as FM
    def w(f, t):
        return max(1, len(t) * 6)
    def h(f, t):
        return 10
    monkeypatch.setattr(FM, 'get_text_width', staticmethod(w))
    monkeypatch.setattr(FM, 'get_text_height', staticmethod(h))

    strat = HorizontalRenderingStrategy()
    text = 'one two three four five six seven eight nine ten'
    img, mask, (w_pixels, h_pixels) = strat.render_text(font, text, max_width=60, max_height=20)
    # Height should be <= 20 and width <= 60
    assert w_pixels <= 60
    assert h_pixels <= 20
