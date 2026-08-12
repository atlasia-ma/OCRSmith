"""Shared fixtures.

Deliberately small: most tests build exactly the objects they need, because a fixture that
serves ten tests tends to drift into serving none of them well.
"""

import os
import tempfile

import pytest
from PIL import Image

FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "fonts")


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_text_image():
    return Image.new("RGBA", (100, 50), (0, 0, 0, 255))


@pytest.fixture
def sample_background_image():
    return Image.new("RGBA", (200, 100), (255, 255, 255, 255))


@pytest.fixture
def sample_csv_file(temp_dir):
    """A small bilingual CSV, including a fully vocalised Arabic row."""
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "text": [
                "Hello world",
                "يَا أَيُّهَا النَّاسُ اعْبُدُوا رَبَّكُمُ الَّذِي خَلَقَكُمْ",
                "Test text",
                "النص العربي",
            ],
            "language": ["en", "ar", "en", "ar"],
        }
    )
    path = os.path.join(temp_dir, "test_data.csv")
    frame.to_csv(path, index=False)
    return path
