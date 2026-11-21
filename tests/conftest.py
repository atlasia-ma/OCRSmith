"""Shared fixtures.

Deliberately small: most tests build exactly the objects they need, because a fixture that
serves ten tests tends to drift into serving none of them well.
"""

import os
import tempfile

import pytest
from PIL import Image

FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "fonts")


@pytest.fixture(autouse=True, scope="session")
def _isolated_font_cache():
    """Run the suite against a throwaway font-coverage cache.

    Coverage answers are cached on disk between runs, keyed by the font's bytes. That is
    what keeps a hundred-face pool cheap in production and it is exactly what a test must
    not depend on: a cached answer would let a coverage test pass against a bug in the
    code that produces the answer. Each session gets its own directory, so every run
    computes coverage for real and nothing is left behind in the working tree.
    """
    with tempfile.TemporaryDirectory(prefix="ocrsmith-fontcache-") as cache:
        previous = os.environ.get("OCRSMITH_CACHE_DIR")
        os.environ["OCRSMITH_CACHE_DIR"] = cache
        try:
            yield cache
        finally:
            if previous is None:
                os.environ.pop("OCRSMITH_CACHE_DIR", None)
            else:
                os.environ["OCRSMITH_CACHE_DIR"] = previous


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
