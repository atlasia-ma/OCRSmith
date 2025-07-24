import pytest
import tempfile
import os
from PIL import Image
import pandas as pd
import json

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def sample_text_image():
    """Create a sample text image for testing"""
    img = Image.new('RGBA', (100, 50), (0, 0, 0, 255))
    return img

@pytest.fixture
def sample_background_image():
    """Create a sample background image for testing"""
    img = Image.new('RGBA', (200, 100), (255, 255, 255, 255))
    return img

@pytest.fixture
def sample_csv_file(temp_dir):
    """Create a sample CSV file for testing"""
    data = {
        'text': ['Hello world', 'يَا أَيُّهَا النَّاسُ اعْبُدُوا رَبَّكُمُ الَّذِي خَلَقَكُمْ وَالَّذِينَ مِن قَبْلِكُمْ لَعَلَّكُمْ تَتَّقُونَ', 'Test text', 'النص العربي'],
        'language': ['en', 'ar', 'en', 'ar']
    }
    df = pd.DataFrame(data)
    csv_path = os.path.join(temp_dir, 'test_data.csv')
    df.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'backgrounds': {
            'solid': {'enabled': True, 'colors': ['#FFFFFF', '#000000']},
            'gradient': {'enabled': True, 'directions': ['horizontal', 'vertical']}
        },
        'fonts': {
            'default_size': 24,
            'paths': ['assets/fonts']
        },
        'placement': {
            'default_strategy': 'random',
            'margins': {'x': 20, 'y': 20}
        }
    }
