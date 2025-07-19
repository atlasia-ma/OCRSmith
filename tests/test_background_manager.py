import pytest
from ocrsmith.core.background_manager import BackgroundManager
from ocrsmith.core.backgrounds.BackgroundFactory import BackgroundFactory
from ocrsmith.core.backgrounds.creators.ColorBackground import ColorBackground
from ocrsmith.core.backgrounds.creators.ImageBackground import ImageBackground

@pytest.fixture
def factory():
    f = BackgroundFactory()
    f.register_creator('solid', ColorBackground)
    f.register_creator('image', ImageBackground)
    return f

def test_manager_creates_backgrounds(factory):
    configs = [
        {'type': 'solid', 'color': '#FFFFFF'},
        {'type': 'image', 'image_path': 'assets/bg1.jpg'}
    ]
    manager = BackgroundManager(configs, factory)
    assert len(manager.backgrounds) == 2
    assert any(isinstance(bg, ColorBackground) for bg in manager.backgrounds)
    assert any(isinstance(bg, ImageBackground) for bg in manager.backgrounds)

def test_get_random_background(factory):
    configs = [{'type': 'solid', 'color': '#000000'}]
    manager = BackgroundManager(configs, factory)
    bg = manager.get_random_background()
    assert isinstance(bg, ColorBackground)
    assert bg.color == '#000000'

def test_empty_config_raises(factory):
    with pytest.raises(ValueError):
        BackgroundManager([], factory)
        