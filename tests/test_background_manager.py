import pytest
from ocrsmith.core.background_manager import BackgroundManager
from ocrsmith.core.backgrounds.BackgroundFactory import BackgroundFactory
from ocrsmith.core.backgrounds.creators.SolidColorBackground import SolidColorBackground
from ocrsmith.core.backgrounds.creators.ImageBackground import ImageBackground
from ocrsmith.core.backgrounds.creators.NoiseBackground import NoiseBackground
from ocrsmith.core.backgrounds.creators.GradientBackground import GradientBackground
from ocrsmith.config import load_config
from ocrsmith.config import AppConfig

@pytest.fixture
def factory():
    f = BackgroundFactory()
    f.register_creator('solid', SolidColorBackground)
    f.register_creator('image', ImageBackground)
    f.register_creator('noise', NoiseBackground)
    f.register_creator('gradient', GradientBackground)
    return f

def test_manager_creates_backgrounds(factory):
    
    configs = load_config()
    print("*"*20)
    print(type(configs))
    manager = BackgroundManager(configs, factory)
    assert len(manager.backgrounds) == 5
    assert any(isinstance(bg, SolidColorBackground) for bg in manager.backgrounds)
    assert any(isinstance(bg, ImageBackground) for bg in manager.backgrounds)
    assert any(isinstance(bg, NoiseBackground) for bg in manager.backgrounds)
    assert any(isinstance(bg, GradientBackground) for bg in manager.backgrounds)

def test_empty_config_raises(factory):
    with pytest.raises(ValueError):
        BackgroundManager(AppConfig(), factory)
        