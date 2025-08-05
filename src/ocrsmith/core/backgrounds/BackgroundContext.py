# src/ocrsmith/core/backgrounds/BackgroundContext.py
class BackgroundContext:
    """Context class that uses a background strategy to render backgrounds."""
    
    def __init__(self, strategy=None):
        self._strategy = strategy
    
    def set_strategy(self, strategy):
        """Change the background strategy at runtime."""
        self._strategy = strategy
    
    def render(self, width: int, height: int, **kwargs):
        """Render background using the current strategy."""
        if not self._strategy:
            raise ValueError("No background strategy set")
        return self._strategy.render(width, height, **kwargs)
    
if __name__=="__main__":
    """Basic usage of all background strategies."""
    from .strategies import ImageBackground, SolidColorBackground, GradientBackground, TextureBackground
    context = BackgroundContext()

    # 1. Image Background with your exact logic
    image_strategy = ImageBackground()
    context.set_strategy(image_strategy)

    # Use different images and modes
    img1 = context.render(800, 600, image_path='assets/backgrounds/test_default_background.png', mode='crop')
    img2 = context.render(800, 600, image_path='assets/backgrounds/test_default_background.png', mode='tile')

    # 2. Solid Background
    solid_strategy = SolidColorBackground()
    context.set_strategy(solid_strategy)
    solid_img = context.render(800, 600, color=(128, 128, 128))

    # 3. Gradient Background
    gradient_strategy = GradientBackground()
    context.set_strategy(gradient_strategy)
    gradient_img = context.render(800, 600, 
                                start_color=(255, 0, 0),
                                end_color=(0, 0, 255),
                                direction='horizontal')

    
    # 4. Texture Background
    texture_strategy = TextureBackground()
    context.set_strategy(texture_strategy)
    texture_img = context.render(800, 600, 
                            base_color=(200, 200, 200),
                            noise_level=10)
    texture_img.show()