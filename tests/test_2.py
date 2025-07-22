from ocrsmith.core.BackgroundManager import BackgroundManager
from ocrsmith.core.backgrounds.BackgroundFactory import BackgroundFactory
from ocrsmith.core.backgrounds.creators.SolidColorBackground import SolidColorBackground
from ocrsmith.core.backgrounds.creators.ImageBackground import ImageBackground
from ocrsmith.core.backgrounds.creators.NoiseBackground import NoiseBackground
from ocrsmith.core.backgrounds.creators.GradientBackground import GradientBackground
from ocrsmith.config import load_config
from ocrsmith.config import AppConfig
from ocrsmith.core.FontManager import FontManager
from ocrsmith.core.TextRenderer import TextRenderer
from ocrsmith.core.text_renderers.renderers.HorizontalRenderingStrategy import HorizontalRenderingStrategy

factory = BackgroundFactory()
factory.register_creator('solid', SolidColorBackground)
factory.register_creator('image', ImageBackground)
factory.register_creator('noise', NoiseBackground)
factory.register_creator('gradient', GradientBackground)

configs = load_config()
manager = BackgroundManager(configs, factory)

background_creator = manager.get_random_background()
print(f"Selected background type: {type(background_creator).__name__}")

font_manager = FontManager(font_paths=["assets/fonts"],default_size=24)

# Factory pattern - FontLoader creates fonts with caching
font = font_manager.load_font(font_size=18)
print(f"Loaded font: {font}")

text = "يل لاختبار توليد الصورة بالنص العربي."

text_renderer = TextRenderer(HorizontalRenderingStrategy())
image, mask, (width, height) = text_renderer.generate_text_image(font, text)

image.show()  # Display the generated text image
background_image = background_creator.render(width, height)
background_image.show()  # Display the generated background image

background_image = background_creator.render(width, height).convert("RGBA")

# Paste the text image onto the background using its alpha channel
background_image.paste(image, (0, 0), image)  # (0, 0) is the top-left corner

# Show or save the final composition
background_image.show()
background_image.save("final_output.png")
