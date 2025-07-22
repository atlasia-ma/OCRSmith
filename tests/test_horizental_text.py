# tests/test_horizontal_text_renderer.py
from pathlib import Path
from PIL import ImageFont
from ocrsmith.core.text_renderers.renderers.HorizontalTextRenderer import HorizontalStrategy
from ocrsmith.core.FontManager import FontManager

def test_horizontal_renderer():
    # Initialize FontManager and get a font
    font_manager = FontManager(font_paths=["assets/fonts"],default_size=24)

# Factory pattern - FontLoader creates fonts with caching
    font = font_manager.load_font(font_size=18)
    print(f"Loaded font: {font}")

    # Text to render
    text = "Hello OCRSmith,byan lya f pdgd"
    renderer = HorizontalStrategy()

    # Generate text image
    img, mask = renderer.generate_text_image(
        font=font,
        text=text,
        character_spacing=1,
        text_color="#FF0000",
        stroke_width=0
    )

    # Save images for manual verification
    output_dir = Path("tests/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    img.save(output_dir / "horizontal_text.png")
    mask.save(output_dir / "horizontal_mask.png")

    print(f"Generated images saved to {output_dir}")

if __name__ == "__main__":
    test_horizontal_renderer()
