# tests/test_horizontal_text_renderer.py
from pathlib import Path
from PIL import ImageFont
from ocrsmith.core.text_renderers.renderers.VerticalRenderingStrategy import VerticalRenderingStrategy
from ocrsmith.core.FontManager import FontManager

def test_horizontal_renderer():
    # Initialize FontManager and get a font
    font_manager = FontManager(font_paths=["assets/fonts"],default_size=24)

# Factory pattern - FontLoader creates fonts with caching
    
    font = font_manager.load_font("Amiri-BoldItalic.ttf",font_size=180)
    
    print(font.getname())
    print(f"Loaded font: {font}")

    # Text to render
    text = "Hello haitam"
    renderer = VerticalRenderingStrategy()

    # Generate text image
    img, mask = renderer.render_text(
        font=font,
        text=text,
        character_spacing=0,
        text_color="#000000",
        stroke_width=0
    )

    # Save images for manual verification
    output_dir = Path("tests/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    img.save(output_dir / "vertical_text.png")
    mask.save(output_dir / "vertical_mask.png")

    print(f"Generated images saved to {output_dir}")

if __name__ == "__main__":
    test_horizontal_renderer()
