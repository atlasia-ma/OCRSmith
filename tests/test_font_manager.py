from ocrsmith.core.FontManager import FontManager
font_manager = FontManager(font_paths=["assets/fonts"],default_size=24)

# Factory pattern - FontLoader creates fonts with caching
font1 = font_manager.load_font(font_size=18)
print(f"Loaded font: {font1}")

# Load same font again - should use cached version (Singleton pattern)
font2 = font_manager.load_font(font_size=18)
print(f"Loaded same font again: {font2}")

# Check cache info
cache_info = font_manager.get_cache_info()
print(f"Cache info: {cache_info}")

# Load different size - will create new font instance
font3 = font_manager.load_font(font_size=32)
print(f"Different size font: {font3}")

# Measure text with loaded font
text = "Hello, World!"
width, height = FontManager.get_text_dimensions(font1, text)
print(f"Text '{text}' dimensions: {width}x{height}")

# Clear cache
font_manager.clear_cache()
print(f"Cache cleared. New cache info: {font_manager.get_cache_info()}")
