# src/ocrsmith/core/text_renderers/renderers/HorizontalTextRenderer.py

from .BaseTextRenderer import BaseTextRenderer

from PIL import Image, ImageDraw

class HorizontalStrategy(BaseTextRenderer):
    
    def generate_text_image(self, font, text):
        
        # Calculate dimensions
        metrics = self.get_text_metrics(font, text)
        text_width = metrics["width"] + (len(text) - 1) * style.character_spacing
        text_height = metrics["height"]
        
        # Create images
        img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
        mask = Image.new("RGB", (text_width, text_height), (0, 0, 0))
        
        draw = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)
        
        # Generate colors
        fill = self.generate_colors(style.text_color)
        stroke_fill = self.generate_colors(style.stroke_fill)
        
        # Draw text character by character
        x_pos = 0
        for i, char in enumerate(text):
            draw.text((x_pos, 0), char, fill=fill, font=font,
                     stroke_width=style.stroke_width, stroke_fill=stroke_fill)
            draw_mask.text((x_pos, 0), char, 
                          fill=((i + 1) // (255 * 255), (i + 1) // 255, (i + 1) % 255),
                          font=font, stroke_width=style.stroke_width, stroke_fill=stroke_fill)
            
            char_width = self.get_text_metrics(font, char)["width"]
            x_pos += char_width + style.character_spacing
        
        if layout.fit_to_content:
            bbox = img.getbbox()
            if bbox:
                img = img.crop(bbox)
                mask = mask.crop(bbox)
        
        return img, mask

