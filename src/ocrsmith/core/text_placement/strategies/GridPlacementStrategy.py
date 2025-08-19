# src/ocrsmith/core/text_placement/strategies/GridPlacementStrategy.py

from ..TextPlacementStrategy import TextPlacementStrategy
from ..PlacementResult import PlacementResult

class GridPlacementStrategy(TextPlacementStrategy):
    def __init__(self, rows=3, cols=3):
        self.rows = rows
        self.cols = cols
        self.current_position = 0
    
    def place_text(self, text_image, background_image, **kwargs) -> PlacementResult:
        bg_w, bg_h = background_image.size
        text_w, text_h = text_image.size
        
        # Calculate grid position
        row = (self.current_position // self.cols) % self.rows
        col = self.current_position % self.cols
        
        cell_w = bg_w // self.cols
        cell_h = bg_h // self.rows
        
        x = col * cell_w + (cell_w - text_w) // 2
        y = row * cell_h + (cell_h - text_h) // 2
        
        # Compose image
        composed_image = background_image.copy()
        composed_image.paste(text_image, (x, y), text_image)
        
        # Create bounding box
        bbox = (x, y, x + text_w, y + text_h)
        
        # Create metadata
        metadata = {
            'placement_type': 'grid',
            'position': (x, y),
            'grid_cell': (row, col),
            'grid_size': (self.rows, self.cols),
            'cell_position': self.current_position
        }
        
        self.current_position += 1
        return PlacementResult(composed_image, bbox, metadata)