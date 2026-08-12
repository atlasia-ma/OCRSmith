# src/ocrsmith/core/text_placement/PlacementResult.py


class PlacementResult:
    """Container for placement results"""

    def __init__(self, composed_image, bbox, metadata=None):
        self.composed_image = composed_image
        self.bbox = bbox
        self.metadata = metadata or {}  # Additional placement info
