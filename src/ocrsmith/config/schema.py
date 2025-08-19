# src/ocrsmith/config/schema.py
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Tuple, Union, Literal
import yaml

class FontConfig(BaseModel):
    name: Optional[str] = None
    path: str
    size: Optional[int] = 24

class SolidBackgroundConfig(BaseModel):
    type: Literal["solid"]
    color: Union[str, Tuple[int, int, int]] = (255, 255, 255)
    weight: float = 1.0

class ImageBackgroundConfig(BaseModel):
    type: Literal["image"]
    image_path: str
    mode: Literal["stretch", "crop", "tile", "center"] = "stretch"
    weight: float = 1.0

class GradientBackgroundConfig(BaseModel):
    type: Literal["gradient"]
    start_color: Tuple[int, int, int] = (255, 255, 255)
    end_color: Tuple[int, int, int] = (200, 200, 200)
    direction: Literal["horizontal", "vertical", "diagonal"] = "horizontal"
    weight: float = 1.0

class TextureBackgroundConfig(BaseModel):
    type: Literal["texture"]
    base_color: Tuple[int, int, int] = (240, 240, 240)
    noise_level: int = Field(default=20, ge=0, le=255)
    weight: float = 1.0

BackgroundConfigUnion = Union[
    SolidBackgroundConfig,
    ImageBackgroundConfig,
    GradientBackgroundConfig,
    TextureBackgroundConfig
]

class SimpleTextRendererConfig(BaseModel):
    type: Literal["simple"]
    color: Tuple[int, int, int] = (0, 0, 0)
    weight: float = 1.0

class OutlinedTextRendererConfig(BaseModel):
    type: Literal["outlined"]
    fill_color: Tuple[int, int, int] = (255, 255, 255)
    outline_color: Tuple[int, int, int] = (0, 0, 0)
    outline_width: int = 2
    weight: float = 1.0

class ShadowedTextRendererConfig(BaseModel):
    type: Literal["shadowed"]
    text_color: Tuple[int, int, int] = (0, 0, 0)
    shadow_color: Tuple[int, int, int] = (128, 128, 128)
    shadow_offset: Tuple[int, int] = (2, 2)
    weight: float = 1.0

class GradientTextRendererConfig(BaseModel):
    type: Literal["gradient"]
    start_color: Tuple[int, int, int] = (255, 0, 0)
    end_color: Tuple[int, int, int] = (0, 0, 255)
    weight: float = 1.0

class HorizontalTextRendererConfig(BaseModel):
    type: Literal["horizontal"]
    weight: float = 1.0

TextRendererConfigUnion = Union[
    HorizontalTextRendererConfig,
    SimpleTextRendererConfig,
    OutlinedTextRendererConfig,
    ShadowedTextRendererConfig,
    GradientTextRendererConfig
]

class CenterPlacementConfig(BaseModel):
    type: Literal["center"]
    padding: int = 20
    weight: float = 1.0

class RandomPlacementConfig(BaseModel):
    type: Literal["random"]
    margin: int = 50
    weight: float = 1.0

class GridPlacementConfig(BaseModel):
    type: Literal["grid"]
    rows: int = 3
    cols: int = 3
    padding: int = 10
    weight: float = 1.0

class PageNumberPlacementConfig(BaseModel):
    type: Literal["page_number"]
    position: Literal["bottom_left", "bottom_right", "bottom_center"] = "bottom_right"
    margin: int = 20
    weight: float = 1.0

class PageTitlePlacementConfig(BaseModel):
    type: Literal["page_title"]
    position: Literal["top_left", "top_right", "top_center"] = "top_center"
    margin: int = 30
    weight: float = 1.0

TextPlacementConfigUnion = Union[
    CenterPlacementConfig,
    RandomPlacementConfig,
    GridPlacementConfig,
    PageNumberPlacementConfig,
    PageTitlePlacementConfig
]

class BaseAugmentationConfig(BaseModel):
    probability: Optional[float] = 1.0
    enabled: Optional[bool] = True
    weight: float = 1.0

class BlurAugmentationConfig(BaseAugmentationConfig):
    type: Literal["blur"]
    blur_radius: Union[float, Tuple[float, float]] = 1.0

class NoiseAugmentationConfig(BaseAugmentationConfig):
    type: Literal["noise"]
    noise_factor: Union[float, Tuple[float, float]] = 0.1

class RotationAugmentationConfig(BaseAugmentationConfig):
    type: Literal["rotation"]
    max_angle: Union[float, Tuple[float, float]] = 5.0

class BrightnessAugmentationConfig(BaseAugmentationConfig):
    type: Literal["brightness"]
    brightness_factor: Union[float, Tuple[float, float]] = 0.8

AugmentationConfigUnion = Union[
    BlurAugmentationConfig,
    NoiseAugmentationConfig,
    RotationAugmentationConfig,
    BrightnessAugmentationConfig
]

class LayoutConfig(BaseModel):
    type: str
    padding: Optional[int] = 50
    max_width: Optional[int] = None
    max_height: Optional[int] = None
    min_width: Optional[int] = None
    min_height: Optional[int] = None

class OutputConfig(BaseModel):
    images_dir: str
    metadata_file: str

class DatasetConfig(BaseModel):
    source: str
    path: str

class TextDataConfig(BaseModel):
    """Configuration for loading text data from various sources."""
    source_type: Literal['csv', 'parquet', 'huggingface']
    source_path: str
    text_column: str = 'text'
    title_column: Optional[str] = None
    # Optional fields, mainly for Hugging Face datasets
    split: Optional[str] = 'train'
    name: Optional[str] = None # For datasets with multiple configurations (e.g., 'wikitext-103-raw-v1')
    data_dir: Optional[str] = None # For datasets that require manual download

class AppConfig(BaseModel):
    fonts: List[FontConfig]
    backgrounds: List[BackgroundConfigUnion]
    text_renderers: Optional[List[TextRendererConfigUnion]] = []
    text_placements: Optional[List[TextPlacementConfigUnion]] = []
    augmentations: Optional[List[AugmentationConfigUnion]] = []
    augmentation_order: Literal['random','fixed'] = 'random'
    layout: LayoutConfig
    output: OutputConfig
    text_data: Optional[TextDataConfig] = None
    seed: Optional[int] = None

