# src/ocrsmith/config/schema.py
from typing import Literal

from pydantic import BaseModel, Field


class FontConfig(BaseModel):
    name: str | None = None
    path: str
    size: int | None = 24


class SolidBackgroundConfig(BaseModel):
    type: Literal["solid"]
    color: str | tuple[int, int, int] = (255, 255, 255)
    weight: float = 1.0


class ImageBackgroundConfig(BaseModel):
    type: Literal["image"]
    image_path: str
    mode: Literal["stretch", "crop", "tile", "center"] = "stretch"
    weight: float = 1.0


class GradientBackgroundConfig(BaseModel):
    type: Literal["gradient"]
    start_color: tuple[int, int, int] = (255, 255, 255)
    end_color: tuple[int, int, int] = (200, 200, 200)
    direction: Literal["horizontal", "vertical", "diagonal"] = "horizontal"
    weight: float = 1.0


class TextureBackgroundConfig(BaseModel):
    type: Literal["texture"]
    base_color: tuple[int, int, int] = (240, 240, 240)
    noise_level: int = Field(default=20, ge=0, le=255)
    weight: float = 1.0


BackgroundConfigUnion = (
    SolidBackgroundConfig | ImageBackgroundConfig | GradientBackgroundConfig | TextureBackgroundConfig
)


class SimpleTextRendererConfig(BaseModel):
    type: Literal["simple"]
    color: tuple[int, int, int] = (0, 0, 0)
    weight: float = 1.0


class OutlinedTextRendererConfig(BaseModel):
    type: Literal["outlined"]
    fill_color: tuple[int, int, int] = (255, 255, 255)
    outline_color: tuple[int, int, int] = (0, 0, 0)
    outline_width: int = 2
    weight: float = 1.0


class ShadowedTextRendererConfig(BaseModel):
    type: Literal["shadowed"]
    text_color: tuple[int, int, int] = (0, 0, 0)
    shadow_color: tuple[int, int, int] = (128, 128, 128)
    shadow_offset: tuple[int, int] = (2, 2)
    weight: float = 1.0


class GradientTextRendererConfig(BaseModel):
    type: Literal["gradient"]
    start_color: tuple[int, int, int] = (255, 0, 0)
    end_color: tuple[int, int, int] = (0, 0, 255)
    weight: float = 1.0


class HorizontalTextRendererConfig(BaseModel):
    type: Literal["horizontal"]
    weight: float = 1.0


TextRendererConfigUnion = (
    HorizontalTextRendererConfig
    | SimpleTextRendererConfig
    | OutlinedTextRendererConfig
    | ShadowedTextRendererConfig
    | GradientTextRendererConfig
)


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


TextPlacementConfigUnion = (
    CenterPlacementConfig
    | RandomPlacementConfig
    | GridPlacementConfig
    | PageNumberPlacementConfig
    | PageTitlePlacementConfig
)


class BaseAugmentationConfig(BaseModel):
    probability: float | None = 1.0
    enabled: bool | None = True
    weight: float = 1.0


class BlurAugmentationConfig(BaseAugmentationConfig):
    type: Literal["blur"]
    blur_radius: float | tuple[float, float] = 1.0


class NoiseAugmentationConfig(BaseAugmentationConfig):
    type: Literal["noise"]
    noise_factor: float | tuple[float, float] = 0.1


class RotationAugmentationConfig(BaseAugmentationConfig):
    type: Literal["rotation"]
    max_angle: float | tuple[float, float] = 5.0


class BrightnessAugmentationConfig(BaseAugmentationConfig):
    type: Literal["brightness"]
    brightness_factor: float | tuple[float, float] = 0.8


AugmentationConfigUnion = (
    BlurAugmentationConfig
    | NoiseAugmentationConfig
    | RotationAugmentationConfig
    | BrightnessAugmentationConfig
)


class LayoutConfig(BaseModel):
    type: str
    padding: int | None = 50
    max_width: int | None = None
    max_height: int | None = None
    min_width: int | None = None
    min_height: int | None = None


class OutputConfig(BaseModel):
    images_dir: str
    metadata_file: str


class DatasetConfig(BaseModel):
    source: str
    path: str


class TextDataConfig(BaseModel):
    """Configuration for loading text data from various sources."""

    source_type: Literal["csv", "parquet", "huggingface"]
    source_path: str
    text_column: str = "text"
    title_column: str | None = None
    # Optional fields, mainly for Hugging Face datasets
    split: str | None = "train"
    name: str | None = None  # For datasets with multiple configurations (e.g., 'wikitext-103-raw-v1')
    data_dir: str | None = None  # For datasets that require manual download


class AppConfig(BaseModel):
    fonts: list[FontConfig]
    backgrounds: list[BackgroundConfigUnion]
    text_renderers: list[TextRendererConfigUnion] | None = []
    text_placements: list[TextPlacementConfigUnion] | None = []
    augmentations: list[AugmentationConfigUnion] | None = []
    augmentation_order: Literal["random", "fixed"] = "random"
    layout: LayoutConfig
    output: OutputConfig
    text_data: TextDataConfig | None = None
    seed: int | None = None
