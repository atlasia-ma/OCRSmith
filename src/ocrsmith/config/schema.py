# src/ocrsmith/config/schema.py
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import yaml

class FontConfig(BaseModel):
    path: str
    size: int

class BackgroundConfig(BaseModel):
    type: str
    color: Optional[str] = None
    image_path: Optional[str] = None

class LayoutConfig(BaseModel):
    type: str

class OutputConfig(BaseModel):
    images_dir: str
    metadata_file: str

class DatasetConfig(BaseModel):
    source: str
    path: str

class AppConfig(BaseModel):
    fonts: List[FontConfig]
    backgrounds: List[BackgroundConfig]
    layout: LayoutConfig
    output: OutputConfig
    dataset: DatasetConfig
