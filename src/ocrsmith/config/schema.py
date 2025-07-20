# src/ocrsmith/config/schema.py
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Tuple, Union
import yaml

class FontConfig(BaseModel):
    path: str
    size: int

class BackgroundConfig(BaseModel):
    type: str
    
    color: Optional[Union[str, Tuple[int, int, int]]] = None
    
    image_path: Optional[str] = None
    mode: Optional[str] = None
    
    start_color: Optional[Tuple[int, int, int]] = None
    end_color: Optional[Tuple[int, int, int]] = None
    direction: Optional[str] = None
    
    noise_type: Optional[str] = None
    intensity: Optional[float] = None
    base_color: Optional[Tuple[int, int, int]] = None
    

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
