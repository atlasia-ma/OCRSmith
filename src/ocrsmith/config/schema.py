# src/ocrsmith/config/schema.py
from pydantic import BaseModel, field_validator, model_validator
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
    
    @model_validator(mode="after")
    def validate_fields_based_on_type(self):
        bg_type = self.type

        allowed_fields = {
            "solid": ["color"],
            "image": ["image_path", "mode"],
            "gradient": ["start_color", "end_color", "direction"],
            "noise": ["noise_type", "intensity", "base_color"],
        }

        if bg_type not in allowed_fields:
            raise ValueError(f"Invalid background type: {bg_type}")

        # Check that only allowed fields are set
        for field_name, value in self.__dict__.items():
            if field_name != "type" and value is not None:
                if field_name not in allowed_fields[bg_type]:
                    raise ValueError(
                        f"Field '{field_name}' is not allowed when type='{bg_type}'"
                    )
                    
        if bg_type == "image" and not self.image_path:
            raise ValueError("Field 'image_path' is required when type='image'")
        
        return self  
    

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
