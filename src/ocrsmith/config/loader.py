# src/ocrsmith/config/loader.py
from .schema import AppConfig
import yaml
import os

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "default_config.yaml")
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    config = AppConfig(**config_dict)
    return config