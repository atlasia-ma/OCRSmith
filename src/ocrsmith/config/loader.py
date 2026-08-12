# src/ocrsmith/config/loader.py
import os

import yaml

from .schema import AppConfig


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "default_config.yaml")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    config = AppConfig(**config_dict)
    return config
