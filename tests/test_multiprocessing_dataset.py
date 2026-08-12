from unittest.mock import patch

import pytest

from ocrsmith.config.schema import AppConfig
from ocrsmith.core.engine import OCRSmithEngine


@pytest.fixture
def small_app_config(tmp_path):
    yaml_dict = {
        "fonts": [{"path": "assets/fonts/dummy.ttf"}],
        "text_data": {"source_type": "csv", "source_path": str(tmp_path / "data.csv"), "text_column": "text"},
        "backgrounds": [{"type": "solid", "color": [255, 255, 255]}],
        "text_renderers": [{"type": "horizontal"}],
        "text_placements": [{"type": "center"}],
        "augmentations": [],
        "layout": {"type": "simple"},
        "output": {"images_dir": "outputs/images", "metadata_file": "outputs/metadata.jsonl"},
        "seed": 42,
    }
    import pandas as pd

    df = pd.DataFrame({"text": ["a", "b", "c", "d"]})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    return AppConfig.model_validate(yaml_dict)


def test_generate_dataset_single_process(monkeypatch, small_app_config, tmp_path):
    eng = OCRSmithEngine(small_app_config)
    with (
        patch.object(eng.text_data_manager, "load_from_source"),
        patch.object(eng.text_data_manager, "get_random_text", side_effect=["a", "b", "c"]),
    ):
        anns = eng.generate_test_dataset(num_samples=3, output_dir=str(tmp_path), workers=1)
        assert len(anns) == 3
