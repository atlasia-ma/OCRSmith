from unittest.mock import patch

import pandas as pd
import pytest

from ocrsmith.datasets import CSVTextLoader, HuggingFaceTextLoader, ParquetTextLoader, TextDataManager
from ocrsmith.text import strip_diacritics


class TestCSVTextLoader:
    """Test CSVTextLoader"""

    def test_load_texts_from_csv(self, sample_csv_file):
        loader = CSVTextLoader(text_column="text")

        texts = loader.load_texts(sample_csv_file)

        assert len(texts) == 4
        assert "Hello world" in texts
        # The vocalised Arabic row must survive the round trip with its diacritics.
        assert any(strip_diacritics(text) != text for text in texts)

    def test_iterator(self, sample_csv_file):
        loader = CSVTextLoader(text_column="text")
        loader.load_texts(sample_csv_file)

        texts = list(loader)

        assert len(texts) == 4
        assert texts[0] == "Hello world"


class TestParquetTextLoader:
    """Test ParquetTextLoader"""

    @patch("pandas.read_parquet")
    def test_load_texts_from_parquet(self, mock_read_parquet):
        mock_df = pd.DataFrame({"text": ["text1", "text2", "text3"]})
        mock_read_parquet.return_value = mock_df

        loader = ParquetTextLoader(text_column="text")
        texts = loader.load_texts("dummy.parquet")

        assert texts == ["text1", "text2", "text3"]
        mock_read_parquet.assert_called_once_with("dummy.parquet")


class TestHuggingFaceTextLoader:
    """Test HuggingFaceTextLoader"""

    @patch("ocrsmith.datasets.loaders.huggingface_loader.load_dataset")
    def test_load_texts_from_huggingface(self, mock_load_dataset):
        mock_dataset = [
            {"text": "text1", "label": "A"},
            {"text": "text2", "label": "B"},
            {"other": "text3"},  # Should be filtered out
        ]
        mock_load_dataset.return_value = mock_dataset

        loader = HuggingFaceTextLoader(text_column="text")
        texts = loader.load_texts("test/dataset", split="train")

        assert texts == ["text1", "text2"]
        mock_load_dataset.assert_called_once_with("test/dataset", split="train")


class TestTextDataManager:
    """Test TextDataManager"""

    def test_load_from_csv_source(self, sample_csv_file):
        manager = TextDataManager()

        texts = manager.load_from_source("csv", sample_csv_file, text_column="text")

        assert len(texts) == 4
        assert isinstance(manager.current_loader, CSVTextLoader)

    @patch("ocrsmith.datasets.loaders.huggingface_loader.load_dataset")
    def test_load_from_huggingface_source(self, mock_load_dataset):
        mock_dataset = [{"text": "test text"}]
        mock_load_dataset.return_value = mock_dataset

        manager = TextDataManager()
        texts = manager.load_from_source("huggingface", "test/dataset")

        assert texts == ["test text"]
        assert isinstance(manager.current_loader, HuggingFaceTextLoader)

    def test_unsupported_source_type(self):
        manager = TextDataManager()

        with pytest.raises(ValueError, match="Unsupported source type"):
            manager.load_from_source("unsupported", "path")

    @patch("random.choice")
    def test_get_random_text(self, mock_choice, sample_csv_file):
        manager = TextDataManager()
        manager.load_from_source("csv", sample_csv_file)

        mock_choice.return_value = "selected text"

        result = manager.get_random_text()

        assert result == "selected text"
        mock_choice.assert_called_once()

    def test_get_random_text_no_loader(self):
        manager = TextDataManager()

        result = manager.get_random_text()

        assert result is None
