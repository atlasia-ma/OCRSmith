# src/ocrsmith/datasets/loaders/ParquetTextLoader.py

import pandas as pd

from .BaseTextDataLoader import BaseTextDataLoader


class ParquetTextLoader(BaseTextDataLoader):
    """Loads text from Parquet files"""

    def __init__(self, text_column="text", title_column=None):
        super().__init__(text_column=text_column, title_column=title_column)

    def load_texts(self, parquet_path, **kwargs):
        try:
            df = pd.read_parquet(parquet_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}") from None

        # Ensure text column exists
        if self.text_column not in df.columns:
            self.text_column = None

        # Drop rows where text is missing
        df = df.dropna(subset=[self.text_column])

        if self.title_column and self.title_column in df.columns:
            self.texts = [
                {"content": str(txt), "title": str(title)}
                for txt, title in zip(df[self.text_column], df[self.title_column].fillna(""), strict=False)
            ]
        else:
            self.texts = df[self.text_column].astype(str).tolist()

        return self.texts

    def __iter__(self):
        return iter(self.texts)
