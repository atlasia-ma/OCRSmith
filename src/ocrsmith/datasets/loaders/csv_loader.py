# src/ocrsmith/datasets/loaders/CSVTextLoader.py

import pandas as pd

from .base import BaseTextDataLoader


class CSVTextLoader(BaseTextDataLoader):
    def __init__(self, text_column="text", title_column=None):
        super().__init__(text_column=text_column, title_column=title_column)

    def load_texts(self, csv_path, **kwargs):
        encoding = kwargs.get("encoding")
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_path}") from None

        if self.text_column not in df.columns:
            raise ValueError(f"Column '{self.text_column}' not found in CSV. Available: {list(df.columns)}")

        if self.title_column and self.title_column not in df.columns:
            self.title_column = None

        if self.title_column:
            df = df.dropna(subset=[self.text_column])
            self.texts = [
                {"content": str(txt), "title": str(title)}
                for txt, title in zip(df[self.text_column], df[self.title_column].fillna(""), strict=False)
            ]
        else:
            self.texts = df[self.text_column].dropna().astype(str).tolist()

        return self.texts

    def __iter__(self):
        return iter(self.texts)

    def iter_texts(self, csv_path, chunksize=10000, **kwargs):
        """Stream texts from a large CSV in chunks."""
        encoding = kwargs.get("encoding")
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, encoding=encoding):
            if self.text_column not in chunk.columns:
                raise ValueError(f"Column '{self.text_column}' not found in CSV chunk.")

            if self.title_column and self.title_column not in chunk.columns:
                self.title_column = None

            if self.title_column:
                chunk = chunk.dropna(subset=[self.text_column])
                for txt, title in zip(
                    chunk[self.text_column], chunk[self.title_column].fillna(""), strict=False
                ):
                    yield {"content": str(txt), "title": str(title)}
            else:
                for txt in chunk[self.text_column].dropna().astype(str).tolist():
                    yield txt
