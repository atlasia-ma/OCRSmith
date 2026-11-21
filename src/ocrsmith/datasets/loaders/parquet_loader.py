"""Reading a corpus out of a Parquet file.

`pandas` is imported inside the functions that need it, not at module scope. It is an
optional dependency — `ocrsmith[data]` — and a top-level import made `import ocrsmith`
itself fail on a base install, because the package imports the pipeline, which imports
the writers, which imported every loader. A plain `pip install ocrsmith` produced a
package whose own `--help` crashed.
"""

from .base import BaseTextDataLoader


class ParquetTextLoader(BaseTextDataLoader):
    """Loads text from Parquet files"""

    def __init__(self, text_column="text", title_column=None):
        super().__init__(text_column=text_column, title_column=title_column)

    def load_texts(self, parquet_path, **kwargs):
        try:
            import pandas as pd

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
