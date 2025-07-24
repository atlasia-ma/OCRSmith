# src/ocrsmith/datasets/loaders/ParquetTextLoader.py

from .BaseTextDataLoader import BaseTextDataLoader

class ParquetTextLoader(BaseTextDataLoader):
    """Loads text from Parquet files"""
    def __init__(self, text_column='text'):
        self.text_column = text_column
        self.texts = []
    
    def load_texts(self, parquet_path, **kwargs):
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        self.texts = df[self.text_column].dropna().tolist()
        return self.texts
    
    def __iter__(self):
        return iter(self.texts)