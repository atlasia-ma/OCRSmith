# src/ocrsmith/datasets/loaders/CSVTextLoader.py

from .BaseTextDataLoader import BaseTextDataLoader

import pandas as pd

class CSVTextLoader(BaseTextDataLoader):
    def __init__(self, text_column='text'):
        self.text_column = text_column
        self.texts = []
    
    def load_texts(self, csv_path, **kwargs):
        df = pd.read_csv(csv_path)
        self.texts = df[self.text_column].dropna().tolist()
        return self.texts
    
    def __iter__(self):
        return iter(self.texts)
    