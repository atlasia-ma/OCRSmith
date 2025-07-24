# src/ocrsmith/datasets/loaders/HuggingFaceTextLoader.py

from .BaseTextDataLoader import BaseTextDataLoader

class HuggingFaceTextLoader(BaseTextDataLoader):
    def __init__(self, text_column='text'):
        self.text_column = text_column
        self.texts = []
    
    def load_texts(self, dataset_name, split='train', **kwargs):
        from datasets import load_dataset
        dataset = load_dataset(dataset_name, split=split)
        self.texts = [item[self.text_column] for item in dataset if self.text_column in item]
        return self.texts
    
    def __iter__(self):
        return iter(self.texts)
    