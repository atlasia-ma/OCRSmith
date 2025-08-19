# src/ocrsmith/datasets/loaders/HuggingFaceTextLoader.py

from .BaseTextDataLoader import BaseTextDataLoader
from datasets import load_dataset, DatasetDict
from typing import Optional

class HuggingFaceTextLoader(BaseTextDataLoader):
    def __init__(self, text_column='text', title_column=None):
        super().__init__(text_column=text_column, title_column=title_column)

    def load_texts(self, dataset_name, split='train', **kwargs):
        """Load texts from a Hugging Face dataset.
        If authentication is required, instruct the user to login and/or set a token.
        Accepts optional 'token' in kwargs.
        """
        token: Optional[str] = kwargs.get('token')
        try:
            if token:
                dataset = load_dataset(dataset_name, split=split, token=token)
            else:
                dataset = load_dataset(dataset_name, split=split)
        except Exception as e:
            msg = str(e)
            hint = ""
            if any(x in msg for x in ['401', '403', 'Unauthorized', 'Permission', 'Access']):
                hint = (
                    " Authentication may be required. Run: 'huggingface-cli login' "
                    "or pass a token via TextDataManager.load_from_source(..., token=YOUR_HF_TOKEN) "
                    "or set the HF_TOKEN env var."
                )
            raise RuntimeError(f"Failed to load dataset '{dataset_name}' (split='{split}').{hint} Original error: {msg}")
        
        if isinstance(dataset, DatasetDict):
            if split in dataset:
                dataset = dataset[split]
            else:
                raise ValueError(f"Split '{split}' not found. Available splits: {list(dataset.keys())}")

        if self.text_column not in dataset.column_names:
            raise ValueError(f"Column '{self.text_column}' not found. Available columns: {dataset.column_names}")

        # Drop rows with missing text
        filtered = dataset.filter(lambda x: x[self.text_column] is not None)
        if self.title_column:
            if self.title_column not in dataset.column_names:
                self.title_column = None

            self.texts = [
                {"content": str(txt), "title": str(title) if title is not None else ""}
                for txt, title in zip(filtered[self.text_column], filtered[self.title_column])
            ]
        else:
            self.texts = [str(txt) for txt in filtered[self.text_column]]

        return self.texts

    def __iter__(self):
        return iter(self.texts)
