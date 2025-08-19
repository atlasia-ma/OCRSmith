# src/

from .loaders import CSVTextLoader, ParquetTextLoader, HuggingFaceTextLoader

class TextDataManager:
    """Manages different text data sources"""
    def __init__(self):
        self.loaders = {
            'csv': CSVTextLoader,
            'parquet': ParquetTextLoader,
            'huggingface': HuggingFaceTextLoader
        }
        self.current_loader = None
    
    def load_from_source(self, source_type, source_path, text_column='text', title_column=None, **kwargs):
        """Load texts from specified source"""
        if source_type not in self.loaders:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        loader_class = self.loaders[source_type]
        self.current_loader = loader_class(text_column=text_column, title_column=title_column)
        
        return self.current_loader.load_texts(source_path, **kwargs)
    
    def get_random_text(self):
        """Get a random text from the loaded dataset"""
        import random
        if self.current_loader and self.current_loader.texts:
            return random.choice(self.current_loader.texts)
        return None
    
    def __iter__(self):
        if self.current_loader:
            return iter(self.current_loader)
        return iter([])
