# src/ocrsmith/core/fonts/FontCache.py

from typing import Optional, Tuple
from PIL.ImageFont import FreeTypeFont
import threading

class FontCache:
    """Singleton pattern for font caching with thread safety."""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._cache = {}
                    cls._instance._cache_lock = threading.Lock()
        return cls._instance
    
    def get(self, key: Tuple[str, int]) -> Optional[FreeTypeFont]:
        with self._cache_lock:
            return self._cache.get(key)
    
    def set(self, key: Tuple[str, int], font: FreeTypeFont) -> None:
        with self._cache_lock:
            self._cache[key] = font
    
    def clear(self) -> None:
        with self._cache_lock:
            self._cache.clear()
    
    def size(self) -> int:
        with self._cache_lock:
            return len(self._cache)
    
    def get_cache_info(self) -> dict:
        with self._cache_lock:
            return {
                'size': len(self._cache),
                'keys': list(self._cache.keys())
            }