# src/ocrsmith/datasets/loaders/HuggingFaceTextLoader.py

from collections.abc import Iterator, Mapping
from typing import Any

from .BaseTextDataLoader import BaseTextDataLoader

_AUTH_MARKERS = ("401", "403", "Unauthorized", "Permission", "Access")
_AUTH_HINT = (
    " Authentication may be required. Run 'huggingface-cli login', pass a token via "
    "TextDataManager.load_from_source(..., token=YOUR_HF_TOKEN), or set the HF_TOKEN env var."
)


def load_dataset(*args, **kwargs):  # pragma: no cover - thin lazy proxy
    """Proxy to ``datasets.load_dataset`` imported at call time.

    Importing ``datasets`` eagerly costs seconds of start-up and pulls a heavy optional
    dependency into every ``import ocrsmith``. Keeping it behind a module-level function
    also gives tests a single, stable patch target.
    """
    from datasets import load_dataset as _load_dataset

    return _load_dataset(*args, **kwargs)


class HuggingFaceTextLoader(BaseTextDataLoader):
    """Loads text from a Hugging Face dataset.

    Any object that iterates over mappings is accepted, so in-memory sequences can stand
    in for a real ``Dataset`` in tests and in offline pipelines.
    """

    def __init__(self, text_column: str = "text", title_column: str | None = None):
        super().__init__(text_column=text_column, title_column=title_column)

    def load_texts(self, dataset_name: str, split: str = "train", **kwargs) -> list:
        dataset = self._open(dataset_name, split, **kwargs)
        dataset = self._resolve_split(dataset, split)
        self._check_columns(dataset)
        self.texts = list(self._iter_records(dataset))
        return self.texts

    def iter_texts(self, dataset_name: str, split: str = "train", **kwargs) -> Iterator:
        """Stream records without materialising the whole dataset.

        Pass ``streaming=True`` through ``kwargs`` to also avoid downloading it in full.
        """
        dataset = self._open(dataset_name, split, **kwargs)
        dataset = self._resolve_split(dataset, split)
        self._check_columns(dataset)
        yield from self._iter_records(dataset)

    def __iter__(self):
        return iter(self.texts)

    # -- internals ---------------------------------------------------------

    def _open(self, dataset_name: str, split: str, **kwargs):
        token = kwargs.pop("token", None)
        try:
            if token:
                return load_dataset(dataset_name, split=split, token=token, **kwargs)
            return load_dataset(dataset_name, split=split, **kwargs)
        except Exception as exc:
            msg = str(exc)
            hint = _AUTH_HINT if any(marker in msg for marker in _AUTH_MARKERS) else ""
            raise RuntimeError(
                f"Failed to load dataset '{dataset_name}' (split='{split}').{hint} Original error: {msg}"
            ) from exc

    @staticmethod
    def _resolve_split(dataset: Any, split: str) -> Any:
        """Unwrap a ``DatasetDict``-like mapping of splits down to a single split.

        ``load_dataset`` returns every split when ``split`` is ignored by the builder,
        in which case the result is a dict keyed by split name.
        """
        if not isinstance(dataset, dict):
            return dataset
        if split not in dataset:
            raise ValueError(f"Split '{split}' not found. Available splits: {list(dataset)}")
        return dataset[split]

    def _check_columns(self, dataset: Any) -> None:
        """Validate declared columns when the source exposes them."""
        columns = getattr(dataset, "column_names", None)
        if not isinstance(columns, (list, tuple)):
            return  # duck-typed source: validated per record instead
        if self.text_column not in columns:
            raise ValueError(f"Column '{self.text_column}' not found. Available columns: {list(columns)}")
        if self.title_column and self.title_column not in columns:
            self.title_column = None

    def _iter_records(self, dataset: Any) -> Iterator:
        """Yield one record per usable row, skipping rows without text."""
        for row in dataset:
            if not isinstance(row, Mapping):
                continue
            value = row.get(self.text_column)
            if value is None:
                continue
            if self.title_column:
                title = row.get(self.title_column)
                yield {"content": str(value), "title": "" if title is None else str(title)}
            else:
                yield str(value)
