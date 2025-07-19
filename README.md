# OCRSmith

**OCRSmith** is a Python library for generating **synthetic OCR datasets** with support for **Arabic and Latin text**.
It enables users to create datasets from raw text, CSV files, Hugging Face datasets, or Parquet files.

---

## Features

- **Synthetic text generation** with configurable fonts and backgrounds.
- **Page layout simulation**: title pages, middle content, and footer with page numbers.
- **Supports Arabic and Latin text rendering**.
- **Multiple dataset input formats**:
  - CSV files
  - Hugging Face datasets
  - Parquet files
- **Background text rendering** (e.g., writing on existing page templates).
- **Push datasets directly to Hugging Face Hub**.

---

## Installation

```bash
pip install ocrsmith
```

## To clone the repo

git clone repo-id

cp .env .env.exemple

fill with your varaibles

conda create -n ocrsmith python=3.11
