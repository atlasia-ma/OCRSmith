# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Development tooling: ruff lint/format configuration, pytest configuration with
  `pythonpath = ["src"]` (so a clean clone is testable without installing), coverage
  settings, and a GitHub Actions CI matrix over Linux/Windows and Python 3.10/3.12.
- `CONTRIBUTING.md` describing the TDD workflow, branch naming, and release process.
- Regression tests covering rendering without layout constraints.

### Changed

- `pyproject.toml`: dependency pins relaxed to compatible ranges, heavy tabular and
  Hugging Face dependencies moved into the `data` extra, project metadata and URLs
  completed.
- `HuggingFaceTextLoader` accepts any iterable of mappings, streams records via
  `iter_texts`, and defers the `datasets` import to call time.

### Fixed

- `wrap_text_by_pixels` raised `TypeError` when `layout.max_width` / `layout.max_height`
  were unset, which made every sample fail for configs that omit them.
- Whitespace-only input no longer produces a zero-sized canvas.
