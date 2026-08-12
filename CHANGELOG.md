# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ocrsmith.core.rendering`: a text renderer that emits pixels and their annotation in
  the same pass, so the two cannot disagree.
  - Words are drawn individually, which yields exact per-word boxes and is typographically
    safe for Arabic (letters never join across a space).
  - `visual_word_order` implements word-level bidi run reordering, so mixed
    Arabic/Latin/digit lines are drawn in the right place while the label stays logical.
  - Wrapping is lossless: no word is silently dropped, oversized words are broken rather
    than overflowed, and lines that do not fit the height budget are *reported* so the
    label can be trimmed to exactly what was drawn.
  - `TextStyle` covers alignment (including natural RTL), line/word spacing, stroke,
    underline, strikethrough, synthetic bold/italic, and baseline/word-gap jitter, with an
    injectable `random.Random` so any sample is reproducible.
- `ocrsmith.text`: a text subsystem that owns everything happening to a string before it
  becomes pixels.
  - Script and base-direction detection (`detect_script`, `detect_direction`).
  - `NormalizationPolicy` — opt-in, idempotent transforms for diacritics, tatweel,
    alef/ya/ta-marbuta unification, numeral systems and whitespace. Each one changes the
    label, so each one is explicit and recorded.
  - Bidi/shaping with two interchangeable backends: `TransparentShaper` when Pillow has
    Raqm, `ReshaperBidiShaper` (arabic-reshaper + python-bidi) otherwise. Both keep the
    logical string as the label so datasets are identical across machines.
  - Font glyph coverage via fontTools `cmap` (`supports_text`, `fonts_supporting`), so a
    font that cannot draw a character is rejected instead of silently emitting tofu.

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
