# Contributing to OCRSmith

Thanks for helping build OCRSmith. This document describes how the project is
developed so that contributions land smoothly.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[data,dev]"
pytest
```

`pytest` works from a clean clone without an install because `pythonpath = ["src"]`
is set in `pyproject.toml`.

## Test-driven workflow

Every behavioural change starts with a test.

1. Write a failing test under `tests/` that expresses the desired behaviour.
2. Run it and watch it fail for the right reason.
3. Write the smallest implementation that makes it pass.
4. Refactor with the suite green.

Rendering code is tested against *properties*, not golden images: bounding boxes
must sit inside the page, text must round-trip through shaping, generated
samples must carry a label. Golden-image tests are brittle across FreeType
versions and are avoided.

## Code style

- `ruff check src tests` and `ruff format src tests` must pass.
- Strategy classes live one-per-file under the subsystem they belong to and are
  re-exported from the package `__init__.py`.
- Public functions carry type hints. Domain objects are frozen dataclasses.
- Prefer generators over materialised lists on any path that can see a large
  dataset.

## Branch and commit conventions

Branches are named `<type>/<slug>`:

| Type       | Use                                        |
| ---------- | ------------------------------------------ |
| `feat/`    | new capability                             |
| `fix/`     | bug fix                                    |
| `refactor/`| behaviour-preserving restructure           |
| `perf/`    | performance work                           |
| `docs/`    | documentation only                         |
| `chore/`   | tooling, CI, dependencies                  |

Commits follow [Conventional Commits](https://www.conventionalcommits.org/):
`feat(text): add bidi-aware line segmentation`.

## Pull requests

One PR per branch, one concern per PR. The description states what changed, why,
and how it was verified. CI (lint + tests on Linux/Windows, Python 3.10 and 3.12)
must be green before merge.

## Releasing

1. Update `CHANGELOG.md` under a new version heading.
2. Bump `version` in `pyproject.toml`.
3. Tag `vX.Y.Z` on `main` and publish the GitHub release.
