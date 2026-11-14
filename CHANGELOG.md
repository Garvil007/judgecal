# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Project skeleton (src/ layout) per the brief's suggested repo structure: `adapters/`, `datasets/`, `judges/`, `metrics/`, `trust/`, `report/` packages under `src/llmjudgecal/`.
- `pyproject.toml` (hatchling build backend), package `llmjudgecal` v0.0.1, Python >=3.11, Apache-2.0 license, `py.typed` marker.
- Minimal core dependencies (pydantic, numpy, pandas, scipy, statsmodels, typer, httpx, tenacity); optional extras `[tracking]`, `[serve]`, `[dashboard]`, `[dev]`.
- Dev tooling: ruff (lint + format), mypy (strict on `src`), pytest + pytest-cov + pytest-asyncio + hypothesis, pre-commit config.
- `judgecal` CLI stub (typer) with working `--version`.
- `Makefile` (install/test/lint/typecheck/format/all), `.gitignore`, `.env.example` (placeholders only).
- GitHub Actions CI (`.github/workflows/ci.yml`) running lint, type-check, and tests on Python 3.11 and 3.12.
- `CLAUDE.md` project conventions doc with a running Decisions log.
- Core data schema (`llmjudgecal.datasets.schema`, pydantic v2, all models frozen): `Response`, `PairwiseItem`, `ScoreItem`, `Dataset`, `TokenUsage`, `JudgeVerdict`, `JudgeScore`. Validators reject blank text/ids, out-of-range scores, inverted scales, duplicate dataset item ids, and mode/item-type mismatches. 28 known-answer tests (`tests/test_schema.py`).
- Dataset loaders (`llmjudgecal.datasets.loaders`): `load_pairwise_csv`, `load_pairwise_jsonl`, `load_score_csv`, `load_score_jsonl`, and a generic `load_dataset(path, mode, column_map=None)` dispatcher. Supports column remapping for non-canonical headers, JSON `metadata` columns, and streams JSONL line by line. Errors raise `DatasetLoadError` naming the file, 1-indexed row, and field. 20 tests (`tests/test_loaders.py`), including two Hypothesis roundtrip properties.
- `examples/pairwise_sample.jsonl` and `examples/score_sample.csv` (10 synthetic rows each).
- `docs/bring_your_own_data.md`: expected CSV/JSONL column formats, `column_map` usage, and `DatasetLoadError` format.
