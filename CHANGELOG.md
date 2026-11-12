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
