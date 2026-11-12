# JudgeCal

## Purpose
JudgeCal is an LLM-as-judge calibration toolkit. It audits how reliable an LLM judge is per metric: agreement with human labels, position/length/self-preference bias, and run-to-run consistency, rolled into a transparent 0-100 trust score with a usage recommendation. It produces CI-friendly reports and can gate a pipeline on trust score.

## Tech Stack
- Python 3.11+
- pydantic v2, pandas/numpy, scipy/statsmodels, scikit-learn, pytest, typer
- MLflow (experiment tracking), Langfuse (tracing/cost), DVC (data/output versioning)
- FastAPI (optional API), Streamlit (dashboard)
- Docker, GitHub Actions

## Commands
- Install: TODO (uv or poetry, pin in pyproject.toml)
- Test: `pytest -q`
- Lint: TODO (ruff)
- Type-check: TODO (mypy or pyright)
- Format: TODO (ruff format)
- Run CLI: TODO (`judgecal audit --config ...`)

## Coding Conventions
- Type hints on every function signature, including return types.
- Pydantic v2 models for all data schemas (dataset rows, judge outputs, configs). No bare dicts crossing module boundaries.
- Public functions get a docstring with a usage example, not just a description.
- No `print()` — use `logging` with module-level loggers.
- I/O (API calls, file reads over network, DB) is async. Pure computation (stats, metrics) stays sync.
- Statistics/metrics functions are pure: same input → same output, no hidden state, no I/O inside them.

## Testing Rules
- TDD for all metrics code: write the test with a known answer first, watch it fail, then implement.
- Known-answer tests: assert exact or tolerance-bounded values, not just "runs without error".
- Synthetic data with planted effects for every bias detector (e.g. inject a known position bias, assert the detector reports it within tolerance).
- No network calls in unit tests. Mock adapters or use recorded fixtures.
- Use pytest markers `slow` and `integration` for anything that's not a fast unit test; default run excludes them.

## Safety Rules
- No hardcoded secrets. API keys only via environment variables, documented in `.env.example`.
- Any paid run (real judge API calls) prints an estimated cost and requires confirmation before proceeding, unless explicitly overridden.
- Caching of judge outputs is on by default.
- Every pipeline/CLI command that calls a judge supports `--limit` to cap sample size.

## Definition of Done
- Code has type hints and passes type-check.
- Tests added/updated and passing (`pytest -q` green), metrics have known-answer or synthetic-effect tests.
- No secrets, no `print()`, no network calls added to unit tests.
- Lint/format clean.
- Brief summary written: what changed, how to run/verify it, what's next.

## Decisions Log
<!-- Append key design decisions here as they're made, newest last. -->
- 2026-11-12: PyPI package name is `llmjudgecal` (`judgecal` taken by unrelated package). Repo dir, CLAUDE.md title, docs stay `judgecal`; Python import path is `llmjudgecal`.
- 2026-11-12: Milestone 1 judge adapter targets a self-hosted OpenAI-compatible endpoint (user has Llama via vLLM/SGLang-style endpoint, no Anthropic/OpenAI keys yet). Anthropic/OpenAI adapters come later behind the same interface.
- 2026-11-12: Dev/test API budget capped at $20 total. Lean on synthetic data, caching, and `--limit` for anything that costs money; avoid large real-dataset runs until later milestones.
- 2026-11-12: Build backend = hatchling, package manager = pip. Hatchling: zero-config PEP 621 src-layout support, no setup.py/setup.cfg boilerplate, fast, actively maintained, standard for new pure-Python packages in 2026. Rejected setuptools: needs more config for src-layout + py.typed and carries legacy baggage we don't need. Package manager = plain pip (`pip install -e ".[dev]"`): zero extra tooling to onboard with, works everywhere including CI runners out of the box. uv is faster but adds a dependency most contributors won't have pre-installed; revisit if install/CI time becomes a real pain point.
