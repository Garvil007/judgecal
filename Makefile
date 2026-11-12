.PHONY: install test lint typecheck format all

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

typecheck:
	mypy src

format:
	ruff format .
	ruff check --fix .

all: format lint typecheck test
