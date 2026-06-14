test:
	uv run mypy src/ tests/
	uv run pytest tests

prettier:
	uv run ruff check --fix . && uv run ruff format .