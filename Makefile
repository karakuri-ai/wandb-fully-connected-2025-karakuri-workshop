.PHONY: format-ruff
format-ruff:
	@uv run ruff format src/

.PHONY: format-ruff-isort
format-ruff-isort:
	@uv run ruff check --select I --fix src/

.PHONY: format
format: format-ruff format-ruff-isort

.PHONY: lint-ruff
lint-ruff:
	-@uv run ruff check src/

.PHONY: lint-pyright
lint-pyright:
	@uv run pyright src/

.PHONY: lint
lint: lint-ruff lint-pyright
