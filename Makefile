.PHONY: help check lint format format-check test run

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "%-18s %s\n", $$1, $$2}'

check: lint format-check test ## Run all local quality gates

lint: ## Lint Python code
	uv run ruff check .

format: ## Format Python code
	uv run ruff format .

format-check: ## Check Python formatting
	uv run ruff format --check .

test: ## Run tests
	uv run pytest

run: ## Run local development server
	uv run uvicorn lab_copilot_gateway.app:app --reload --host 127.0.0.1 --port 8080
