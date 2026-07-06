.PHONY: help check lint typecheck format format-check test complexity mdlint run

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "%-18s %s\n", $$1, $$2}'

check: lint typecheck format-check mdlint test ## Run all local quality gates

lint: ## Lint Python code
	uv run ruff check .

typecheck: ## Static type check (pyright)
	pyright src/

format: ## Format Python code
	uv run ruff format .

format-check: ## Check Python formatting
	uv run ruff format --check .

complexity: ## Cognitive complexity check (complexipy, max 15)
	uv run complexipy src/ --max-complexity-allowed 15

mdlint: ## Lint Markdown files with markdownlint-cli2 (config: .markdownlint.json)
	@# markdownlint-cli2 is a local npm devDependency; npx resolves it.
	npx --no-install markdownlint-cli2 '**/*.md' '#node_modules' '#.git' '#.venv' || { \
		echo "markdownlint-cli2 not installed — run: npm install"; exit 1; }

test: ## Run tests
	uv run pytest

run: ## Run local development server
	uv run uvicorn lab_copilot_gateway.app:app --reload --host 127.0.0.1 --port 8080
