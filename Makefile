.PHONY: help check lint typecheck format format-check test complexity mdlint run test-opencloning-contract

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "%-30s %s\n", $$1, $$2}'

check: lint typecheck format-check complexity mdlint test ## Run all local quality gates

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
	@# .plans/ contains internal handoff docs, not public documentation.
	npx --no-install markdownlint-cli2 '**/*.md' '#node_modules' '#.git' '#.venv' '#.plans' || { \
		echo "markdownlint-cli2 not installed — run: npm install"; exit 1; }

test: ## Run tests
	uv run pytest

test-opencloning-contract: ## Run real-container OpenCloning contract tests (requires LAB_COPILOT_OPENCLONING_URL)
	@uv run pytest -n 0 -v tests/test_opencloning_contract.py || { \
		echo ""; \
		echo "Contract tests require a running OpenCloning container."; \
		echo "Set LAB_COPILOT_OPENCLONING_URL=http://localhost:8091 and retry."; \
		echo "Pinned image: manulera/opencloning:v1.3.1-baseurl-opencloning"; \
		exit 1; }

run: ## Run local development server
	uv run uvicorn lab_copilot_gateway.app:app --reload --host 127.0.0.1 --port 8080
