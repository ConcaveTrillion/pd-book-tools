.PHONY: install setup reinstall reset-venv reset-full test lint format pre-commit-check clean help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies and set up development environment
	@echo "📦 Installing dependencies..."
	uv sync --extra dev
	@echo "🪝 Setting up pre-commit hooks..."
	uv run pre-commit install
	@echo "✅ Installation complete!"

setup: install ## Alias for install

reinstall: reset-venv ## Alias for reset-venv (backward compatibility)

remove-venv: ## Remove the virtual environment
	@echo "🗑️  Removing existing virtual environment..."
	rm -rf .venv
	@echo "✅ Virtual environment removed!"

reset: ## Rebuild virtual environment (keeps UV cache)
	@$(MAKE) --no-print-directory clean
	@$(MAKE) --no-print-directory remove-venv
	@$(MAKE) --no-print-directory install
	@echo "✅ Environment Reset!"

reset-full: ## Nuclear option: clear everything and redownload
	@echo "🔄 FULL RESET: Clearing all caches and virtual environment..."
	@$(MAKE) --no-print-directory clean
	@$(MAKE) --no-print-directory remove-venv
	@echo "🧹 Clearing UV cache..."
	uv cache clean
	@echo "⬇️ Dependencies should download fresh now."
	@$(MAKE) --no-print-directory install
	@echo "💥 Full reset complete! Everything is fresh!"

test: ## Run tests
	@echo "🧪 Running tests..."
	uv run pytest

lint: ## Run linting checks
	@echo "🔍 Running linting checks..."
	uv run ruff check --select I --fix
	uv run ruff check --fix

format: ## Format code
	@echo "✨ Formatting code..."
	uv run ruff format

pre-commit-check: ## Run pre-commit on all files
	@echo "🪝 Running pre-commit on all files..."
	uv run pre-commit run --all-files

clean: ## Clean up cache and temporary files (keeps venv and UV cache)
	@echo "🧹 Cleaning Python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "🧹 Cleaning coverage files..."
	rm -rf htmlcov/ 2>/dev/null || true
	rm -f coverage.xml 2>/dev/null || true
	@echo "✅ Cache cleanup complete!"
