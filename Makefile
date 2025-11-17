# ═══════════════════════════════════════════════════════════════════════════════
# Makefile - SYSTÈME RAG HYBRIDE
# Commandes rapides pour développement et déploiement
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: help install install-dev setup run test clean lint format docs

# Couleurs pour output
BLUE=\033[0;34m
GREEN=\033[0;32m
RED=\033[0;31m
NC=\033[0m # No Color

# ───────────────────────────────────────────────────────────────────────────────
# Help - Afficher les commandes disponibles
# ───────────────────────────────────────────────────────────────────────────────
help:
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)  Système RAG Hybride - Commandes Make disponibles$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(GREEN)Installation et setup:$(NC)"
	@echo "  make install          - Installer les dépendances de production"
	@echo "  make install-dev      - Installer les dépendances de développement"
	@echo "  make setup            - Configuration initiale (Google Drive, vector store)"
	@echo ""
	@echo "$(GREEN)Exécution:$(NC)"
	@echo "  make run              - Lancer l'interface Streamlit"
	@echo "  make build-index      - Construire la base vectorielle"
	@echo "  make download-pdfs    - Télécharger les PDFs depuis Google Drive"
	@echo ""
	@echo "$(GREEN)Tests et qualité:$(NC)"
	@echo "  make test             - Exécuter les tests unitaires"
	@echo "  make test-cov         - Tests avec coverage"
	@echo "  make lint             - Linter le code (flake8 + mypy)"
	@echo "  make format           - Formatter le code (black + isort)"
	@echo "  make benchmark        - Benchmarks de performance"
	@echo ""
	@echo "$(GREEN)Documentation:$(NC)"
	@echo "  make docs             - Générer la documentation"
	@echo "  make docs-serve       - Servir la documentation localement"
	@echo ""
	@echo "$(GREEN)Nettoyage:$(NC)"
	@echo "  make clean            - Nettoyer les fichiers temporaires"
	@echo "  make clean-all        - Nettoyage complet (data + cache)"
	@echo ""
	@echo "$(GREEN)Docker (optionnel):$(NC)"
	@echo "  make docker-build     - Construire l'image Docker"
	@echo "  make docker-run       - Lancer le container Docker"
	@echo ""

# ───────────────────────────────────────────────────────────────────────────────
# Installation
# ───────────────────────────────────────────────────────────────────────────────
install:
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "$(GREEN)✓ Installation complete!$(NC)"

install-dev:
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	pip install --upgrade pip
	pip install -r requirements-dev.txt
	@echo "$(GREEN)✓ Development installation complete!$(NC)"

# ───────────────────────────────────────────────────────────────────────────────
# Setup et configuration initiale
# ───────────────────────────────────────────────────────────────────────────────
setup:
	@echo "$(BLUE)Setting up project...$(NC)"
	@if [ ! -f .env ]; then \
		echo "Creating .env from .env.example..."; \
		cp .env.example .env; \
		echo "$(RED)⚠ Please edit .env with your API keys!$(NC)"; \
	else \
		echo ".env already exists"; \
	fi
	@mkdir -p data/raw data/processed data/vector_store data/logs
	@echo "$(GREEN)✓ Setup complete!$(NC)"
	@echo "$(BLUE)Next steps:$(NC)"
	@echo "  1. Edit .env with your API keys"
	@echo "  2. Run: make setup-gdrive (configure Google Drive)"
	@echo "  3. Run: make download-pdfs (download PDFs)"
	@echo "  4. Run: make build-index (build vector store)"
	@echo "  5. Run: make run (start interface)"

setup-gdrive:
	@echo "$(BLUE)Configuring Google Drive...$(NC)"
	python scripts/setup_gdrive.py
	@echo "$(GREEN)✓ Google Drive configured!$(NC)"

# ───────────────────────────────────────────────────────────────────────────────
# Exécution
# ───────────────────────────────────────────────────────────────────────────────
run:
	@echo "$(BLUE)Starting Streamlit interface...$(NC)"
	@echo "$(GREEN)➜ Open browser at: http://localhost:8501$(NC)"
	streamlit run src/interface/app.py

download-pdfs:
	@echo "$(BLUE)Downloading PDFs from Google Drive...$(NC)"
	python scripts/download_pdfs.py
	@echo "$(GREEN)✓ PDFs downloaded!$(NC)"

build-index:
	@echo "$(BLUE)Building vector store...$(NC)"
	python scripts/build_vector_store.py
	@echo "$(GREEN)✓ Vector store built!$(NC)"

# ───────────────────────────────────────────────────────────────────────────────
# Tests
# ───────────────────────────────────────────────────────────────────────────────
test:
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v
	@echo "$(GREEN)✓ Tests complete!$(NC)"

test-cov:
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

test-integration:
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/integration_test.py -v
	@echo "$(GREEN)✓ Integration tests complete!$(NC)"

benchmark:
	@echo "$(BLUE)Running benchmarks...$(NC)"
	python scripts/run_benchmarks.py
	@echo "$(GREEN)✓ Benchmarks complete!$(NC)"

# ───────────────────────────────────────────────────────────────────────────────
# Qualité de code
# ───────────────────────────────────────────────────────────────────────────────
lint:
	@echo "$(BLUE)Linting code...$(NC)"
	@echo "Running flake8..."
	-flake8 src/ tests/ --max-line-length=100 --exclude=__pycache__
	@echo "Running mypy..."
	-mypy src/ --ignore-missing-imports
	@echo "$(GREEN)✓ Linting complete!$(NC)"

format:
	@echo "$(BLUE)Formatting code...$(NC)"
	@echo "Running black..."
	black src/ tests/ --line-length=100
	@echo "Running isort..."
	isort src/ tests/
	@echo "$(GREEN)✓ Formatting complete!$(NC)"

type-check:
	@echo "$(BLUE)Type checking...$(NC)"
	mypy src/ --ignore-missing-imports
	@echo "$(GREEN)✓ Type checking complete!$(NC)"

# ───────────────────────────────────────────────────────────────────────────────
# Documentation
# ───────────────────────────────────────────────────────────────────────────────
docs:
	@echo "$(BLUE)Generating documentation...$(NC)"
	cd docs && make html
	@echo "$(GREEN)✓ Documentation generated in docs/_build/html/$(NC)"

docs-serve:
	@echo "$(BLUE)Serving documentation...$(NC)"
	@echo "$(GREEN)➜ Open browser at: http://localhost:8000$(NC)"
	cd docs/_build/html && python -m http.server

# ───────────────────────────────────────────────────────────────────────────────
# Nettoyage
# ───────────────────────────────────────────────────────────────────────────────
clean:
	@echo "$(BLUE)Cleaning temporary files...$(NC)"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf build/
	rm -rf dist/
	@echo "$(GREEN)✓ Cleaned!$(NC)"

clean-all: clean
	@echo "$(BLUE)Cleaning all data and cache...$(NC)"
	@read -p "This will delete all PDFs, vector store, and logs. Continue? (y/N) " confirm; \
	if [ "$$confirm" = "y" ]; then \
		rm -rf data/raw/*.pdf; \
		rm -rf data/processed/*; \
		rm -rf data/vector_store/*; \
		rm -rf data/logs/*.log; \
		rm -rf data/embeddings_cache/*; \
		echo "$(GREEN)✓ All data cleaned!$(NC)"; \
	else \
		echo "$(RED)Cancelled.$(NC)"; \
	fi

# ───────────────────────────────────────────────────────────────────────────────
# Docker (optionnel)
# ───────────────────────────────────────────────────────────────────────────────
docker-build:
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker-compose build
	@echo "$(GREEN)✓ Docker image built!$(NC)"

docker-run:
	@echo "$(BLUE)Starting Docker container...$(NC)"
	@echo "$(GREEN)➜ Open browser at: http://localhost:8501$(NC)"
	docker-compose up

docker-stop:
	@echo "$(BLUE)Stopping Docker container...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Docker stopped!$(NC)"

# ───────────────────────────────────────────────────────────────────────────────
# Développement
# ───────────────────────────────────────────────────────────────────────────────
notebook:
	@echo "$(BLUE)Starting Jupyter Lab...$(NC)"
	@echo "$(GREEN)➜ JupyterLab will open in your browser$(NC)"
	jupyter lab

pre-commit-install:
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	pre-commit install
	@echo "$(GREEN)✓ Pre-commit hooks installed!$(NC)"

pre-commit-run:
	@echo "$(BLUE)Running pre-commit on all files...$(NC)"
	pre-commit run --all-files
	@echo "$(GREEN)✓ Pre-commit complete!$(NC)"

# ───────────────────────────────────────────────────────────────────────────────
# Monitoring et logs
# ───────────────────────────────────────────────────────────────────────────────
logs:
	@echo "$(BLUE)Showing recent logs...$(NC)"
	tail -f data/logs/app.log

logs-errors:
	@echo "$(BLUE)Showing recent errors...$(NC)"
	tail -f data/logs/errors.log

# ═══════════════════════════════════════════════════════════════════════════════
# NOTES
# ═══════════════════════════════════════════════════════════════════════════════
#
# WORKFLOW RECOMMANDÉ:
# 1. make install          # Installer dépendances
# 2. make setup            # Configuration initiale
# 3. Edit .env             # Ajouter clés API
# 4. make setup-gdrive     # Configurer Google Drive
# 5. make download-pdfs    # Télécharger PDFs
# 6. make build-index      # Construire base vectorielle
# 7. make run              # Lancer interface
#
# DÉVELOPPEMENT:
# - make install-dev       # Setup développement
# - make format            # Avant chaque commit
# - make test              # Tester
# - make lint              # Vérifier qualité
#
# PRODUCTION:
# - make docker-build      # Build image Docker
# - make docker-run        # Lancer en production
#
# ═══════════════════════════════════════════════════════════════════════════════
