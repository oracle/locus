.PHONY: all format lint test tests integration_tests help clean

# Default target
all: help

######################
# TESTING
######################

test tests:
	hatch run test

test-cov:
	hatch run test-cov

test-fast:
	hatch run test-fast

integration_tests:
	hatch run pytest tests/integration/ -v

test-all:
	hatch run pytest tests/ -v

######################
# LINTING AND FORMATTING
######################

lint:
	hatch run lint

format:
	hatch run ruff format src/ tests/
	hatch run ruff check --select I --fix src/ tests/

typecheck:
	hatch run typecheck

######################
# DEVELOPMENT
######################

install:
	pip install hatch
	hatch env create

install-dev:
	pip install hatch
	hatch env create
	hatch run pre-commit install
	hatch run pre-commit install --hook-type commit-msg

clean:
	hatch env prune
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

######################
# BUILD & PUBLISH
######################

build:
	hatch build

publish:
	hatch publish

######################
# DOCUMENTATION
######################

docs-serve:
	hatch run docs:serve

docs-build:
	hatch run docs:build

######################
# HELP
######################

help:
	@echo 'Locus Development Commands'
	@echo '========================='
	@echo ''
	@echo 'Testing:'
	@echo '  make test              - run unit tests'
	@echo '  make test-cov          - run tests with coverage'
	@echo '  make test-fast         - run tests in parallel'
	@echo '  make integration_tests - run integration tests'
	@echo '  make test-all          - run all tests'
	@echo ''
	@echo 'Linting & Formatting:'
	@echo '  make lint              - run linters (ruff, mypy)'
	@echo '  make format            - format code with ruff'
	@echo '  make typecheck         - run mypy type checking'
	@echo ''
	@echo 'Development:'
	@echo '  make install           - install hatch and create env'
	@echo '  make install-dev       - install with pre-commit hooks'
	@echo '  make clean             - remove build artifacts and envs'
	@echo ''
	@echo 'Build & Publish:'
	@echo '  make build             - build package'
	@echo '  make publish           - publish to PyPI'
	@echo ''
	@echo 'Documentation:'
	@echo '  make docs-serve        - serve docs locally'
	@echo '  make docs-build        - build documentation'
