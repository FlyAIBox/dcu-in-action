# DCU-in-Action Makefile
# Automation for development, testing, and deployment

.PHONY: help install test clean lint format docs docker

# Default target
help:
	@echo "Available targets:"
	@echo "  install    - Install dependencies"
	@echo "  test       - Run tests"
	@echo "  lint       - Run linting"
	@echo "  format     - Format code"
	@echo "  clean      - Clean build artifacts"
	@echo "  docs       - Generate documentation"
	@echo "  docker     - Build Docker images"

# Environment setup
install:
	pip install -r requirements.txt
	pip install -e .

# Testing
test:
	python -m pytest tests/ -v --cov=common

test-quick:
	python -m pytest tests/ -x

# Code quality
lint:
	flake8 common/ examples/ tests/
	mypy common/

format:
	black common/ examples/ tests/
	isort common/ examples/ tests/

# Documentation
docs:
	@echo "Generating documentation..."
	mkdir -p docs/api
	python -m pydoc -w common

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/

# Docker operations
docker:
	docker build -t dcu-in-action .

docker-dev:
	docker-compose up -d

docker-down:
	docker-compose down

# DCU specific
check-dcu:
	python -c "from common.dcu import DCUManager; print(DCUManager().get_system_info())"

# Quick start
quick-start: install check-dcu
	@echo "DCU-in-Action setup complete!" 