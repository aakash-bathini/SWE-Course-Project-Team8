.PHONY: help install-dev black-check black-format flake mypy check test

help:
	@echo "Common tasks:"
	@echo "  make install-dev   # pip install -r requirements-dev.txt"
	@echo "  make black-check   # run black --check ."
	@echo "  make black-format  # run black ."
	@echo "  make flake         # run flake8 ."
	@echo "  make mypy          # run mypy per mypy.ini"
	@echo "  make check         # black --check + flake8 + mypy"

install-dev:
	pip install -r requirements-dev.txt

black-check:
	python -m black --check .

black-format:
	python -m black .

flake:
	flake8 .

mypy:
	mypy . --config-file mypy.ini

check: black-check flake mypy
	@echo "All checks passed."


