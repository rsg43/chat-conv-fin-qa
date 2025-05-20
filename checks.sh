#!/bin/bash

echo "Running checks..."

echo "BLACK"
python3 -m black . --line-length=79

echo "FLAKE8"
python3 -m flake8 .

echo "PYLINT"
python3 -m pylint . --recursive=y

echo "MYPY"
python3 -m mypy . --strict

echo "PYDOCLINT"
pydoclint . --style=sphinx --check-class-attributes=False --skip-checking-short-docstrings=False

echo "SEMGREP"
semgrep scan --error --config auto

echo "PYTEST"
pytest

echo "Checks complete!"
