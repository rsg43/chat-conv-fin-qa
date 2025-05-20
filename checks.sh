#!/bin/bash

echo "Running checks..."

echo "BLACK"
black . --line-length=79

echo "FLAKE8"
flake8 .

echo "PYLINT"
pylint . --recursive=y

echo "MYPY"
mypy . --strict

echo "PYDOCLINT"
pydoclint . --style=sphinx --check-class-attributes=False --skip-checking-short-docstrings=False

echo "SEMGREP"
semgrep scan --error --config auto

echo "PYTEST"
pytest

echo "Checks complete!"
