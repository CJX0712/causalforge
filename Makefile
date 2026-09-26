# CausalForge — developer convenience targets
# Usage:  make setup   |  make lint  |  make test  |  make demo  |  make ci
PYTHON  ?= python
PIP     ?= $(PYTHON) -m pip
PYTEST  ?= $(PYTHON) -m pytest
RUFF    ?= $(PYTHON) -m ruff

.PHONY: help setup lint test demo ci clean

help:
	@echo "CausalForge targets:"
	@echo "  setup   create venv + install requirements.txt"
	@echo "  lint    ruff check"
	@echo "  test    pytest (quiet)"
	@echo "  demo    run Monte-Carlo benchmark -> examples/benchmark.json"
	@echo "  ci      lint + test + demo smoke (local CI equivalent)"
	@echo "  clean   remove caches"

setup:
	$(PYTHON) -m venv .venv
	.venv/Scripts/activate || . .venv/bin/activate
	$(PIP) install -r requirements.txt

lint:
	$(RUFF) check causalforge tests

test:
	$(PYTEST) -q -W ignore::UserWarning

demo:
	$(PYTHON) -m causalforge.examples.run_demo

ci: lint test demo
	@echo "CI equivalent: all green."

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ causalforge/**/__pycache__
