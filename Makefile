.PHONY: help install train test serve all

PORT ?= 8000

help:  ## list targets
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-8s %s\n", $$1, $$2}'

install:  ## install dependencies with uv
	uv sync

train: install  ## download data, train, write docs/model.json + docs/backtest.json
	uv run python train.py

test: install  ## run unit tests
	uv run pytest -q

serve:  ## serve the dashboard at http://localhost:$(PORT)
	@echo "http://localhost:$(PORT)"
	cd docs && uv run python -m http.server $(PORT)

all: train test serve  ## everything, then serve
