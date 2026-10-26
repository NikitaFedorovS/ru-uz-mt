PYTHON  ?= python3
VENV    ?= .venv
BIN     := $(VENV)/bin
CONFIG  ?= configs/train_config.yaml
IMAGE   ?= ru-uz-mt
# Docker build variant: gpu | cpu
VARIANT ?= gpu
# Extra CLI flags, e.g. make train ARGS="--max-train-samples 2000 --max-steps 50"
ARGS    ?=

.PHONY: install download data train eval serve docker test lint format clean

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt
	$(BIN)/pip install -e .

download:
	scripts/download_data.sh

data: download
	$(BIN)/python -m ru_uz_mt.data.prepare --config $(CONFIG) $(ARGS)

train:
	$(BIN)/python -m ru_uz_mt.train.train_lora --config $(CONFIG) $(ARGS)

eval:
	$(BIN)/python -m ru_uz_mt.eval.evaluate --config $(CONFIG) $(ARGS)

serve:
	$(BIN)/uvicorn app.main:app --host 0.0.0.0 --port $${PORT:-8000}

docker:
	docker build --build-arg VARIANT=$(VARIANT) -t $(IMAGE):$(VARIANT) .

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check src app tests
	$(BIN)/black --check src app tests

format:
	$(BIN)/ruff check --fix src app tests
	$(BIN)/black src app tests

clean:
	rm -rf .pytest_cache .ruff_cache build *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
