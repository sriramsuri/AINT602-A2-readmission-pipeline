# One-command pipeline stages (Unix/macOS). Windows users: see run.ps1.
PY ?= python

.PHONY: setup validate train evaluate diagram sample predict test lint all clean

setup:
	$(PY) -m pip install -r requirements.txt

validate:
	$(PY) -m src.validate

train:
	$(PY) -m src.train

evaluate:
	$(PY) -m src.evaluate

diagram:
	$(PY) -m src.make_diagram

sample:
	$(PY) -m src.make_sample

predict: sample
	$(PY) -m src.predict --input data/sample_new_patients.csv --output artifacts/predictions.csv

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests

all: validate train evaluate diagram predict test
	@echo "Pipeline complete."

clean:
	rm -f artifacts/*.joblib artifacts/*.json artifacts/predictions.csv reports/figures/*.png reports/figures/*.svg
