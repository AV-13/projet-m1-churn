# Makefile — cibles principales du projet
# Usage : make setup | make train | make api | make dashboard | make test

PYTHON := .venv/bin/python
PIP := .venv/bin/pip
export PYTHONPATH := src

.PHONY: help setup train api dashboard test clean

help:
	@echo "Cibles disponibles :"
	@echo "  setup      - cree le venv et installe les dependances"
	@echo "  train      - entraine, compare, selectionne et serialise le modele final"
	@echo "  api        - lance l'API FastAPI (http://127.0.0.1:8000)"
	@echo "  dashboard  - lance le dashboard Streamlit"
	@echo "  test       - lance les tests pytest"
	@echo "  report     - genere le PDF du rapport (docs/03-rapport.pdf)"
	@echo "  clean      - supprime les artefacts generes"

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

train:
	$(PYTHON) -m churn.train

api:
	.venv/bin/uvicorn api.main:app --reload --app-dir . --host 127.0.0.1 --port 8000

dashboard:
	.venv/bin/streamlit run dashboard/app.py

test:
	$(PYTHON) -m pytest -q

report:
	$(PYTHON) scripts/build_report_pdf.py

clean:
	rm -f models/*.joblib models/*.json
	rm -f reports/figures/* reports/metrics/*
