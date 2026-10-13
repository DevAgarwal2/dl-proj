# Prefer the project virtualenv when it exists.
VENV_PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
PYTHON ?= $(VENV_PY)
export PYTHONPATH := src

.PHONY: help setup venv doctor prepare smoke train train-chestmnist evaluate predict ingest fetch-corpus query llama llama-stop embed-server embed-server-stop build-gold eval-retrieval eval-answers demo serve ui test clean

help:
	@echo "OpenMed targets:"
	@echo "  setup      install dependencies"
	@echo "  doctor     check environment (torch/MPS/llama-server/model)"
	@echo "  prepare    build train/val/test splits from NIH metadata"
	@echo "  smoke      run a 1-epoch smoke training run"
	@echo "  train      full training run (configs/chestxray14.yaml, needs NIH data)"
	@echo "  train-chestmnist  train on MedMNIST ChestMNIST (no NIH download needed)"
	@echo "  fetch-corpus      fetch PubMed abstracts for the 14 findings"
	@echo "  evaluate   evaluate a checkpoint"
	@echo "  predict    predict a single image"
	@echo "  embed-server      start the EmbeddingGemma embedding server (:8081)"
	@echo "  embed-server-stop stop the embedding server"
	@echo "  ingest     embed the corpus into the vector store"
	@echo "  build-gold        build the retrieval gold set from the corpus"
	@echo "  eval-retrieval    score retrieval (Recall@k / MRR / nDCG)"
	@echo "  eval-answers      judge answer faithfulness with the local LLM"
	@echo "  query      ask the RAG pipeline a question"
	@echo "  demo       run the live end-to-end demo script"
	@echo "  llama      start the local llama.cpp server (LFM2.5-1.2B)"
	@echo "  llama-stop stop the llama.cpp server"
	@echo "  serve      run the FastAPI inference server"
	@echo "  ui         run the Streamlit browser UI"
	@echo "  test       run unit tests"

venv:
	python3 -m venv --system-site-packages .venv
	.venv/bin/python -m pip install --upgrade pip

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

doctor:
	$(PYTHON) -m openmed.cli doctor

prepare:
	$(PYTHON) -m openmed.cli prepare-data --config configs/chestxray14.yaml

smoke:
	$(PYTHON) -m openmed.cli train --config configs/smoke.yaml

train:
	$(PYTHON) -m openmed.cli train --config configs/chestxray14.yaml

train-chestmnist:
	$(PYTHON) -m openmed.cli train --config configs/chestmnist.yaml

fetch-corpus:
	$(PYTHON) scripts/fetch_pubmed.py --per-finding 40 --out data/corpus/pubmed

evaluate:
	$(PYTHON) -m openmed.cli evaluate --config configs/chestxray14.yaml --checkpoint runs/chestxray14/best.pt

predict:
	$(PYTHON) -m openmed.cli predict --checkpoint runs/chestxray14/best.pt --image $(IMAGE)

ingest:
	$(PYTHON) -m openmed.cli ingest --config configs/rag.yaml

build-gold:
	$(PYTHON) -m openmed.cli build-gold --config configs/rag.yaml

eval-retrieval:
	$(PYTHON) -m openmed.cli eval-retrieval --config configs/rag.yaml

eval-answers:
	$(PYTHON) -m openmed.cli eval-answers --config configs/rag.yaml

query:
	$(PYTHON) -m openmed.cli query --config configs/rag.yaml --question "$(Q)"

demo:
	$(PYTHON) scripts/demo.py $(IMAGE)

llama:
	bash scripts/run_llama_server.sh start

llama-stop:
	bash scripts/run_llama_server.sh stop

embed-server:
	bash scripts/run_embed_server.sh start

embed-server-stop:
	bash scripts/run_embed_server.sh stop

serve:
	$(PYTHON) -m uvicorn openmed.api.server:app --host 127.0.0.1 --port 8000

ui:
	$(PYTHON) -m streamlit run app/streamlit_app.py --server.port 8501

test:
	$(PYTHON) -m pytest -q tests || $(PYTHON) -m unittest discover -s tests -v

clean:
	rm -rf runs/checkpoints build dist *.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
