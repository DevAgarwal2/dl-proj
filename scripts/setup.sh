#!/usr/bin/env bash
# Create a virtualenv, install OpenMed and check the environment.
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

if ! command -v llama-server >/dev/null 2>&1; then
  echo "Hint: install the llama.cpp server with: brew install llama.cpp"
fi

PYTHONPATH=src python -m openmed.cli doctor
echo
echo "Next steps:"
echo "  1. Place NIH ChestX-ray14 under data/raw/nih-chestxray14/"
echo "  2. make prepare && make smoke"
echo "  3. Put literature under data/corpus/, then make ingest && make llama && make query Q=\"...\""
