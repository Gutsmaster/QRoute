#!/usr/bin/env bash
# Regenerates everything from scratch: data pipeline -> QPSO -> benchmarks -> plots -> map.
# Needs the dataset in data/new_delhi_traffic_dataset/ (see README.md).
# Usage: ./run_full_pipeline.sh
set -e
cd "$(dirname "$0")"

PY=""
for c in python3.13 python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
    echo "Python 3.11+ is required (found: $(python3 --version 2>&1))."
    echo "Install it with:  brew install python@3.12   then re-run ./run_dashboard.sh"
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "First run: setting up a local environment with $PY (happens once)..."
    "$PY" -m venv venv
fi
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

if [ ! -d "data/new_delhi_traffic_dataset" ]; then
    echo "Dataset not found at data/new_delhi_traffic_dataset/"
    echo "Unzip the Kaggle dataset there first (see README.md), then re-run this script."
    exit 1
fi

echo "=== Running full pipeline (data -> graph -> QPSO -> benchmarks) ==="
python3 run_everything.py

echo "=== Building interactive route map ==="
python3 build_map.py

echo "Done. Now run ./run_dashboard.sh to view the results."
