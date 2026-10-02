#!/usr/bin/env bash
# One-command dashboard launcher.  Usage: ./run_dashboard.sh
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
pip install --quiet streamlit==1.64.0 pandas==3.0.2

if [ ! -f "outputs/results.json" ]; then
    echo "outputs/results.json not found. Run ./run_full_pipeline.sh first (needs the dataset)."
    exit 1
fi
echo "Launching dashboard - your browser should open. If not: http://localhost:8501"
streamlit run dashboard/app.py
