#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create venv if it doesn't exist yet
if [ ! -f ".venv/bin/activate" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate and install/sync dependencies
source .venv/bin/activate

if [ -f requirements.txt ]; then
    pip install -q -r requirements.txt
fi

exec python main.py "$@"
