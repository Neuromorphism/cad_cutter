#!/usr/bin/env bash
# ---------------------------------------------------------------
# setup_venv.sh — Create a Python venv for the CAD Assembly Pipeline
#
# Usage:
#   chmod +x setup_venv.sh
#   ./setup_venv.sh            # creates .venv in the current dir
#   ./setup_venv.sh /path/env  # creates venv at the given path
#
# Requires: Python 3.10+ (cadquery/OCP wheels need >=3.10)
# ---------------------------------------------------------------
set -euo pipefail

VENV_DIR="${1:-.venv}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REQ_FILE="$SCRIPT_DIR/requirements.txt"

# --- Check Python version ---------------------------------------------------
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: $PYTHON not found. Install Python 3.10+ first."
    exit 1
fi

PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "ERROR: Python >= 3.10 required (found $PY_VER)"
    exit 1
fi

echo "Using $PYTHON ($PY_VER)"

# --- Create venv -------------------------------------------------------------
echo "Creating virtual environment at: $VENV_DIR"
"$PYTHON" -m venv "$VENV_DIR"

# --- Activate and install ----------------------------------------------------
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip --quiet

echo "Installing dependencies from requirements.txt..."
pip install -r "$REQ_FILE"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Activate with:  source $VENV_DIR/bin/activate"
echo "============================================"
echo ""
echo "Quick start:"
echo "  python assemble.py outer_1.step inner_1.stl -o assembly.step --render assembly.png"
echo ""
echo "Run tests:"
echo "  xvfb-run -a python -m pytest test_assemble.py -v"
echo "  (on a headed system, drop xvfb-run)"
