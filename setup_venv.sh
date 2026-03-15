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
TOPOPT_REQ_FILE="$SCRIPT_DIR/requirements-topopt.txt"
TOPOPT_DL4TO_REQ_FILE="$SCRIPT_DIR/requirements-topopt-dl4to.txt"
TOPOPT_PYMOTO_REQ_FILE="$SCRIPT_DIR/requirements-topopt-pymoto.txt"
INSTALL_TOPOPT="${INSTALL_TOPOPT:-0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
TOPOPT_SOLVERS="${TOPOPT_SOLVERS:-both}"

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

# --- Ensure OpenGL runtime for cadquery/OCP ---------------------------------
if [ ! -e /usr/lib/x86_64-linux-gnu/libGL.so.1 ] && [ ! -e /lib/x86_64-linux-gnu/libGL.so.1 ]; then
    if command -v apt-get &>/dev/null && [ "$(id -u)" -eq 0 ]; then
        echo "Installing system dependency: libgl1 (provides libGL.so.1)..."
        apt-get update
        apt-get install -y libgl1
    else
        echo "WARNING: libGL.so.1 not found. Install system package 'libgl1' before running tests."
    fi
fi

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

if [ "$INSTALL_TOPOPT" = "1" ]; then
    echo "Installing optional topology-optimization stack..."
    case "$TOPOPT_SOLVERS" in
        both)
            if [ -n "$TORCH_INDEX_URL" ]; then
                echo "Installing torch/torchvision from custom index: $TORCH_INDEX_URL"
                pip install torch torchvision --index-url "$TORCH_INDEX_URL"
            else
                echo "Installing torch/torchvision from default index"
                pip install torch torchvision
            fi
            pip install -r "$TOPOPT_REQ_FILE"
            ;;
        dl4to)
            if [ -n "$TORCH_INDEX_URL" ]; then
                echo "Installing torch/torchvision from custom index: $TORCH_INDEX_URL"
                pip install torch torchvision --index-url "$TORCH_INDEX_URL"
            else
                echo "Installing torch/torchvision from default index"
                pip install torch torchvision
            fi
            pip install -r "$TOPOPT_DL4TO_REQ_FILE"
            ;;
        pymoto)
            pip install -r "$TOPOPT_PYMOTO_REQ_FILE"
            ;;
        *)
            echo "ERROR: TOPOPT_SOLVERS must be one of: both, dl4to, pymoto"
            exit 1
            ;;
    esac
fi

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Activate with:  source $VENV_DIR/bin/activate"
echo "============================================"
echo ""
echo "Quick start:"
echo "  python assemble.py outer_1.step inner_1.stl -o assembly.step --render assembly.png"
echo ""
echo "Optional native topology optimization:"
echo "  INSTALL_TOPOPT=1 TOPOPT_SOLVERS=dl4to TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 ./setup_venv.sh"
echo "  INSTALL_TOPOPT=1 TOPOPT_SOLVERS=pymoto ./setup_venv.sh"
echo ""
echo "Run tests:"
echo "  xvfb-run -a python -m pytest test_assemble.py -v"
echo "  (on a headed system, drop xvfb-run)"
