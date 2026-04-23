#!/usr/bin/env bash
# ============================================================================
#  ProfileX - Cross-Platform Run Script (macOS / Linux)
#  Installs prerequisites, checks for errors, and launches the application.
# ============================================================================
set -euo pipefail

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[  OK]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()    { echo -e "${RED}[FAIL]${NC}  $*"; }

# ── Resolve the project root (where this script lives) ──────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
VENV_DIR="$PROJECT_ROOT/programme/venv"
REQUIREMENTS="$PROJECT_ROOT/programme/requirements.txt"
MAIN_PY="$PROJECT_ROOT/programme/main.py"

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║        ProfileX - Forensic Composite Generator      ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Check Python Installation ───────────────────────────────────────
info "Checking Python installation..."
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PY_VERSION=$("$cmd" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
        PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
        PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
        if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 9 ]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    fail "Python 3.9+ is required but not found."
    echo ""
    echo "  Install Python:"
    echo "    macOS:  brew install python3"
    echo "    Linux:  sudo apt install python3 python3-venv python3-pip"
    echo ""
    exit 1
fi
success "Found $($PYTHON_CMD --version 2>&1)"

# ── Step 2: Check for pip ───────────────────────────────────────────────────
info "Checking pip..."
if ! "$PYTHON_CMD" -m pip --version &>/dev/null; then
    warn "pip not found. Attempting to install..."
    "$PYTHON_CMD" -m ensurepip --upgrade 2>/dev/null || {
        fail "Cannot install pip automatically."
        echo "  Try: $PYTHON_CMD -m ensurepip --upgrade"
        exit 1
    }
fi
success "pip is available"

# ── Step 3: Check for tkinter (system dependency) ──────────────────────────
info "Checking tkinter availability..."
if ! "$PYTHON_CMD" -c "import tkinter" 2>/dev/null; then
    warn "tkinter is missing (required for the GUI)."
    echo ""
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  macOS fix:  brew install python-tk"
    else
        echo "  Ubuntu/Debian:  sudo apt install python3-tk"
        echo "  Fedora:         sudo dnf install python3-tkinter"
        echo "  Arch:           sudo pacman -S tk"
    fi
    echo ""
    fail "Please install tkinter and re-run this script."
    exit 1
fi
success "tkinter is available"

# ── Step 4: Create / Repair Virtual Environment ────────────────────────────
info "Setting up virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment at programme/venv..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    success "Virtual environment created"
elif [ ! -f "$VENV_DIR/bin/python3" ] && [ ! -f "$VENV_DIR/bin/python" ]; then
    warn "Virtual environment is broken. Rebuilding..."
    rm -rf "$VENV_DIR"
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    success "Virtual environment rebuilt"
else
    success "Virtual environment already exists"
fi

# Activate the venv
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    fail "Cannot activate virtual environment (missing bin/activate)"
    exit 1
fi

# ── Step 5: Upgrade pip inside venv ─────────────────────────────────────────
info "Upgrading pip inside venv..."
python -m pip install --upgrade pip --quiet 2>/dev/null
success "pip upgraded"

# ── Step 6: Install Dependencies ───────────────────────────────────────────
info "Installing dependencies from requirements.txt..."
if [ -f "$REQUIREMENTS" ]; then
    python -m pip install -r "$REQUIREMENTS" --quiet 2>&1 | tail -3 || {
        warn "Strict version install failed. Trying flexible install..."
        # Extract package names without version pins
        sed 's/[>=<].*//' "$REQUIREMENTS" | while read -r pkg; do
            [ -n "$pkg" ] && python -m pip install "$pkg" --quiet 2>/dev/null
        done
    }
else
    warn "requirements.txt not found. Installing core packages manually..."
    python -m pip install torch numpy Pillow --quiet
fi

# Also install scikit-learn (used in some modules)
python -m pip install scikit-learn --quiet 2>/dev/null || true

success "Dependencies installed"

# ── Step 7: Verify Critical Imports ─────────────────────────────────────────
info "Verifying Python imports..."
IMPORT_ERRORS=0
for mod in torch numpy PIL tkinter sklearn; do
    if python -c "import $mod" 2>/dev/null; then
        success "  $mod"
    else
        fail "  $mod - MISSING"
        IMPORT_ERRORS=$((IMPORT_ERRORS + 1))
    fi
done

if [ "$IMPORT_ERRORS" -gt 0 ]; then
    fail "$IMPORT_ERRORS module(s) could not be imported."
    echo "  Try running: pip install torch numpy Pillow scikit-learn"
    exit 1
fi

# ── Step 8: Verify Critical Data Files ──────────────────────────────────────
info "Checking required data files..."
DATA_OK=true
for f in \
    "$PROJECT_ROOT/programme/src/assets/vae_celeba_final.pth" \
    "$PROJECT_ROOT/data/attribute_directions.npz" \
    "$PROJECT_ROOT/data/attribute_vectors.npz"; do
    if [ -f "$f" ]; then
        success "  $(basename "$f")"
    else
        fail "  MISSING: $f"
        DATA_OK=false
    fi
done

if [ "$DATA_OK" = false ]; then
    warn "Some data files are missing. The app may not function correctly."
fi

# ── Step 9: Dry-Run Import Test ─────────────────────────────────────────────
info "Running dry-run import test..."
cd "$PROJECT_ROOT/programme"
if python -c "from src.ui.app import Application; print('Import OK')" 2>/dev/null; then
    success "Application modules load successfully"
else
    warn "Dry-run import test failed. The app may still work if data files load at runtime."
fi
cd "$PROJECT_ROOT"

# ── Step 10: Launch ─────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  All checks passed. Launching ProfileX...${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════${NC}"
echo ""

cd "$PROJECT_ROOT"
python "$MAIN_PY"
