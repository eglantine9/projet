#!/usr/bin/env bash
# ============================================================================
#  ProfileX Release Builder
#  Creates a portable ZIP archive that runs on both macOS and Windows.
#
#  Usage:  bash build_release.sh
#  Output: ./release/ProfileX-v1.0.zip
# ============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[  OK]${NC}  $*"; }
fail()    { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VERSION="v1.0"
RELEASE_NAME="ProfileX-${VERSION}"
STAGING_DIR="/tmp/${RELEASE_NAME}"
OUTPUT_DIR="$SCRIPT_DIR/release"
OUTPUT_ZIP="$OUTPUT_DIR/${RELEASE_NAME}.zip"

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║       ProfileX Release Builder ${VERSION}             ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Verify critical files exist ─────────────────────────────────────
info "Verifying source files..."

REQUIRED_FILES=(
    "programme/main.py"
    "programme/requirements.txt"
    "programme/src/core/logic.py"
    "programme/src/core/genetic_algorithm.py"
    "programme/src/core/CVAE_architecture.py"
    "programme/src/ui/app.py"
    "programme/src/assets/vae_celeba_final.pth"
    "data/attribute_directions.npz"
    "data/attribute_vectors.npz"
    "run.sh"
    "run.bat"
    "ProfileX.command"
)

for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        fail "Missing required file: $f"
    fi
done

if [ ! -d "data/celeba/img_align_celeba" ]; then
    fail "Missing directory: data/celeba/img_align_celeba"
fi
success "All required files present"

# ── Step 2: Clean staging area ──────────────────────────────────────────────
info "Preparing staging area..."
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"
success "Staging area ready"

# ── Step 3: Copy source code ───────────────────────────────────────────────
info "Copying source code..."

# Programme directory (exclude venv and __pycache__)
rsync -a \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    programme/ "$STAGING_DIR/programme/"

success "Source code copied"

# ── Step 4: Copy data files ────────────────────────────────────────────────
info "Copying data files..."

rsync -a \
    --exclude='.DS_Store' \
    data/ "$STAGING_DIR/data/"

success "Data files copied"

# ── Step 5: Copy launchers and docs ─────────────────────────────────────────
info "Copying launchers and documentation..."

cp run.sh           "$STAGING_DIR/"
cp run.bat          "$STAGING_DIR/"
cp ProfileX.command "$STAGING_DIR/"
cp README.md        "$STAGING_DIR/"
cp LICENSE          "$STAGING_DIR/"

# Make shell scripts executable (they keep permissions inside ZIP on macOS)
chmod +x "$STAGING_DIR/run.sh"
chmod +x "$STAGING_DIR/ProfileX.command"

success "Launchers and docs copied"

# ── Step 6: Build the ZIP archive ──────────────────────────────────────────
info "Building release archive..."
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_ZIP"

cd /tmp
zip -r -9 "$OUTPUT_ZIP" "$RELEASE_NAME" \
    -x "*.DS_Store" \
    -x "*__pycache__*" \
    -x "*.pyc" \
    > /dev/null

cd "$SCRIPT_DIR"
success "Archive created"

# ── Step 7: Cleanup staging ────────────────────────────────────────────────
rm -rf "$STAGING_DIR"

# ── Step 8: Summary ───────────────────────────────────────────────────────
ARCHIVE_SIZE=$(du -sh "$OUTPUT_ZIP" | cut -f1)

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║  Release built successfully!                     ║${NC}"
echo -e "${BOLD}╠══════════════════════════════════════════════════╣${NC}"
echo -e "${BOLD}║${NC}  Archive : ${CYAN}${OUTPUT_ZIP}${NC}"
echo -e "${BOLD}║${NC}  Size    : ${CYAN}${ARCHIVE_SIZE}${NC}"
echo -e "${BOLD}╠══════════════════════════════════════════════════╣${NC}"
echo -e "${BOLD}║${NC}  ${GREEN}Windows${NC} → Extract ZIP → double-click ${BOLD}run.bat${NC}"
echo -e "${BOLD}║${NC}  ${GREEN}macOS${NC}   → Extract ZIP → double-click ${BOLD}ProfileX.command${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════╝${NC}"
echo ""
