#!/usr/bin/env bash
# ============================================================================
#  ProfileX - macOS Installer (.dmg) Generation Script
#  Requires: homebrew, create-dmg, pyinstaller
# ============================================================================
set -e

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}Building ProfileX Application for macOS...${NC}"

# Step 1: Ensure PyInstaller is run to build the .app bundle
if [ ! -d "dist/ProfileX.app" ]; then
    echo "Running PyInstaller..."
    programme/venv/bin/python3 -m pip install pyinstaller Pillow numpy torch scikit-learn
    programme/venv/bin/python3 -m PyInstaller ProfileX.spec
else
    echo "dist/ProfileX.app already exists. Skipping PyInstaller build."
fi

# Step 2: Check for create-dmg
if ! command -v create-dmg &> /dev/null; then
    echo "create-dmg not found. Attempting to install via Homebrew..."
    if command -v brew &> /dev/null; then
        brew install create-dmg
    else
        echo "Homebrew is not installed. Please install it to use create-dmg, or package manually."
        exit 1
    fi
fi

# Step 3: Create the DMG
DMG_OUTPUT="dist/Installers/ProfileX_macOS.dmg"
mkdir -p dist/Installers
rm -f "${DMG_OUTPUT}"

echo -e "${CYAN}Generating styled DMG via create-dmg...${NC}"

create-dmg \
  --volname "ProfileX Installer" \
  --volicon "programme/src/assets/icon.icns" \
  --window-pos 200 120 \
  --window-size 800 400 \
  --icon-size 100 \
  --icon "ProfileX.app" 200 190 \
  --hide-extension "ProfileX.app" \
  --app-drop-link 600 190 \
  --eula "LICENSE" \
  "${DMG_OUTPUT}" \
  "dist/ProfileX.app"

echo -e "${GREEN}Successfully generated ${DMG_OUTPUT}!${NC}"
