#!/usr/bin/env bash
# ============================================================================
#  4BiM - Rapid Source-Based Installer Builder
#  Generates macOS (.pkg) and Windows (.exe) instantly without PyInstaller.
# ============================================================================
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}Creating fast extracting installers for 4BiM...${NC}"

mkdir -p dist/Installers

# ---------------------------------------------------------
# 1. macOS Installer (.pkg)
# ---------------------------------------------------------
echo "Building macOS Installer (.pkg)..."

PAYLOAD_DIR="dist/mac_payload"
APP_DIR="${PAYLOAD_DIR}/Applications/4BiM"

mkdir -p "$APP_DIR"
cp -r programme data run.sh LICENSE README.md "$APP_DIR"
chmod +x "$APP_DIR/run.sh"

# Create a simple AppleScript wrapper so it sits in Applications nicely
osacompile -o "$APP_DIR/4BiM.app" -e 'tell application "Terminal"
    do script (POSIX path of (path to me)) & "/Contents/Resources/run.sh"
    activate
end tell'

# Move the actual run.sh inside the .app bundle so the user can just click it
cp "$APP_DIR/run.sh" "$APP_DIR/4BiM.app/Contents/Resources/run.sh"
# Copy an icon if it exists
if [ -f "programme/src/assets/icon.icns" ]; then
    cp "programme/src/assets/icon.icns" "$APP_DIR/4BiM.app/Contents/Resources/applet.icns"
fi

rm "$APP_DIR/run.sh"

pkgbuild --root "$PAYLOAD_DIR" \
         --identifier "com.4bim.app" \
         --version "1.0.0" \
         --install-location "/" \
         "dist/Installers/4BiM_macOS.pkg"

echo -e "${GREEN}-> Created 4BiM_macOS.pkg successfully!${NC}"

# ---------------------------------------------------------
# 2. Windows Installer (.exe)
# ---------------------------------------------------------
echo "Building Windows Installer (.exe) using NSIS..."

if ! command -v makensis &> /dev/null; then
    echo "NSIS compiler not found. Installing via Homebrew..."
    if command -v brew &> /dev/null; then
        brew install nsis
    else
        echo -e "${RED}Homebrew not found. Unable to build Windows .exe. Skip.${NC}"
        exit 1
    fi
fi

makensis installer_windows.nsi

echo -e "${GREEN}-> Created 4BiM_Setup_Windows.exe successfully!${NC}"
echo -e "${CYAN}Done. Installers are located in dist/Installers/.${NC}"
