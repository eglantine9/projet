#!/usr/bin/env bash
# ============================================================================
#  ProfileX - Rapid Source-Based Installer Builder
#  Generates macOS (.pkg) and Windows (.exe) instantly without PyInstaller.
# ============================================================================
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}Creating fast extracting installers for ProfileX...${NC}"

mkdir -p dist/Installers

# ---------------------------------------------------------
# 1. macOS Installer (.pkg)
# ---------------------------------------------------------
echo "Building macOS Installer (.pkg)..."

PAYLOAD_DIR="dist/mac_payload"
mkdir -p "$PAYLOAD_DIR/Applications"
APP_BUNDLE="${PAYLOAD_DIR}/Applications/ProfileX.app"

# 1A. Create the Application Bundle first
osacompile -o "$APP_BUNDLE" -e 'tell application "Terminal"
    do script "\"" & POSIX path of (path to me) & "Contents/Resources/run.sh\""
    activate
end tell'

# 1B. Inject all core code inside the app bundle's resources
APP_DIR="$APP_BUNDLE/Contents/Resources"
rsync -aP --exclude 'venv/' --exclude '__pycache__/' programme "$APP_DIR/"
cp -r data run.sh LICENSE README.md "$APP_DIR/"
chmod +x "$APP_DIR/run.sh"

# Copy an icon if it exists
if [ -f "programme/src/assets/icon.icns" ]; then
    cp "programme/src/assets/icon.icns" "$APP_DIR/applet.icns"
fi

pkgbuild --root "$PAYLOAD_DIR" \
         --identifier "com.profilex.app" \
         --version "1.0.0" \
         --install-location "/" \
         "dist/Installers/ProfileX_macOS.pkg"

echo -e "${GREEN}-> Created ProfileX_macOS.pkg successfully!${NC}"

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

echo -e "${GREEN}-> Created ProfileX_Setup_Windows.exe successfully!${NC}"
echo -e "${CYAN}Done. Installers are located in dist/Installers/.${NC}"
