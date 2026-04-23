#!/usr/bin/env bash
# ============================================================================
#  4BiM - GitHub Automated Release script
#  Requires: GitHub CLI (gh)
# ============================================================================
set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

if ! command -v gh &> /dev/null; then
    echo -e "${RED}Error: GitHub CLI (gh) is not installed.${NC}"
    exit 1
fi

VERSION="v1.0.0"

# Check for artifacts
MAC_PKG="dist/Installers/ProfileX_macOS.pkg"
WIN_EXE="dist/Installers/ProfileX_Setup_Windows.exe"

ARTIFACTS=""
[ -f "$MAC_PKG" ] && ARTIFACTS="$ARTIFACTS $MAC_PKG" && echo "Found macOS PKG."
[ -f "$WIN_EXE" ] && ARTIFACTS="$ARTIFACTS $WIN_EXE" && echo "Found Windows EXE."

echo -e "${CYAN}Creating GitHub Release ${VERSION}...${NC}"

# Ensure tag exists locally or gh release create will create it automatically
gh release create "$VERSION" $ARTIFACTS \
    --title "ProfileX ($VERSION)" \
    --notes "Welcome to ProfileX!
    
### Fast Extracting Installers
These installers are highly optimized. They deploy the core modules and bootstrap everything dynamically on first launch.

### Installation
- **macOS Users:** Download \`ProfileX_macOS.pkg\` and run the package installer. It will place ProfileX in your Applications.
- **Windows Users:** Download \`ProfileX_Setup_Windows.exe\` and run the wizard. It will place a shortcut directly on your desktop.

### Credits
Created internally by Nathanael Pons, Axel Mondesir, Ambre Lesurtel, Skander Zouhir, Eglantine Salomé (Licensed under MIT).
"

echo -e "${GREEN}Release ${VERSION} created and pushed to GitHub successfully!${NC}"
