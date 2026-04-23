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
MAC_PKG="dist/Installers/4BiM_macOS.pkg"
WIN_EXE="dist/Installers/4BiM_Setup_Windows.exe"

ARTIFACTS=""
[ -f "$MAC_PKG" ] && ARTIFACTS="$ARTIFACTS $MAC_PKG" && echo "Found macOS PKG."
[ -f "$WIN_EXE" ] && ARTIFACTS="$ARTIFACTS $WIN_EXE" && echo "Found Windows EXE."

echo -e "${CYAN}Creating GitHub Release ${VERSION}...${NC}"

# Ensure tag exists locally or gh release create will create it automatically
gh release create "$VERSION" $ARTIFACTS \
    --title "4BiM Forensic Composite Generator - $VERSION" \
    --notes "Thank you for using 4BiM.
    
### Fast Extracting Installers!
These installers are highly optimized. They simply extract the main modules to your computer and set everything up natively on first launch.

### Installation
- **macOS Users:** Download \`4BiM_macOS.pkg\` and run the package installer. It will place 4BiM in your Applications.
- **Windows Users:** Download \`4BiM_Setup_Windows.exe\` and run the wizard. It will place a shortcut directly on your desktop.

### Credits
Created internally by Nathanael Pons, Axel Mondesir, Ambre Lesurtel, Skander Zouhir, Eglantine Salomé (Licensed under MIT).
"

echo -e "${GREEN}Release ${VERSION} created and pushed to GitHub successfully!${NC}"
