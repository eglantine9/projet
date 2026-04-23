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
    echo "Install with: brew install gh"
    exit 1
fi

if ! gh auth status &> /dev/null; then
    echo -e "${RED}Error: Not authenticated with GitHub CLI.${NC}"
    echo "Please run: gh auth login"
    exit 1
fi

VERSION="v1.0.0"
NOTES="Release notes for version ${VERSION}"

# Check for artifacts
MAC_DMG="dist/Installers/4BiM_macOS.dmg"
WIN_EXE="dist/Installers/4BiM_Setup_Windows.exe"

ARTIFACTS=""
if [ -f "$MAC_DMG" ]; then
    ARTIFACTS="$ARTIFACTS $MAC_DMG"
    echo "Found macOS DMG."
fi

if [ -f "$WIN_EXE" ]; then
    ARTIFACTS="$ARTIFACTS $WIN_EXE"
    echo "Found Windows Setup EXE."
fi

if [ -z "$ARTIFACTS" ]; then
    echo -e "${RED}Warning: No installers found in dist/Installers/.${NC}"
    read -p "Do you want to create a release without artifacts? (y/n): " choice
    if [[ ! "$choice" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${CYAN}Creating GitHub Release ${VERSION}...${NC}"

# Ensure tag exists locally or gh release create will create it automatically
gh release create "$VERSION" $ARTIFACTS \
    --title "4BiM Forensic Composite Generator - $VERSION" \
    --notes "Thank you for using 4BiM.
    
### Core Features
- Real-time latent face fusion (10 visual generation loops).
- Norm-Preserving Hypersphere projection technology.
- Custom alignment GUI (CropDialog / 51% Y threshold).
- Advanced dynamic attribute escalation limits.

### Installation
- **macOS Users:** Download \`4BiM_macOS.dmg\` and drag it to Applications.
- **Windows Users:** Download \`4BiM_Setup_Windows.exe\` and run the wizard.

### Credits
Created internally by Nathanael Pons, Axel Mondesir, Ambre Lesurtel, Skander Zouhir, Eglantine Salomé (Licensed under MIT).
"

echo -e "${GREEN}Release ${VERSION} created and pushed to GitHub successfully!${NC}"
