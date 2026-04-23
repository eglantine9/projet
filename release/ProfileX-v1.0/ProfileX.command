#!/usr/bin/env bash
# ============================================================================
#  ProfileX - Double-Click Launcher for macOS
#  This .command file opens in Terminal when double-clicked in Finder.
# ============================================================================

# Navigate to the directory where this script lives
cd "$(dirname "$0")"

# Hand off to the main run script
exec bash ./run.sh
