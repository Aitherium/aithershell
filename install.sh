#!/usr/bin/env bash
# AitherShell Installation Script for macOS/Linux
# Usage: curl -fsSL https://install.aitherium.com/aithershell | bash

set -e

REPO="Aitherium/aithershell"
VERSION="1.0.0"
INSTALL_DIR="${HOME}/.local/bin"

# Detect OS
OS=$(uname -s)
case $OS in
    Darwin)
        if [[ $(uname -m) == "arm64" ]]; then
            BINARY="aither-macos-arm64"
            OS_NAME="macOS (Apple Silicon)"
        else
            BINARY="aither-macos-x64"
            OS_NAME="macOS (Intel)"
        fi
        ;;
    Linux)
        BINARY="aither-linux-x64"
        OS_NAME="Linux x86_64"
        ;;
    *)
        echo "❌ Unsupported OS: $OS"
        exit 1
        ;;
esac

echo "🔷 AitherShell Installation Script"
echo "   OS: $OS_NAME"
echo "   Version: $VERSION"
echo ""

# Create install directory
mkdir -p "$INSTALL_DIR"

# Download binary
DOWNLOAD_URL="https://github.com/${REPO}/releases/download/v${VERSION}/${BINARY}"
BINARY_PATH="${INSTALL_DIR}/aither"

echo "📥 Downloading AitherShell..."
curl -fsSL "$DOWNLOAD_URL" -o "$BINARY_PATH"

# Make executable
chmod +x "$BINARY_PATH"

echo "✅ Installation complete!"
echo ""
echo "📖 Quick start:"
echo "   1. Set your license key:"
echo "      export AITHERIUM_LICENSE_KEY=\"your-key\""
echo ""
echo "   2. Run AitherShell:"
echo "      aither --help"
echo "      aither prompt \"Hello, AitherOS!\""
echo "      aither shell"
echo ""
echo "🔑 Get your free license key at: https://aitherium.com/free"
echo ""

# Check if path is in $PATH
if [[ ":$PATH:" == *":${INSTALL_DIR}:"* ]]; then
    echo "✅ ${INSTALL_DIR} is in your PATH"
else
    echo "⚠️  Add to your PATH:"
    echo "   export PATH=\"${INSTALL_DIR}:\$PATH\""
fi
