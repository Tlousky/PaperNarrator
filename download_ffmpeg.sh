#!/bin/bash
# Download and setup ffmpeg for Linux/Mac

set -e

echo "=== FFmpeg Setup Script ==="

# Detect OS
OS=$(uname -s)
ARCH=$(uname -m)

echo "Detected OS: $OS"
echo "Detected Architecture: $ARCH"

# Determine download URL based on OS and architecture
if [ "$OS" = "Linux" ]; then
    if [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "amd64" ]; then
        FILENAME="ffmpeg-master-latest-linux64-gpl.tar.xz"
        URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/${FILENAME}"
    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        FILENAME="ffmpeg-master-latest-linuxarm64-gpl.tar.xz"
        URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/${FILENAME}"
    else
        echo "Unsupported architecture: $ARCH"
        exit 1
    fi
elif [ "$OS" = "Darwin" ]; then
    if [ "$ARCH" = "x86_64" ]; then
        FILENAME="ffmpeg-master-latest-macos64-gpl.tar.xz"
        URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/${FILENAME}"
    elif [ "$ARCH" = "arm64" ]; then
        FILENAME="ffmpeg-master-latest-macosarm64-gpl.tar.xz"
        URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/${FILENAME}"
    else
        echo "Unsupported architecture: $ARCH"
        exit 1
    fi
else
    echo "Unsupported OS: $OS"
    exit 1
fi

# Create ffmpeg directory
mkdir -p .ffmpeg

# Download
echo "Downloading ffmpeg..."
curl -L -o "${FILENAME}" "${URL}"

# Extract
echo "Extracting..."
tar -xf "${FILENAME}" --strip-components=1 -C .ffmpeg

# Clean up
echo "Cleaning up..."
rm -f "${FILENAME}"

# Add to PATH (temporary - for current session)
export PATH="$(pwd)/.ffmpeg/bin:${PATH}"

# Verify
if command -v ffmpeg &> /dev/null; then
    echo "FFmpeg installed successfully:"
    ffmpeg -version | head -n 2
    echo ""
    echo "To permanently add to PATH, add this to your ~/.bashrc or ~/.zshrc:"
    echo "export PATH=\"$(pwd)/.ffmpeg/bin:\${PATH\""
else
    echo "Error: Could not install ffmpeg"
    exit 1
fi

echo "FFmpeg setup complete!"
