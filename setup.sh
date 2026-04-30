#!/bin/bash
# Setup script for PaperNarrator
# Downloads and configures all dependencies: VibeVoice model and ffmpeg

set -e

echo "=== PaperNarrator Setup ==="
echo ""

# Check if we're in the project directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the PaperNarrator project root directory"
    exit 1
fi

# Step 1: Download and setup VibeVoice model
echo "[Step 1/2] Setting up VibeVoice TTS model..."
echo ""
if [ -f "setup_vibevoice.sh" ]; then
    bash setup_vibevoice.sh
else
    echo "Warning: setup_vibevoice.sh not found. Skipping VibeVoice setup."
fi

echo ""
echo ""

# Step 2: Download and setup ffmpeg
echo "[Step 2/2] Setting up ffmpeg..."
echo ""
if [ -f "download_ffmpeg.sh" ]; then
    bash download_ffmpeg.sh
else
    echo "Warning: download_ffmpeg.sh not found. Skipping ffmpeg setup."
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Available commands:"
echo "  uv run python app.py              # Start Gradio server"
echo "  uv run pytest                     # Run tests"
echo ""
echo "Note: If using MP3 output, ensure ffmpeg is in your PATH:"
echo "  Linux:  export PATH=\"$(pwd)/.ffmpeg/bin:\${PATH\""
echo "  Mac:    export PATH=\"$(pwd)/.ffmpeg/bin:\${PATH\""
echo ""
