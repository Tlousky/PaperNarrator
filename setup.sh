#!/bin/bash
# Setup script for PaperNarrator
# Downloads and configures all dependencies: VibeVoice model, ffmpeg, and Python packages

set -e

echo "=== PaperNarrator Setup ==="
echo ""

# Check if we're in the project directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the PaperNarrator project root directory"
    exit 1
fi

# Step 1: Install Python dependencies
echo "[Step 1/3] Installing Python dependencies..."
echo ""
if command -v uv &> /dev/null; then
    # Create venv with Python 3.11 if available (better package compatibility)
    if command -v python3.11 &> /dev/null; then
        uv venv --python 3.11 .venv 2>/dev/null || uv venv .venv
    else
        uv venv .venv
    fi
    uv pip install -e . --compile-bytecode
    echo "Python dependencies installed successfully."
    echo ""
    echo "[Step 1b/3] Installing CUDA-enabled PyTorch (cu128)..."
    uv pip install --python .venv torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu128
    echo "CUDA PyTorch installed."
else
    echo "Warning: 'uv' not found. Install it with: pip install uv"
    pip install -e .
    echo ""
    echo "Installing CUDA-enabled PyTorch..."
    pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu128
fi

echo ""
echo ""

# Step 2: Download and setup VibeVoice model
echo "[Step 2/3] Setting up VibeVoice TTS model..."
echo ""
if [ -f "setup_vibevoice.sh" ]; then
    bash setup_vibevoice.sh
else
    echo "Warning: setup_vibevoice.sh not found. Skipping VibeVoice setup."
fi

echo ""
echo ""

# Step 3: Download and setup ffmpeg
echo "[Step 3/3] Setting up ffmpeg..."
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
echo "  Windows: export PATH=\"%CD%\\.ffmpeg\\bin;%PATH%\""
echo ""
