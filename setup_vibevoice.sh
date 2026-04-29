#!/bin/bash
set -e

echo "=== VibeVoice-1.5B Setup Script ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MODEL_NAME="microsoft/VibeVoice-1.5B"
VOICE_NAMES="Carter Wayne Avery Carter"
PYTHON_VENV="${PYTHON_VENV:-.venv}"

echo "${YELLOW}Checking Python environment...${NC}"
if [ ! -d "$PYTHON_VENV" ]; then
    echo "${RED}Error: Python virtual environment '$PYTHON_VENV' not found.${NC}"
    echo "${YELLOW}Creating one now...${NC}"
    python3 -m venv "$PYTHON_VENV"
fi

# Activate venv
if [ -f "$PYTHON_VENV/bin/activate" ]; then
    source "$PYTHON_VENV/bin/activate"
else
    echo "${RED}Error: Could not activate virtual environment.${NC}"
    exit 1
fi

echo "${GREEN}Using Python: $(which python)${NC}"
echo "${GREEN}Python version: $(python --version)${NC}"

# Install dependencies
echo ""
echo "${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip
pip install huggingface_hub soundfile pydub transformers torch

# Check for ffmpeg (required by pydub/soundfile)
if ! command -v ffmpeg &> /dev/null; then
    echo "${YELLOW}Warning: ffmpeg not found. Installing...${NC}"
    if [ "$(uname)" == "Darwin" ]; then
        brew install ffmpeg
    elif [ "$(uname)" == "Linux" ]; then
        if [ -f /etc/debian_version ]; then
            apt-get update && apt-get install -y ffmpeg
        elif [ -f /etc/redhat-release ]; then
            dnf install -y ffmpeg
        else
            echo "${RED}Could not install ffmpeg automatically. Please install manually.${NC}"
            exit 1
        fi
    fi
fi

# Download model
echo ""
echo "${YELLOW}Downloading VibeVoice-1.5B model (${MODEL_NAME})...${NC}"
echo "${YELLOW}This may take 5-10 minutes depending on your connection.${NC}"

# Download using huggingface-cli
huggingface-cli download \
    "$MODEL_NAME" \
    --local-dir "./models/$MODEL_NAME" \
    --local-dir-use-symlinks false \
    --resume-download

if [ -d "./models/$MODEL_NAME" ]; then
    echo "${GREEN}Model downloaded successfully to ./models/$MODEL_NAME${NC}"
else
    echo "${RED}Failed to download model.${NC}"
    exit 1
fi

# Download voice samples
echo ""
echo "${YELLOW}Downloading voice samples...${NC}"
for VOICE in $VOICE_NAMES; do
    echo "Downloading $VOICE..."
    huggingface-cli download \
        "$MODEL_NAME" \
        "voices/$VOICE.pt" \
        --local-dir "./models/$MODEL_NAME/voices" \
        --local-dir-use-symlinks false \
        --resume-download
done

echo "${GREEN}Voice samples downloaded to ./models/$MODEL_NAME/voices${NC}"

# Create test text file
echo ""
echo "${YELLOW}Creating test text file...${NC}"
mkdir -p test_data
cat > test_data/sample.txt << 'EOF'
Hello, this is a test of the VibeVoice text-to-speech system. 
This sample text is short enough to generate quickly but long enough to demonstrate the quality.
Thank you for listening.
EOF

echo "${GREEN}Created test_data/sample.txt${NC}"

# Print summary
echo ""
echo "=== Setup Complete ==="
echo "Model location: ./models/$MODEL_NAME"
echo "Voice samples: ./models/$MODEL_NAME/voices/"
echo "Test file: test_data/sample.txt"
echo ""
echo "Available voices: $VOICE_NAMES"
echo ""
echo "To test:"
echo "  python -c 'from tts.vibevoice import VibeVoiceTTS; tts = VibeVoiceTTS(model_name=\"./models/microsoft/VibeVoice-1.5B\", speaker_name=\"Carter\"); tts.generate_audio(\"Hello world\", \"test.wav\")'"

echo "${GREEN}Setup successful!${NC}"
