# PaperNarrator

Convert scientific papers to narrated EP3 audiobooks with AI-powered text cleaning and visual descriptions.

## Features

- **EP3 Output**: Generate audiobooks with embedded chapter markers and metadata (EP3 format with MP3 audio)
- **Multiple Input Sources**: Accept papers via URL, PDF upload, or raw text
- **LLM-Powered Text Cleaning**: Automatically clean and structure academic text for natural narration
- **VLM Figure Descriptions**: Vision Language Models describe figures and tables for accessibility
- **Multi-Provider LLM Support**: OpenAI, Google Gemini, Anthropic Claude, or local Ollama
- **VibeVoice TTS**: High-quality neural text-to-speech with natural prosody

## Quick Start

**Requirements**: Python 3.11+ (3.11 recommended for best package compatibility), 8GB+ RAM, NVIDIA GPU recommended for TTS

### Option 1: Docker (Recommended)

```bash
# CPU mode (slow TTS)
docker-compose --profile cpu up

# GPU mode (fast TTS - requires NVIDIA)
docker-compose --profile gpu up

# With Ollama (local LLM)
docker-compose --profile ollama --profile cpu up
```

### Option 2: Local Setup with UV

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run setup script (installs deps, downloads VibeVoice model, installs ffmpeg)
bash setup.sh  # Linux/Mac
setup.bat      # Windows

# Or manually:
uv venv --python 3.11 .venv  # Use Python 3.11 for best compatibility
source .venv/bin/activate    # Windows: .venv\Scripts\activate
uv pip install -e .

# Copy and configure environment
cp .env.example .env
# Edit .env and add your API keys

# Run the application
uv run python app.py
```

## Configuration

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider (openai, gemini, anthropic, ollama) | `openai` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `GEMINI_API_KEY` | Google Gemini API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `OPENAI_MODEL` | OpenAI model name | `gpt-4o-mini` |
| `GEMINI_MODEL` | Gemini model name | `gemini-1.5-pro` |
| `ANTHROPIC_MODEL` | Claude model name | `claude-3-5-sonnet-20241022` |
| `OLLAMA_MODEL` | Ollama model name | `llama3.2` |
| `VIBEVOICE_DEVICE` | TTS device (cuda:0, cpu) | `cuda:0` |
| `OUTPUT_FORMAT` | Output format (ep3, mp3) | `ep3` |
| `MP3_BITRATE` | Audio bitrate in kbps | `128` |
| `VLM_ENABLED` | Enable figure descriptions | `false` |

## Setup VibeVoice Models

Before running, download the VibeVoice-1.5B model and voice samples:

### Linux/Mac

```bash
./setup_vibevoice.sh
```

### Windows

```bash
setup_vibevoice.bat
```

This downloads:
- **Model**: `microsoft/VibeVoice-1.5B` (~2GB) to `./models/microsoft/VibeVoice-1.5B/`
- **Voice samples**: Carter, Wayne, Avery Carter (~10MB each)
- **Test data**: Sample text file

### Manual Download

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download('microsoft/VibeVoice-1.5B', local_dir='./models/microsoft/VibeVoice-1.5B', resume_download=True)
"
```

## Quick Test

```bash
python -c "
from tts.vibevoice import VibeVoiceTTS

tts = VibeVoiceTTS(
    model_name='./models/microsoft/VibeVoice-1.5B',
    speaker_name='Carter'
)

tts.generate_audio(
    text='Hello world. This is a test.',
    output_path='output.wav',
    word_count=8
)
print('Saved to output.wav')
"
```

## Usage

1. Launch the Gradio interface
2. Input your paper via URL, upload PDF, or paste text
3. Select LLM provider and configure options
4. Click "Generate" to create your audiobook
5. Download the EP3 file with embedded chapters

## Architecture

- **Gradio**: Web interface for user interaction
- **LangGraph**: Workflow orchestration for multi-step processing
- **Transformers/Torch**: ML model inference
- **PyMuPDF**: PDF parsing and text extraction
- **PyDub**: Audio processing and concatenation
- **EbookLib**: EP3 audiobook generation

## License

MIT
