# PaperNarrator

Convert scientific papers to narrated M4B audiobooks with AI-powered text cleaning and visual descriptions.

## Features

- **M4B Audiobook Output**: Generate audiobooks with embedded chapter markers and metadata (AAC audio in M4B container).
- **Multiple Formats**: Support for M4B, MP3, and WAV outputs.
- **Multiple Input Sources**: Accept papers via URL, PDF upload, or raw text.
- **High-Fidelity PDF Extraction**: Robust block-based text extraction resolves character-spacing distortions.
- **LLM-Powered Text Cleaning**: 
    - **Citation Removal**: Specialized skills to remove academic citations (APA, MLA, IEEE, etc.) while preserving narrative mentions.
    - **Figure & Diagram Debris Cleaning**: Targeted removal of non-narrative technical artifacts (chart coordinates, figure labels).
- **VLM Figure Descriptions**: Vision Language Models describe figures and tables for accessibility.
- **Langfuse Observability**: Full session tracking and monitoring of LLM traces, costs, and quality.
- **Multi-Provider LLM Support**: OpenAI, Google Gemini, Anthropic Claude, or local Ollama.
- **VibeVoice TTS**: High-quality neural text-to-speech with natural prosody.

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
```

## Configuration

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
| `OUTPUT_FORMAT` | Output format (m4b, mp3, wav) | `m4b` |
| `MP3_BITRATE` | Audio bitrate in kbps | `128k` |
| `VLM_ENABLED` | Enable figure descriptions | `false` |
| `LANGFUSE_HOST` | Langfuse server URL | `http://localhost:3000` |

## Setup VibeVoice Models

Before running, download the VibeVoice-Realtime-0.5B model and voice samples:

### Windows

```bash
setup_vibevoice.bat
```

This downloads:
- **Model**: `microsoft/VibeVoice-Realtime-0.5B` to `./models/microsoft/VibeVoice-Realtime-0.5B/`
- **Voice samples**: Carter, Davis, Emma

## Observability

PaperNarrator integrates with **Langfuse** for real-time monitoring of:
- **Sessions**: Track multiple paper generations per user session.
- **Costs**: monitor LLM token usage and estimated cost per run.
- **Traces**: Inspect the multi-step cleaning and chunking logic.

To view traces locally:
1. Start the Langfuse server (see `langfuse/docker-compose.yml`).
2. Access the dashboard at `http://localhost:3000`.

## Architecture

- **LangGraph**: Workflow orchestration for multi-step processing.
- **Transformers/Torch**: ML model inference for TTS.
- **PyMuPDF (fitz)**: Robust block-based PDF text extraction.
- **FFmpeg**: Audio encoding and M4B packaging with metadata.
- **Gradio**: Web interface for user interaction.

## License

MIT
