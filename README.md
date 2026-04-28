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

### Option 1: Docker (Recommended)

```bash
# Clone and build
git clone https://github.com/yourusername/PaperNarrator.git
cd PaperNarrator
docker build -t paper-narrator .

# Run with environment variables
docker run -p 7860:7860 -e OPENAI_API_KEY=your-key-here paper-narrator
```

### Option 2: Local Setup with UV

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .

# Copy and configure environment
cp .env.example .env
# Edit .env and add your API keys

# Run the application
python -m app
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
