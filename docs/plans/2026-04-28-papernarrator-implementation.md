# PaperNarrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task.

**Goal:** Build a Gradio web app that converts scientific papers into EP3 audiobooks using VibeVoice TTS, with autonomous LLM cleaning via LangGraph.

**Architecture:**
- **Frontend:** Gradio (3 input tabs, streaming status chat, file download)
- **Orchestration:** LangGraph state machine (9 states)
- **LLM Layer:** OpenAI (default), Google Gemini, Anthropic, or Local (Ollama)
- **TTS:** Microsoft VibeVoice-1.5B (60-min limit per chunk)
- **Output:** EP3 (EPUB 3 with Media Overlays) with chapter navigation

**Tech Stack:** Python 3.11+, Gradio 4.x, LangGraph, HuggingFace Transformers, OpenAI/Anthropic/Gemini SDKs, ebooklib, PyMuPDF, UV, Docker

---

## File Structure

```
F:/code/PaperNarrator/
├── app.py                      # Gradio frontend
├── config.py                   # Configuration
├── langgraph_pipeline/         # State machine
│   ├── state.py               # Pydantic states
│   ├── tools.py               # LLM tools
│   ├── workflow.py            # Graph definition
│   └── ep3_builder.py         # EP3 generation
├── tts/vibevoice.py           # TTS wrapper
├── llm/                       # Provider implementations
│   ├── base.py
│   ├── openai_provider.py
│   ├── gemini_provider.py
│   ├── anthropic_provider.py
│   └── ollama_provider.py
├── tests/
├── scripts/                    # Install scripts
├── docker/                     # Dockerfile
├── pyproject.toml
└── README.md
```

---

## Phase 1: Foundation (Tasks 1-4)

### Task 1: Project Setup
**Files:** Create `pyproject.toml`, `.env.example`, `README.md`

**Steps:**
1. Create `pyproject.toml` with dependencies: gradio, langgraph, transformers, torch, pymupdf, pydub, ebooklib, openai, google-generativeai, anthropic, ollama, requests
2. Create `.env.example` with LLM_PROVIDER, API keys, VIBEVOICE_DEVICE, OUTPUT_FORMAT=ep3
3. Update `.gitignore` to exclude .env, *.epub, *.mp3, *.wav, *.pyc, __pycache__/
4. Create README.md with Quick Start (Docker and Local sections)

**Commit:** `feat: initial project setup with UV and multi-provider support`

---

### Task 2: Configuration Layer
**Files:** Create `config.py`, `tests/test_config.py`

**Steps:**
1. Write tests for config loading from environment
2. Implement `config.py` with Config class containing:
   - LLM_PROVIDER, OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY
   - Model names for each provider
   - VIBEVOICE_DEVICE, OUTPUT_FORMAT, MP3_BITRATE
   - VLM_ENABLED, VLM_PROVIDER, VLM_MODEL
3. Add validate_config() that raises ValueError if provider selected but API key missing
4. Run tests, verify 2 passed

**Commit:** `feat: configuration layer with provider validation`

---

### Task 3: LLM Provider Abstraction
**Files:** Create `llm/base.py`, `llm/openai_provider.py`, `llm/gemini_provider.py`, `llm/anthropic_provider.py`, `llm/ollama_provider.py`, `llm/__init__.py`

**Steps:**
1. Create abstract base `LLMProvider` with methods: `call_with_tools()`, `call_simple()`, `get_cost_per_million_tokens()`
2. Implement OpenAIProvider using openai.AsyncOpenAI, with cost tracking per model (gpt-4o-mini: $0.15/1M input, $0.60/1M output)
3. Implement GeminiProvider using google.generativeai, convert OpenAI-style tools to Gemini format
4. Implement AnthropicProvider using anthropic.AsyncAnthropic, handle tool_use blocks
5. Implement OllamaProvider using ollama.AsyncClient, cost=0 (local)
6. Create factory function `get_provider(name)` in __init__.py

**Commit:** `feat: LLM providers (OpenAI/Gemini/Anthropic/Ollama) with cost tracking`

---

### Task 4: LangGraph State Machine
**Files:** Create `langgraph_pipeline/state.py`, `langgraph_pipeline/workflow.py`, `langgraph_pipeline/__init__.py`

**Steps:**
1. Define Pydantic models:
   - PaperSection: title, content, word_count
   - TextChunk: text, word_count, section_names
   - PipelineState: source_type, content, temp_path, raw_text, cleaned_sections, chunks, audio_files, final_output, total_cost, status, error
2. Create StateGraph with nodes: extracting_text, describing_figures, cleaning_with_llm, chunking_text, generating_audio, concatenating_audio, packaging_ep3
3. Set edges between nodes, conditional edge for EP3 packaging
4. Compile graph

**Commit:** `feat: LangGraph state machine skeleton`

---

## Phase 2: Core Pipeline (Tasks 5-7)

### Task 5: PDF Extraction
**Files:** Create `langgraph_pipeline/tools.py`

**Steps:**
1. Implement `extract_pdf_text(path)` using PyMuPDF (fitz.open), return joined page text
2. Implement `extract_figures(path)` returning List[PIL.Image] from PDF images
3. Update workflow.py `extracting_text()` node:
   - If URL: download with requests, save to temp file
   - If file: use temp_path directly
   - Extract text, check if empty (scanned PDF error)
   - Set status_message

**Commit:** `feat: PDF extraction with PyMuPDF`

---

### Task 6: LLM Cleaning Agent
**Files:** Modify `langgraph_pipeline/tools.py`, `langgraph_pipeline/workflow.py`

**Steps:**
1. Add tools to tools.py:
   - `extract_sections(text)`: Regex patterns for Abstract, Introduction, Methods, Results, Discussion, Conclusion. Return List[{title, content, word_count}]
   - `remove_citations(text)`: Regex remove [1], (Author, 2023)
   - `remove_metadata(text)`: Remove text after Keywords:, References, Acknowledgements, Funding
   - `smooth_for_tts(text)`: Expand "et al." to "and colleagues", "Fig." to "Figure"
2. Update workflow.py `cleaning_with_llm()` node:
   - Call extract_sections on raw_text
   - For each section: apply remove_citations, remove_metadata, smooth_for_tts
   - Store in state.cleaned_sections, update state.total_words

**Commit:** `feat: LLM cleaning tools (sections, citations, metadata, smoothing)`

---

### Task 7: Text Chunking
**Files:** Modify `langgraph_pipeline/workflow.py`

**Steps:**
1. Implement `chunking_text()` node with algorithm:
   - MAX_WORDS = 8500 (~60 min at 150 wpm)
   - Greedy packing: iterate sections, add to current chunk if fits
   - If section doesn't fit but < MAX_WORDS: save current, start new chunk with this section
   - If section > MAX_WORDS: split by paragraphs, create sub-chunks
   - Never split section title from content
2. Store chunks in state.chunks with text, word_count, section_names

**Commit:** `feat: section-aware text chunking for TTS limits`

---

## Phase 3: TTS & Audio (Tasks 8-9)

### Task 8: VibeVoice TTS Integration
**Files:** Create `tts/vibevoice.py`

**Steps:**
1. Load model: `microsoft/VibeVoice-1.5B` using HuggingFace Transformers
2. Implement `generate_audio(text, output_path)` that:
   - Validates word count < 10000 (else raise error)
   - Runs pipeline(text) -> WAV
   - Saves to output_path
3. Update workflow.py `generating_audio()` node:
   - Iterate state.chunks
   - Call vibevoice.generate_audio for each
   - Store paths in state.audio_files
   - Track progress in status_message

**Commit:** `feat: VibeVoice TTS integration with 60-min limit`

---

### Task 9: Audio Processing
**Files:** Modify `langgraph_pipeline/workflow.py`

**Steps:**
1. Implement `concatenating_audio()` node:
   - Use pydub.AudioSegment to concatenate all WAV files in state.audio_files
   - If OUTPUT_FORMAT == "mp3": export as MP3 (128kbps), delete WAVs
   - If OUTPUT_FORMAT == "wav": keep concatenated WAV
   - Store path in state.final_output

**Commit:** `feat: audio concatenation with pydub and format conversion`

---

## Phase 4: EP3 & UI (Tasks 10-12)

### Task 10: EP3 Generation
**Files:** Create `langgraph_pipeline/ep3_builder.py`

**Steps:**
1. Implement `create_ep3(audio_path, cleaned_text, sections, output_path)`:
   - Create EPUB3 container (zipfile)
   - Generate XHTML content documents from cleaned text (one per section)
   - Generate SMIL file (synchronization of audio with text)
   - Generate OPF package file (metadata, manifest, spine)
   - Add nav.xhtml (chapter navigation)
   - Package with ebooklib or manual zipfile
2. Update workflow.py `packaging_ep3()` node:
   - Call create_ep3 with concatenated audio and cleaned_sections
   - Store EP3 path in state.final_output

**Commit:** `feat: EP3 audiobook packaging with chapter navigation`

---

### Task 11: Gradio Frontend
**Files:** Create `app.py`

**Steps:**
1. Create Gradio interface with 3 tabs (URL, File Upload, Text Paste)
2. Add controls: Checkbox (VLM), Radio (EP3/MP3/WAV), Textbox (LLM endpoint)
3. Implement async process function that:
   - Creates PipelineState from input
   - Runs graph.invoke(state, stream=True)
   - Yields status updates to chatbot
   - Returns final_output file
4. Add cost tracking display

**Commit:** `feat: Gradio frontend with streaming status`

---

### Task 12: Installation Scripts
**Files:** Create `scripts/install.bat`, `scripts/install.sh`, `scripts/download_models.py`

**Steps:**
1. Check for OPENAI_API_KEY in env; if missing, prompt user
2. If no API key, ask if user wants Ollama; if yes, download Ollama and pull llama3.2
3. Create venv with `uv venv`
4. Install dependencies with `uv pip install -e .`
5. Download VibeVoice model (check HF token if required)
6. Create .env file with user inputs

**Commit:** `feat: auto-install scripts with provider detection`

---

## Phase 5: Deployment (Tasks 13-14)

### Task 13: Docker Support
**Files:** Create `docker/Dockerfile`, `docker/docker-compose.yml`

**Steps:**
1. Multi-stage Dockerfile:
   - Stage 1: python:3.11-slim (CPU)
   - Stage 2: nvidia/cuda:12.4.0 (GPU variant)
2. Install UV, install dependencies
3. Expose port 7860
4. HEALTHCHECK endpoint
5. docker-compose.yml for Ollama service (optional)

**Commit:** `feat: multi-arch Docker support (CPU/GPU)`

---

### Task 14: Testing
**Files:** Create `tests/test_extract_sections.py`, `tests/test_chunking.py`, `tests/test_ep3_builder.py`, `tests/test_tts.py`

**Steps:**
1. Test extract_sections with sample paper text
2. Test chunking algorithm with edge cases (section > MAX_WORDS)
3. Test EP3 builder creates valid .epub file
4. Test TTS with short text (mock or actual)

**Commit:** `feat: unit tests for core components`

---

## Progress Status (as of 2026-04-28)

**Completed Tasks:** 5/14 (36%)

| Task | Status | Notes |
|------|--------|-------|
| Task 1 | ✅ Complete | `pyproject.toml`, `.env.example`, `README.md` created |
| Task 2 | ✅ Complete | `config.py` with validation, 11 tests passing |
| Task 3 | ✅ Complete | 4 providers (OpenAI/Gemini/Anthropic/Ollama) + factory |
| Task 4 | ✅ Complete | State machine with 7 nodes, 7 tests passing |
| Task 5 | ✅ Complete | PDF extraction + cleaning tools, 12 tests |
| Task 6-14 | ❌ Pending | LLM cleaning, chunking, TTS, EP3, UI, Docker, Testing |

**Next Task:** Task 6 - LLM Cleaning Agent (implement `cleaning_with_llm` node)

---

## Execution

**Plan complete.** Two execution options:

1. **Subagent-Driven (recommended)** - Dispatch subagent per task using `subagent-driven-development`
2. **Inline Execution** - Execute tasks in this session using `executing-plans`

Which approach?