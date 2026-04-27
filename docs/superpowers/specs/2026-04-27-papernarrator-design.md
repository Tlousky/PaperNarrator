---
title: PaperNarrator - Scientific Paper to Audio Narration System
date: 2026-04-27
status: draft
author: [User]
---

# PaperNarrator Design Specification

## Overview

**Purpose:** Web application that converts scientific papers (PDF/HTML/text) into narrated EP3 audiobooks (EPUB 3 with Media Overlays) using Microsoft's VibeVoice TTS model, with LLM-based text cleaning and optional figure description via VLM.

**Key features:**
- Multi-input support: URL, file upload, or text paste
- Autonomous LLM agent (LangGraph) that cleans papers using tools (citation removal, section extraction, VLM figure descriptions)
- Intelligent text chunking (≤8,500 words) at section boundaries for 60-min VibeVoice limit
- **EP3 output (default):** Chapter navigation, embedded text, read-along capability, playback in any modern audiobook app
- Synchronous processing: Input paper → Wait → Download EP3
- Streaming SSE status updates showing LLM reasoning and progress

**Success criteria:** User can upload a 20-page scientific paper and receive a properly structured EP3 audiobook within 10-30 minutes, with natural-sounding speech that excludes references, author lists, and formatted citations, and includes chapter markers for easy navigation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Gradio Interface                         │
│  ┌─────────┐  ┌─────────────┐  ┌──────────┐                │
│  │   URL   │  │ File Upload │  │ Text     │                │
│  │  Input  │  │    Input    │  │  Paste   │                │
│  └────┬────┘  └──────┬──────┘  └────┬─────┘                │
│       │              │              │                      │
│       └──────────────┴──────────────┘                      │
│                       │                                    │
│              ┌────────▼────────┐                          │
│              │  Controls       │                          │
│              │ • Describe      │                          │
│              │   Figures? [ ]  │                          │
│              │ • Format: MP3   │                          │
│              │   WAV          │                          │
│              │ • LLM Endpoint │                          │
│              └────────┬────────┘                          │
│                       │                                    │
│              ┌────────▼────────┐                          │
│              │ Streaming Chat  │ ← SSE (LLM reasoning)    │
│              │   Status Bar    │                          │
│              └────────┬────────┘                          │
│                       │                                    │
│              ┌────────▼────────┐                          │
│              │  Audio Player   │                          │
│              │  Download       │                          │
│              └─────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 LangGraph Orchestration                     │
│                                                             │
│  input_received → extracting_text → describing_figures? →   │
│  cleaning_with_llm → chunking_text → generating_audio →    │
│  concatenating_audio → complete                            │
│                                                             │
│  [Error handling: retries, state rollback]                 │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Tool Layer                               │
│  • extract_sections() - Section detection                   │
│  • remove_citations() - Citation marker removal             │
│  • remove_figures_references() - Caption handling           │
│  • smooth_for_tts() - Natural language smoothing            │
│  • describe_figure() - VLM figure description (optional)    │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   VibeVoice TTS                             │
│  Model: microsoft/VibeVoice-1.5B                            │
│  Limit: 60 min audio per generation (≈8,500 words)          │
│  Output: WAV → MP3 (128kbps) or WAV (lossless)             │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. Gradio Interface

**Input Tabs (3):**
- **URL Tab:** `gr.Textbox` for HTTP(S) URLs (arXiv, journal PDFs/HTML)
- **File Upload Tab:** `gr.File` accepting `.pdf`, `.html`
- **Text Paste Tab:** `gr.Textbox` (large) for raw text

**Controls:**
- `gr.Checkbox` "Describe figures using VLM" (default: False, requires GPU)
- `gr.Radio` choices=["EP3 (Audiobook)", "MP3 (Audio only)", "WAV (Lossless)"] (default: EP3)
- `gr.Textbox` "LLM endpoint URL" (optional, overrides `.env`)

**Output:**
- `gr.Chatbot` or streaming `gr.Textbox` for SSE status updates
- `gr.File` for download (EP3 or audio file)
- `gr.Audio` for quick preview (extracts audio from EP3 or plays directly)

**Progress:**
- Streaming updates via generator yielding LangGraph events
- Format: "📥 Received → 🔍 Extracting → 🧹 Cleaning → 🎵 Generating (2/5) → ✅ Complete"

### 2. LangGraph State Machine

**States:**

1. **`input_received`**
   - Data: `source_type: str` (url/file/text), `content: bytes|str`, `temp_path: str`
   - Validation: Check file size <100MB, URL accessible, text length >100 chars

2. **`extracting_text`**
   - Tool: `extract_pdf_text(path: str) -> str` (PyMuPDF/pdfplumber)
   - Data: `raw_text: str`, `figures: List[PIL.Image]` (if VLM enabled)
   - Error: Return error if scanned PDF (0 text layers)

3. **`describing_figures`** (optional, only if checkbox checked)
   - Tool: `describe_figure(image: PIL.Image, context: str) -> str`
   - LLM: Local VLM (LLaVA or Qwen-VL via llama.cpp)
   - Inserts descriptions inline at figure references: "Figure 1 [Description: ...] shows..."

4. **`cleaning_with_llm`**
   - Autonomous agent with tools:
     - `extract_sections(text: str) -> List[{title, content, word_count}]`
     - `remove_citations(text: str) -> str`
     - `remove_metadata(text: str) -> str` (authors, affiliations, funding)
     - `smooth_for_tts(text: str) -> str`
   - Data: `cleaned_sections: List[Section]`, `total_words: int`
   - Output: Plain text, no LaTeX, no Markdown, paragraphs separated by blank lines

5. **`chunking_text`**
   - Algorithm: Greedy packing at section boundaries, max 8,500 words/chunk
   - Data: `chunks: List[{text: str, word_count: int, section_names: List[str]}]`
   - Logic: Never split a section; if section >8,500 words, split by paragraph

6. **`generating_audio`**
   - Tool: `vibe_voice_generate(text: str) -> Path` (WAV)
   - Model: `microsoft/VibeVoice-1.5B`
   - Data: `audio_files: List[Path]`, `chunk_index: int`
   - Error: Retry once on OOM, fallback to CPU if available

7. **`concatenating_audio`**
   - Tool: `concat_wav(files: List[Path], output_format: str) -> Path`
   - Library: `pydub` or `ffmpeg`
   - Cleanup: Delete intermediate WAVs if output_format != "ep3"
   - Data: `final_audio_path: Path`

8. **`packaging_ep3`** (only if output_format == "ep3")
   - Tool: `create_ep3(audio_path: Path, cleaned_text: str, sections: List[Section], chunks: List[Chunk]) -> Path`
   - Generates: EPUB 3 with Media Overlays (SMIL), chapter navigation, embedded text
   - Data: `ep3_path: Path`, `ep3_size: int`

9. **`complete`**
   - Data: `output_path: Path`, `format: str`, `duration: float`
   - Return: EP3 or audio file to Gradio

**Error State:** Any state can transition to `error` with `error_message: str`

### 3. LLM Cleaning Tools (The "Skill")

**Skill File:** `skills/paper-narrator-cleaning/SKILL.md`

**Goal:** Transform academic paper text into natural narration text.

**Tool Definitions:**

```python
def extract_sections(text: str) -> List[dict]:
    """Identify standard paper sections using regex and heuristics."""
    # Returns: [{title: "Abstract", content: "...", word_count: 150}, ...]
    
def remove_citations(text: str) -> str:
    """Remove [1], [2], (Author, 2023), superscript markers."""
    # Regex: \[\d+\], (\w+, \d{4}), ^\d+$ patterns
    
def remove_metadata(text: str) -> str:
    """Remove author lists, affiliations, acknowledgments, funding statements."""
    # Heuristics: Text before first newline after title, "We thank", "Funding"
    
def smooth_for_tts(text: str) -> str:
    """Convert academic passive to natural speech.
    - "et al." → "and colleagues"
    - "Fig. 1" → "Figure 1"
    - "→" → "leads to"
    - Expand abbreviations (e.g., "ANOVA" → "ANOVA")
    - Rewrite: "X was analyzed using Y" → "We analyzed X using Y"
    """
    
def describe_figure(image: PIL.Image, context: str) -> str:
    """Use VLM to describe the figure in plain language."""
    # Returns: "A bar chart comparing treatment groups A and B, showing..."
```

**Prompt Template:**
```
You are a scientific paper narrator. Your task is to clean text for audio narration.

Rules:
1. Keep only narrative text (Abstract, Introduction, Methods, Results, Discussion, Conclusion)
2. Remove: References, Authors, Acknowledgments, Funding, Figure captions (unless inline)
3. Remove citation markers: [1], (Smith et al., 2023)
4. Expand: "et al." → "and colleagues", "Fig." → "Figure"
5. Convert math to speech: "$E=mc^2$" → "E equals m c squared"
6. Make passive voice active where natural

Use the available tools to process the text step by step.
```

### 4. Chunking Algorithm

**Pseudocode:**
```python
def chunk_text(sections: List[Section]) -> List[Chunk]:
    MAX_WORDS = 8500
    chunks = []
    current = []
    current_count = 0
    
    for section in sections:
        if current_count + section.word_count <= MAX_WORDS:
            # Fits in current chunk
            current.append(section)
            current_count += section.word_count
        elif section.word_count <= MAX_WORDS:
            # Save current, start new with this section
            chunks.append(Chunk(text='\n\n'.join(s.content for s in current)))
            current = [section]
            current_count = section.word_count
        else:
            # Section too big, split by paragraphs
            chunks.append(Chunk(...))  # current
            sub_chunks = split_by_paragraphs(section, MAX_WORDS)
            chunks.extend(sub_chunks)
            current = []
            current_count = 0
    
    if current:
        chunks.append(Chunk(...))
    return chunks
```

**Constraint:** Never split a section title from its content. If a section cannot fit in remaining space, it starts a new chunk.

### 5. VibeVoice Integration

**Installation:**
```python
from transformers import AutoProcessor, VibeVoicePipeline

processor = AutoProcessor.from_pretrained("microsoft/VibeVoice-1.5B")
pipeline = VibeVoicePipeline(model="microsoft/VibeVoice-1.5B", device="cuda")

wav = pipeline("Text to narrate...")
```

**Limits:**
- Max 60 minutes per generation (enforced by model)
- Input text limit: ~10,000 words at 150 wpm
- GPU memory: ~3-4GB for 1.5B model

**Output:** WAV file at 24kHz, mono.

**Note:** VibeVoice generates audio in chunks. These are concatenated into a continuous audio stream, then packaged into EP3 with chapter markers aligned to section boundaries.

### 5. EP3 Generation

**EP3 Structure (ISO/IEC 23736):**
- **Package file** (`content.opf`): Metadata, manifest, spine (reading order)
- **Content documents** (XHTML): The cleaned paper text, split into sections
- **Audio files**: MP3-encoded chunks (128kbps) or single concatenated MP3
- **SMIL files**: Synchronization of audio with text paragraphs/chapters
- **Navigation** (`nav.xhtml`): Chapter list (Abstract, Intro, Methods, Results, Discussion, Conclusion)

**Generation Process:**
1. Concatenate WAV chunks → Convert to MP3 (128kbps) using `ffmpeg`
2. Create XHTML content documents from cleaned text (one per section or one combined)
3. Create SMIL file describing audio timing (paragraph-level or section-level sync)
4. Create OPF package file linking content, audio, and SMIL
5. Zip container with `mimetype` and `container.xml`

**Chapter Markers:**
- Aligned to section boundaries from `extract_sections()` tool
- Each chapter corresponds to a paper section (Abstract, Introduction, etc.)
- Navigation allows jumping to any section

**Text Embedding:**
- Full cleaned text embedded in XHTML for read-along capability
- Highlighting support in compatible readers (Calibre, Apple Books)
- Searchable text within the audiobook

**Python Libraries:**
- `ebooklib` (EPUB creation)
- `xml.etree.ElementTree` (SMIL/OPF generation)
- `zipfile` (OCF container)

### 6. Error Handling

| Failure Mode | Detection | Recovery |
|-------------|-----------|----------|
| Invalid URL | 404/SSL error on fetch | Return error immediately |
| Scanned PDF | PyMuPDF returns 0 text chars | Return: "Please use text-based PDF" |
| LLM offline | ConnectionError on endpoint | Retry 3x (exponential backoff), then error |
| VibeVoice OOM | `torch.cuda.OutOfMemoryError` | Try CPU, or error with suggestion |
| Audio concat fail | `pydub` raises | Return error with failed chunk index |
| File >100MB | Gradio upload limit | Configured in `gr.File(max_size="100MB")` |

### 7. Configuration

**`.env` file:**
```env
LLM_ENDPOINT=http://localhost:11434/api/chat  # Ollama or llama.cpp
LLM_MODEL=llama3.2                           # For cleaning
VLM_ENDPOINT=http://localhost:11434/api/chat  # Optional, for figures
VLM_MODEL=llava-v1.6-34b                     # Optional
VIBEVOICE_DEVICE=cuda:0                      # or cpu
OUTPUT_FORMAT=ep3                            # default: ep3, mp3, or wav
MP3_BITRATE=128
EP3_TITLE="PaperNarrator Output"            # Default title for EP3 metadata
```

**Requirements:**
- `gradio>=4.0`
- `langgraph`
- `pymupdf` (PyMuPDF)
- `transformers` (for VibeVoice)
- `torch`
- `pydub` (audio concatenation)
- `ebooklib` (EP3 generation)
- `ffmpeg` (system dependency for audio conversion and pydub)

---

## Implementation Plan

**Phase 1: Core Pipeline**
1. Gradio UI with 3 input tabs, no processing yet
2. LangGraph state machine skeleton (all states, dummy transitions)
3. PDF extraction tool (PyMuPDF)
4. Dummy LLM cleaning (identity function)
5. VibeVoice integration with single chunk
6. Audio concatenation

**Phase 2: LLM Integration**
7. Implement LangGraph tools (extract_sections, remove_citations, etc.)
8. Integrate local LLM (Ollama/llama.cpp)
9. Implement chunking logic
10. Multi-chunk VibeVoice generation

**Phase 3: Polish**
11. VLM figure description
12. SSE streaming to Gradio
13. Error handling and retries
14. EP3 packaging (SMIL, OPF, chapter navigation)
15. Testing with real papers (5k, 10k, 20k words)
16. MP3/WAV fallback options

**Phase 4: Deployment**
16. Dockerfile creation
17. GPU memory optimization
18. Load testing (concurrent users)

---

## Open Questions

1. **LLM model choice:** Which specific model for text cleaning? (Llama 3.2 3B, Mistral 7B, or something else?)
2. **VibeVoice speed:** Approx. 0.5x realtime? Need to estimate total processing time for UX.
3. **Concurrent users:** If two users upload papers, do we queue or error? (Current design assumes single-user or FIFO queue.)
4. **Figure extraction:** Which VLM? LLaVA 34B requires ~20GB VRAM. May need cloud API fallback.
5. **Math handling:** How to handle equations? Current design says "convert to speech" but complex math may be unreadable. May need to skip or say "Equation one..."
6. **EP3 SMIL granularity:** Should SMIL sync be paragraph-level (more precise, larger file) or section-level (simpler, smaller)? Default: section-level.

---

## References

- VibeVoice: https://github.com/microsoft/VibeVoice
- VibeVoice Model: https://huggingface.co/microsoft/VibeVoice-1.5B
- LangGraph: https://langchain-ai.github.io/langgraph/
- Gradio: https://www.gradio.app/
- PyMuPDF (fitz): https://pymupdf.readthedocs.io/
- EPUB 3 Standard: https://www.w3.org/publishing/epub3/
- DAISY Consortium (EP3/SMIL): https://daisy.org/
- ebooklib: https://github.com/aerkalov/ebooklib
