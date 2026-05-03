from langgraph.graph import StateGraph, END
import os
import re
import shutil
import subprocess
import math
from datetime import datetime
from typing import TypedDict, Annotated, List, Optional

from .state import PipelineState, PipelineStatus, PaperSection, TextChunk, ChapterInfo
from .tools import (
    _extract_pdf_text,
    _extract_sections,
    _remove_citations,
    _remove_metadata,
    _smooth_for_tts
)
import tempfile
import os
import requests
import logging

logger = logging.getLogger(__name__)

import sys
from pathlib import Path
# Ensure project root is in path for observability import
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import tracer for observability
try:
    from observability.tracer import get_tracer
    LANGFUSE_AVAILABLE = True
except ImportError as e:
    print(f"[WorkflowBuilder] Langfuse tracing disabled: {e}")
    LANGFUSE_AVAILABLE = False


class WorkflowBuilder:
    """Builder for the PaperNarrator LangGraph workflow."""
    
    def __init__(self, config=None):
        self.config = config
        self.graph = None
    
    def create_graph(self) -> StateGraph:
        """Create the LangGraph state machine with 7 nodes."""
        # Define the graph with PipelineState
        workflow = StateGraph(PipelineState)
        
        # Add nodes
        workflow.add_node("setup_run", self.setup_run)
        workflow.add_node("extracting_text", self.extracting_text)
        workflow.add_node("describing_figures", self.describing_figures)
        workflow.add_node("cleaning_with_llm", self.cleaning_with_llm)
        workflow.add_node("chunking_text", self.chunking_text)
        workflow.add_node("generating_audio", self.generating_audio)
        workflow.add_node("concatenating_audio", self.concatenating_audio)
        workflow.add_node("packaging_m4b", self.packaging_m4b)
        workflow.add_node("finalize_run", self.finalize_run)
        
        # Set entry point
        workflow.set_entry_point("setup_run")
        
        # Set conditional edges based on OUTPUT_FORMAT
        workflow.add_conditional_edges(
            "concatenating_audio",
            self.should_package_as_m4b,
            {
                "package": "packaging_m4b",
                "skip": "finalize_run"
            }
        )
        
        # Set standard edges
        workflow.add_edge("setup_run", "extracting_text")
        workflow.add_edge("extracting_text", "describing_figures")
        workflow.add_edge("describing_figures", "cleaning_with_llm")
        workflow.add_edge("cleaning_with_llm", "chunking_text")
        workflow.add_edge("chunking_text", "generating_audio")
        workflow.add_edge("generating_audio", "concatenating_audio")
        workflow.add_edge("packaging_m4b", "finalize_run")
        workflow.add_edge("finalize_run", END)
        
        self.graph = workflow.compile()
        
        # Add LangFuse tracing if available
        if LANGFUSE_AVAILABLE:
            try:
                tracer = get_tracer()
                self.graph = tracer.trace_graph(self.graph, "paper-narrator-pipeline")
            except Exception as e:
                logger.error(f"[WorkflowBuilder] Failed to initialize tracing: {e}")
        
        return self.graph
    
    def should_package_as_m4b(self, state: PipelineState) -> str:
        """Route to M4B packaging or finish if other format."""
        output_format = getattr(self.config, 'OUTPUT_FORMAT', 'm4b').lower()
        if output_format == "m4b":
            return "package"
        return "skip"
    
    async def setup_run(self, state: PipelineState) -> PipelineState:
        """Task 4: Initialize run directory and ID for the paper trail."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state.run_id = timestamp
        logger.info(f"Setting up run: {timestamp}")
        
        # Create run directory
        run_dir = os.path.join("outputs", "runs", timestamp)
        os.makedirs(run_dir, exist_ok=True)
        state.run_dir = run_dir
        
        # Create subdirectories
        os.makedirs(os.path.join(run_dir, "audio_chunks"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "text_chunks"), exist_ok=True)
        
        state.status_message = f"Initialized run: {timestamp}"
        logger.info(f"Run directory created: {run_dir}")
        return state
    
    async def extracting_text(self, state: PipelineState) -> PipelineState:
        """Task 5: Extract text from PDF (URL or file)."""
        state.status_message = "Extracting text from PDF..."
        logger.info(f"Node [extracting_text] starting for source: {state.source_type}")
        
        pdf_path = None
        
        try:
            # Handle URL input
            if state.source_type == "url":
                logger.info(f"Downloading PDF from: {state.content}")
                state.status_message = "Downloading PDF..."
                response = requests.get(state.content, timeout=30)
                response.raise_for_status()
                
                # Save to temp file
                with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
                    f.write(response.content)
                    pdf_path = f.name
                state.temp_path = pdf_path
                logger.info(f"PDF downloaded to temp: {pdf_path}")
            elif state.source_type == "file":
                pdf_path = state.temp_path
                logger.info(f"Using uploaded file: {pdf_path}")
            else:
                logger.error(f"Invalid source_type: {state.source_type}")
                state.status = PipelineStatus.FAILED
                state.error = "Invalid source_type. Must be 'url' or 'file'."
                return state
            
            # Extract text
            logger.info(f"Extracting text from {pdf_path} using PyMuPDF")
            state.status_message = "Extracting text..."
            raw_text = await _extract_pdf_text(pdf_path)
            
            # Check for scanned PDF (empty or very short text)
            if not raw_text or len(raw_text.strip()) < 50:
                logger.error("Extracted text is too short or empty (possible scanned PDF)")
                state.status = PipelineStatus.FAILED
                state.error = "No text extracted. PDF may be image-only (scanned). Please use OCR or provide text PDF."
                return state
            
            state.raw_text = raw_text
            state.status_message = f"Extracted {len(raw_text)} characters"
            logger.info(f"Text extraction complete. {len(raw_text)} characters extracted.")
            
        except Exception as e:
            logger.error(f"Failed to process PDF: {str(e)}", exc_info=True)
            state.status = PipelineStatus.FAILED
            state.error = f"Failed to process PDF: {str(e)}"
        
        return state
    
    async def describing_figures(self, state: PipelineState) -> PipelineState:
        """VLM (Vision Language Model) to describe figures if enabled."""
        if state.status == PipelineStatus.FAILED: return state
        state.status_message = "Describing figures..."
        # TODO: Implement VLM if VLM_ENABLED
        return state
    
    async def cleaning_with_llm(self, state: PipelineState) -> PipelineState:
        """Task 6: LLM-based cleaning (extract sections, remove citations, smooth for TTS)."""
        if state.status == PipelineStatus.FAILED: return state
        state.status_message = "Cleaning text with LLM..."
        logger.info(f"Node [cleaning_with_llm] starting. Raw text length: {len(state.raw_text)}")
        
        try:
            if not state.raw_text:
                state.status = PipelineStatus.FAILED
                state.error = "No raw text available. Previous extraction step may have failed."
                return state
            
            # 1. Extract sections using regex first
            state.status_message = "Extracting sections..."
            logger.info("Extracting sections using regex...")
            sections_raw = await _extract_sections(state.raw_text)
            
            if not sections_raw:
                logger.error("No sections extracted from raw text.")
                state.status = PipelineStatus.FAILED
                state.error = "No sections extracted from text."
                return state
            
            logger.info(f"Extracted {len(sections_raw)} sections: {[s['title'] for s in sections_raw]}")
            
            # 2. Initialize OpenAI and Langfuse for cleaning
            from openai import AsyncOpenAI
            logger.info(f"Initializing AsyncOpenAI (Base URL: {os.getenv('OPENAI_API_BASE')})")
            client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE")
            )
            
            # Initialize Langfuse client for tracing LLM calls
            langfuse_client = None
            cleaning_span_ctx = None
            try:
                from langfuse import Langfuse
                langfuse_client = Langfuse(
                    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                    host=os.getenv("LANGFUSE_HOST")
                )
                # In v4, start_as_current_observation creates a span that becomes part of the trace
                cleaning_span_ctx = langfuse_client.start_as_current_observation(
                    name="cleaning_with_llm",
                    input={"pdf_path": state.temp_path, "num_sections": len(sections_raw)},
                    metadata={
                        "run_id": state.run_id,
                        "session_id": state.session_id,
                        "user_id": state.user_id
                    }
                )
                # Also propagate attributes if using a separate client instance
                from langfuse import propagate_attributes
                prop_ctx = propagate_attributes(
                    session_id=state.session_id,
                    user_id=state.user_id,
                    tags=state.tags
                )
                logger.info("Langfuse span and propagation initialized for cleaning_with_llm")
            except Exception as lf_err:
                logger.warning(f"Langfuse tracing not available: {lf_err}")
                prop_ctx = None
            
            cleaned_sections = []
            total_words = 0
            model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
            
            # Use cleaning_span_ctx if available
            async def process_sections():
                nonlocal total_words
                for i, sec in enumerate(sections_raw):
                    title = sec["title"]
                    content = sec["content"]
                    
                    state.status_message = f"Cleaning section {i+1}/{len(sections_raw)}: {title}..."
                    logger.info(f"Cleaning section {i+1}/{len(sections_raw)}: {title} ({len(content.split())} words)")
                    
                    # Preliminary regex cleaning
                    content = await _remove_citations(content)
                    content = await _remove_metadata(content)
                    content = await _smooth_for_tts(content)
                    
                    from .skills import CITATION_REMOVAL_SKILL, FIGURE_CLEANING_SKILL
                    
                    system_msg = "You are an expert editor preparing scientific papers for high-quality TTS audiobooks. You remove technical debris and academic citations while preserving the full narrative text. Use your specialized skills to handle citations and figures properly."
                    user_msg = f"""CLEANING TASK: PREPARE TEXT FOR TTS AUDIOBOOK

I have a section of a scientific paper extracted from a PDF. It contains the core narrative but also has 'debris' from PDF parsing and vector chart extraction.

CRITICAL INSTRUCTION:
- DO NOT TRUNCATE the text. Your output must contain the FULL narrative of the section.
- DO NOT SUMMARIZE. The word count of the output should be similar to the input (minus the debris).
- If the text ends abruptly, try to keep it exactly as it is, do not try to "finish" it or cut it further.

TECHNICAL DEBRIS TO REMOVE:
{FIGURE_CLEANING_SKILL}

{CITATION_REMOVAL_SKILL}

YOUR GOAL:
Remove all such debris and academic citations. Keep only the readable, narrative scientific text. 
Maintain the logical flow and sentence structure.

SECTION TITLE: {title}
RAW CONTENT:
---
{content}
---

Cleaned text for TTS:"""
                    
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ]
                    
                    # Create a Langfuse generation span
                    generation = None
                    if langfuse_client:
                        generation = langfuse_client.start_observation(
                            name=f"clean_section_{i+1}_{title.replace(' ', '_')}",
                            as_type="generation",
                            model=model_name,
                            model_parameters={"temperature": 0.0},
                            input=messages
                        )
                    
                    try:
                        logger.info(f"Sending LLM request for section: {title}...")
                        response = await client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            temperature=0.0,
                            max_tokens=16384  # Ensure we don't truncate long sections
                        )
                        cleaned_content = response.choices[0].message.content.strip()
                        
                        if generation:
                            generation.update(
                                output=cleaned_content,
                                usage_details={
                                    "prompt_tokens": response.usage.prompt_tokens,
                                    "completion_tokens": response.usage.completion_tokens,
                                    "total_tokens": response.usage.total_tokens
                                }
                            )
                            generation.end()
                        logger.info(f"LLM cleaning successful for section: {title}")
                    except Exception as llm_err:
                        logger.warning(f"LLM pass failed for section {title}: {llm_err}. Using basic cleaned text.")
                        if generation:
                            generation.update(output=f"ERROR: {llm_err}", level="ERROR")
                            generation.end()
                        cleaned_content = content
                    
                    cleaned_content = re.sub(r"^Cleaned text for TTS:\s*", "", cleaned_content, flags=re.IGNORECASE)
                    word_count = len(cleaned_content.split())
                    total_words += word_count
                    
                    cleaned_sections.append(PaperSection(
                        title=title,
                        content=cleaned_content,
                        word_count=word_count
                    ))

            # Execute the processing
            async def run_with_propagation():
                if prop_ctx:
                    with prop_ctx:
                        await process_sections()
                else:
                    await process_sections()

            if cleaning_span_ctx:
                with cleaning_span_ctx as span:
                    await run_with_propagation()
                    span.update(output={"sections_cleaned": len(cleaned_sections), "total_words": total_words})
            else:
                await run_with_propagation()

            # Finalize Langfuse if client exists
            if langfuse_client:
                langfuse_client.flush()
                logger.info("Langfuse trace flushed.")
            
            state.cleaned_sections = cleaned_sections
            state.total_words = total_words
            state.status_message = f"LLM cleaning complete: {len(cleaned_sections)} sections ({total_words} words)"
            logger.info(f"Node [cleaning_with_llm] complete. Total words: {total_words}")
            
        except Exception as e:
            logger.error(f"Failed to clean text with LLM: {str(e)}", exc_info=True)
            
            # Fallback: Just use the regex-cleaned sections we had before the LLM pass
            if not state.cleaned_sections and 'sections_raw' in locals():
                logger.warning("LLM cleaning stage failed completely. Falling back to basic regex cleaning.")
                state.status_message = "LLM cleaning failed. Using regex fallback."
                fallback_sections = []
                for sec in sections_raw:
                    content = sec["content"]
                    content = await _remove_citations(content)
                    content = await _remove_metadata(content)
                    content = await _smooth_for_tts(content)
                    fallback_sections.append(PaperSection(
                        title=sec["title"],
                        content=content,
                        word_count=len(content.split())
                    ))
                state.cleaned_sections = fallback_sections
                state.total_words = sum(s.word_count for s in fallback_sections)
        
        return state
    
    async def chunking_text(self, state: PipelineState) -> PipelineState:
        """Task 7: Balanced, section-aware chunking for TTS (max 4000 words per chunk)."""
        if state.status == PipelineStatus.FAILED: return state
        state.status_message = "Chunking text into balanced chapters..."
        logger.info(f"Node [chunking_text] starting. Total words: {state.total_words}")
        
        try:
            if not state.cleaned_sections:
                state.status = PipelineStatus.FAILED
                state.error = "No cleaned sections available."
                return state
            
            MAX_WORDS = 4000
            all_chunks = []
            
            for section in state.cleaned_sections:
                section_title = section.title
                section_text = section.content
                section_word_count = section.word_count
                
                if section_word_count == 0:
                    continue
                
                logger.info(f"Chunking section: {section_title} ({section_word_count} words)")
                
                # 1. Split section into sentences (more robustly)
                # This pattern looks for punctuation followed by space or end of string, 
                # but avoids splitting on common abbreviations if possible.
                # However, since we smoothed text earlier, it's mostly safe.
                sentence_matches = list(re.finditer(r'.*?[.!?](?=\s+[A-Z]|\s*$)', section_text, re.DOTALL))
                sentences = []
                last_pos = 0
                for m in sentence_matches:
                    sentences.append(m.group(0).strip())
                    last_pos = m.end()
                
                # Catch any trailing text that doesn't end with sentence punctuation
                if last_pos < len(section_text):
                    trailing = section_text[last_pos:].strip()
                    if trailing:
                        sentences.append(trailing)
                
                if not sentences:
                    # Fallback: if no punctuation found, just take the whole thing as one "sentence"
                    sentences = [section_text.strip()]
                
                # 2. Determine how many chunks this section needs
                # We use a lower MAX_WORDS for safer TTS processing if needed, but 4000 is okay.
                num_chunks = math.ceil(section_word_count / MAX_WORDS)
                target_words_per_chunk = math.ceil(section_word_count / num_chunks)
                logger.info(f"Target: {num_chunks} chunks of ~{target_words_per_chunk} words each.")
                
                # 3. Distribute sentences into chunks
                section_chunks = []
                current_sentences = []
                current_word_count = 0
                
                for sent in sentences:
                    sent_words = len(sent.split())
                    
                    # If adding this sentence exceeds MAX_WORDS, we MUST save current chunk
                    # OR if we have reached the target and it's not the last chunk
                    if (current_word_count + sent_words > MAX_WORDS) or \
                       (current_word_count >= target_words_per_chunk and len(section_chunks) < num_chunks - 1):
                        
                        if current_sentences:
                            chunk_id = f"{section_title}_chunk_{len(section_chunks)+1}"
                            section_chunks.append(TextChunk(
                                text=" ".join(current_sentences),
                                word_count=current_word_count,
                                section_names=[section_title],
                                chunk_id=chunk_id
                            ))
                            logger.info(f"Created chunk {chunk_id}: {current_word_count} words.")
                            current_sentences = []
                            current_word_count = 0
                    
                    current_sentences.append(sent)
                    current_word_count += sent_words
                
                # Save the last chunk of the section
                if current_sentences:
                    chunk_id = f"{section_title}_chunk_{len(section_chunks)+1}"
                    section_chunks.append(TextChunk(
                        text=" ".join(current_sentences),
                        word_count=current_word_count,
                        section_names=[section_title],
                        chunk_id=chunk_id
                    ))
                    logger.info(f"Created final section chunk {chunk_id}: {current_word_count} words.")
                
                all_chunks.extend(section_chunks)
            
            state.chunks = all_chunks
            total_words = sum(c.word_count for c in all_chunks)
            state.status_message = f"Created {len(all_chunks)} chunks from {len(state.cleaned_sections)} sections ({total_words} total words)"
            logger.info(f"Node [chunking_text] complete. {len(all_chunks)} chunks created.")
            
        except Exception as e:
            logger.error(f"Failed to chunk text: {str(e)}", exc_info=True)
            state.status = PipelineStatus.FAILED
            state.error = f"Failed to chunk text: {str(e)}"
        
        return state
    
    async def generating_audio(self, state: PipelineState) -> PipelineState:
        """Task 8: Generate audio for each chunk using VibeVoice."""
        if state.status == PipelineStatus.FAILED: return state
        state.status_message = "Generating audio for chunks..."
        logger.info(f"Node [generating_audio] starting for {len(state.chunks)} chunks.")
        
        try:
            if not state.chunks:
                logger.error("No chunks available for TTS.")
                state.status = PipelineStatus.FAILED
                state.error = "No chunks available for TTS."
                return state
            
            from tts.vibevoice import VibeVoiceTTS
            from pydub import AudioSegment
            import tempfile
            import os
            
            # Initialize TTS (lazy loads model)
            logger.info("Initializing VibeVoiceTTS engine...")
            tts = VibeVoiceTTS(
                model_name=os.getenv("VIBEVOICE_MODEL_PATH", "./models/microsoft/VibeVoice-Realtime-0.5B"),
                speaker_name=state.voice_profile or "Emma",
                device=os.getenv("VIBEVOICE_DEVICE", None),
                cfg_scale=float(os.getenv("VIBEVOICE_CFG_SCALE", 1.5)),
                num_steps=int(os.getenv("VIBEVOICE_NUM_STEPS", 20))
            )
            logger.info(f"TTS Engine initialized. Speaker: {state.voice_profile or 'Emma'}, Steps: {os.getenv('VIBEVOICE_NUM_STEPS', 20)}")
            
            audio_files = []
            temp_dir = tempfile.mkdtemp(prefix="papernarrator_tts_")
            last_section = None
            
            # Create silence segment (2 seconds)
            silence = AudioSegment.silent(duration=2000)
            
            for i, chunk in enumerate(state.chunks):
                chunk_num = i + 1
                total_chunks = len(state.chunks)
                
                current_section = chunk.section_names[0] if chunk.section_names else None
                is_new_section = current_section and current_section != last_section
                
                state.status_message = f"Generating audio {chunk_num}/{total_chunks} ({chunk.word_count} words)..."
                
                # 1. Generate audio for this chunk's content
                content_wav = os.path.join(temp_dir, f"chunk_{chunk_num:03d}_content.wav")
                result_path = tts.generate_audio(
                    text=chunk.text,
                    output_path=content_wav,
                    word_count=chunk.word_count
                )
                
                if not os.path.exists(result_path):
                    logger.error(f"TTS generation failed for chunk {chunk_num} content.")
                    state.status = PipelineStatus.FAILED
                    state.error = f"TTS failed for content of chunk {chunk_num}"
                    return state
                
                # 2. If it's a new section, generate title intro and combine
                final_chunk_wav = os.path.join(temp_dir, f"chunk_{chunk_num:03d}.wav")
                
                if is_new_section:
                    state.status_message = f"Adding intro for section: {current_section}..."
                    title_wav = os.path.join(temp_dir, f"chunk_{chunk_num:03d}_title.wav")
                    tts.generate_audio(
                        text=current_section,
                        output_path=title_wav,
                        word_count=len(current_section.split())
                    )
                    
                    if os.path.exists(title_wav):
                        title_seg = AudioSegment.from_file(title_wav, format="wav")
                        content_seg = AudioSegment.from_file(content_wav, format="wav")
                        combined = title_seg + silence + content_seg
                        combined.export(final_chunk_wav, format="wav")
                        
                        # Cleanup intermediate
                        os.remove(title_wav)
                        os.remove(content_wav)
                    else:
                        # Fallback to just content
                        os.rename(content_wav, final_chunk_wav)
                else:
                    # No intro needed, just rename content to final
                    os.rename(content_wav, final_chunk_wav)
                
                if os.path.exists(final_chunk_wav):
                    audio_files.append(final_chunk_wav)
                    # Also copy to run_dir for paper trail
                    if state.run_dir:
                        dest = os.path.join(state.run_dir, "audio_chunks", os.path.basename(final_chunk_wav))
                        shutil.copy2(final_chunk_wav, dest)
                else:
                    state.status = PipelineStatus.FAILED
                    state.error = f"Failed to finalize audio for chunk {chunk_num}"
                    return state
                
                last_section = current_section
            
            state.audio_files = audio_files
            state.temp_path = temp_dir
            state.status_message = f"Generated {len(audio_files)} audio files with intros"
            state.status = PipelineStatus.COMPLETED
            
        except Exception as e:
            state.status = PipelineStatus.FAILED
            state.error = f"TTS generation failed: {str(e)}"
        
        return state
    
    async def concatenating_audio(self, state: PipelineState) -> PipelineState:
        """Task 9: Concatenate audio chunks and calculate chapter marks."""
        if state.status == PipelineStatus.FAILED: return state
        state.status_message = "Concatenating audio chunks and calculating chapters..."
        
        try:
            if not state.audio_files:
                state.status = PipelineStatus.FAILED
                state.error = "No audio files generated."
                return state
            
            from pydub import AudioSegment
            
            combined = AudioSegment.empty()
            chapters = []
            current_time_ms = 0
            
            for i, (chunk, audio_path) in enumerate(zip(state.chunks, state.audio_files)):
                segment = AudioSegment.from_file(audio_path, format="wav")
                duration_ms = len(segment)
                
                # Each chunk corresponds to a section (or part of one)
                # If it's the first chunk of a new section, create a chapter
                # Note: Our new chunking logic forces a new chunk for each section boundary.
                title = ", ".join(chunk.section_names) if chunk.section_names else f"Chapter {i+1}"
                
                # If this chunk's title is different from last, or it's the first, add chapter
                if not chapters or chapters[-1].title != title:
                    # If last chapter was part of same section, we don't necessarily want a new chapter,
                    # but usually paper sections are distinct.
                    chapters.append(ChapterInfo(
                        title=title,
                        start_ms=current_time_ms,
                        end_ms=current_time_ms + duration_ms
                    ))
                else:
                    # Extend last chapter
                    chapters[-1].end_ms += duration_ms
                
                combined += segment
                current_time_ms += duration_ms
            
            state.chapters = chapters
            
            # Determine output format and path
            output_format = getattr(self.config, 'OUTPUT_FORMAT', 'm4b').lower()
            output_dir = os.path.dirname(state.audio_files[0])
            
            if output_format == "mp3":
                output_path = os.path.join(output_dir, "combined.mp3")
                bitrate = getattr(self.config, 'MP3_BITRATE', '128k')
                combined.export(output_path, format="mp3", bitrate=bitrate)
                state.status_message = f"Exported MP3 at {bitrate}"
            elif output_format == "wav":
                output_path = os.path.join(output_dir, "combined.wav")
                combined.export(output_path, format="wav")
                state.status_message = "Exported WAV file"
            else:  # m4b (default) - keep as WAV for now, M4B builder will handle it
                output_path = os.path.join(output_dir, "concatenated_temp.wav")
                combined.export(output_path, format="wav")
                state.status_message = "Audio concatenated, ready for M4B packaging"
            
            state.final_output = output_path
            state.status_message = f"Audio concatenation complete: {len(chapters)} chapters identified"
            
        except Exception as e:
            state.status = PipelineStatus.FAILED
            state.error = f"Audio concatenation failed: {str(e)}"
        
        return state
    
    async def packaging_m4b(self, state: PipelineState) -> PipelineState:
        """Task 10: Package as M4B audiobook with chapters and metadata."""
        if state.status == PipelineStatus.FAILED: return state
        state.status_message = "Packaging as M4B audiobook..."
        
        try:
            if not state.final_output:
                state.status = PipelineStatus.FAILED
                state.error = "No audio file available for M4B packaging."
                return state
            
            import subprocess
            import tempfile
            import os
            
            # 1. Prepare Metadata File (FFMETADATA1)
            title = "Unknown Paper"
            artist = "PaperNarrator AI"
            
            # Try to extract title from content or filename
            if state.cleaned_sections:
                # Often the first section might be the title if extraction worked well
                if "title" in state.cleaned_sections[0].title.lower():
                    title = state.cleaned_sections[0].content[:100].strip()
                else:
                    title = state.cleaned_sections[0].title
            
            if not title or title == "Unknown":
                if state.temp_path:
                    title = os.path.basename(state.temp_path).replace(".pdf", "").replace("_", " ")
            
            metadata = [";FFMETADATA1", f"title={title}", f"artist={artist}", "album=PaperNarrator Audiobooks", "genre=Audiobook"]
            
            for chapter in state.chapters:
                metadata.append("[CHAPTER]")
                metadata.append("TIMEBASE=1/1000")
                metadata.append(f"START={chapter.start_ms}")
                metadata.append(f"END={chapter.end_ms}")
                metadata.append(f"title={chapter.title}")
            
            metadata_content = "\n".join(metadata)
            
            # 2. Write metadata to temp file
            output_dir = os.path.dirname(state.final_output)
            metadata_path = os.path.join(output_dir, "metadata.txt")
            with open(metadata_path, "w", encoding="utf-8") as f:
                f.write(metadata_content)
            
            # 3. Run FFmpeg to create M4B
            # We convert the concatenated WAV to AAC (m4a/m4b)
            final_m4b_path = os.path.join(output_dir, "audiobook.m4b")
            
            # ffmpeg -i input.wav -i metadata.txt -map_metadata 1 -c:a aac -b:a 128k output.m4b
            bitrate = getattr(self.config, 'MP3_BITRATE', '128k')
            cmd = [
                "ffmpeg", "-y",
                "-i", state.final_output,
                "-i", metadata_path,
                "-map_metadata", "1",
                "-c:a", "aac",
                "-b:a", bitrate,
                final_m4b_path
            ]
            
            # Run command
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise Exception(f"FFmpeg failed: {process.stderr}")
            
            # 4. Cleanup
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            if state.final_output and os.path.exists(state.final_output) and "temp" in state.final_output:
                os.remove(state.final_output)
            
            state.final_output = final_m4b_path
            state.status_message = f"M4B audiobook created: {len(state.chapters)} chapters embedded"
            state.status = PipelineStatus.COMPLETED
            
        except Exception as e:
            state.status = PipelineStatus.FAILED
            state.error = f"M4B packaging failed: {str(e)}"
        
        return state
    
    async def finalize_run(self, state: PipelineState) -> PipelineState:
        """Task 11: Collect all artifacts and finalize the run trail."""
        if not state.run_dir: return state
        
        state.status_message = "Finalizing run and collecting artifacts..."
        
        try:
            import json
            import shutil
            
            # 1. Save original text
            if state.raw_text:
                with open(os.path.join(state.run_dir, "original_text.txt"), "w", encoding="utf-8") as f:
                    f.write(state.raw_text)
            
            # 2. Save cleaned text
            if state.cleaned_sections:
                cleaned_full = "\n\n".join([f"# {s.title}\n{s.content}" for s in state.cleaned_sections])
                with open(os.path.join(state.run_dir, "cleaned_text.md"), "w", encoding="utf-8") as f:
                    f.write(cleaned_full)
            
            # 3. Save chunks (text)
            if state.chunks:
                os.makedirs(os.path.join(state.run_dir, "text_chunks"), exist_ok=True)
                for i, chunk in enumerate(state.chunks):
                    with open(os.path.join(state.run_dir, "text_chunks", f"chunk_{i+1:03d}.txt"), "w", encoding="utf-8") as f:
                        f.write(chunk.text)
            
            # 4. Save chapters metadata
            if state.chapters:
                chapters_data = [c.model_dump() for c in state.chapters]
                with open(os.path.join(state.run_dir, "chapters.json"), "w", encoding="utf-8") as f:
                    json.dump(chapters_data, f, indent=2)
            
            # 5. Copy final output to run_dir (if not already there)
            if state.final_output and os.path.exists(state.final_output):
                final_dest = os.path.join(state.run_dir, os.path.basename(state.final_output))
                if os.path.abspath(state.final_output) != os.path.abspath(final_dest):
                    shutil.copy2(state.final_output, final_dest)
                    state.final_output = final_dest
            
            # 6. Save full state as JSON
            state_json = state.model_dump()
            with open(os.path.join(state.run_dir, "pipeline_state.json"), "w", encoding="utf-8") as f:
                json.dump(state_json, f, indent=2, default=str)
                
            state.status_message = f"Run finalized. Artifacts saved to {state.run_id}"
            
        except Exception as e:
            # Assumes logger is available in scope or needs to be imported/defined
            print(f"Failed to finalize run: {str(e)}")
            state.status_message = f"Warning: Failed to collect all artifacts: {str(e)}"
            
        return state