from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Optional
from langchain_core.tools import tool

from .state import PipelineState, PipelineStatus, PaperSection, TextChunk
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
        workflow.add_node("extracting_text", self.extracting_text)
        workflow.add_node("describing_figures", self.describing_figures)
        workflow.add_node("cleaning_with_llm", self.cleaning_with_llm)
        workflow.add_node("chunking_text", self.chunking_text)
        workflow.add_node("generating_audio", self.generating_audio)
        workflow.add_node("concatenating_audio", self.concatenating_audio)
        workflow.add_node("packaging_ep3", self.packaging_ep3)
        
        # Set entry point
        workflow.set_entry_point("extracting_text")
        
        # Set conditional edges based on OUTPUT_FORMAT
        workflow.add_conditional_edges(
            "concatenating_audio",
            self.should_package_as_ep3,
            {
                "ep3": "packaging_ep3",
                "audio": END
            }
        )
        
        # Set standard edges
        workflow.add_edge("extracting_text", "describing_figures")
        workflow.add_edge("describing_figures", "cleaning_with_llm")
        workflow.add_edge("cleaning_with_llm", "chunking_text")
        workflow.add_edge("chunking_text", "generating_audio")
        workflow.add_edge("generating_audio", "concatenating_audio")
        workflow.add_edge("packaging_ep3", END)
        
        self.graph = workflow.compile()
        return self.graph
    
    def should_package_as_ep3(self, state: PipelineState) -> str:
        """Conditional edge: route to EP3 packaging if OUTPUT_FORMAT is ep3."""
        output_format = getattr(self.config, 'OUTPUT_FORMAT', 'ep3').lower()
        return "ep3" if output_format == "ep3" else "audio"
    
    async def extracting_text(self, state: PipelineState) -> PipelineState:
        """Task 5: Extract text from PDF (URL or file)."""
        state.status_message = "Extracting text from PDF..."
        
        pdf_path = None
        
        try:
            # Handle URL input
            if state.source_type == "url":
                state.status_message = "Downloading PDF..."
                response = requests.get(state.content, timeout=30)
                response.raise_for_status()
                
                # Save to temp file
                with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
                    f.write(response.content)
                    pdf_path = f.name
                state.temp_path = pdf_path
            elif state.source_type == "file":
                pdf_path = state.temp_path
            else:
                state.status = PipelineStatus.FAILED
                state.error = "Invalid source_type. Must be 'url' or 'file'."
                return state
            
            # Extract text
            state.status_message = "Extracting text..."
            raw_text = await _extract_pdf_text(pdf_path)
            
            # Check for scanned PDF (empty or very short text)
            if not raw_text or len(raw_text.strip()) < 50:
                state.status = PipelineStatus.FAILED
                state.error = "No text extracted. PDF may be image-only (scanned). Please use OCR or provide text PDF."
                return state
            
            state.raw_text = raw_text
            state.status_message = f"Extracted {len(raw_text)} characters"
            
        except Exception as e:
            state.status = PipelineStatus.FAILED
            state.error = f"Failed to process PDF: {str(e)}"
        
        return state
    
    async def describing_figures(self, state: PipelineState) -> PipelineState:
        """VLM (Vision Language Model) to describe figures if enabled."""
        state.status_message = "Describing figures..."
        # TODO: Implement VLM if VLM_ENABLED
        return state
    
    async def cleaning_with_llm(self, state: PipelineState) -> PipelineState:
        """Task 6: LLM-based cleaning (extract sections, remove citations, smooth for TTS)."""
        state.status_message = "Cleaning text with LLM..."
        
        try:
            if not state.raw_text:
                state.status = PipelineStatus.FAILED
                state.error = "No raw text available. Previous extraction step may have failed."
                return state
            
            # Extract sections
            state.status_message = "Extracting sections..."
            sections_raw = await _extract_sections(state.raw_text)
            
            if not sections_raw:
                state.status = PipelineStatus.FAILED
                state.error = "No sections extracted from text."
                return state
            
            # Clean each section
            state.status_message = "Removing citations and metadata..."
            cleaned_sections = []
            total_words = 0
            
            for sec in sections_raw:
                title = sec["title"]
                content = sec["content"]
                
                # Apply cleaning pipeline
                content = await _remove_citations(content)
                content = await _remove_metadata(content)
                content = await _smooth_for_tts(content)
                
                word_count = len(content.split())
                total_words += word_count
                
                cleaned_sections.append(PaperSection(
                    title=title,
                    content=content,
                    word_count=word_count
                ))
            
            state.cleaned_sections = cleaned_sections
            state.total_words = total_words
            state.status_message = f"Cleaned {len(cleaned_sections)} sections ({total_words} words)"
            
        except Exception as e:
            state.status = PipelineStatus.FAILED
            state.error = f"Failed to clean text: {str(e)}"
        
        return state
    
    async def chunking_text(self, state: PipelineState) -> PipelineState:
        """Task 7: Section-aware chunking for TTS (max 8500 words per chunk)."""
        state.status_message = "Chunking text for TTS..."
        
        try:
            if not state.cleaned_sections:
                state.status = PipelineStatus.FAILED
                state.error = "No cleaned sections available."
                return state
            
            MAX_WORDS = 8500  # ~60 min at 150 wpm
            chunks = []
            current_chunk_text = []
            current_word_count = 0
            current_sections = []
            
            def save_chunk():
                """Helper to save current chunk and reset."""
                nonlocal current_chunk_text, current_word_count, current_sections
                if current_chunk_text:
                    chunks.append(TextChunk(
                        text="\n\n".join(current_chunk_text),
                        word_count=current_word_count,
                        section_names=current_sections.copy(),
                        chunk_id=f"{current_sections[0]}_chunk_{len(chunks)+1}"
                    ))
                current_chunk_text = []
                current_word_count = 0
                current_sections = []
            
            for section in state.cleaned_sections:
                section_word_count = section.word_count
                section_text = section.content
                
                if section_word_count <= MAX_WORDS:
                    # Section fits - try to add to current chunk
                    if current_word_count + section_word_count <= MAX_WORDS:
                        current_chunk_text.append(section_text)
                        current_word_count += section_word_count
                        if section.title not in current_sections:
                            current_sections.append(section.title)
                    else:
                        # Save current, start new with this section
                        save_chunk()
                        current_chunk_text = [section_text]
                        current_word_count = section_word_count
                        current_sections = [section.title]
                else:
                    # Section too large - split by paragraphs
                    paragraphs = [p.strip() for p in section_text.split('\n\n') if p.strip()]
                    
                    for para in paragraphs:
                        para_words = len(para.split())
                        
                        if para_words <= MAX_WORDS:
                            # Paragraph fits - try to add
                            if current_word_count + para_words <= MAX_WORDS:
                                current_chunk_text.append(para)
                                current_word_count += para_words
                            else:
                                save_chunk()
                                current_chunk_text = [para]
                                current_word_count = para_words
                            if section.title not in current_sections:
                                current_sections.append(section.title)
                        else:
                            # Paragraph too large - split by sentences (preserving punctuation)
                            import re
                            sentences = re.split(r'(?<=[.!?])\s+', para)
                            sentences = [s.strip() for s in sentences if s.strip()]
                            
                            for sent in sentences:
                                sent_words = len(sent.split())
                                
                                if sent_words <= MAX_WORDS:
                                    if current_word_count + sent_words <= MAX_WORDS:
                                        current_chunk_text.append(sent)
                                        current_word_count += sent_words
                                    else:
                                        save_chunk()
                                        current_chunk_text = [sent]
                                        current_word_count = sent_words
                                    if section.title not in current_sections:
                                        current_sections.append(section.title)
                                else:
                                    # Sentence too large - split by words
                                    words = sent.split()
                                    
                                    for word in words:
                                        if current_word_count + 1 <= MAX_WORDS:
                                            if current_chunk_text:
                                                current_chunk_text[-1] += " " + word
                                            else:
                                                current_chunk_text = [word]
                                            current_word_count += 1
                                        else:
                                            save_chunk()
                                            current_chunk_text = [word]
                                            current_word_count = 1
                                        if section.title not in current_sections:
                                            current_sections.append(section.title)
            
            # Last chunk
            save_chunk()
            
            state.chunks = chunks
            state.status_message = f"Created {len(chunks)} chunks ({sum(c.word_count for c in chunks)} total words)"
            
        except Exception as e:
            state.status = PipelineStatus.FAILED
            state.error = f"Failed to chunk text: {str(e)}"
        
        return state
    
    async def generating_audio(self, state: PipelineState) -> PipelineState:
        """Task 8: VibeVoice TTS generation for each chunk."""
        state.status_message = "Generating audio with VibeVoice..."
        
        try:
            if not state.chunks:
                state.status = PipelineStatus.FAILED
                state.error = "No chunks available for TTS."
                return state
            
            from tts.vibevoice import VibeVoiceTTS
            import tempfile
            import os
            
            # Initialize TTS (lazy loads model)
            tts = VibeVoiceTTS(
                model_name=os.getenv("VIBEVOICE_MODEL_PATH", "./models/microsoft/VibeVoice-Realtime-0.5B"),
                device=os.getenv("VIBEVOICE_DEVICE", "cuda"),
                speaker_name=os.getenv("VIBEVOICE_SPEAKER", "Carter"),
                cfg_scale=float(os.getenv("VIBEVOICE_CFG_SCALE", 1.5))
            )
            
            audio_files = []
            temp_dir = tempfile.mkdtemp(prefix="papernarrator_tts_")
            
            for i, chunk in enumerate(state.chunks):
                chunk_num = i + 1
                total_chunks = len(state.chunks)
                
                state.status_message = f"Generating audio {chunk_num}/{total_chunks} ({chunk.word_count} words)..."
                
                # Generate audio for this chunk
                output_path = os.path.join(temp_dir, f"chunk_{chunk_num:03d}.wav")
                
                result_path = tts.generate_audio(
                    text=chunk.text,
                    output_path=output_path,
                    word_count=chunk.word_count
                )
                
                if os.path.exists(result_path):
                    audio_files.append(result_path)
                else:
                    state.status = PipelineStatus.FAILED
                    state.error = f"TTS failed for chunk {chunk_num}: {chunk.chunk_id}"
                    return state
            
            state.audio_files = audio_files
            state.temp_path = temp_dir  # Store temp dir for cleanup later
            state.status_message = f"Generated {len(audio_files)} audio files"
            state.status = PipelineStatus.COMPLETED
            
        except Exception as e:
            state.status = PipelineStatus.FAILED
            state.error = f"TTS generation failed: {str(e)}"
        
        return state
    
    async def concatenating_audio(self, state: PipelineState) -> PipelineState:
        """Task 9: Concatenate audio files and convert to output format."""
        state.status_message = "Concatenating audio..."
        
        try:
            if not state.audio_files:
                state.status = PipelineStatus.FAILED
                state.error = "No audio files to concatenate."
                return state
            
            from pydub import AudioSegment
            import os
            import shutil
            
            # Concatenate all WAV files
            state.status_message = f"Concatenating {len(state.audio_files)} audio segments..."
            
            # Start with first file
            combined = AudioSegment.from_file(state.audio_files[0], format="wav")
            
            # Append remaining files
            for i, audio_path in enumerate(state.audio_files[1:], 2):
                segment = AudioSegment.from_file(audio_path, format="wav")
                combined += segment
            
            # Determine output format and path
            output_format = getattr(self.config, 'OUTPUT_FORMAT', 'ep3').lower()
            output_dir = os.path.dirname(state.audio_files[0])
            
            if output_format == "mp3":
                output_path = os.path.join(output_dir, "combined.mp3")
                bitrate = getattr(self.config, 'MP3_BITRATE', '128kbps')
                combined.export(output_path, format="mp3", bitrate=bitrate)
                state.status_message = f"Exported MP3 at {bitrate}"
            elif output_format == "wav":
                output_path = os.path.join(output_dir, "combined.wav")
                combined.export(output_path, format="wav")
                state.status_message = "Exported WAV file"
            else:  # ep3 - keep as WAV for now, EP3 builder will handle it
                output_path = os.path.join(output_dir, "combined.wav")
                combined.export(output_path, format="wav")
                state.status_message = "Exported WAV for EP3 packaging"
            
            state.final_output = output_path
            state.status_message = f"Audio concatenation complete: {os.path.basename(output_path)}"
            state.status = PipelineStatus.COMPLETED
            
        except Exception as e:
            state.status = PipelineStatus.FAILED
            state.error = f"Audio concatenation failed: {str(e)}"
        
        return state
    
    async def packaging_ep3(self, state: PipelineState) -> PipelineState:
        """Task 10: Package as EP3 (EPUB 3 with Media Overlays)."""
        state.status_message = "Packaging as EP3 audiobook..."
        # TODO: Implement in Task 10
        return state