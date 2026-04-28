from langgraph.graph import StateGraph, END
from langgraph.graph.message import BaseMessage
from typing import TypedDict, Annotated, List, Optional
from langchain_core.tools import tool

from .state import PipelineState, PipelineStatus, PaperSection, TextChunk
from .tools import _extract_pdf_text
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
    
    # Node implementations (stubs for now, will be filled in Tasks 5-7)
    
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
        # TODO: Implement in Task 6
        # - Extract sections (Abstract, Introduction, etc.)
        # - Remove citations [1], (Author, 2023)
        # - Remove metadata (Keywords, References)
        # - Smooth for TTS (expand "et al.", "Fig.")
        return state
    
    async def chunking_text(self, state: PipelineState) -> PipelineState:
        """Task 7: Section-aware chunking for TTS (max 8500 words per chunk)."""
        state.status_message = "Chunking text for TTS..."
        # TODO: Implement in Task 7
        # - Greedy packing algorithm
        # - Never split sections across chunks
        # - Handle sections > 8500 words by splitting paragraphs
        return state
    
    async def generating_audio(self, state: PipelineState) -> PipelineState:
        """Task 8: VibeVoice TTS generation for each chunk."""
        state.status_message = "Generating audio with VibeVoice..."
        # TODO: Implement in Task 8
        return state
    
    async def concatenating_audio(self, state: PipelineState) -> PipelineState:
        """Task 9: Concatenate audio files and convert to output format."""
        state.status_message = "Concatenating audio..."
        # TODO: Implement in Task 9
        # - Use pydub to concatenate WAVs
        # - Convert to MP3 if needed
        return state
    
    async def packaging_ep3(self, state: PipelineState) -> PipelineState:
        """Task 10: Package as EP3 (EPUB 3 with Media Overlays)."""
        state.status_message = "Packaging as EP3 audiobook..."
        # TODO: Implement in Task 10
        return state