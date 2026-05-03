from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum


class PipelineStatus(str, Enum):
    """Status of the pipeline execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PaperSection(BaseModel):
    """Represents a section of a scientific paper (Abstract, Methods, etc.)."""
    title: str = Field(..., description="Section title (e.g., 'Abstract', 'Introduction')")
    content: str = Field(..., description="Cleaned text content of the section")
    word_count: int = Field(0, description="Number of words in this section")


class TextChunk(BaseModel):
    """Represents a chunk of text optimized for TTS (max ~8500 words)."""
    text: str = Field(..., description="The text content for TTS generation")
    word_count: int = Field(..., description="Word count (must be < 10000)")
    section_names: List[str] = Field(default_factory=list, description="List of section titles included in this chunk")
    chunk_id: Optional[str] = Field(None, description="Unique identifier for this chunk")
    
    @field_validator("word_count")
    @classmethod
    def validate_word_count(cls, v):
        if v > 10000:
            raise ValueError(f"Word count {v} exceeds maximum allowed 10000 words")
        return v


class ChapterInfo(BaseModel):
    """Represents a chapter in the final M4B audiobook."""
    title: str = Field(..., description="Chapter title")
    start_ms: int = Field(..., description="Start time in milliseconds")
    end_ms: int = Field(..., description="End time in milliseconds")


class PipelineState(BaseModel):
    """State of the LangGraph pipeline."""
    # Input
    source_type: str = Field(..., description="Input type: 'url', 'file', or 'text'")
    content: str = Field(..., description="The actual content (URL, file path, or raw text)")
    temp_path: Optional[str] = Field(None, description="Temporary file path if downloaded or uploaded")
    voice_profile: str = Field(default="Emma", description="Voice profile to use for TTS")
    
    # Processing stages
    raw_text: Optional[str] = Field(None, description="Raw extracted text from PDF")
    cleaned_sections: Optional[List[PaperSection]] = Field(None, description="Extracted and cleaned sections")
    chunks: Optional[List[TextChunk]] = Field(None, description="Text chunks ready for TTS")
    audio_files: List[str] = Field(default_factory=list, description="Paths to generated audio files (WAV)")
    chapters: List[ChapterInfo] = Field(default_factory=list, description="List of chapters with timestamps")
    
    # Output
    final_output: Optional[str] = Field(None, description="Path to final output file (M4B/MP3/WAV)")
    
    # Metadata
    run_id: Optional[str] = Field(None, description="Unique ID for this run (timestamp-based)")
    session_id: Optional[str] = Field(None, description="Session ID for grouping traces in Langfuse")
    user_id: Optional[str] = Field(None, description="User ID for tracking usage in Langfuse")
    tags: List[str] = Field(default_factory=list, description="Tags for categorizing traces in Langfuse")
    version: Optional[str] = Field(None, description="Application version for Langfuse tracing")
    run_dir: Optional[str] = Field(None, description="Directory for intermediate and final outputs")
    total_cost: float = Field(0.0, description="Total LLM API cost in USD")
    status: PipelineStatus = Field(default=PipelineStatus.PENDING, description="Current pipeline status")
    error: Optional[str] = Field(None, description="Error message if failed")
    status_message: Optional[str] = Field(None, description="Human-readable status update")
    total_words: Optional[int] = Field(None, description="Total word count after cleaning")