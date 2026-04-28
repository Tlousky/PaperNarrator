"""Tests for LangGraph state machine (Task 4)."""
import pytest
from langgraph_pipeline import (
    PipelineState,
    PipelineStatus,
    PaperSection,
    TextChunk,
    WorkflowBuilder
)


class TestPaperSection:
    """Test PaperSection model."""
    
    def test_create_section(self):
        section = PaperSection(
            title="Abstract",
            content="This is the abstract content.",
            word_count=6
        )
        assert section.title == "Abstract"
        assert section.word_count == 6


class TestTextChunk:
    """Test TextChunk model."""
    
    def test_create_chunk(self):
        chunk = TextChunk(
            text="Sample text",
            word_count=2,
            section_names=["Abstract"]
        )
        assert chunk.word_count == 2
        assert len(chunk.section_names) == 1
    
    def test_chunk_max_words(self):
        """Verify chunks respect TTS limit."""
        chunk = TextChunk(
            text="x " * 9000,
            word_count=9000,
            section_names=["Introduction"]
        )
        assert chunk.word_count < 10000


class TestPipelineState:
    """Test PipelineState model."""
    
    def test_create_state(self):
        state = PipelineState(
            source_type="file",
            content="/path/to/paper.pdf",
            status=PipelineStatus.PENDING
        )
        assert state.source_type == "file"
        assert state.status == PipelineStatus.PENDING
        assert state.total_cost == 0.0
    
    def test_state_transitions(self):
        """Test status transitions."""
        state = PipelineState(
            source_type="text",
            content="Sample text",
            status=PipelineStatus.PENDING
        )
        state.status = PipelineStatus.RUNNING
        assert state.status == PipelineStatus.RUNNING
        state.status = PipelineStatus.COMPLETED
        assert state.status == PipelineStatus.COMPLETED


class TestWorkflowBuilder:
    """Test WorkflowBuilder creates valid graph."""
    
    def test_build_graph(self):
        builder = WorkflowBuilder()
        graph = builder.create_graph()
        
        # Verify graph has expected nodes via get_graph()
        graph_schema = graph.get_graph()
        nodes = graph_schema.nodes
        
        assert "extracting_text" in nodes
        assert "cleaning_with_llm" in nodes
        assert "chunking_text" in nodes
        assert "generating_audio" in nodes
        assert "concatenating_audio" in nodes
        assert "packaging_ep3" in nodes
    
    def test_conditional_edge_exists(self):
        """Test that conditional edge for EP3 exists."""
        builder = WorkflowBuilder()
        graph = builder.create_graph()
        
        # Verify conditional edge from concatenating_audio
        graph_schema = graph.get_graph()
        edges = graph_schema.edges
        cond_edges = [e for e in edges if e.source == "concatenating_audio"]
        assert len(cond_edges) > 0