"""Tests for text chunking (Task 7)."""
import pytest
from langgraph_pipeline.state import PipelineState, PaperSection, TextChunk
from langgraph_pipeline.workflow import WorkflowBuilder


class TestChunkingTextNode:
    """Test the chunking_text node."""
    
    @pytest.mark.asyncio
    async def test_simple_chunking(self):
        """Test basic chunking with small sections."""
        builder = WorkflowBuilder()
        
        state = PipelineState(
            source_type="file",
            content="test.pdf",
            cleaned_sections=[
                PaperSection(title="Abstract", content="This is the abstract text with some words.", word_count=10),
                PaperSection(title="Introduction", content="This is the introduction text with more words here.", word_count=12),
                PaperSection(title="Methods", content="This is the methods section with additional content.", word_count=10)
            ]
        )
        
        result = await builder.chunking_text(state)
        
        assert result.chunks is not None
        assert len(result.chunks) >= 1
        assert all(c.word_count <= 8500 for c in result.chunks)
    
    @pytest.mark.asyncio
    async def test_section_split_across_chunks(self):
        """Test that sections are split when they exceed MAX_WORDS."""
        builder = WorkflowBuilder()
        
        # Create a very long section with actual sentences
        sentence = "This is a test sentence with some words. "
        long_text = sentence * 500  # ~5000 words but with sentence boundaries
        
        state = PipelineState(
            source_type="file",
            content="test.pdf",
            cleaned_sections=[
                PaperSection(title="Abstract", content="Short abstract.", word_count=3),
                PaperSection(title="Introduction", content=long_text, word_count=len(long_text.split())),
                PaperSection(title="Methods", content="Short methods.", word_count=3)
            ]
        )
        
        result = await builder.chunking_text(state)
        
        assert result.chunks is not None
        # No chunk should exceed limit
        assert all(c.word_count <= 8500 for c in result.chunks)
        # Total words should be preserved
        assert sum(c.word_count for c in result.chunks) == len(long_text.split()) + 6
    
    @pytest.mark.asyncio
    async def test_greedy_packing(self):
        """Test that multiple small sections are packed into one chunk."""
        builder = WorkflowBuilder()
        
        small_sections = [
            PaperSection(title=f"Section{i}", content=f"Content of section {i} with 20 words here.", word_count=20)
            for i in range(400)  # 400 * 20 = 8000 words, fits in one chunk
        ]
        
        state = PipelineState(
            source_type="file",
            content="test.pdf",
            cleaned_sections=small_sections
        )
        
        result = await builder.chunking_text(state)
        
        assert result.chunks is not None
        # Should fit in 1 chunk (8000 words < 8500)
        assert len(result.chunks) == 1
        assert result.chunks[0].word_count == 8000
    
    @pytest.mark.asyncio
    async def test_no_sections(self):
        """Test handling of empty cleaned_sections."""
        builder = WorkflowBuilder()
        
        state = PipelineState(
            source_type="file",
            content="test.pdf",
            cleaned_sections=[]
        )
        
        result = await builder.chunking_text(state)
        
        assert result.status.value == "failed"
        assert "No cleaned sections" in result.error
    
    @pytest.mark.asyncio
    async def test_paragraph_splitting(self):
        """Test that large sections are split by paragraphs first."""
        builder = WorkflowBuilder()
        
        # Create section with 3 paragraphs of 4000 words each
        para = "Word " * 4000
        long_section = f"{para}\n\n{para}\n\n{para}"
        
        state = PipelineState(
            source_type="file",
            content="test.pdf",
            cleaned_sections=[
                PaperSection(title="Abstract", content="Short.", word_count=2),
                PaperSection(title="Methods", content=long_section, word_count=12000)
            ]
        )
        
        result = await builder.chunking_text(state)
        
        assert result.chunks is not None
        # Greedy packing: Abstract + para1 + para2 (8002), para3 (4000) = 2 chunks
        assert len(result.chunks) == 2
        assert all(c.word_count <= 8500 for c in result.chunks)
        # Verify word counts
        assert result.chunks[0].word_count == 8002
        assert result.chunks[1].word_count == 4000