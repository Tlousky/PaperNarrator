"""Tests for PDF extraction (Task 5)."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langgraph_pipeline.tools import (
    _extract_pdf_text,
    _extract_figures,
    _extract_sections,
    _remove_citations,
    _remove_metadata,
    _smooth_for_tts
)
from langgraph_pipeline.state import PipelineState, PipelineStatus
from langgraph_pipeline.workflow import WorkflowBuilder
import tempfile
import os


class TestExtractPDFText:
    """Test PDF text extraction."""
    
    @pytest.mark.asyncio
    async def test_extract_text_from_pdf(self):
        """Test extraction creates temp PDF and extracts text."""
        # Mock fitz.open since fitz is imported inside the function
        with patch('fitz.open', new_callable=MagicMock) as mock_open:
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_page.get_text.return_value = "Test content from PDF that is long enough to pass validation"
            mock_doc.__iter__ = lambda self: iter([mock_page])
            mock_doc.close = MagicMock()
            mock_open.return_value = mock_doc
            
            result = await _extract_pdf_text("fake.pdf")
            assert "Test content from PDF" in result


class TestExtractSections:
    """Test section extraction."""
    
    @pytest.mark.asyncio
    async def test_extract_abstract(self):
        text = """Abstract
        This is the abstract content.
        
        Introduction
        This is the introduction.
        """
        sections = await _extract_sections(text)
        abstracts = [s for s in sections if s["title"] == "Abstract"]
        assert len(abstracts) == 1
        assert "abstract content" in abstracts[0]["content"].lower()
    
    @pytest.mark.asyncio
    async def test_extract_introduction(self):
        text = """Introduction
        This paper presents a new method.
        
        Methods
        We used the following approach.
        """
        sections = await _extract_sections(text)
        intros = [s for s in sections if s["title"] == "Introduction"]
        assert len(intros) == 1


class TestRemoveCitations:
    """Test citation removal."""
    
    @pytest.mark.asyncio
    async def test_remove_bracket_citations(self):
        text = "This is a test [1] with citations [2-5] and more [10]."
        result = await _remove_citations(text)
        assert "[1]" not in result
        assert "[2-5]" not in result
        assert "[10]" not in result
    
    @pytest.mark.asyncio
    async def test_remove_author_year_citations(self):
        text = "Smith et al. (Smith, 2023) showed that (Johnson et al., 2024) this works."
        result = await _remove_citations(text)
        assert "(Smith, 2023)" not in result
        assert "(Johnson et al., 2024)" not in result


class TestRemoveMetadata:
    """Test metadata removal."""
    
    @pytest.mark.asyncio
    async def test_remove_keywords(self):
        text = "Some text.\nKeywords: machine learning, AI, deep learning.\nMore text."
        result = await _remove_metadata(text)
        assert "Keywords:" not in result
        assert "machine learning" not in result


class TestSmoothForTTS:
    """Test TTS smoothing."""
    
    @pytest.mark.asyncio
    async def test_expand_et_al(self):
        text = "Smith et al. showed that..."
        result = await _smooth_for_tts(text)
        assert "and colleagues" in result
    
    @pytest.mark.asyncio
    async def test_expand_fig(self):
        text = "See Fig. 1 for details."
        result = await _smooth_for_tts(text)
        assert "Figure 1" in result
    
    @pytest.mark.asyncio
    async def test_expand_vs(self):
        text = "Model A vs Model B"
        result = await _smooth_for_tts(text)
        assert "versus" in result


class TestExtractingTextNode:
    """Test the extracting_text node in workflow."""
    
    @pytest.mark.asyncio
    async def test_extract_from_url(self):
        """Test URL download and extraction."""
        builder = WorkflowBuilder()
        
        # Mock where functions are used (imported into workflow module)
        with patch('langgraph_pipeline.workflow.requests') as mock_requests, \
             patch('langgraph_pipeline.workflow._extract_pdf_text', new_callable=AsyncMock) as mock_extract:
            
            # Setup mocks
            mock_response = MagicMock()
            mock_response.content = b"fake pdf content"
            mock_response.raise_for_status = MagicMock()
            mock_requests.get.return_value = mock_response
            mock_extract.return_value = "Extracted text content here. This is a longer text that contains more than fifty characters so it will pass the validation check in the workflow node and not be treated as an empty or scanned PDF document."
            
            # Create state
            state = PipelineState(
                source_type="url",
                content="https://example.com/paper.pdf"
            )
            
            # Run node
            result = await builder.extracting_text(state)
            
            assert "Extracted text content here" in result.raw_text
            assert result.status_message.startswith("Extracted")
            assert os.path.exists(result.temp_path)
            
            # Cleanup
            os.unlink(result.temp_path)
    
    @pytest.mark.asyncio
    async def test_extract_from_file(self):
        """Test file upload extraction."""
        builder = WorkflowBuilder()
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as f:
            f.write("test pdf")
            temp_path = f.name
        
        try:
            with patch('langgraph_pipeline.workflow._extract_pdf_text', new_callable=AsyncMock) as mock_extract:
                mock_extract.return_value = "File content extracted from the PDF document. This is a longer text that contains more than fifty characters so it will pass the validation check in the workflow node and not be treated as an empty or scanned PDF document."
                
                state = PipelineState(
                    source_type="file",
                    content="test.pdf",
                    temp_path=temp_path
                )
                
                result = await builder.extracting_text(state)
                
                assert "File content extracted" in result.raw_text
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_empty_text_raises_error(self):
        """Test that scanned PDF detection works."""
        builder = WorkflowBuilder()
        
        with patch('langgraph_pipeline.workflow._extract_pdf_text', new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = "   "  # Empty/whitespace
            
            state = PipelineState(
                source_type="file",
                content="test.pdf",
                temp_path="fake.pdf"
            )
            
            result = await builder.extracting_text(state)
            
            assert result.status == PipelineStatus.FAILED
            assert "scanned" in result.error.lower() or "no text" in result.error.lower()