"""Tests for packaging_ep3 workflow node."""

import pytest
import asyncio
import os
import zipfile
from unittest.mock import MagicMock, patch

from langgraph_pipeline.state import PipelineState, PipelineStatus
from langgraph_pipeline.workflow import WorkflowBuilder


class TestPackagingEP3:
    """Tests for the packaging_ep3 node (Task 10)."""
    
    @pytest.fixture
    def workflow_builder(self):
        """Create WorkflowBuilder with config set to EP3."""
        config = MagicMock()
        config.OUTPUT_FORMAT = "ep3"
        return WorkflowBuilder(config=config)
    
    @pytest.fixture
    def state_with_audio(self, tmp_path):
        """Create state with audio file and sections."""
        # Create a fake audio file
        audio_path = str(tmp_path / "combined.wav")
        with open(audio_path, 'wb') as f:
            f.write(b'RIFF')
            f.write((36 + 24000).to_bytes(4, 'little'))
            f.write(b'WAVEfmt ')
            f.write((16).to_bytes(4, 'little'))
            f.write((1).to_bytes(2, 'little'))
            f.write((1).to_bytes(2, 'little'))
            f.write((24000).to_bytes(4, 'little'))
            f.write((48000).to_bytes(4, 'little'))
            f.write((2).to_bytes(2, 'little'))
            f.write((16).to_bytes(2, 'little'))
            f.write(b'data')
            f.write((24000).to_bytes(4, 'little'))
            f.write(b'\x00' * 24000)
        
        # Create PaperSection objects
        from langgraph_pipeline.state import PaperSection
        sections = [
            PaperSection(title="Abstract", content="Test abstract.", word_count=2),
            PaperSection(title="Introduction", content="Test intro.", word_count=2)
        ]
        
        return PipelineState(
            source_type="file",
            content="",
            final_output=audio_path,
            cleaned_sections=sections
        )
    
    @pytest.mark.asyncio
    async def test_packages_ep3_successfully(self, workflow_builder, state_with_audio):
        """Test that EP3 packaging succeeds and creates valid file."""
        result = await workflow_builder.packaging_ep3(state_with_audio)
        
        assert result.status == PipelineStatus.COMPLETED
        assert result.final_output is not None
        assert result.final_output.endswith('.epub')
        assert os.path.exists(result.final_output)
        
        # Verify it's a valid EPUB
        with zipfile.ZipFile(result.final_output, 'r') as epub:
            names = epub.namelist()
            assert any('content.opf' in n for n in names), "Missing OPF"
            assert any('nav.xhtml' in n for n in names), "Missing nav"
    
    @pytest.mark.asyncio
    async def test_handles_missing_audio(self, workflow_builder):
        """Test error when no audio file provided."""
        state = PipelineState(
            source_type="file",
            content="",
            final_output=None,
            cleaned_sections=[]
        )
        
        result = await workflow_builder.packaging_ep3(state)
        
        assert result.status == PipelineStatus.FAILED
        assert "No audio file" in result.error
    
    @pytest.mark.asyncio
    async def test_creates_ep3_with_sections(self, workflow_builder, state_with_audio):
        """Test that EP3 contains all sections."""
        result = await workflow_builder.packaging_ep3(state_with_audio)
        
        assert result.status == PipelineStatus.COMPLETED
        
        # Check EP3 contents
        with zipfile.ZipFile(result.final_output, 'r') as epub:
            names = epub.namelist()
            # Should have chapter files for Abstract and Introduction
            chapter_files = [n for n in names if 'chapter_' in n and n.endswith('.xhtml')]
            assert len(chapter_files) == 2, f"Expected 2 chapters, got {len(chapter_files)}"
