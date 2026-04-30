"""Tests for EP3 (EPUB3) audiobook generation."""

import pytest
import zipfile
import os
from pathlib import Path


class TestEP3Builder:
    """Tests for EP3 generation."""
    
    @pytest.fixture
    def sample_audio(self, tmp_path):
        """Create a sample WAV file for testing."""
        audio_path = tmp_path / "test.wav"
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
        return str(audio_path)
    
    def test_creates_ep3_file(self, tmp_path, sample_audio):
        """Test that create_ep3 generates a valid EPUB3 file."""
        from langgraph_pipeline.ep3_builder import create_ep3
        
        sections = [
            {"title": "Abstract", "content": "This is a test abstract."},
            {"title": "Introduction", "content": "This is the introduction."}
        ]
        
        cleaned_text = "This is a test abstract. This is the introduction."
        output_path = str(tmp_path / "output.epub")
        
        result_path = create_ep3(
            audio_path=sample_audio,
            cleaned_text=cleaned_text,
            sections=sections,
            output_path=output_path
        )
        
        assert os.path.exists(result_path)
        assert result_path.endswith('.epub') or result_path.endswith('.ep3')
        
        # Verify it's a valid ZIP file (EPUB is ZIP-based)
        with zipfile.ZipFile(result_path, 'r') as ep3:
            names = ep3.namelist()
            assert any('content.opf' in name for name in names), "Missing OPF file"
            assert any('nav.xhtml' in name for name in names), "Missing navigation file"
    
    def test_contains_all_sections(self, tmp_path, sample_audio):
        """Test that all sections are included in the EP3."""
        from langgraph_pipeline.ep3_builder import create_ep3
        
        sections = [
            {"title": "Abstract", "content": "Abstract content."},
            {"title": "Introduction", "content": "Intro content."},
            {"title": "Methods", "content": "Methods content."}
        ]
        
        output_path = str(tmp_path / "output.epub")
        result_path = create_ep3(
            audio_path=sample_audio,
            cleaned_text="Abstract content. Intro content. Methods content.",
            sections=sections,
            output_path=output_path
        )
        
        with zipfile.ZipFile(result_path, 'r') as ep3:
            names = ep3.namelist()
            # Should have 3 chapter files
            chapter_files = [n for n in names if 'chapter_' in n and n.endswith('.xhtml')]
            assert len(chapter_files) == 3, f"Expected 3 chapters, got {len(chapter_files)}"
    
    def test_contains_audio_and_smil(self, tmp_path, sample_audio):
        """Test that audio and SMIL files are included."""
        from langgraph_pipeline.ep3_builder import create_ep3
        
        sections = [{"title": "Section 1", "content": "Content."}]
        output_path = str(tmp_path / "output.epub")
        
        create_ep3(
            audio_path=sample_audio,
            cleaned_text="Content.",
            sections=sections,
            output_path=output_path
        )
        
        with zipfile.ZipFile(output_path, 'r') as ep3:
            names = ep3.namelist()
            assert any('audio.wav' in n for n in names), "Missing audio file"
            assert any('overlay.smil' in n for n in names), "Missing SMIL file"
    
    def test_returns_correct_path(self, tmp_path, sample_audio):
        """Test that function returns the correct output path."""
        from langgraph_pipeline.ep3_builder import create_ep3
        
        sections = [{"title": "Test", "content": "Content."}]
        output_path = str(tmp_path / "result.epub")
        
        result = create_ep3(
            audio_path=sample_audio,
            cleaned_text="Content.",
            sections=sections,
            output_path=output_path
        )
        
        assert result == output_path
