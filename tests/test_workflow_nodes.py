"""Tests for LangGraph workflow nodes (generating_audio, concatenating_audio)."""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path

from langgraph_pipeline.state import PipelineState, PipelineStatus, TextChunk
from langgraph_pipeline.workflow import WorkflowBuilder


# Mock VibeVoiceTTS for tests
class MockVibeVoiceTTS:
    """Mock TTS that creates fake audio files."""
    
    def __init__(self, model_name=None, device=None, speaker_name=None, cfg_scale=None):
        self.model_name = model_name
        self.device = device
        self.speaker_name = speaker_name
        self.cfg_scale = cfg_scale
    
    def generate_audio(self, text, output_path, word_count):
        """Create a fake WAV file (just a header)."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Create minimal WAV header (44 bytes) + some dummy data
        with open(output_path, 'wb') as f:
            # RIFF header
            f.write(b'RIFF')
            f.write((32 + word_count * 48000).to_bytes(4, 'little'))  # File size - 8
            f.write(b'WAVE')
            f.write(b'fmt ')  # fmt chunk
            f.write((16).to_bytes(4, 'little'))  # chunk size
            f.write((1).to_bytes(2, 'little'))   # PCM format
            f.write((1).to_bytes(2, 'little'))   # mono
            f.write((24000).to_bytes(4, 'little'))  # sample rate
            f.write((48000).to_bytes(4, 'little'))  # byte rate
            f.write((2).to_bytes(2, 'little'))   # block align
            f.write((16).to_bytes(2, 'little'))  # bits per sample
            f.write(b'data')  # data chunk
            f.write((word_count * 48000 * 2).to_bytes(4, 'little'))
            # Dummy audio data
            f.write(b'\x00' * (word_count * 48000 * 2))
        return output_path


# Patch tts.vibevoice
import sys
from unittest.mock import MagicMock

mock_tts_module = MagicMock()
sys.modules['tts'] = MagicMock()
sys.modules['tts.vibevoice'] = mock_tts_module
mock_tts_module.VibeVoiceTTS = MockVibeVoiceTTS


@pytest.fixture
def workflow_builder():
    """Create WorkflowBuilder with mock config."""
    config = MagicMock()
    config.OUTPUT_FORMAT = "wav"
    config.MP3_BITRATE = "128kbps"
    return WorkflowBuilder(config=config)


@pytest.fixture
def state_with_chunks():
    """Create state with sample chunks."""
    state = PipelineState(
        source_type="file",
        content="",
        chunks=[
            TextChunk(
                text="Hello world. This is a test.",
                word_count=7,
                section_names=["Introduction"]
            ),
            TextChunk(
                text="The quick brown fox jumps over the lazy dog.",
                word_count=10,
                section_names=["Methods"]
            )
        ]
    )
    return state


class TestGeneratingAudio:
    """Tests for the generating_audio node (Task 8)."""
    
    @pytest.mark.asyncio
    async def test_generates_audio_for_chunks(self, workflow_builder, state_with_chunks, tmp_path):
        """Test that audio files are generated for each chunk."""
        # Override temp_path handling - we'll use tmp_path
        workflow_builder.config.OUTPUT_DIR = str(tmp_path)
        
        result = await workflow_builder.generating_audio(state_with_chunks)
        
        assert result.status == PipelineStatus.COMPLETED
        assert len(result.audio_files) == 2
        assert all(os.path.exists(f) for f in result.audio_files)
        assert "Generating audio" in result.status_message or "Generated" in result.status_message
    
    @pytest.mark.asyncio
    async def test_handles_empty_chunks(self, workflow_builder):
        """Test error handling when no chunks provided."""
        state = PipelineState(source_type="file", content="", chunks=[])
        
        result = await workflow_builder.generating_audio(state)
        
        assert result.status == PipelineStatus.FAILED
        assert "No chunks available" in result.error
    
    @pytest.mark.asyncio
    async def test_tracks_progress_in_status(self, workflow_builder, state_with_chunks):
        """Test that status messages track progress."""
        result = await workflow_builder.generating_audio(state_with_chunks)
        
        # Status should mention generated files
        assert "Generated" in result.status_message or "audio" in result.status_message.lower()


class TestConcatenatingAudio:
    """Tests for the concatenating_audio node (Task 9)."""
    
    @pytest.mark.asyncio
    async def test_concatenates_wav_files(self, workflow_builder, tmp_path):
        """Test concatenation of multiple WAV files."""
        # Create fake audio files using MockVibeVoiceTTS method
        audio_files = []
        for i in range(3):
            path = os.path.join(tmp_path, f"chunk_{i+1:03d}.wav")
            # Create valid minimal WAV (44 bytes header + some data)
            with open(path, 'wb') as f:
                # RIFF header
                f.write(b'RIFF')
                f.write((36 + 48000).to_bytes(4, 'little'))  # File size - 8
                f.write(b'WAVEfmt ')
                f.write((16).to_bytes(4, 'little'))  # chunk size
                f.write((1).to_bytes(2, 'little'))   # PCM
                f.write((1).to_bytes(2, 'little'))   # mono
                f.write((24000).to_bytes(4, 'little'))  # sample rate
                f.write((48000).to_bytes(4, 'little'))  # byte rate
                f.write((2).to_bytes(2, 'little'))   # block align
                f.write((16).to_bytes(2, 'little'))  # bits per sample
                f.write(b'data')
                f.write((48000).to_bytes(4, 'little'))  # 1 second of audio
                f.write(b'\x00' * 48000)  # silence
            audio_files.append(path)
        
        state = PipelineState(
            source_type="file",
            content="",
            audio_files=audio_files,
            temp_path=str(tmp_path)
        )
        
        # Need to mock pydub or skip if not available
        try:
            from pydub import AudioSegment
            result = await workflow_builder.concatenating_audio(state)
            
            assert result.status == PipelineStatus.COMPLETED
            assert result.final_output is not None
            assert os.path.exists(result.final_output)
            assert os.path.basename(result.final_output) == "combined.wav"
        except ImportError:
            pytest.skip("pydub not installed")
    
    @pytest.mark.asyncio
    async def test_converts_to_mp3_when_configured(self, tmp_path):
        """Test MP3 export when OUTPUT_FORMAT=mp3."""
        config = MagicMock()
        config.OUTPUT_FORMAT = "mp3"
        config.MP3_BITRATE = "128kbps"
        
        workflow_builder = WorkflowBuilder(config=config)
        
        # Create valid WAV file
        audio_path = os.path.join(tmp_path, "chunk_001.wav")
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
        
        state = PipelineState(
            source_type="file",
            content="",
            audio_files=[audio_path],
            temp_path=str(tmp_path)
        )
        
        try:
            result = await workflow_builder.concatenating_audio(state)
            
            # Check if ffmpeg is available for MP3 encoding
            if "ffmpeg" in str(result.error).lower() or "encoder" in str(result.error).lower():
                pytest.skip("ffmpeg not available for MP3 encoding")
            
            assert result.status == PipelineStatus.COMPLETED
            assert result.final_output.endswith('.mp3')
        except ImportError:
            pytest.skip("pydub not installed")
    
    @pytest.mark.asyncio
    async def test_handles_empty_audio_files(self, workflow_builder):
        """Test error when no audio files to concatenate."""
        state = PipelineState(
            source_type="file",
            content="",
            audio_files=[]
        )
        
        result = await workflow_builder.concatenating_audio(state)
        
        assert result.status == PipelineStatus.FAILED
        assert "No audio files" in result.error


class TestWorkflowIntegration:
    """Integration tests for workflow nodes."""
    
    @pytest.mark.asyncio
    async def test_full_audio_pipeline(self, workflow_builder, tmp_path):
        """Test generating_audio -> concatenating_audio pipeline."""
        # Setup state with chunks
        state = PipelineState(
            source_type="file",
            content="",
            chunks=[
                TextChunk(
                    text="Test sentence one.",
                    word_count=3,
                    section_names=["Section 1"]
                ),
                TextChunk(
                    text="Test sentence two.",
                    word_count=3,
                    section_names=["Section 1"]
                )
            ]
        )
        
        # Step 1: Generate audio
        result1 = await workflow_builder.generating_audio(state)
        assert result1.status == PipelineStatus.COMPLETED
        assert len(result1.audio_files) == 2
        
        # Step 2: Concatenate
        result2 = await workflow_builder.concatenating_audio(result1)
        assert result2.status == PipelineStatus.COMPLETED
        assert result2.final_output is not None
        assert os.path.exists(result2.final_output)
