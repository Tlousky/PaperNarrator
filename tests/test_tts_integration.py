"""Integration tests for VibeVoice TTS generation.

These tests require the VibeVoice-1.5B model to be downloaded.
Run setup_vibevoice.sh or setup_vibevoice.bat first.
"""

import os
import tempfile
import pytest
from pathlib import Path

# Skip if models not available
MODELS_DIR = Path("./models/microsoft/VibeVoice-1.5B")
pytestmark = pytest.mark.skipif(
    not MODELS_DIR.exists(),
    reason=f"VibeVoice models not found. Run setup_vibevoice.sh/bat first. Expected at: {MODELS_DIR}"
)


@pytest.mark.forked
class TestVibeVoiceIntegration:
    """Integration tests requiring actual model inference."""
    
    @pytest.fixture
    def tts(self):
        """Create TTS instance with local model."""
        from tts.vibevoice import VibeVoiceTTS
        
        tts = VibeVoiceTTS(
            model_name=str(MODELS_DIR),
            speaker_name="Carter",
            cfg_scale=1.5
        )
        return tts
    
    @pytest.fixture
    def temp_audio_dir(self):
        """Create temporary directory for audio outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_generate_short_text(self, tts, temp_audio_dir):
        """Test generation with short text (< 100 words)."""
        text = "Hello world. This is a simple test."
        output_path = temp_audio_dir / "test_short.wav"
        
        result_path = tts.generate_audio(
            text=text,
            output_path=str(output_path),
            word_count=9
        )
        
        assert os.path.exists(result_path), "Output file should exist"
        assert result_path == str(output_path), "Should return correct path"
        
        # Verify it's a valid WAV file
        import soundfile as sf
        audio_data, sr = sf.read(result_path)
        assert len(audio_data) > 0, "Audio should not be empty"
        assert sr == 24000, f"Sample rate should be 24000, got {sr}"
        assert audio_data.max() <= 1.0 and audio_data.min() >= -1.0, "Audio should be normalized"
    
    def test_generate_with_word_count(self, tts, temp_audio_dir):
        """Test generation with explicit word count."""
        text = "The quick brown fox jumps over the lazy dog."
        output_path = temp_audio_dir / "test_count.wav"
        
        result_path = tts.generate_audio(
            text=text,
            output_path=str(output_path),
            word_count=10  # Actual word count
        )
        
        assert os.path.exists(result_path)
        
        import soundfile as sf
        audio_data, _ = sf.read(result_path)
        # ~10 words at ~3 words/sec = ~3.3 seconds
        duration = len(audio_data) / 24000
        assert 2.0 < duration < 6.0, f"Duration {duration}s seems wrong for 10 words"
    
    def test_different_speakers(self, tts, temp_audio_dir):
        """Test generation with different voice presets."""
        text = "Testing different voice presets."
        
        for speaker in ["Carter", "Wayne"]:
            tts.speaker_name = speaker
            tts._loaded = False  # Force reload
            
            output_path = temp_audio_dir / f"test_{speaker}.wav"
            tts.generate_audio(
                text=text,
                output_path=str(output_path),
                word_count=6
            )
            
            assert os.path.exists(output_path), f"Failed for speaker {speaker}"
            
            # Verify audio is different (rough check)
            import soundfile as sf
            data, _ = sf.read(output_path)
            assert len(data) > 1000, f"Audio too short for {speaker}"
    
    def test_word_count_validation(self, tts, temp_audio_dir):
        """Test that word count > 12000 raises ValueError."""
        text = "word " * 13000  # 13000 words
        
        with pytest.raises(ValueError, match="exceeds 12000 words"):
            tts.generate_audio(
                text=text,
                output_path=str(temp_audio_dir / "test_large.wav"),
                word_count=13000
            )
    
    def test_output_directory_creation(self, tts, temp_audio_dir):
        """Test that output directory is created if it doesn't exist."""
        text = "Test."
        nested_path = temp_audio_dir / "nested" / "dir" / "output.wav"
        
        result = tts.generate_audio(
            text=text,
            output_path=str(nested_path),
            word_count=1
        )
        
        assert os.path.exists(nested_path)
        assert result == str(nested_path)
    
    def test_longer_text_chunk(self, tts, temp_audio_dir):
        """Test generation with longer text (~200 words)."""
        text = """
        The quick brown fox jumps over the lazy dog. 
        Pack my box with five dozen liquor jugs.
        How vexingly quick daft zebras jump!
        The five boxing wizards jump quickly.
        Sphinx of black quartz, judge my vow.
        """ * 3  # ~180 words
        
        output_path = temp_audio_dir / "test_long.wav"
        
        result = tts.generate_audio(
            text=text.strip(),
            output_path=str(output_path),
            word_count=180
        )
        
        import soundfile as sf
        audio_data, sr = sf.read(result)
        duration = len(audio_data) / sr
        
        # ~180 words at ~150 wpm = ~1.2 minutes = 72 seconds
        assert 60 < duration < 120, f"Duration {duration}s out of expected range for 180 words"
