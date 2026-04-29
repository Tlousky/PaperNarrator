"""Integration tests for VibeVoice TTS generation.

These tests require the VibeVoice-Realtime-0.5B model to be downloaded.
Run setup_vibevoice.sh or setup_vibevoice.bat first.

Audio outputs are saved to ./outputs/test_audio/ for inspection.
"""

import os
import pytest
from pathlib import Path

# Skip if models not available
MODELS_DIR = Path("./models/microsoft/VibeVoice-Realtime-0.5B")
pytestmark = pytest.mark.skipif(
    not MODELS_DIR.exists(),
    reason=f"VibeVoice models not found. Run setup_vibevoice.sh/bat first. Expected at: {MODELS_DIR}"
)

# Output directory for test audio files (persists after tests)
OUTPUT_DIR = Path("./outputs/test_audio")


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
    
    @pytest.fixture(autouse=True)
    def setup_outputs(self):
        """Create output directory before tests."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        yield
    
    def test_generate_short_text(self, tts):
        """Test generation with short text (< 100 words)."""
        text = "Hello world. This is a simple test."
        output_path = OUTPUT_DIR / "test_short.wav"
        
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
    
    def test_generate_with_word_count(self, tts):
        """Test generation with explicit word count."""
        text = "The quick brown fox jumps over the lazy dog."
        output_path = OUTPUT_DIR / "test_count.wav"
        
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
    
    def test_different_speakers(self, tts):
        """Test generation with different voice presets."""
        text = "Testing different voice presets."
        
        # Available voices: Carter, Davis, Emma
        for speaker in ["Carter", "Davis"]:
            tts.speaker_name = speaker
            tts._loaded = False  # Force reload
            
            output_path = OUTPUT_DIR / f"test_{speaker}.wav"
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
    
    def test_word_count_validation(self, tts):
        """Test that word count > 12000 raises ValueError."""
        text = "word " * 13000  # 13000 words
        
        with pytest.raises(ValueError, match="exceeds 12000 words"):
            tts.generate_audio(
                text=text,
                output_path=str(OUTPUT_DIR / "test_large.wav"),
                word_count=13000
            )
    
    def test_output_directory_creation(self, tts):
        """Test that output directory is created if it doesn't exist."""
        text = "Test."
        nested_path = OUTPUT_DIR / "nested" / "dir" / "output.wav"
        
        result = tts.generate_audio(
            text=text,
            output_path=str(nested_path),
            word_count=1
        )
        
        assert os.path.exists(nested_path)
        assert result == str(nested_path)
    
    def test_longer_text_chunk(self, tts):
        """Test generation with longer text (~50 words)."""
        text = """
        The quick brown fox jumps over the lazy dog. 
        Pack my box with five dozen liquor jugs.
        How vexingly quick daft zebras jump!
        The five boxing wizards jump quickly.
        Sphinx of black quartz, judge my vow.
        """  # ~30 words
        
        output_path = OUTPUT_DIR / "test_long.wav"
        
        result = tts.generate_audio(
            text=text.strip(),
            output_path=str(output_path),
            word_count=30
        )
        
        import soundfile as sf
        audio_data, sr = sf.read(result)
        duration = len(audio_data) / sr
        
        # ~30 words at ~150 wpm = ~12 seconds
        assert 10 < duration < 30, f"Duration {duration}s out of expected range for 30 words"
