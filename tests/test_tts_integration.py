"""Integration tests for VibeVoice TTS generation.

These tests require the VibeVoice-Realtime-0.5B model to be downloaded.
Run download_vibevoice.py first.

Audio outputs are saved to ./outputs/test_audio/ for inspection.
"""

import os
import pytest
import wave
import struct
from pathlib import Path

# Skip if models not available
MODELS_DIR = Path("./models/microsoft/VibeVoice-Realtime-0.5B")
pytestmark = pytest.mark.skipif(
    not MODELS_DIR.exists(),
    reason=f"VibeVoice models not found. Run download_vibevoice.py first. Expected at: {MODELS_DIR}"
)

# Output directory for test audio files (persists after tests)
OUTPUT_DIR = Path("./outputs/test_audio")


class TestVibeVoiceModelCompatibility:
    """
    Regression tests ensuring the vibevoice model classes have the
    expected API surface. These guard against silent API breakages
    (e.g. the 1.5B model that was missing generate()).
    """

    def test_streaming_model_has_generate_method(self):
        """
        Regression: VibeVoiceStreamingForConditionalGenerationInference
        must expose a .generate() method for inference to work.
        See: https://github.com/microsoft/VibeVoice (0.5B streaming demo)
        """
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference,
        )
        assert hasattr(VibeVoiceStreamingForConditionalGenerationInference, "generate"), (
            "VibeVoiceStreamingForConditionalGenerationInference is missing .generate(). "
            "The model class may have changed — check the vibevoice package version."
        )
        assert callable(VibeVoiceStreamingForConditionalGenerationInference.generate)

    def test_vibevoice_tts_max_words_constant_exists(self):
        """VibeVoiceTTS must expose MAX_WORDS so chunking can reference it."""
        import importlib, sys
        # Temporarily remove mock so we load the real module
        sys.modules.pop('tts.vibevoice', None)
        sys.modules.pop('tts', None)
        real_mod = importlib.import_module('tts.vibevoice')
        RealVibeVoiceTTS = real_mod.VibeVoiceTTS
        assert hasattr(RealVibeVoiceTTS, "MAX_WORDS")
        assert RealVibeVoiceTTS.MAX_WORDS > 0

    def test_vibevoice_tts_word_limit_raises_value_error(self):
        """Words exceeding MAX_WORDS must raise ValueError before loading model."""
        import importlib, sys
        sys.modules.pop('tts.vibevoice', None)
        sys.modules.pop('tts', None)
        real_mod = importlib.import_module('tts.vibevoice')
        RealVibeVoiceTTS = real_mod.VibeVoiceTTS

        # Manually construct a minimal instance without loading the model
        tts = object.__new__(RealVibeVoiceTTS)
        tts._loaded = True  # pretend model is loaded to skip _load_model
        tts.model = None
        tts.processor = None
        tts._voice_sample = None
        tts.device = "cpu"
        tts.cfg_scale = 1.5

        over_limit = RealVibeVoiceTTS.MAX_WORDS + 1
        text = "word " * over_limit

        with pytest.raises(ValueError, match=str(RealVibeVoiceTTS.MAX_WORDS)):
            tts.generate_audio(text=text, output_path="/tmp/fake.wav", word_count=over_limit)


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

    def _read_wav(self, path: str):
        """Read a WAV file and return (frames, sample_rate, n_channels)."""
        with wave.open(path, 'rb') as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            return n_frames, sr, n_channels
    
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
        
        # Verify it's a valid WAV file with some audio
        n_frames, sr, _ = self._read_wav(result_path)
        assert n_frames > 0, "Audio should not be empty"
        assert sr in (24000, 16000, 44100), f"Unexpected sample rate: {sr}"
    
    def test_generate_with_word_count(self, tts):
        """Test generation with explicit word count."""
        text = "The quick brown fox jumps over the lazy dog."
        output_path = OUTPUT_DIR / "test_count.wav"
        
        result_path = tts.generate_audio(
            text=text,
            output_path=str(output_path),
            word_count=10
        )
        
        assert os.path.exists(result_path)
        
        n_frames, sr, _ = self._read_wav(result_path)
        duration = n_frames / sr
        # VibeVoice generates at least 1 second for any text
        assert duration > 0.5, f"Duration {duration}s seems too short"
    
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
            
            n_frames, sr, _ = self._read_wav(str(output_path))
            assert n_frames > 1000, f"Audio too short for {speaker}"
    
    def test_word_count_validation(self, tts):
        """Test that word count > MAX_WORDS raises ValueError."""
        over_limit = tts.MAX_WORDS + 1
        text = "word " * over_limit
        
        with pytest.raises(ValueError, match=str(tts.MAX_WORDS)):
            tts.generate_audio(
                text=text,
                output_path=str(OUTPUT_DIR / "test_large.wav"),
                word_count=over_limit
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
        """Test generation with longer text (~30 words)."""
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
        
        n_frames, sr, _ = self._read_wav(result)
        duration = n_frames / sr
        
        # VibeVoice generates audio — just verify we got something non-trivial
        assert duration > 1.0, f"Duration {duration}s too short for 30 words"
