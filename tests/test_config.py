"""Tests for configuration module."""
import os
import unittest
from unittest.mock import patch


class TestConfig(unittest.TestCase):
    """Test configuration loading and validation."""

    def test_config_loads_from_environment(self):
        """Test that config loads API keys from environment variables."""
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key-123",
            "OPENAI_MODEL": "gpt-4o"
        }, clear=False):
            from config import LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL
            self.assertEqual(LLM_PROVIDER, "openai")
            self.assertEqual(OPENAI_API_KEY, "test-key-123")
            self.assertEqual(OPENAI_MODEL, "gpt-4o")

    def test_output_format_defaults_to_ep3(self):
        """Test that OUTPUT_FORMAT defaults to 'ep3'."""
        with patch.dict(os.environ, {}, clear=False):
            from config import OUTPUT_FORMAT
            self.assertEqual(OUTPUT_FORMAT, "ep3")

    def test_validate_config_missing_openai_key(self):
        """Test validate_config raises error when OPENAI_API_KEY is missing."""
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": ""
        }, clear=True):
            # Force reload of config module to pick up new env
            import importlib
            import config
            importlib.reload(config)
            
            with self.assertRaises(ValueError) as ctx:
                config.validate_config()
            self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_validate_config_missing_gemini_key(self):
        """Test validate_config raises error when GEMINI_API_KEY is missing."""
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": ""
        }, clear=True):
            import importlib
            import config
            importlib.reload(config)
            
            with self.assertRaises(ValueError) as ctx:
                config.validate_config()
            self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    def test_validate_config_missing_anthropic_key(self):
        """Test validate_config raises error when ANTHROPIC_API_KEY is missing."""
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": ""
        }, clear=True):
            import importlib
            import config
            importlib.reload(config)
            
            with self.assertRaises(ValueError) as ctx:
                config.validate_config()
            self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))

    def test_validate_config_invalid_provider(self):
        """Test validate_config raises error for invalid LLM provider."""
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "invalid_provider"
        }, clear=True):
            import importlib
            import config
            importlib.reload(config)
            
            with self.assertRaises(ValueError) as ctx:
                config.validate_config()
            self.assertIn("Unknown LLM_PROVIDER", str(ctx.exception))
            self.assertIn("invalid_provider", str(ctx.exception))

    def test_validate_config_invalid_mp3_bitrate_non_integer(self):
        """Test config raises error when MP3_BITRATE is not a valid integer."""
        with patch.dict(os.environ, {
            "MP3_BITRATE": "abc"
        }, clear=True):
            with self.assertRaises((ValueError, TypeError)) as ctx:
                import importlib
                import config
                importlib.reload(config)

    def test_validate_config_invalid_mp3_bitrate_out_of_range(self):
        """Test config raises error when MP3_BITRATE is out of valid range (32-320)."""
        with patch.dict(os.environ, {
            "MP3_BITRATE": "500"
        }, clear=True):
            with self.assertRaises(ValueError) as ctx:
                import importlib
                import config
                importlib.reload(config)
            self.assertIn("MP3_BITRATE", str(ctx.exception))
            self.assertIn("32", str(ctx.exception))
            self.assertIn("320", str(ctx.exception))

    def test_validate_config_invalid_output_format(self):
        """Test config raises error when OUTPUT_FORMAT is not valid."""
        with patch.dict(os.environ, {
            "OUTPUT_FORMAT": "invalid"
        }, clear=True):
            with self.assertRaises(ValueError) as ctx:
                import importlib
                import config
                importlib.reload(config)
            self.assertIn("OUTPUT_FORMAT", str(ctx.exception))
            self.assertIn("ep3", str(ctx.exception))
            self.assertIn("mp3", str(ctx.exception))
            self.assertIn("wav", str(ctx.exception))


class TestConfigValidValues(unittest.TestCase):
    """Test that valid configuration values are accepted."""

    def test_valid_mp3_bitrate_values(self):
        """Test that valid MP3 bitrate values are accepted."""
        for bitrate in [32, 64, 128, 192, 256, 320]:
            with patch.dict(os.environ, {
                "LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "test-key",
                "MP3_BITRATE": str(bitrate)
            }, clear=True):
                import importlib
                import config
                importlib.reload(config)
                self.assertEqual(config.MP3_BITRATE, bitrate)

    def test_valid_output_formats(self):
        """Test that valid output format values are accepted."""
        for fmt in ["ep3", "mp3", "wav"]:
            with patch.dict(os.environ, {
                "LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "test-key",
                "OUTPUT_FORMAT": fmt
            }, clear=True):
                import importlib
                import config
                importlib.reload(config)
                self.assertEqual(config.OUTPUT_FORMAT, fmt)


if __name__ == "__main__":
    unittest.main()
