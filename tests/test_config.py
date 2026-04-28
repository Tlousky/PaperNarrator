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


if __name__ == "__main__":
    unittest.main()
