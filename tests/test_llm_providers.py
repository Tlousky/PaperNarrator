"""Smoke tests for LLM providers."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from llm import get_provider
from llm.base import LLMProvider


class TestProviderFactory:
    """Test provider factory function."""
    
    def test_get_provider_openai(self):
        """Test OpenAI provider instantiation."""
        with patch('llm.openai_provider.AsyncOpenAI'):
            provider = get_provider("openai", model="gpt-4o-mini")
            assert isinstance(provider, LLMProvider)
            assert provider.model == "gpt-4o-mini"
    
    def test_get_provider_gemini(self):
        """Test Gemini provider instantiation."""
        with patch('llm.gemini_provider.genai'):
            provider = get_provider("gemini", model="gemini-1.5-flash", api_key="test-key")
            assert isinstance(provider, LLMProvider)
            assert provider.model == "gemini-1.5-flash"
    
    def test_get_provider_anthropic(self):
        """Test Anthropic provider instantiation."""
        with patch('llm.anthropic_provider.AsyncAnthropic'):
            provider = get_provider("anthropic", model="claude-3-5-sonnet", api_key="test-key")
            assert isinstance(provider, LLMProvider)
            assert provider.model == "claude-3-5-sonnet"
    
    def test_get_provider_ollama(self):
        """Test Ollama provider instantiation."""
        provider = get_provider("ollama", model="llama3.2")
        assert isinstance(provider, LLMProvider)
        assert provider.model == "llama3.2"
    
    def test_get_provider_invalid(self):
        """Test invalid provider raises error."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("invalid-provider")


class TestOpenAIProvider:
    """Test OpenAI provider implementation."""
    
    @pytest.fixture
    def provider(self):
        with patch('llm.openai_provider.AsyncOpenAI'):
            return get_provider("openai")
    
    def test_get_cost_per_million_tokens_default(self, provider):
        """Test default model cost calculation."""
        # gpt-4o-mini: ($0.15 + $0.60) / 2 = $0.375
        assert provider.get_cost_per_million_tokens() == 0.375
    
    def test_get_cost_per_million_tokens_gpt4o(self):
        """Test gpt-4o cost calculation."""
        with patch('llm.openai_provider.AsyncOpenAI'):
            provider = get_provider("openai", model="gpt-4o")
            # ($2.50 + $10.00) / 2 = $6.25
            assert provider.get_cost_per_million_tokens() == 6.25


class TestGeminiProvider:
    """Test Gemini provider implementation."""
    
    @pytest.fixture
    def provider(self):
        with patch('llm.gemini_provider.genai'):
            return get_provider("gemini", api_key="test")
    
    def test_get_cost_per_million_tokens_default(self, provider):
        """Test default model cost calculation."""
        # gemini-1.5-flash: ($0.075 + $0.30) / 2 = $0.1875
        assert provider.get_cost_per_million_tokens() == 0.1875
    
    def test_get_cost_per_million_tokens_pro(self):
        """Test gemini-1.5-pro cost calculation."""
        with patch('llm.gemini_provider.genai'):
            provider = get_provider("gemini", model="gemini-1.5-pro", api_key="test")
            # ($1.25 + $5.00) / 2 = $3.125
            assert provider.get_cost_per_million_tokens() == 3.125


class TestAnthropicProvider:
    """Test Anthropic provider implementation."""
    
    @pytest.fixture
    def provider(self):
        with patch('llm.anthropic_provider.AsyncAnthropic'):
            return get_provider("anthropic", api_key="test")
    
    def test_get_cost_per_million_tokens(self, provider):
        """Test claude-3-5-sonnet cost calculation."""
        # ($3.00 + $15.00) / 2 = $9.00
        assert provider.get_cost_per_million_tokens() == 9.0


class TestOllamaProvider:
    """Test Ollama provider implementation."""
    
    @pytest.fixture
    def provider(self):
        return get_provider("ollama")
    
    def test_get_cost_per_million_tokens(self, provider):
        """Test Ollama cost is zero."""
        assert provider.get_cost_per_million_tokens() == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
