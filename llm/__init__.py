"""LLM Provider Factory and Abstract Base Class."""

from typing import Any, Dict, Type

from llm.anthropic_provider import AnthropicProvider
from llm.base import LLMProvider
from llm.gemini_provider import GeminiProvider
from llm.ollama_provider import OllamaProvider
from llm.openai_provider import OpenAIProvider

PROVIDER_CLASSES: Dict[str, Type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
}


def get_provider(name: str, **kwargs) -> LLMProvider:
    """
    Factory function to get an LLM provider instance.
    
    Args:
        name: Provider name ('openai', 'gemini', 'anthropic', 'ollama')
        **kwargs: Provider-specific arguments (model, api_key, etc.)
        
    Returns:
        Instance of the requested LLMProvider
        
    Raises:
        ValueError: If provider name is not recognized
    """
    name = name.lower()
    if name not in PROVIDER_CLASSES:
        available = ", ".join(PROVIDER_CLASSES.keys())
        raise ValueError(
            f"Unknown provider: {name}. Available providers: {available}"
        )
    
    return PROVIDER_CLASSES[name](**kwargs)


__all__ = ["LLMProvider", "get_provider", "PROVIDER_CLASSES"]
