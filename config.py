"""Configuration module for PaperNarrator."""
import os
from typing import Optional

# Load environment variables from .env file if present
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration class for PaperNarrator."""
    
    # LLM Provider Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    
    # API Keys (loaded from environment)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    # Model names
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    
    # TTS Configuration
    VIBEVOICE_DEVICE: str = os.getenv("VIBEVOICE_DEVICE", "cuda:0")
    
    # Output Configuration
    OUTPUT_FORMAT: str = os.getenv("OUTPUT_FORMAT", "ep3")
    MP3_BITRATE: int = int(os.getenv("MP3_BITRATE", "128"))
    
    # VLM Configuration
    VLM_ENABLED: bool = os.getenv("VLM_ENABLED", "False").lower() in ("true", "1", "yes")


# Module-level constants for easier imports
LLM_PROVIDER = Config.LLM_PROVIDER
OPENAI_API_KEY = Config.OPENAI_API_KEY
GEMINI_API_KEY = Config.GEMINI_API_KEY
ANTHROPIC_API_KEY = Config.ANTHROPIC_API_KEY
OPENAI_MODEL = Config.OPENAI_MODEL
GEMINI_MODEL = Config.GEMINI_MODEL
ANTHROPIC_MODEL = Config.ANTHROPIC_MODEL
VIBEVOICE_DEVICE = Config.VIBEVOICE_DEVICE
OUTPUT_FORMAT = Config.OUTPUT_FORMAT
MP3_BITRATE = Config.MP3_BITRATE
VLM_ENABLED = Config.VLM_ENABLED


def validate_config() -> None:
    """
    Validate configuration settings.
    
    Raises:
        ValueError: If selected LLM provider does not have a valid API key configured.
    """
    provider = Config.LLM_PROVIDER.lower()
    
    if provider == "openai" and not Config.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required when LLM_PROVIDER is 'openai'. "
            "Please set it in your .env file or environment."
        )
    elif provider == "gemini" and not Config.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY environment variable is required when LLM_PROVIDER is 'gemini'. "
            "Please set it in your .env file or environment."
        )
    elif provider == "anthropic" and not Config.ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is required when LLM_PROVIDER is 'anthropic'. "
            "Please set it in your .env file or environment."
        )
    elif provider not in ("openai", "gemini", "anthropic"):
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            "Supported providers: 'openai', 'gemini', 'anthropic'"
        )
