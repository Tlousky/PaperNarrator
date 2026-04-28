from abc import ABC, abstractmethod
from typing import Any, Dict, List


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def call_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: List[Dict[str, Any]],
        tool_choice: str | None = None
    ) -> Dict[str, Any]:
        """
        Call the LLM with function calling capabilities.
        
        Args:
            system_prompt: The system message to set context
            user_message: The user's message
            tools: List of tool definitions in OpenAI format
            tool_choice: Optional tool selection strategy ('auto', 'none', 'required', or tool name)
            
        Returns:
            Dict with keys:
                - content: str - The text response
                - tool_calls: List[Dict] - List of tool call objects with 'name' and 'arguments'
                - cost: float - The cost in dollars for this request
        """
        pass

    @abstractmethod
    async def call_simple(self, prompt: str) -> str:
        """
        Simple text-only call without tools.
        
        Args:
            prompt: The input prompt
            
        Returns:
            The text response from the LLM
        """
        pass

    @abstractmethod
    def get_cost_per_million_tokens(self) -> float:
        """
        Get the cost per million tokens for this provider's default model.
        
        Returns:
            Cost in dollars per million tokens
        """
        pass
