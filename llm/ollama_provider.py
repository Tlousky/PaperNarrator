import os
from typing import Any, Dict, List, Optional

import ollama

from llm.base import LLMProvider


class OllamaProvider(LLMProvider):
    """Local Ollama provider (zero cost)."""
    
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: Optional[str] = None
    ):
        self.model = model
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.client = ollama.AsyncClient(host=self.base_url)
    
    async def call_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: List[Dict[str, Any]],
        tool_choice: Optional[str] = None
    ) -> Dict[str, Any]:
        """Call Ollama with function calling."""
        # Convert tools to Ollama format
        ollama_tools = []
        for tool in tools:
            if "function" not in tool:
                continue
            func = tool["function"]
            ollama_func = {
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {})
            }
            ollama_tools.append(ollama_func)
        
        # Build messages with system prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        
        kwargs = {
            "model": self.model,
            "messages": messages,
        }
        
        if ollama_tools:
            kwargs["tools"] = ollama_tools
        
        response = await self.client.chat(**kwargs)
        message = response.message
        
        # Parse tool calls
        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                })
        
        return {
            "content": message.content or "",
            "tool_calls": tool_calls,
            "cost": 0.0  # Local is free
        }
    
    async def call_simple(self, prompt: str) -> str:
        """Simple text-only call."""
        response = await self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.message.content or ""
    
    def get_cost_per_million_tokens(self) -> float:
        """Get cost per million tokens (zero for local)."""
        return 0.0
