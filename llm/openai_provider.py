import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from llm.base import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider with cost tracking."""
    
    # Cost per million tokens (input/output)
    PRICES = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
    }
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    async def call_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: List[Dict[str, Any]],
        tool_choice: Optional[str] = None
    ) -> Dict[str, Any]:
        """Call OpenAI with function calling."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        
        response = await self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        
        # Calculate cost
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        price = self.PRICES.get(self.model, {"input": 0.15, "output": 0.60})
        cost = (
            (input_tokens / 1_000_000) * price["input"] +
            (output_tokens / 1_000_000) * price["output"]
        )
        
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
            "cost": cost
        }
    
    async def call_simple(self, prompt: str) -> str:
        """Simple text-only call."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content or ""
    
    def get_cost_per_million_tokens(self) -> float:
        """Get average cost per million tokens (input+output)."""
        price = self.PRICES.get(self.model, {"input": 0.15, "output": 0.60})
        return (price["input"] + price["output"]) / 2
