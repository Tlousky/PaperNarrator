import os
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from llm.base import LLMProvider


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider with cost tracking."""
    
    # Cost per million tokens (input/output)
    PRICES = {
        "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
    }
    
    def __init__(
        self,
        model: str = "claude-3-5-sonnet",
        api_key: Optional[str] = None
    ):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable or api_key argument required")
        
        self.client = AsyncAnthropic(api_key=self.api_key)
    
    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style tools to Anthropic format."""
        if not tools:
            return []
        
        anthropic_tools = []
        for tool in tools:
            if "function" not in tool:
                continue
            func = tool["function"]
            params = func.get("parameters", {})
            
            anthropic_tool = {
                "name": func["name"],
                "description": func.get("description", ""),
                "input_schema": {
                    "type": "object",
                    "properties": params.get("properties", {}),
                    "required": params.get("required", [])
                }
            }
            anthropic_tools.append(anthropic_tool)
        
        return anthropic_tools
    
    async def call_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: List[Dict[str, Any]],
        tool_choice: Optional[str] = None
    ) -> Dict[str, Any]:
        """Call Anthropic with function calling."""
        anthropic_tools = self._convert_tools(tools)
        
        # Determine tool choice
        tool_choice_param = None
        if tool_choice == "required" and anthropic_tools:
            tool_choice_param = {"type": "any"}
        elif tool_choice == "auto":
            tool_choice_param = {"type": "auto"}
        elif tool_choice == "none":
            tool_choice_param = {"type": "none"}
        elif tool_choice and anthropic_tools:
            # Specific tool name
            tool_choice_param = {"type": "tool", "name": tool_choice}
        
        response = await self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            tools=anthropic_tools if anthropic_tools else None,
            tool_choice=tool_choice_param,
            max_tokens=4096
        )
        
        # Calculate cost
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        price = self.PRICES.get(self.model, {"input": 3.00, "output": 15.00})
        cost = (
            (input_tokens / 1_000_000) * price["input"] +
            (output_tokens / 1_000_000) * price["output"]
        )
        
        # Parse response
        content = ""
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "name": block.name,
                    "arguments": block.input
                })
        
        return {
            "content": content,
            "tool_calls": tool_calls,
            "cost": cost
        }
    
    async def call_simple(self, prompt: str) -> str:
        """Simple text-only call."""
        response = await self.client.messages.create(
            model=self.model,
            system="",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096
        )
        return response.content[0].text if response.content else ""
    
    def get_cost_per_million_tokens(self) -> float:
        """Get average cost per million tokens (input+output)."""
        price = self.PRICES.get(self.model, {"input": 3.00, "output": 15.00})
        return (price["input"] + price["output"]) / 2
