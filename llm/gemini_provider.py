import os
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from llm.base import LLMProvider


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider with cost tracking."""
    
    # Cost per million tokens (input/output)
    PRICES = {
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    }
    
    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: Optional[str] = None
    ):
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable or api_key argument required")
        
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(
            model_name=model,
            system_instruction=genai.protos.SystemInstruction(
                parts=[genai.protos.Part(text="")]
            )
        )
    
    def _convert_tools(self, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert OpenAI-style tools to Gemini tool format."""
        if not tools:
            return {}
        
        gemini_tools = []
        for tool in tools:
            if "function" not in tool:
                continue
            func = tool["function"]
            gemini_func = {
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            
            # Convert parameters
            params = func.get("parameters", {})
            properties = params.get("properties", {})
            for param_name, param_def in properties.items():
                gemini_func["parameters"]["properties"][param_name] = {
                    "type": param_def.get("type", "string"),
                    "description": param_def.get("description", "")
                }
            gemini_func["parameters"]["required"] = params.get("required", [])
            
            gemini_tools.append({"function_declarations": [gemini_func]})
        
        return gemini_tools[0] if gemini_tools else {}
    
    async def call_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: List[Dict[str, Any]],
        tool_choice: Optional[str] = None
    ) -> Dict[str, Any]:
        """Call Gemini with function calling."""
        # Update system instruction
        self.client.system_instruction = genai.protos.SystemInstruction(
            parts=[genai.protos.Part(text=system_prompt)]
        )
        
        # Convert tools
        gemini_tools = self._convert_tools(tools)
        
        # Build generation config
        generation_config = genai.types.GenerationConfig()
        if tool_choice == "required" and gemini_tools:
            generation_config.tool_choice = genai.types.ToolChoice.ALL
        elif tool_choice == "none":
            generation_config.tool_choice = genai.types.ToolChoice.NONE
        
        response = self.client.generate_content(
            user_message,
            tools=gemini_tools if gemini_tools else None,
            generation_config=generation_config
        )
        
        # Calculate cost
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        price = self.PRICES.get(self.model, {"input": 0.075, "output": 0.30})
        cost = (
            (input_tokens / 1_000_000) * price["input"] +
            (output_tokens / 1_000_000) * price["output"]
        )
        
        # Parse response
        content = response.text or ""
        tool_calls = []
        
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    fc = part.function_call
                    tool_calls.append({
                        "name": fc.name,
                        "arguments": fc.args
                    })
        
        return {
            "content": content,
            "tool_calls": tool_calls,
            "cost": cost
        }
    
    async def call_simple(self, prompt: str) -> str:
        """Simple text-only call."""
        self.client.system_instruction = genai.protos.SystemInstruction(parts=[])
        response = self.client.generate_content(prompt)
        return response.text or ""
    
    def get_cost_per_million_tokens(self) -> float:
        """Get average cost per million tokens (input+output)."""
        price = self.PRICES.get(self.model, {"input": 0.075, "output": 0.30})
        return (price["input"] + price["output"]) / 2
