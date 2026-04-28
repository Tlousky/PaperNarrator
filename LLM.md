Example of local LLM setup we can use for testing and local runs of e2e agent logic:
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=os.environ.get("LLM_MODEL", "qwopus-27b-v3"),
    temperature=0,
    max_tokens=2000,
    base_url=os.environ.get("OPENAI_API_BASE", "http://localhost:8080/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", "sk-no-key-required"),
)
```