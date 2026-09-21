from typing import List, Dict, Optional, Any
import os
from openai import OpenAI
from anthropic import Anthropic


class LLMClient:
    """
    Thin wrapper to talk to either a local OpenAI-compatible endpoint (e.g., Ollama)
    or Anthropic. Defaults to local to avoid rate limits/credits.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "openai_local")).lower()

        if self.provider.startswith("anthropic"):
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("ANTHROPIC_API_KEY not set. Please add it to .env")
            self.client = Anthropic(api_key=key)
            self.model = model or os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")
            self.mode = "anthropic"
        else:
            # Default to local OpenAI-compatible endpoint (e.g., Ollama)
            base = base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
            key = api_key or os.getenv("LLM_API_KEY", "ollama")
            self.client = OpenAI(base_url=base, api_key=key)
            self.model = model or os.getenv("LLM_MODEL", "phi3")
            self.mode = "openai"

    def chat(self, messages: List[Dict[str, Any]], max_tokens: int = 512) -> str:
        """Send chat messages and return the text content."""
        if self.mode == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages,
            )
            return self._extract_anthropic(response)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return self._extract_openai(response)

    def _extract_openai(self, response) -> str:
        """Extract text from OpenAI-style response."""
        if not response or not getattr(response, "choices", None):
            return ""

        choice = response.choices[0]
        content = getattr(choice.message, "content", "")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []
            for part in content:
                if hasattr(part, "text"):
                    parts.append(part.text)
                elif isinstance(part, str):
                    parts.append(part)
            return "".join(parts).strip()

        return str(content).strip()

    def _extract_anthropic(self, response) -> str:
        """Extract text from Anthropic response."""
        if hasattr(response, "content") and isinstance(response.content, list):
            for block in response.content:
                if getattr(block, "type", "") == "text" and hasattr(block, "text"):
                    return block.text.strip()
        if hasattr(response, "text"):
            return response.text.strip()
        return ""
