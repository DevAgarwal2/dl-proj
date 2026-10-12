"""OpenAI-compatible client for a local llama.cpp server (LFM2.5 GGUF)."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import requests

from openmed.utils import get_logger

LOGGER = get_logger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LLMError(RuntimeError):
    pass


def strip_reasoning(text: str) -> str:
    """Remove ``<think>...</think>`` blocks emitted by reasoning models."""
    cleaned = _THINK_RE.sub("", text)
    # Drop a dangling opening tag if the server truncated the trace.
    if "<think>" in cleaned and "<｜end▁of▁thinking｜>" not in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned.strip()


class LLMClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "LFM2.5-1.2B-Instruct-Q6_K.gguf",
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 512,
        timeout: int = 120,
        strip_reasoning_output: bool = True,
        api_key: str = "not-needed",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.strip_reasoning_output = strip_reasoning_output
        self.api_key = api_key

    # ------------------------------------------------------------------ health
    def health(self) -> bool:
        try:
            response = requests.get(
                f"{self.base_url.replace('/v1', '')}/health", timeout=5
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    # ------------------------------------------------------------------- chat
    def chat(self, messages: List[Dict[str, str]], **overrides) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": overrides.get("temperature", self.temperature),
            "top_p": overrides.get("top_p", self.top_p),
            "max_tokens": overrides.get("max_tokens", self.max_tokens),
        }
        # Optional structured-output hint (e.g. {"type": "json_object"}).
        response_format = overrides.get("response_format")
        if response_format is not None:
            payload["response_format"] = response_format
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(
                f"llama-server request failed ({self.base_url}). "
                "Is it running? Try: make llama"
            ) from exc

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected response: {data}") from exc
        return strip_reasoning(content) if self.strip_reasoning_output else content

    def complete(self, prompt: str, system: Optional[str] = None, **overrides) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **overrides)
