"""
Gemini: base_url https://generativelanguage.googleapis.com/v1beta/openai
Grok:   base_url https://api.x.ai/v1
"""

from __future__ import annotations

import httpx


class ChatAPIGenerator:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        context_limit: int = 4096,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        self.name = model
        self.context_limit = context_limit
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._model = model
        self._temperature = temperature
        self._timeout = timeout

    def count_tokens(self, text: str) -> int:
        return len(text) // 4 + 1

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        response = httpx.post(
            self._url,
            headers=self._headers,
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self._temperature,
                "max_tokens": max_new_tokens,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"]).strip()
