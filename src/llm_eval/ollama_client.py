from __future__ import annotations

from typing import Optional, Tuple

import requests


class OllamaClient:
    def __init__(self, base_url: str, timeout_seconds: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = max(1, int(timeout_seconds))

    def generate(self, model: str, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - surface all client/HTTP errors
            return None, f"{type(exc).__name__}: {exc}"

        if not isinstance(data, dict) or "response" not in data:
            return None, f"Unexpected Ollama response: {data}"

        return data.get("response"), None
