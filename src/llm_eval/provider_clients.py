from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import requests


def _coerce_timeout(timeout_seconds: int) -> int:
    return max(1, int(timeout_seconds))


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                chunks.append(text)
        return "\n".join(chunk for chunk in chunks if chunk).strip()
    return ""


def _normalize_model_catalog_entries(raw_models: Any, *, label_keys: tuple[str, ...]) -> list[dict[str, str]]:
    if not isinstance(raw_models, list):
        return []

    seen_ids: set[str] = set()
    normalized: list[dict[str, str]] = []
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id or model_id in seen_ids:
            continue
        seen_ids.add(model_id)

        label = ""
        for key in label_keys:
            candidate = str(item.get(key) or "").strip()
            if candidate:
                label = candidate
                break
        if not label:
            label = model_id

        normalized.append({"id": model_id, "label": label})

    normalized.sort(key=lambda entry: entry["id"].lower())
    return normalized


class OpenAIClient:
    def __init__(self, *, api_key: str, base_url: str = "https://api.openai.com/v1", timeout_seconds: int = 120) -> None:
        self.api_key = str(api_key).strip()
        self.base_url = str(base_url or "https://api.openai.com/v1").rstrip("/")
        self.timeout_seconds = _coerce_timeout(timeout_seconds)

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        format_json: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        if not self.api_key:
            return None, "OpenAI API key is missing."
        url = f"{self.base_url}/chat/completions"
        messages = []
        if isinstance(system, str) and system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if format_json:
            payload["response_format"] = {"type": "json_object"}
        if isinstance(options, dict):
            if options.get("temperature") is not None:
                payload["temperature"] = options.get("temperature")
            if options.get("max_tokens") is not None:
                payload["max_tokens"] = options.get("max_tokens")
            if options.get("max_completion_tokens") is not None:
                payload["max_completion_tokens"] = options.get("max_completion_tokens")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout_seconds)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001 - surface provider and HTTP failures
            return None, f"{type(exc).__name__}: {exc}"

        if not isinstance(data, dict):
            return None, f"Unexpected OpenAI response payload: {data}"
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return None, f"OpenAI response missing choices: {data}"
        message = choices[0].get("message") if isinstance(choices[0], dict) else {}
        content = message.get("content") if isinstance(message, dict) else ""
        text = _extract_text_content(content)
        if not text:
            return None, f"OpenAI response missing message content: {data}"
        return text, None

    def list_models(self) -> Tuple[Optional[list[dict[str, str]]], Optional[str]]:
        if not self.api_key:
            return None, "OpenAI API key is missing."

        url = f"{self.base_url}/models"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.get(url, headers=headers, timeout=self.timeout_seconds)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001 - surface provider and HTTP failures
            return None, f"{type(exc).__name__}: {exc}"

        if not isinstance(data, dict):
            return None, f"Unexpected OpenAI response payload: {data}"
        normalized = _normalize_model_catalog_entries(data.get("data"), label_keys=("id",))
        return normalized, None


class AnthropicClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.anthropic.com/v1",
        timeout_seconds: int = 120,
        anthropic_version: str = "2023-06-01",
    ) -> None:
        self.api_key = str(api_key).strip()
        self.base_url = str(base_url or "https://api.anthropic.com/v1").rstrip("/")
        self.timeout_seconds = _coerce_timeout(timeout_seconds)
        self.anthropic_version = str(anthropic_version).strip() or "2023-06-01"

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        format_json: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        if not self.api_key:
            return None, "Anthropic API key is missing."
        url = f"{self.base_url}/messages"
        user_content = str(prompt or "")
        if format_json:
            user_content = (
                f"{user_content}\n\n"
                "Return valid JSON only."
            )
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": 900,
        }
        if isinstance(system, str) and system.strip():
            payload["system"] = system
        if isinstance(options, dict):
            if options.get("temperature") is not None:
                payload["temperature"] = options.get("temperature")
            if options.get("max_tokens") is not None:
                payload["max_tokens"] = options.get("max_tokens")

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout_seconds)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001 - surface provider and HTTP failures
            return None, f"{type(exc).__name__}: {exc}"

        if not isinstance(data, dict):
            return None, f"Unexpected Anthropic response payload: {data}"
        content = data.get("content")
        text = _extract_text_content(content)
        if not text:
            return None, f"Anthropic response missing text content: {data}"
        return text, None

    def list_models(self) -> Tuple[Optional[list[dict[str, str]]], Optional[str]]:
        if not self.api_key:
            return None, "Anthropic API key is missing."

        url = f"{self.base_url}/models"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "Content-Type": "application/json",
        }
        try:
            response = requests.get(url, headers=headers, timeout=self.timeout_seconds)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001 - surface provider and HTTP failures
            return None, f"{type(exc).__name__}: {exc}"

        if not isinstance(data, dict):
            return None, f"Unexpected Anthropic response payload: {data}"
        normalized = _normalize_model_catalog_entries(data.get("data"), label_keys=("display_name", "id"))
        return normalized, None
