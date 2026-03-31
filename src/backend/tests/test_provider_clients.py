import unittest
from unittest.mock import patch

import requests

from llm_eval.provider_clients import AnthropicClient, OpenAIClient


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}")

    def json(self) -> dict:
        return self._payload


class ProviderClientTests(unittest.TestCase):
    def test_openai_generate_parses_message_content(self) -> None:
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": "{\"ok\":true}",
                    }
                }
            ]
        }
        with patch("llm_eval.provider_clients.requests.post", return_value=_FakeResponse(response_payload)) as post_mock:
            client = OpenAIClient(api_key="sk-test", base_url="https://api.openai.com/v1", timeout_seconds=45)
            text, error = client.generate(
                model="gpt-4.1-mini",
                prompt="Return JSON",
                system="You are helpful.",
                format_json=True,
                options={"temperature": 0},
            )

        self.assertIsNone(error)
        self.assertEqual(text, "{\"ok\":true}")
        self.assertIn("/chat/completions", str(post_mock.call_args.args[0]))

    def test_anthropic_generate_parses_content_blocks(self) -> None:
        response_payload = {
            "content": [
                {"type": "text", "text": "{\"mode\":\"ok\"}"}
            ]
        }
        with patch("llm_eval.provider_clients.requests.post", return_value=_FakeResponse(response_payload)) as post_mock:
            client = AnthropicClient(api_key="claude-test", timeout_seconds=60)
            text, error = client.generate(
                model="claude-sonnet-4-5",
                prompt="Return JSON",
                system="System",
                format_json=True,
                options={"max_tokens": 256},
            )

        self.assertIsNone(error)
        self.assertEqual(text, "{\"mode\":\"ok\"}")
        self.assertIn("/messages", str(post_mock.call_args.args[0]))

    def test_openai_generate_returns_error_when_key_missing(self) -> None:
        client = OpenAIClient(api_key="")
        text, error = client.generate(model="gpt-4.1-mini", prompt="x")
        self.assertIsNone(text)
        self.assertIn("missing", str(error).lower())

    def test_openai_list_models_parses_and_deduplicates(self) -> None:
        response_payload = {
            "data": [
                {"id": "gpt-4.1-mini"},
                {"id": "gpt-4.1"},
                {"id": "gpt-4.1-mini"},
            ]
        }
        with patch("llm_eval.provider_clients.requests.get", return_value=_FakeResponse(response_payload)) as get_mock:
            client = OpenAIClient(api_key="sk-test", base_url="https://api.openai.com/v1")
            models, error = client.list_models()

        self.assertIsNone(error)
        self.assertEqual(models, [{"id": "gpt-4.1", "label": "gpt-4.1"}, {"id": "gpt-4.1-mini", "label": "gpt-4.1-mini"}])
        self.assertIn("/models", str(get_mock.call_args.args[0]))

    def test_anthropic_list_models_uses_display_name(self) -> None:
        response_payload = {
            "data": [
                {"id": "claude-sonnet-4-5", "display_name": "Claude Sonnet 4.5"},
                {"id": "claude-opus-4-1", "display_name": "Claude Opus 4.1"},
            ]
        }
        with patch("llm_eval.provider_clients.requests.get", return_value=_FakeResponse(response_payload)) as get_mock:
            client = AnthropicClient(api_key="claude-test", base_url="https://api.anthropic.com/v1")
            models, error = client.list_models()

        self.assertIsNone(error)
        self.assertEqual(
            models,
            [
                {"id": "claude-opus-4-1", "label": "Claude Opus 4.1"},
                {"id": "claude-sonnet-4-5", "label": "Claude Sonnet 4.5"},
            ],
        )
        self.assertIn("/models", str(get_mock.call_args.args[0]))

    def test_anthropic_list_models_returns_error_when_key_missing(self) -> None:
        client = AnthropicClient(api_key="")
        models, error = client.list_models()
        self.assertIsNone(models)
        self.assertIn("missing", str(error).lower())


if __name__ == "__main__":
    unittest.main()
