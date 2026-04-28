from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException

from backend.app import main


class _DummyQuery:
    def __init__(self, first_value=None) -> None:
        self._first_value = first_value

    def filter(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return self

    def order_by(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return self

    def first(self):
        return self._first_value


class _DummySession:
    def __init__(self) -> None:
        self.query_calls = 0

    def query(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.query_calls += 1
        return _DummyQuery(None)

    def close(self) -> None:
        return None


def _resolved_settings() -> dict:
    return {
        "active_model": {"provider": "openai", "model": "gpt-4.1-mini"},
        "custom_instruction": "Stay grounded.",
    }


def _chat_request(*, message: str, spec_id: int = 11, latest_run: dict | None = None) -> main.SpecAssistantChatRequest:
    return main.SpecAssistantChatRequest(
        spec_id=spec_id,
        message=message,
        thread=[
            {"role": "user", "content": "What failed in the last run?"},
            {"role": "assistant", "content": "TC-01 failed with 422."},
        ],
        context_snapshot={
            "parsed_spec": {
                "title": "Sample API",
                "version": "1.0.0",
                "base_url": "https://api.example.com",
                "endpoints": [{"method": "GET", "path": "/users", "operation_id": "listUsers"}],
            },
            "generated_tests": [
                {
                    "test_id": "TC-01",
                    "title": "List users",
                    "category": "happy_path",
                    "method": "GET",
                    "path": "/users",
                    "expected_result": {"status_code": 200},
                }
            ],
            "latest_run": latest_run,
        },
    )


class LlmChatEndpointTests(unittest.TestCase):
    def test_chat_returns_grounded_payload_when_llm_succeeds(self) -> None:
        runtime_client = Mock()
        runtime_client.generate.return_value = ("TC-01 failed due to status mismatch.", None)
        runtime = {"provider": "openai", "model": "gpt-4.1-mini", "client": runtime_client, "options": {}}

        with (
            patch.object(main, "SessionLocal", return_value=_DummySession()),
            patch.object(main, "_get_owned_spec", return_value=SimpleNamespace(id=11, title="Sample API", version="1.0.0")),
            patch.object(main, "_resolve_effective_llm_settings", return_value=_resolved_settings()),
            patch.object(main, "_build_runtime_from_active_model", return_value=runtime),
        ):
            response = main.chat_with_spec_assistant(
                req=_chat_request(message="Why did TC-01 fail?"),
                current_user=SimpleNamespace(id=7),
            )

        assistant = response["assistantMessage"]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["context"], {"mode": "spec", "specId": 11})
        self.assertIn("status mismatch", assistant["content"])
        self.assertEqual(response["meta"]["provider"], "openai")
        self.assertEqual(response["meta"]["model"], "gpt-4.1-mini")
        self.assertEqual(response["meta"]["test_count"], 1)
        generate_kwargs = runtime_client.generate.call_args.kwargs
        self.assertIn("spec-grounded assistant", str(generate_kwargs.get("system", "")).lower())
        self.assertNotIn("Stay grounded.", str(generate_kwargs.get("system", "")))

    def test_chat_rejects_empty_message(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            main.chat_with_spec_assistant(
                req=_chat_request(message="   "),
                current_user=SimpleNamespace(id=7),
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("cannot be empty", str(ctx.exception.detail).lower())

    def test_chat_rejects_non_owned_spec(self) -> None:
        with (
            patch.object(main, "SessionLocal", return_value=_DummySession()),
            patch.object(main, "_get_owned_spec", side_effect=HTTPException(status_code=404, detail="Specification not found.")),
        ):
            with self.assertRaises(HTTPException) as ctx:
                main.chat_with_spec_assistant(
                    req=_chat_request(message="Show endpoint list.", spec_id=99),
                    current_user=SimpleNamespace(id=7),
                )

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIn("not found", str(ctx.exception.detail).lower())

    def test_chat_out_of_scope_returns_refusal_without_model_call(self) -> None:
        with (
            patch.object(main, "SessionLocal", return_value=_DummySession()),
            patch.object(main, "_get_owned_spec", return_value=SimpleNamespace(id=11, title="Sample API", version="1.0.0")),
            patch.object(main, "_resolve_effective_llm_settings", return_value=_resolved_settings()),
            patch.object(main, "_build_runtime_from_active_model") as build_runtime,
        ):
            response = main.chat_with_spec_assistant(
                req=_chat_request(message="What's the weather in Dublin tomorrow?"),
                current_user=SimpleNamespace(id=7),
            )

        self.assertTrue(bool(response["meta"]["out_of_scope"]))
        self.assertIn("selected api spec context", response["assistantMessage"]["content"].lower())
        build_runtime.assert_not_called()

    def test_chat_handles_no_run_context(self) -> None:
        runtime_client = Mock()
        runtime_client.generate.return_value = ("No run data exists yet for this spec.", None)
        runtime = {"provider": "openai", "model": "gpt-4.1-mini", "client": runtime_client, "options": {}}

        with (
            patch.object(main, "SessionLocal", return_value=_DummySession()),
            patch.object(main, "_get_owned_spec", return_value=SimpleNamespace(id=11, title="Sample API", version="1.0.0")),
            patch.object(main, "_resolve_effective_llm_settings", return_value=_resolved_settings()),
            patch.object(main, "_build_runtime_from_active_model", return_value=runtime),
        ):
            response = main.chat_with_spec_assistant(
                req=_chat_request(message="Do we have run results yet?", latest_run=None),
                current_user=SimpleNamespace(id=7),
            )

        self.assertFalse(bool(response["meta"]["has_latest_run"]))
        self.assertIn("no run data", response["assistantMessage"]["content"].lower())

    def test_chat_propagates_provider_rate_limit_error(self) -> None:
        runtime_client = Mock()
        runtime_client.generate.return_value = (
            None,
            {
                "kind": "rate_limit",
                "message": "Rate limit reached",
                "status_code": 429,
                "provider_error_code": "rate_limit_exceeded",
            },
        )
        runtime = {"provider": "openai", "model": "gpt-4.1-mini", "client": runtime_client, "options": {}}

        with (
            patch.object(main, "SessionLocal", return_value=_DummySession()),
            patch.object(main, "_get_owned_spec", return_value=SimpleNamespace(id=11, title="Sample API", version="1.0.0")),
            patch.object(main, "_resolve_effective_llm_settings", return_value=_resolved_settings()),
            patch.object(main, "_build_runtime_from_active_model", return_value=runtime),
        ):
            with self.assertRaises(HTTPException) as ctx:
                main.chat_with_spec_assistant(
                    req=_chat_request(message="Explain the latest failed test."),
                    current_user=SimpleNamespace(id=7),
                )

        self.assertEqual(ctx.exception.status_code, 429)
        detail = ctx.exception.detail
        self.assertEqual(detail["failure_kind"], "rate_limit")
        self.assertIn("rate limit", str(detail["guidance"]).lower())


if __name__ == "__main__":
    unittest.main()
