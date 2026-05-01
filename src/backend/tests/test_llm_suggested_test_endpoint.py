from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from backend.app import main


class _DummySession:
    def close(self) -> None:
        return None


def _bundle() -> dict:
    return {
        "case_result": {"outcome": "FAIL", "actual_status": 200},
        "settings": {
            "base_url": "http://localhost:11434",
            "timeout_seconds": 120,
            "model": "qwen3-coder:latest",
            "retry_invalid_output": 1,
            "suggestion_timeout_seconds": 45,
            "suggestion_retry_invalid_output": 0,
            "ollama_options": {},
        },
        "evidence": {
            "case_id": "api.example.com::TC-EX-001",
            "test_context": {"category": "negative_invalid"},
            "execution": {
                "response_received": {"status": 422, "body_snippet": '{"detail":"invalid input"}'},
                "assertion_failures": [{"expected": 404, "actual_status": 422}],
            },
        },
        "prompt_bundle": {"case_evidence": {}, "pipeline_context": {}},
        "test_case": {
            "test_id": "TC-EX-001",
            "title": "Sample case",
            "category": "negative_invalid",
            "method": "GET",
            "path": "/users",
            "priority": "medium",
            "steps": [{"step_number": 1, "action": "Execute request", "input_data": {}}],
            "expected_result": {"status_code": 404, "description": "Not found"},
        },
        "suite_data": {"test_cases": [{"test_id": "TC-EX-001"}]},
    }


class LlmSuggestedTestEndpointTests(unittest.TestCase):
    def test_suggest_test_returns_429_for_provider_rate_limit(self) -> None:
        failed_payload = {
            "mode": "suggest_test",
            "reason": "rate limited",
            "signal": "validation",
            "external_failure": False,
            "warning_external": False,
            "warning": "",
            "can_apply": False,
            "suggested_test_case": None,
            "model": "gpt-4.1-mini",
            "used_fallback": False,
            "llm_error": "HTTPError: 429 Client Error: Too Many Requests",
            "failure_mode": "rate_limit",
            "failure_kind": "rate_limit",
            "status_code": 429,
            "error_meta": {"kind": "rate_limit", "status_code": 429},
            "attempts": ["llm_error:HTTPError: 429 Client Error: Too Many Requests"],
        }
        with (
            patch.object(main, "SessionLocal", return_value=_DummySession()),
            patch.object(main, "_load_run_case_bundle", return_value=_bundle()),
            patch.object(main, "_resolve_effective_llm_settings", return_value={"active_model_id": "user:model:1"}),
            patch.object(
                main,
                "_build_runtime_from_active_model",
                return_value={"provider": "openai", "client": object(), "model": "gpt-4.1-mini", "options": {}},
            ),
            patch.object(main, "_generate_failure_explanation_or_raise") as generate_explanation,
            patch.object(main, "generate_suggested_test", return_value=failed_payload),
        ):
            with self.assertRaises(HTTPException) as ctx:
                main.suggest_test_for_failure(run_id=10, test_id="TC-EX-001", current_user=SimpleNamespace(id=7))

        self.assertEqual(ctx.exception.status_code, 429)
        self.assertIsInstance(ctx.exception.detail, dict)
        self.assertEqual(ctx.exception.detail.get("failure_kind"), "rate_limit")
        self.assertEqual(ctx.exception.detail.get("provider"), "openai")
        self.assertEqual(ctx.exception.detail.get("model"), "gpt-4.1-mini")
        generate_explanation.assert_not_called()

    def test_suggest_test_keeps_fallback_payload_for_non_rate_limit_failures(self) -> None:
        fallback_payload = {
            "mode": "suggest_test",
            "reason": "fallback",
            "signal": "validation",
            "external_failure": False,
            "warning_external": False,
            "warning": "",
            "can_apply": True,
            "suggested_test_case": {"test_id": "TC-X"},
            "model": "qwen3-coder:latest",
            "used_fallback": True,
            "llm_error": "parse_error:invalid_json",
            "failure_mode": "invalid_schema",
        }
        with (
            patch.object(main, "SessionLocal", return_value=_DummySession()),
            patch.object(main, "_load_run_case_bundle", return_value=_bundle()),
            patch.object(main, "_resolve_effective_llm_settings", return_value={"active_model_id": "builtin:ollama:qwen3-coder:latest"}),
            patch.object(
                main,
                "_build_runtime_from_active_model",
                return_value={"provider": "ollama", "client": object(), "model": "qwen3-coder:latest", "options": {}},
            ),
            patch.object(main, "_generate_failure_explanation_or_raise") as generate_explanation,
            patch.object(main, "generate_suggested_test", return_value=fallback_payload),
        ):
            response = main.suggest_test_for_failure(run_id=10, test_id="TC-EX-001", current_user=SimpleNamespace(id=7))

        self.assertEqual(response["run_id"], 10)
        self.assertEqual(response["test_id"], "TC-EX-001")
        self.assertEqual(response["payload"], fallback_payload)
        generate_explanation.assert_not_called()

    def test_suggest_test_does_not_prime_explanation_when_request_context_missing(self) -> None:
        suggestion_payload = {
            "mode": "suggest_test",
            "reason": "fallback",
            "signal": "validation",
            "external_failure": False,
            "warning_external": False,
            "warning": "",
            "can_apply": True,
            "suggested_test_case": {"test_id": "TC-X"},
            "model": "gpt-4.1-mini",
            "used_fallback": False,
            "llm_error": None,
            "failure_mode": "none",
        }
        with (
            patch.object(main, "SessionLocal", return_value=_DummySession()),
            patch.object(main, "_load_run_case_bundle", return_value=_bundle()),
            patch.object(main, "_resolve_effective_llm_settings", return_value={"active_model_id": "user:model:1"}),
            patch.object(
                main,
                "_build_runtime_from_active_model",
                return_value={"provider": "openai", "client": object(), "model": "gpt-4.1-mini", "options": {}},
            ) as build_runtime,
            patch.object(main, "_generate_failure_explanation_or_raise") as generate_explanation,
            patch.object(main, "generate_suggested_test", return_value=suggestion_payload) as generate_suggested_test,
        ):
            response = main.suggest_test_for_failure(run_id=10, test_id="TC-EX-001", current_user=SimpleNamespace(id=7))

        self.assertEqual(response["run_id"], 10)
        self.assertEqual(response["test_id"], "TC-EX-001")
        self.assertEqual(response["payload"], suggestion_payload)
        generate_explanation.assert_not_called()
        build_runtime.assert_called_once()
        generate_suggested_test.assert_called_once()
        called_context = generate_suggested_test.call_args.kwargs.get("explanation_context")
        self.assertIsInstance(called_context, dict)
        self.assertEqual(str(called_context.get("signal")), "schema_value")
        self.assertFalse(bool(called_context.get("explanation")))

    def test_suggest_test_uses_llm_generation_path_for_non_input_signal(self) -> None:
        soft_defer_payload = {
            "mode": "suggest_test",
            "reason": "This failure does not appear input-related. AI is deferring a follow-up test suggestion for now.",
            "signal": "status_mismatch",
            "external_failure": False,
            "warning_external": False,
            "warning": "",
            "can_apply": False,
            "suggested_test_case": None,
            "model": "qwen3-coder:latest",
            "used_fallback": False,
            "llm_error": None,
            "failure_mode": "none",
            "skipped": True,
            "skip_reason": "This failure does not appear input-related. AI is deferring a follow-up test suggestion for now.",
            "eligible_for_generation": False,
        }
        with (
            patch.object(main, "SessionLocal", return_value=_DummySession()),
            patch.object(main, "_load_run_case_bundle", return_value=_bundle()),
            patch.object(
                main,
                "_resolve_effective_llm_settings",
                return_value={"active_model_id": "builtin:ollama:qwen3-coder:latest"},
            ),
            patch.object(
                main,
                "_build_runtime_from_active_model",
                return_value={"provider": "ollama", "client": object(), "model": "qwen3-coder:latest", "options": {}},
            ) as build_runtime,
            patch.object(main, "_generate_failure_explanation_or_raise") as generate_explanation,
            patch.object(main, "generate_suggested_test", return_value=soft_defer_payload) as generate_suggested_test,
        ):
            response = main.suggest_test_for_failure(
                run_id=10,
                test_id="TC-EX-001",
                req=main.SuggestTestRequest(explanation={"signal": "status_mismatch"}),
                current_user=SimpleNamespace(id=7),
            )

        self.assertEqual(response["run_id"], 10)
        self.assertEqual(response["test_id"], "TC-EX-001")
        payload = response["payload"]
        self.assertTrue(bool(payload.get("skipped")))
        self.assertFalse(bool(payload.get("eligible_for_generation")))
        self.assertFalse(payload.get("can_apply"))
        self.assertIn("input-related", str(payload.get("reason") or ""))
        build_runtime.assert_called_once()
        generate_explanation.assert_not_called()
        generate_suggested_test.assert_called_once()


if __name__ == "__main__":
    unittest.main()
