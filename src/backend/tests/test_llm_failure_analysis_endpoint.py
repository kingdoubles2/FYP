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
        "case_result": {"outcome": "FAIL", "actual_status": 422},
        "settings": {
            "base_url": "http://localhost:11434",
            "timeout_seconds": 120,
            "model": "qwen3-coder:latest",
            "word_target": 55,
            "word_max": 100,
            "retry_invalid_output": 1,
            "max_items_per_section": 12,
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


class LlmFailureAnalysisEndpointTests(unittest.TestCase):
    def test_analyze_failure_returns_explanation_and_llm_suggestion(self) -> None:
        explanation_payload = {
            "ok": True,
            "mode": "explanation",
            "signal": "validation",
            "explanation": "Sample explanation text.",
            "contract": {"cause": "Invalid input"},
            "prompt_snapshot": {"system": "System prompt text", "user": "User prompt text"},
        }
        suggestion_payload = {
            "mode": "suggest_test",
            "signal": "validation",
            "reason": "deterministic",
            "can_apply": True,
            "suggested_test_case": {"test_id": "TC-EX-001-FOLLOWUP"},
            "skipped": False,
            "eligible_for_generation": True,
        }
        with (
            patch.object(main, "SessionLocal", return_value=_DummySession()),
            patch.object(main, "_load_run_case_bundle", return_value=_bundle()),
            patch.object(main, "_resolve_effective_llm_settings", return_value={"active_model_id": "builtin:ollama:qwen3-coder:latest"}),
            patch.object(
                main,
                "_generate_failure_explanation_or_raise",
                return_value={
                    "runtime": {"provider": "ollama", "model": "qwen3-coder:latest", "client": object(), "options": {}},
                    "payload": explanation_payload,
                },
            ),
            patch.object(main, "generate_suggested_test", return_value=suggestion_payload) as llm_suggest,
        ):
            response = main.analyze_failure_with_suggestion(
                run_id=10,
                test_id="TC-EX-001",
                current_user=SimpleNamespace(id=7),
            )

        self.assertEqual(response["run_id"], 10)
        self.assertEqual(response["test_id"], "TC-EX-001")
        self.assertEqual(response["mode"], "analysis")
        self.assertEqual(response["payload"]["explanation"], explanation_payload)
        self.assertEqual(response["payload"]["suggestion"], suggestion_payload)
        llm_suggest.assert_called_once()

    def test_analyze_failure_propagates_explanation_failure_http_error(self) -> None:
        expected_error = HTTPException(
            status_code=429,
            detail={"message": "LLM explanation failed after retry attempts.", "failure_kind": "rate_limit"},
        )
        with (
            patch.object(main, "SessionLocal", return_value=_DummySession()),
            patch.object(main, "_load_run_case_bundle", return_value=_bundle()),
            patch.object(main, "_resolve_effective_llm_settings", return_value={"active_model_id": "user:model:1"}),
            patch.object(main, "_generate_failure_explanation_or_raise", side_effect=expected_error),
        ):
            with self.assertRaises(HTTPException) as ctx:
                main.analyze_failure_with_suggestion(
                    run_id=10,
                    test_id="TC-EX-001",
                    current_user=SimpleNamespace(id=7),
                )

        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(ctx.exception.detail, expected_error.detail)

    def test_analyze_failure_explanation_matches_explanation_endpoint_payload(self) -> None:
        explanation_payload = {
            "ok": True,
            "mode": "explanation",
            "signal": "validation",
            "explanation": "Sample explanation text.",
            "contract": {"cause": "Invalid input"},
            "prompt_snapshot": {"system": "System prompt text", "user": "User prompt text"},
        }
        suggestion_payload = {
            "mode": "suggest_test",
            "signal": "validation",
            "reason": "deterministic",
            "can_apply": True,
            "suggested_test_case": {"test_id": "TC-EX-001-FOLLOWUP"},
            "skipped": False,
            "eligible_for_generation": True,
        }
        with (
            patch.object(main, "SessionLocal", return_value=_DummySession()),
            patch.object(main, "_load_run_case_bundle", return_value=_bundle()),
            patch.object(main, "_resolve_effective_llm_settings", return_value={"active_model_id": "builtin:ollama:qwen3-coder:latest"}),
            patch.object(
                main,
                "_generate_failure_explanation_or_raise",
                return_value={
                    "runtime": {"provider": "ollama", "model": "qwen3-coder:latest", "client": object(), "options": {}},
                    "payload": explanation_payload,
                },
            ),
            patch.object(main, "generate_suggested_test", return_value=suggestion_payload),
        ):
            explanation_response = main.explain_failed_case(
                run_id=10,
                test_id="TC-EX-001",
                current_user=SimpleNamespace(id=7),
            )
            analysis_response = main.analyze_failure_with_suggestion(
                run_id=10,
                test_id="TC-EX-001",
                current_user=SimpleNamespace(id=7),
            )

        self.assertEqual(
            explanation_response["payload"],
            analysis_response["payload"]["explanation"],
        )


if __name__ == "__main__":
    unittest.main()
