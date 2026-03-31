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
        "case_result": {"outcome": "FAIL"},
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
        "evidence": {"case_id": "api.example.com::TC-EX-001"},
        "prompt_bundle": {"case_evidence": {}, "pipeline_context": {}},
    }


class LlmExplanationEndpointTests(unittest.TestCase):
    def test_explain_failed_case_returns_payload_when_llm_succeeds(self) -> None:
        expected_payload = {
            "ok": True,
            "mode": "explanation",
            "explanation": "Confirmed cause: sample text",
            "used_fallback": False,
            "llm_error": None,
            "contract": {"cause": "sample"},
            "attempts": [{"prompt": "Case:", "raw_response": "{}", "parse_error": None, "validation_errors": []}],
        }

        with (
            patch.object(main, "SessionLocal", return_value=_DummySession()),
            patch.object(main, "_load_run_case_bundle", return_value=_bundle()),
            patch.object(main, "_resolve_effective_llm_settings", return_value={"active_model_id": "builtin:ollama:qwen3-coder:latest"}),
            patch.object(
                main,
                "_generate_failure_explanation_or_raise",
                return_value={
                    "runtime": {"provider": "ollama", "model": "qwen3-coder:latest"},
                    "payload": expected_payload,
                },
            ),
        ):
            response = main.explain_failed_case(run_id=10, test_id="TC-EX-001", current_user=SimpleNamespace(id=7))

        self.assertEqual(response["run_id"], 10)
        self.assertEqual(response["test_id"], "TC-EX-001")
        self.assertEqual(response["payload"], expected_payload)

    def test_explain_failed_case_returns_502_with_diagnostics_when_llm_fails(self) -> None:
        expected_error = HTTPException(
            status_code=502,
            detail={"message": "LLM explanation failed after retry attempts.", "llm_error": "empty response"},
        )
        with (
            patch.object(main, "SessionLocal", return_value=_DummySession()),
            patch.object(main, "_load_run_case_bundle", return_value=_bundle()),
            patch.object(main, "_resolve_effective_llm_settings", return_value={"active_model_id": "builtin:ollama:qwen3-coder:latest"}),
            patch.object(
                main,
                "_generate_failure_explanation_or_raise",
                side_effect=expected_error,
            ),
        ):
            with self.assertRaises(HTTPException) as ctx:
                main.explain_failed_case(run_id=10, test_id="TC-EX-001", current_user=SimpleNamespace(id=7))

        self.assertEqual(ctx.exception.status_code, 502)
        self.assertEqual(ctx.exception.detail, expected_error.detail)

    def test_explain_failed_case_returns_429_with_provider_metadata_for_rate_limit(self) -> None:
        expected_error = HTTPException(
            status_code=429,
            detail={
                "message": "LLM explanation failed after retry attempts.",
                "failure_kind": "rate_limit",
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "guidance": "Provider rate limit reached. Wait and retry, or switch models in Settings.",
            },
        )
        with (
            patch.object(main, "SessionLocal", return_value=_DummySession()),
            patch.object(main, "_load_run_case_bundle", return_value=_bundle()),
            patch.object(main, "_resolve_effective_llm_settings", return_value={"active_model_id": "user:model:1"}),
            patch.object(
                main,
                "_generate_failure_explanation_or_raise",
                side_effect=expected_error,
            ),
        ):
            with self.assertRaises(HTTPException) as ctx:
                main.explain_failed_case(run_id=10, test_id="TC-EX-001", current_user=SimpleNamespace(id=7))

        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(ctx.exception.detail, expected_error.detail)

    def test_load_run_case_bundle_surfaces_settings_error(self) -> None:
        with patch.object(main, "load_backend_llm_settings", side_effect=ValueError("bad config")):
            with self.assertRaises(HTTPException) as ctx:
                main._load_run_case_bundle(
                    db=object(),
                    user_id=1,
                    run_id=1,
                    test_id="TC-EX-001",
                    selection_reason="user_requested_explanation",
                )
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("LLM settings error", str(ctx.exception.detail))


if __name__ == "__main__":
    unittest.main()
