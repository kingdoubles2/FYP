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
                "_build_runtime_from_active_model",
                return_value={"client": object(), "model": "qwen3-coder:latest", "options": {}},
            ),
            patch.object(main, "generate_failure_explanation", return_value=expected_payload),
        ):
            response = main.explain_failed_case(run_id=10, test_id="TC-EX-001", current_user=SimpleNamespace(id=7))

        self.assertEqual(response["run_id"], 10)
        self.assertEqual(response["test_id"], "TC-EX-001")
        self.assertEqual(response["payload"], expected_payload)

    def test_explain_failed_case_returns_502_with_diagnostics_when_llm_fails(self) -> None:
        failed_payload = {
            "ok": False,
            "mode": "explanation",
            "used_fallback": False,
            "llm_error": "empty response",
            "validation_errors": ["parse_error"],
            "attempts": [
                {
                    "prompt": "Case:",
                    "raw_response": "",
                    "llm_error": None,
                    "parse_error": "empty response",
                    "validation_errors": [],
                }
            ],
        }

        with (
            patch.object(main, "SessionLocal", return_value=_DummySession()),
            patch.object(main, "_load_run_case_bundle", return_value=_bundle()),
            patch.object(main, "_resolve_effective_llm_settings", return_value={"active_model_id": "builtin:ollama:qwen3-coder:latest"}),
            patch.object(
                main,
                "_build_runtime_from_active_model",
                return_value={"client": object(), "model": "qwen3-coder:latest", "options": {}},
            ),
            patch.object(main, "generate_failure_explanation", return_value=failed_payload),
        ):
            with self.assertRaises(HTTPException) as ctx:
                main.explain_failed_case(run_id=10, test_id="TC-EX-001", current_user=SimpleNamespace(id=7))

        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIsInstance(ctx.exception.detail, dict)
        self.assertIn("llm_error", ctx.exception.detail)
        self.assertEqual(ctx.exception.detail["llm_error"], "empty response")
        self.assertIn("attempt_diagnostics", ctx.exception.detail)
        self.assertTrue(ctx.exception.detail["attempt_diagnostics"])

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
