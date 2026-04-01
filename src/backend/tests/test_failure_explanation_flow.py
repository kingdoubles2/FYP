import os
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_eval.failure_assistant import generate_failure_explanation, load_backend_llm_settings


def _sample_evidence() -> dict:
    return {
        "case_id": "api.example.com::TC-EX-001",
        "spec": {
            "operation": {
                "method": "GET",
                "path": "/user",
                "operation_id": "users/get-user",
                "summary": "Get user",
                "description": "",
            },
            "security_requirements": ["bearerAuth"],
            "request_constraints": {},
            "response_contract": {
                "declared_statuses": [200, 401, 404],
                "primary_expected_for_test": [404],
            },
        },
        "test_context": {
            "test_id": "TC-EX-001",
            "title": "GET /user negative path",
            "category": "negative_invalid",
            "intent": "negative",
            "generator_rule": "negative._invalid_value_cases",
            "expected_outcome": {"status_code": 404, "description": "Not found"},
            "generated_input": {"path_params": {}, "query_params": {}, "headers": {}, "body": None},
        },
        "execution": {
            "outcome": "FAIL",
            "assertion_failures": [
                {"type": "status_mismatch", "expected": 404, "actual_status": 200, "actual_error": ""}
            ],
            "request_sent": {
                "final_url": "https://api.example.com/user",
                "path_params": {},
                "query_params": {},
                "headers": {"Authorization": "<redacted>"},
                "body": None,
            },
            "response_received": {
                "status": 200,
                "headers": {"Content-Type": "application/json"},
                "body_snippet": "{\"login\":\"demo-user\",\"id\":1001}",
            },
            "duration_ms": 42.5,
        },
        "related_context": {"same_endpoint_results_summary": {"pass": 0, "fail": 1, "skip": 0}, "similar_failures": []},
        "ir_context": {"parser_warnings": [], "unsupported_spec_warnings": []},
        "missing_evidence": ["server_logs"],
    }


def _negative_type_evidence(*, executed_latitude: str) -> dict:
    evidence = _sample_evidence()
    evidence["spec"]["operation"] = {
        "method": "GET",
        "path": "/v1/forecast",
        "operation_id": "forecast/get-forecast",
        "summary": "Forecast",
        "description": "",
    }
    evidence["spec"]["request_constraints"] = {
        "query_param_rules": {
            "latitude": {"type": "number", "format": "double"},
            "longitude": {"type": "number", "format": "double"},
        }
    }
    evidence["ir_context"] = {
        "endpoint_ir": {
            "query_params": [
                {"name": "latitude", "schema": {"type": "number", "format": "double"}},
                {"name": "longitude", "schema": {"type": "number", "format": "double"}},
            ]
        },
        "parser_warnings": [],
        "unsupported_spec_warnings": [],
    }
    evidence["test_context"]["title"] = "GET /v1/forecast - Negative type: query latitude='not_a_number'"
    evidence["test_context"]["category"] = "negative_type"
    evidence["test_context"]["intent"] = "negative"
    evidence["test_context"]["expected_outcome"] = {"status_code": 400, "description": "Bad request"}
    evidence["test_context"]["generated_input"] = {
        "path_params": {},
        "query_params": {"latitude": "not_a_number", "longitude": 52.52},
        "headers": {},
        "body": None,
    }
    evidence["execution"]["assertion_failures"] = [
        {"type": "status_mismatch", "expected": 400, "actual_status": 200, "actual_error": ""}
    ]
    evidence["execution"]["request_sent"] = {
        "final_url": "https://api.example.com/v1/forecast",
        "path_params": {},
        "query_params": {"latitude": executed_latitude, "longitude": 52.52},
        "headers": {},
        "body": None,
    }
    evidence["execution"]["response_received"] = {
        "status": 200,
        "headers": {"Content-Type": "application/json"},
        "body_snippet": '{"latitude":52.52,"longitude":52.52}',
    }
    return evidence


def _valid_contract_json() -> str:
    return (
        "{"
        "\"cause\":\"Request hit an existing user and returned 200 instead of the expected 404.\","
        "\"why_likely\":\"Execution shows status 200 and a user payload body_snippet for GET /user.\","
        "\"check_next\":\"Use a token tied to a non-existent account or adjust the expected status.\","
        "\"confidence\":\"confirmed\","
        "\"expected_negative_behavior\":false,"
        "\"evidence_quotes\":[\"status\\\": 200\",\"body_snippet\\\": \\\"{\\\\\\\"login\\\\\\\"\\\"\"]"
        "}"
    )


def _minimal_contract_json(cause: str) -> str:
    return (
        "{"
        f"\"cause\":\"{cause}\","
        "\"why_likely\":\"Model-supplied why likely.\","
        "\"check_next\":\"Model-supplied next check.\","
        "\"confidence\":\"likely\","
        "\"expected_negative_behavior\":false,"
        "\"evidence_quotes\":[]"
        "}"
    )


class _StubOllamaClient:
    def __init__(self, responses: list[tuple[str | None, object]]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def generate(self, **_: object) -> tuple[str | None, object]:
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return None, "stub_exhausted"


class FailureExplanationFlowTests(unittest.TestCase):
    def test_generate_explanation_success_on_first_try_uses_sample_prompt_shape(self) -> None:
        client = _StubOllamaClient([( _valid_contract_json(), None )])
        output = generate_failure_explanation(
            client=client,
            model="qwen3-coder:latest",
            evidence=_sample_evidence(),
            prompt_bundle={"case_evidence": {"x": 1}, "pipeline_context": {"ignored": True}},
            word_target=55,
            word_max=100,
            retry_invalid_output=1,
            max_items_per_section=12,
            ollama_options={},
        )

        self.assertTrue(output["ok"])
        self.assertFalse(output["used_fallback"])
        self.assertEqual(output["llm_error"], None)
        self.assertIn("contract", output)
        self.assertIn("explanation", output)
        self.assertEqual(len(output["attempts"]), 1)

        prompt_text = str(output["attempts"][0]["prompt"])
        self.assertIn("Case:", prompt_text)
        self.assertIn("Evidence:", prompt_text)
        self.assertIn("Task:", prompt_text)
        self.assertIn("Return STRICT JSON", prompt_text)
        self.assertNotIn("pipeline_context", prompt_text)

    def test_generate_explanation_retries_after_invalid_json_then_succeeds(self) -> None:
        client = _StubOllamaClient([("not json", None), (_valid_contract_json(), None)])
        output = generate_failure_explanation(
            client=client,
            model="qwen3-coder:latest",
            evidence=_sample_evidence(),
            prompt_bundle={},
            word_target=55,
            word_max=100,
            retry_invalid_output=1,
            max_items_per_section=12,
            ollama_options={},
        )

        self.assertTrue(output["ok"])
        self.assertEqual(len(output["attempts"]), 2)
        self.assertIn("response is not valid JSON object", str(output["attempts"][0]["parse_error"]))
        self.assertEqual(output["attempts"][1]["parse_error"], None)

    def test_generate_explanation_returns_structured_failure_without_fallback(self) -> None:
        client = _StubOllamaClient([("", None), ("", None)])
        output = generate_failure_explanation(
            client=client,
            model="qwen3-coder:latest",
            evidence=_sample_evidence(),
            prompt_bundle={},
            word_target=55,
            word_max=100,
            retry_invalid_output=1,
            max_items_per_section=12,
            ollama_options={},
        )

        self.assertFalse(output["ok"])
        self.assertFalse(output["used_fallback"])
        self.assertEqual(output["explanation"], "")
        self.assertIn("empty response", str(output["llm_error"]))
        self.assertEqual(output["contract"], {})
        self.assertEqual(len(output["attempts"]), 2)

    def test_generate_explanation_openai_rate_limit_hard_fails_without_retry_loop(self) -> None:
        client = _StubOllamaClient([
            (
                None,
                {
                    "kind": "rate_limit",
                    "message": "HTTPError: 429 Client Error: Too Many Requests",
                    "status_code": 429,
                    "provider_error_code": "rate_limit_exceeded",
                    "provider_error_type": "rate_limit_error",
                    "retryable": True,
                },
            ),
            (_valid_contract_json(), None),
        ])
        output = generate_failure_explanation(
            client=client,
            model="gpt-4.1-mini",
            evidence=_sample_evidence(),
            prompt_bundle={},
            word_target=55,
            word_max=100,
            retry_invalid_output=3,
            max_items_per_section=12,
            ollama_options={},
        )

        self.assertFalse(output["ok"])
        self.assertFalse(output["used_fallback"])
        self.assertEqual(output["failure_kind"], "rate_limit")
        self.assertEqual(output["status_code"], 429)
        self.assertEqual(len(output["attempts"]), 1)
        self.assertEqual(client.calls, 1)

    def test_generate_explanation_hard_overrides_when_negative_setup_drift_detected(self) -> None:
        client = _StubOllamaClient([(_minimal_contract_json("Model cause that should be overridden."), None)])
        output = generate_failure_explanation(
            client=client,
            model="qwen3-coder:latest",
            evidence=_negative_type_evidence(executed_latitude="52.52"),
            prompt_bundle={},
            word_target=55,
            word_max=120,
            retry_invalid_output=0,
            max_items_per_section=12,
            ollama_options={},
        )

        self.assertTrue(output["ok"])
        contract = output.get("contract")
        self.assertIsInstance(contract, dict)
        self.assertIn("setup drift", str(contract.get("cause") or "").lower())
        self.assertEqual(contract.get("confidence"), "confirmed")
        self.assertIn("restore intentionally invalid latitude", str(contract.get("check_next") or "").lower())
        self.assertIn("drift_detected\": true", str(output["attempts"][0]["prompt"]).lower())
        self.assertIn(
            "execution.request_sent as the authoritative executed input",
            str(output["attempts"][0]["prompt"]),
        )

    def test_generate_explanation_does_not_override_when_no_negative_setup_drift(self) -> None:
        model_cause = "Model cause should remain unchanged."
        client = _StubOllamaClient([(_minimal_contract_json(model_cause), None)])
        output = generate_failure_explanation(
            client=client,
            model="qwen3-coder:latest",
            evidence=_negative_type_evidence(executed_latitude="not_a_number"),
            prompt_bundle={},
            word_target=55,
            word_max=120,
            retry_invalid_output=0,
            max_items_per_section=12,
            ollama_options={},
        )

        self.assertTrue(output["ok"])
        contract = output.get("contract")
        self.assertIsInstance(contract, dict)
        self.assertEqual(contract.get("cause"), model_cause)

    def test_load_backend_settings_raises_if_missing_config_and_no_env_model(self) -> None:
        missing = Path("src/backend/tests/_missing_llm_settings.yaml")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CONTRACTGUARD_LLM_MODEL", None)
            with self.assertRaises(ValueError):
                load_backend_llm_settings(config_path=str(missing))


if __name__ == "__main__":
    unittest.main()
