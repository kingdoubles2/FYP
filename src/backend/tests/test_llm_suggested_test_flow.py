import json
import unittest

from llm_eval.failure_assistant import (
    _build_compact_suggestion_prompt_bundle,
    build_deterministic_suggested_test_payload,
    generate_suggested_test,
)


def _sample_evidence() -> dict:
    return {
        "spec": {"operation": {"method": "GET", "path": "/user"}},
        "test_context": {"category": "negative_invalid"},
        "execution": {
            "response_received": {"status": 404, "body_snippet": '{"message":"Not Found"}'},
            "assertion_failures": [{"type": "status_mismatch", "expected": 404, "actual_status": 200}],
        },
    }


def _sample_original_test_case() -> dict:
    return {
        "test_id": "TC-GET-user-002",
        "title": "GET /user should return 404 for missing user",
        "category": "negative_invalid",
        "requirement_ref": "REQ-USER-404",
        "method": "GET",
        "path": "/user",
        "priority": "medium",
        "preconditions": ["Auth token is configured"],
        "steps": [
            {
                "step_number": 1,
                "action": "Execute request",
                "input_data": {
                    "path_params": {},
                    "query_params": {"username": "missing-user"},
                    "headers": {},
                    "body": None,
                },
            }
        ],
        "expected_result": {"status_code": 404, "description": "Not Found"},
    }


def _sample_positive_original_test_case() -> dict:
    return {
        "test_id": "TC-GET-forecast-001",
        "title": "GET /v1/forecast should return 200 for valid coordinates",
        "category": "happy_path",
        "requirement_ref": "REQ-FORECAST-200",
        "method": "GET",
        "path": "/v1/forecast",
        "priority": "high",
        "preconditions": ["API is available"],
        "steps": [
            {
                "step_number": 1,
                "action": "Execute request",
                "input_data": {
                    "path_params": {},
                    "query_params": {"latitude": 52.52, "longitude": 13.41},
                    "headers": {},
                    "body": None,
                },
            }
        ],
        "expected_result": {"status_code": 200, "description": "Successful response"},
    }


def _sample_case_result() -> dict:
    return {
        "test_id": "TC-GET-user-002",
        "outcome": "FAIL",
        "expected_status": 404,
        "actual_status": 200,
        "error_message": "Expected 404 but got 200",
        "response_snippet": '{"id":1,"username":"existing-user"}',
    }


def _sample_positive_case_result() -> dict:
    return {
        "test_id": "TC-GET-forecast-001",
        "outcome": "FAIL",
        "expected_status": 200,
        "actual_status": 422,
        "error_message": "Expected 200 but got 422",
        "response_snippet": '{"reason":"Latitude must be in range of -90 to 90. Given: 9e+07."}',
    }


def _negative_type_drift_evidence(*, executed_latitude: str) -> dict:
    evidence = _sample_evidence()
    evidence["spec"] = {
        "operation": {"method": "GET", "path": "/v1/forecast"},
        "request_constraints": {
            "query_param_rules": {
                "latitude": {"type": "number", "format": "double"},
                "longitude": {"type": "number", "format": "double"},
            }
        },
    }
    evidence["ir_context"] = {
        "endpoint_ir": {
            "query_params": [
                {"name": "latitude", "schema": {"type": "number", "format": "double"}},
                {"name": "longitude", "schema": {"type": "number", "format": "double"}},
            ]
        }
    }
    evidence["test_context"] = {
        "category": "negative_type",
        "intent": "negative",
        "title": "GET /v1/forecast - Negative type: query latitude='not_a_number'",
        "generated_input": {
            "path_params": {},
            "query_params": {"latitude": "not_a_number", "longitude": 52.52},
            "headers": {},
            "body": None,
        },
    }
    evidence["execution"] = {
        "response_received": {"status": 200, "body_snippet": '{"latitude":52.52}'},
        "assertion_failures": [{"type": "status_mismatch", "expected": 400, "actual_status": 200}],
        "request_sent": {
            "path_params": {},
            "query_params": {"latitude": executed_latitude, "longitude": 52.52},
            "headers": {},
            "body": None,
        },
    }
    return evidence


def _negative_type_original_test_case(*, executed_latitude: str) -> dict:
    return {
        "test_id": "TC-GET-v1-forecast-002",
        "title": "GET /v1/forecast - Negative type: query latitude='not_a_number'",
        "category": "negative_type",
        "requirement_ref": "GET /v1/forecast",
        "method": "GET",
        "path": "/v1/forecast",
        "priority": "medium",
        "preconditions": [],
        "steps": [
            {
                "step_number": 1,
                "action": "Execute request",
                "input_data": {
                    "path_params": {},
                    "query_params": {"latitude": executed_latitude, "longitude": 52.52},
                    "headers": {},
                    "body": None,
                },
            }
        ],
        "expected_result": {"status_code": 400, "description": "Bad request"},
    }


def _sample_prompt_bundle() -> dict:
    tests_full = [{"test_id": f"TC-{index:03d}", "method": "GET", "path": "/user"} for index in range(80)]
    results_full = [
        {
            "test_id": f"TC-{index:03d}",
            "outcome": "FAIL" if index % 2 == 0 else "PASS",
            "actual_status": 404 if index % 2 == 0 else 200,
            "response_snippet": "x" * 180,
        }
        for index in range(80)
    ]
    return {
        "case_evidence": {
            "spec": {"operation": {"method": "GET", "path": "/user"}},
            "test_context": {"test_id": "TC-GET-user-002"},
            "execution": {"outcome": "FAIL"},
        },
        "pipeline_context": {
            "test_case_full": _sample_original_test_case(),
            "tests_for_endpoint": tests_full[:5],
            "tests_full": tests_full,
            "results_for_endpoint": results_full[:5],
            "results_full": results_full,
            "ir_full": {
                "api_title": "Sample API",
                "api_version": "1.0.0",
                "base_url": "https://api.example.com",
                "endpoint_for_case": {
                    "method": "GET",
                    "path": "/user",
                    "request_schema": {"type": "object"},
                },
            },
            "spec_full": {
                "operation": {"operationId": "users/get-user", "summary": "Get user"},
                "path_item": {"get": {"summary": "Get user path item"}},
            },
            "execution": {
                "result_full": _sample_case_result(),
                "same_endpoint_summary": {"pass": 2, "fail": 3, "skip": 0},
            },
        },
    }


class _StubOllamaClient:
    def __init__(self, responses: list[tuple[str | None, object]], timeout_seconds: int = 120) -> None:
        self._responses = list(responses)
        self.timeout_seconds = int(timeout_seconds)
        self.calls = 0
        self.prompts: list[str] = []

    def generate(self, **kwargs: object) -> tuple[str | None, object]:
        self.calls += 1
        self.prompts.append(str(kwargs.get("prompt") or ""))
        if self._responses:
            return self._responses.pop(0)
        return None, "stub_exhausted"


def _valid_case_payload_json(*, key_name: str = "suggested_test_case") -> str:
    payload = {
        "reason": "Add one follow-up negative case for the same endpoint.",
        key_name: {
            "test_id": "TC-GET-user-followup",
            "title": "Follow-up negative check",
            "category": "llm_followup",
            "method": "get",
            "path": "/user",
            "priority": "medium",
            "steps": [
                {
                    "step_number": 1,
                    "action": "Execute request",
                    "input_data": {"query_params": {"username": "missing-user-2"}},
                }
            ],
            "expected_result": {"status_code": 404, "description": "Not Found"},
        },
    }
    return json.dumps(payload)


class SuggestedTestFlowTests(unittest.TestCase):
    def test_deterministic_reason_describes_test_behavior_without_explanation_repetition(self) -> None:
        output = build_deterministic_suggested_test_payload(
            model="qwen3-coder:latest",
            evidence=_sample_evidence(),
            original_test_case=_sample_original_test_case(),
            case_result=_sample_case_result(),
            existing_test_ids=["TC-GET-user-002"],
            explanation_context={
                "signal": "validation",
                "likely_cause": "Input is malformed",
                "why_likely": "Status mismatch indicates validation issue",
                "check_next": "Verify input constraints",
            },
        )

        reason = str(output.get("reason") or "")
        self.assertIn("Adds a follow-up", reason)
        self.assertIn("GET /user", reason)
        self.assertNotIn("Likely cause:", reason)
        self.assertNotIn("Why likely:", reason)
        self.assertNotIn("Check next:", reason)

    def test_deterministic_negative_mode_preserves_invalid_request_shape(self) -> None:
        evidence = _sample_evidence()
        evidence["test_context"]["generated_input"] = {
            "path_params": {},
            "query_params": {"latitude": 52.52, "longitude": 52.52},
            "headers": {},
            "body": None,
        }
        evidence["execution"]["request_sent"] = {
            "path_params": {},
            "query_params": {"latitude": 90000000, "longitude": 52.52},
            "headers": {},
            "body": None,
        }
        evidence["execution"]["response_received"]["body_snippet"] = (
            '{"reason":"Latitude must be in range of -90 to 90. Given: 9e+07.","error":true}'
        )

        original_case = _sample_original_test_case()
        original_case["path"] = "/v1/forecast"
        original_case["requirement_ref"] = "GET /v1/forecast"
        original_case["steps"][0]["input_data"]["query_params"] = {"latitude": 52.52, "longitude": 52.52}

        output = build_deterministic_suggested_test_payload(
            model="qwen3-coder:latest",
            evidence=evidence,
            original_test_case=original_case,
            case_result=_sample_case_result(),
            existing_test_ids=["TC-GET-user-002"],
            explanation_context={
                "signal": "validation",
                "why_likely": "Latitude must be in range of -90 to 90. Given: 9e+07.",
            },
        )

        suggested = output.get("suggested_test_case")
        self.assertIsInstance(suggested, dict)
        steps = suggested.get("steps")
        self.assertIsInstance(steps, list)
        self.assertTrue(steps)
        first_step = steps[0]
        self.assertIsInstance(first_step, dict)
        input_data = first_step.get("input_data")
        self.assertIsInstance(input_data, dict)
        query = input_data.get("query_params")
        self.assertIsInstance(query, dict)
        self.assertEqual(query.get("latitude"), 90000000)
        self.assertIn('"latitude": 90000000', str(first_step.get("action") or ""))
        self.assertIn("latitude=90000000", str(output.get("reason") or ""))
        expected_result = suggested.get("expected_result")
        self.assertIsInstance(expected_result, dict)
        self.assertEqual(expected_result.get("status_code"), 404)
        self.assertIn("remains invalid", str(expected_result.get("description") or "").lower())

    def test_deterministic_extracts_invalid_value_from_reason_when_request_payload_is_missing(self) -> None:
        evidence = _sample_evidence()
        evidence["spec"]["operation"] = {"method": "GET", "path": "/v1/forecast"}
        evidence["test_context"]["generated_input"] = {
            "path_params": {},
            "query_params": {"latitude": 52.52, "longitude": 52.52},
            "headers": {},
            "body": None,
        }
        evidence["execution"]["request_sent"] = {}
        evidence["execution"]["response_received"]["body_snippet"] = (
            '{"reason":"Latitude must be in range of -90 to 90. Given: 9e+07.","error":true}'
        )

        original_case = _sample_original_test_case()
        original_case["path"] = "/v1/forecast"
        original_case["requirement_ref"] = "GET /v1/forecast"
        original_case["steps"][0]["input_data"]["query_params"] = {"latitude": 52.52, "longitude": 52.52}

        output = build_deterministic_suggested_test_payload(
            model="qwen3-coder:latest",
            evidence=evidence,
            original_test_case=original_case,
            case_result=_sample_case_result(),
            existing_test_ids=["TC-GET-user-002"],
            explanation_context={
                "signal": "validation",
                "explanation": "Latitude must be in range of -90 to 90. Given: 9e+07.",
            },
        )

        suggested = output.get("suggested_test_case")
        self.assertIsInstance(suggested, dict)
        steps = suggested.get("steps")
        self.assertIsInstance(steps, list)
        self.assertTrue(steps)
        first_step = steps[0]
        self.assertIsInstance(first_step, dict)
        query = (
            first_step.get("input_data", {}).get("query_params")
            if isinstance(first_step.get("input_data"), dict)
            else {}
        )
        self.assertIsInstance(query, dict)
        self.assertEqual(query.get("latitude"), 90000000)
        expected_result = suggested.get("expected_result")
        self.assertIsInstance(expected_result, dict)
        self.assertIn("latitude remains invalid", str(expected_result.get("description") or "").lower())

    def test_deterministic_headers_fall_back_when_request_sent_has_only_redacted_auth(self) -> None:
        evidence = _sample_evidence()
        evidence["test_context"]["generated_input"] = {
            "path_params": {},
            "query_params": {"username": "missing-user"},
            "headers": {"Accept": "application/json"},
            "body": None,
        }
        evidence["execution"]["request_sent"] = {
            "path_params": {},
            "query_params": {"username": "missing-user"},
            "headers": {"Authorization": "<redacted>"},
            "body": None,
        }
        original_case = _sample_original_test_case()
        original_case["steps"][0]["input_data"]["headers"] = {"Accept": "application/json"}

        output = build_deterministic_suggested_test_payload(
            model="qwen3-coder:latest",
            evidence=evidence,
            original_test_case=original_case,
            case_result=_sample_case_result(),
            existing_test_ids=["TC-GET-user-002"],
            explanation_context={"signal": "validation"},
        )

        suggested = output.get("suggested_test_case")
        self.assertIsInstance(suggested, dict)
        first_step = suggested.get("steps")[0]
        self.assertIsInstance(first_step, dict)
        headers = first_step.get("input_data", {}).get("headers")
        self.assertIsInstance(headers, dict)
        self.assertEqual(headers.get("Accept"), "application/json")
        self.assertNotIn("Authorization", headers)

    def test_deterministic_headers_preserve_non_auth_and_strip_redacted_auth(self) -> None:
        evidence = _sample_evidence()
        evidence["test_context"]["generated_input"] = {
            "path_params": {},
            "query_params": {"username": "missing-user"},
            "headers": {"Accept": "application/json"},
            "body": None,
        }
        evidence["execution"]["request_sent"] = {
            "path_params": {},
            "query_params": {"username": "missing-user"},
            "headers": {
                "Authorization": "<redacted>",
                "X-API-Key": "<redacted>",
                "X-Trace-Id": "trace-123",
            },
            "body": None,
        }

        output = build_deterministic_suggested_test_payload(
            model="qwen3-coder:latest",
            evidence=evidence,
            original_test_case=_sample_original_test_case(),
            case_result=_sample_case_result(),
            existing_test_ids=["TC-GET-user-002"],
            explanation_context={"signal": "validation"},
        )

        suggested = output.get("suggested_test_case")
        self.assertIsInstance(suggested, dict)
        first_step = suggested.get("steps")[0]
        self.assertIsInstance(first_step, dict)
        headers = first_step.get("input_data", {}).get("headers")
        self.assertIsInstance(headers, dict)
        self.assertEqual(headers.get("X-Trace-Id"), "trace-123")
        self.assertNotIn("Authorization", headers)
        self.assertNotIn("X-API-Key", headers)
        self.assertNotIn("Accept", headers)

    def test_deterministic_positive_mode_repairs_failing_field_and_preserves_expected_status(self) -> None:
        evidence = _sample_evidence()
        evidence["spec"] = {
            "operation": {"method": "GET", "path": "/v1/forecast"},
            "request_constraints": {
                "query_param_rules": {
                    "latitude": {"type": "number", "format": "double"},
                    "longitude": {"type": "number", "format": "double"},
                }
            },
        }
        evidence["ir_context"] = {
            "endpoint_ir": {
                "query_params": [
                    {"name": "latitude", "schema": {"type": "number", "format": "double"}},
                    {"name": "longitude", "schema": {"type": "number", "format": "double"}},
                ]
            }
        }
        evidence["test_context"]["category"] = "happy_path"
        evidence["test_context"]["intent"] = "positive"
        evidence["test_context"]["generated_input"] = {
            "path_params": {},
            "query_params": {"latitude": 52.52, "longitude": 13.41},
            "headers": {},
            "body": None,
        }
        evidence["execution"]["request_sent"] = {
            "path_params": {},
            "query_params": {"latitude": 90000000, "longitude": 13.41},
            "headers": {},
            "body": None,
        }
        evidence["execution"]["response_received"]["body_snippet"] = (
            '{"reason":"Latitude must be in range of -90 to 90. Given: 9e+07.","error":true}'
        )
        original_case = _sample_positive_original_test_case()
        output = build_deterministic_suggested_test_payload(
            model="qwen3-coder:latest",
            evidence=evidence,
            original_test_case=original_case,
            case_result=_sample_positive_case_result(),
            existing_test_ids=["TC-GET-forecast-001"],
            explanation_context={
                "signal": "validation",
                "why_likely": "Latitude must be in range of -90 to 90. Given: 9e+07.",
            },
        )

        suggested = output.get("suggested_test_case")
        self.assertIsInstance(suggested, dict)
        first_step = suggested.get("steps")[0]
        self.assertIsInstance(first_step, dict)
        query = first_step.get("input_data", {}).get("query_params")
        self.assertIsInstance(query, dict)
        self.assertEqual(query.get("latitude"), 52.52)
        expected_result = suggested.get("expected_result")
        self.assertIsInstance(expected_result, dict)
        self.assertEqual(expected_result.get("status_code"), 200)
        self.assertIn("after repairing latitude", str(expected_result.get("description") or "").lower())

    def test_deterministic_changes_expected_status_only_with_explicit_phrase(self) -> None:
        baseline_output = build_deterministic_suggested_test_payload(
            model="qwen3-coder:latest",
            evidence=_sample_evidence(),
            original_test_case=_sample_original_test_case(),
            case_result=_sample_case_result(),
            existing_test_ids=["TC-GET-user-002"],
            explanation_context={
                "signal": "validation",
                "explanation": "There is a status mismatch; verify contract behavior.",
            },
        )
        baseline_case = baseline_output.get("suggested_test_case")
        self.assertIsInstance(baseline_case, dict)
        baseline_expected = baseline_case.get("expected_result")
        self.assertIsInstance(baseline_expected, dict)
        self.assertEqual(baseline_expected.get("status_code"), 404)

        override_output = build_deterministic_suggested_test_payload(
            model="qwen3-coder:latest",
            evidence=_sample_evidence(),
            original_test_case=_sample_original_test_case(),
            case_result=_sample_case_result(),
            existing_test_ids=["TC-GET-user-002"],
            explanation_context={
                "signal": "validation",
                "explanation": "Please change expected status to 422 for this case.",
            },
        )
        override_case = override_output.get("suggested_test_case")
        self.assertIsInstance(override_case, dict)
        override_expected = override_case.get("expected_result")
        self.assertIsInstance(override_expected, dict)
        self.assertEqual(override_expected.get("status_code"), 422)

    def test_deterministic_negative_drift_restores_invalid_hint_and_expected_status(self) -> None:
        evidence = _negative_type_drift_evidence(executed_latitude="52.52")
        original_case = _negative_type_original_test_case(executed_latitude="52.52")
        case_result = {
            "test_id": "TC-GET-v1-forecast-002",
            "outcome": "FAIL",
            "expected_status": 400,
            "actual_status": 200,
            "error_message": "Expected 400 but got 200",
            "response_snippet": '{"latitude":52.52}',
        }
        output = build_deterministic_suggested_test_payload(
            model="qwen3-coder:latest",
            evidence=evidence,
            original_test_case=original_case,
            case_result=case_result,
            existing_test_ids=["TC-GET-v1-forecast-002"],
            explanation_context={"signal": "schema_type"},
        )

        suggested = output.get("suggested_test_case")
        self.assertIsInstance(suggested, dict)
        first_step = suggested.get("steps")[0]
        self.assertIsInstance(first_step, dict)
        query = first_step.get("input_data", {}).get("query_params")
        self.assertIsInstance(query, dict)
        self.assertEqual(query.get("latitude"), "not_a_number")
        expected_result = suggested.get("expected_result")
        self.assertIsInstance(expected_result, dict)
        self.assertEqual(expected_result.get("status_code"), 400)
        self.assertIn("restoring negative setup drift", str(expected_result.get("description") or "").lower())
        self.assertIn("restoring negative setup drift", str(output.get("reason") or "").lower())

    def test_non_input_signal_soft_defers_after_llm_generation(self) -> None:
        client = _StubOllamaClient(responses=[(_valid_case_payload_json(), None)], timeout_seconds=120)
        non_input_evidence = {
            "spec": {"operation": {"method": "GET", "path": "/user"}},
            "execution": {
                "response_received": {"status": 500, "body_snippet": '{"message":"Internal"}'},
                "assertion_failures": [{"type": "status_mismatch", "expected": 404, "actual_status": 500}],
            },
        }
        output = generate_suggested_test(
            client=client,
            model="qwen3-coder:latest",
            evidence=non_input_evidence,
            prompt_bundle=_sample_prompt_bundle(),
            original_test_case=_sample_original_test_case(),
            case_result=_sample_case_result(),
            existing_test_ids=["TC-GET-user-002"],
            retry_invalid_output=1,
            ollama_options={},
        )

        self.assertEqual(client.calls, 1)
        self.assertTrue(bool(output.get("skipped")))
        self.assertFalse(bool(output.get("eligible_for_generation")))
        self.assertFalse(output["can_apply"])

    def test_external_signal_soft_defers_with_external_warning(self) -> None:
        client = _StubOllamaClient(responses=[(_valid_case_payload_json(), None)], timeout_seconds=120)
        output = generate_suggested_test(
            client=client,
            model="qwen3-coder:latest",
            evidence=_sample_evidence(),
            prompt_bundle=_sample_prompt_bundle(),
            original_test_case=_sample_original_test_case(),
            case_result=_sample_case_result(),
            existing_test_ids=["TC-GET-user-002"],
            retry_invalid_output=0,
            explanation_context={"signal": "auth"},
            ollama_options={},
        )

        self.assertEqual(client.calls, 1)
        self.assertTrue(bool(output.get("skipped")))
        self.assertFalse(output["can_apply"])
        self.assertTrue(bool(output.get("external_failure")))
        self.assertTrue(bool(output.get("warning_external")))
        self.assertIn("external", str(output.get("reason") or "").lower())

    def test_parse_error_soft_defers_without_fallback_case(self) -> None:
        client = _StubOllamaClient(responses=[("not json", None)], timeout_seconds=120)
        output = generate_suggested_test(
            client=client,
            model="qwen3-coder:latest",
            evidence=_sample_evidence(),
            prompt_bundle=_sample_prompt_bundle(),
            original_test_case=_sample_original_test_case(),
            case_result=_sample_case_result(),
            existing_test_ids=["TC-GET-user-002"],
            retry_invalid_output=0,
            ollama_options={},
        )

        self.assertEqual(client.calls, 1)
        self.assertFalse(output["used_fallback"])
        self.assertFalse(output["can_apply"])
        self.assertTrue(bool(output.get("skipped")))
        self.assertEqual(output.get("failure_mode"), "invalid_schema")
        self.assertIn("parse_error", str(output.get("llm_error") or ""))
        self.assertEqual(output.get("suggested_test_case"), None)

    def test_invalid_schema_soft_defers_with_single_attempt(self) -> None:
        client = _StubOllamaClient(
            responses=[
                (json.dumps({"summary": "not a test case"}), None),
            ],
            timeout_seconds=120,
        )
        output = generate_suggested_test(
            client=client,
            model="qwen3-coder:latest",
            evidence=_sample_evidence(),
            prompt_bundle=_sample_prompt_bundle(),
            original_test_case=_sample_original_test_case(),
            case_result=_sample_case_result(),
            existing_test_ids=["TC-GET-user-002"],
            retry_invalid_output=1,
            ollama_options={},
        )

        self.assertEqual(client.calls, 1)
        self.assertFalse(output["used_fallback"])
        self.assertFalse(output["can_apply"])
        self.assertTrue(bool(output.get("skipped")))
        self.assertEqual(output["failure_mode"], "invalid_schema")
        self.assertIn("validation_error:invalid_test_case", str(output["llm_error"]))
        self.assertEqual(client.timeout_seconds, 120)

    def test_timeout_on_first_attempt_aborts_retries_and_returns_timeout_mode(self) -> None:
        client = _StubOllamaClient(
            responses=[
                (None, "ReadTimeout: HTTPConnectionPool(host='localhost', port=11434): Read timed out. (read timeout=120)"),
                (_valid_case_payload_json(), None),
            ],
            timeout_seconds=120,
        )
        output = generate_suggested_test(
            client=client,
            model="qwen3-coder:latest",
            evidence=_sample_evidence(),
            prompt_bundle=_sample_prompt_bundle(),
            original_test_case=_sample_original_test_case(),
            case_result=_sample_case_result(),
            existing_test_ids=["TC-GET-user-002"],
            retry_invalid_output=3,
            ollama_options={},
        )

        self.assertEqual(client.calls, 1)
        self.assertFalse(output["used_fallback"])
        self.assertFalse(output["can_apply"])
        self.assertTrue(bool(output.get("skipped")))
        self.assertEqual(output["failure_mode"], "timeout")
        self.assertIn("ReadTimeout", str(output["llm_error"]))

    def test_explicit_llm_soft_defer_with_null_case_returns_non_actionable_payload(self) -> None:
        client = _StubOllamaClient(
            responses=[
                (json.dumps({"reason": "Insufficient evidence for a reliable extra test.", "suggested_test_case": None}), None),
            ],
            timeout_seconds=120,
        )
        output = generate_suggested_test(
            client=client,
            model="qwen3-coder:latest",
            evidence=_sample_evidence(),
            prompt_bundle=_sample_prompt_bundle(),
            original_test_case=_sample_original_test_case(),
            case_result=_sample_case_result(),
            existing_test_ids=["TC-GET-user-002"],
            retry_invalid_output=0,
            ollama_options={},
        )

        self.assertEqual(client.calls, 1)
        self.assertFalse(output["used_fallback"])
        self.assertFalse(output["can_apply"])
        self.assertTrue(bool(output.get("skipped")))
        self.assertEqual(output.get("suggested_test_case"), None)
        self.assertIn("insufficient evidence", str(output.get("reason") or "").lower())

    def test_openai_rate_limit_returns_hard_fail_without_fallback(self) -> None:
        client = _StubOllamaClient(
            responses=[
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
                (_valid_case_payload_json(), None),
            ],
            timeout_seconds=120,
        )
        output = generate_suggested_test(
            client=client,
            model="gpt-4.1-mini",
            evidence=_sample_evidence(),
            prompt_bundle=_sample_prompt_bundle(),
            original_test_case=_sample_original_test_case(),
            case_result=_sample_case_result(),
            existing_test_ids=["TC-GET-user-002"],
            retry_invalid_output=3,
            ollama_options={},
        )

        self.assertEqual(client.calls, 1)
        self.assertFalse(output["used_fallback"])
        self.assertFalse(output["can_apply"])
        self.assertEqual(output["failure_kind"], "rate_limit")
        self.assertEqual(output["status_code"], 429)
        self.assertTrue(bool(output.get("eligible_for_generation")))

    def test_explanation_context_signal_can_enable_generation(self) -> None:
        client = _StubOllamaClient(
            responses=[(_valid_case_payload_json(), None)],
            timeout_seconds=120,
        )
        non_input_evidence = {
            "spec": {"operation": {"method": "GET", "path": "/user"}},
            "execution": {
                "response_received": {"status": 500, "body_snippet": '{"message":"Internal"}'},
                "assertion_failures": [{"type": "status_mismatch", "expected": 404, "actual_status": 500}],
            },
        }
        output = generate_suggested_test(
            client=client,
            model="qwen3-coder:latest",
            evidence=non_input_evidence,
            prompt_bundle=_sample_prompt_bundle(),
            original_test_case=_sample_original_test_case(),
            case_result=_sample_case_result(),
            existing_test_ids=["TC-GET-user-002"],
            retry_invalid_output=0,
            explanation_context={"signal": "validation", "likely_cause": "Input payload is malformed"},
            ollama_options={},
        )

        self.assertEqual(client.calls, 1)
        self.assertFalse(output["used_fallback"])
        self.assertTrue(output["can_apply"])
        self.assertFalse(bool(output.get("skipped")))

    def test_llm_prompt_includes_authoritative_input_rule_and_drift_context(self) -> None:
        client = _StubOllamaClient(responses=[(_valid_case_payload_json(), None)], timeout_seconds=120)
        output = generate_suggested_test(
            client=client,
            model="qwen3-coder:latest",
            evidence=_negative_type_drift_evidence(executed_latitude="52.52"),
            prompt_bundle=_sample_prompt_bundle(),
            original_test_case=_negative_type_original_test_case(executed_latitude="52.52"),
            case_result={
                "test_id": "TC-GET-v1-forecast-002",
                "outcome": "FAIL",
                "expected_status": 400,
                "actual_status": 200,
                "error_message": "Expected 400 but got 200",
                "response_snippet": '{"latitude":52.52}',
            },
            existing_test_ids=["TC-GET-v1-forecast-002"],
            retry_invalid_output=0,
            ollama_options={},
        )

        self.assertTrue(output["can_apply"])
        self.assertGreaterEqual(len(client.prompts), 1)
        prompt = client.prompts[0]
        self.assertIn("execution.request_sent as the authoritative executed input", prompt)
        self.assertIn('"drift_detected": true', prompt.lower())

    def test_llm_normalization_preserves_original_expected_status_when_model_omits_status(self) -> None:
        response_json = json.dumps(
            {
                "reason": "Follow-up generated by model.",
                "suggested_test_case": {
                    "test_id": "TC-GET-user-followup",
                    "title": "Follow-up",
                    "category": "llm_followup",
                    "method": "GET",
                    "path": "/user",
                    "steps": [
                        {
                            "step_number": 1,
                            "action": "Execute request",
                            "input_data": {"query_params": {"username": "missing-user-2"}},
                        }
                    ],
                    "expected_result": {"description": "Model omitted explicit status"},
                },
            }
        )
        client = _StubOllamaClient(responses=[(response_json, None)], timeout_seconds=120)
        output = generate_suggested_test(
            client=client,
            model="qwen3-coder:latest",
            evidence=_sample_evidence(),
            prompt_bundle=_sample_prompt_bundle(),
            original_test_case=_sample_original_test_case(),
            case_result=_sample_case_result(),
            existing_test_ids=["TC-GET-user-002"],
            retry_invalid_output=0,
            ollama_options={},
        )

        self.assertFalse(output["used_fallback"])
        self.assertTrue(output["can_apply"])
        suggested = output.get("suggested_test_case")
        self.assertIsInstance(suggested, dict)
        expected_result = suggested.get("expected_result")
        self.assertIsInstance(expected_result, dict)
        self.assertEqual(expected_result.get("status_code"), 404)
        self.assertEqual(suggested.get("category"), _sample_original_test_case().get("category"))

    def test_alias_and_root_object_parsing_produce_valid_suggestion(self) -> None:
        scenarios = [
            ("alias_key", _valid_case_payload_json(key_name="suggested_test")),
            (
                "root_object",
                json.dumps(
                    {
                        "test_id": "TC-ROOT-01",
                        "title": "Root object suggestion",
                        "method": "post",
                        "path": "/user",
                        "category": "llm_followup",
                        "steps": [
                            {
                                "step_number": 1,
                                "action": "Execute request",
                                "input_data": {"body": {"username": "bad-user"}},
                            }
                        ],
                        "expected_result": {"status_code": 404, "description": "Not Found"},
                    }
                ),
            ),
        ]

        for name, response_json in scenarios:
            with self.subTest(name=name):
                client = _StubOllamaClient(responses=[(response_json, None)], timeout_seconds=120)
                output = generate_suggested_test(
                    client=client,
                    model="qwen3-coder:latest",
                    evidence=_sample_evidence(),
                    prompt_bundle=_sample_prompt_bundle(),
                    original_test_case=_sample_original_test_case(),
                    case_result=_sample_case_result(),
                    existing_test_ids=["TC-GET-user-002"],
                    retry_invalid_output=0,
                    ollama_options={},
                )

                self.assertFalse(output["used_fallback"])
                self.assertTrue(output["can_apply"])
                self.assertEqual(output["failure_mode"], "none")
                suggested = output["suggested_test_case"]
                self.assertIsInstance(suggested, dict)
                self.assertIn("steps", suggested)
                self.assertEqual(str(suggested.get("method") or "").upper(), str(suggested.get("method") or ""))

    def test_compact_prompt_bundle_drops_full_collections_and_reduces_size(self) -> None:
        prompt_bundle = _sample_prompt_bundle()
        compact_bundle = _build_compact_suggestion_prompt_bundle(prompt_bundle)
        original_size = len(json.dumps(prompt_bundle, ensure_ascii=True))
        compact_size = len(json.dumps(compact_bundle, ensure_ascii=True))

        compact_pipeline = compact_bundle.get("pipeline_context") if isinstance(compact_bundle, dict) else {}
        self.assertIsInstance(compact_pipeline, dict)
        self.assertNotIn("tests_full", compact_pipeline)
        self.assertNotIn("results_full", compact_pipeline)
        self.assertLess(compact_size, int(original_size * 0.75))

        client = _StubOllamaClient(responses=[(_valid_case_payload_json(), None)], timeout_seconds=120)
        output = generate_suggested_test(
            client=client,
            model="qwen3-coder:latest",
            evidence=_sample_evidence(),
            prompt_bundle=prompt_bundle,
            original_test_case=_sample_original_test_case(),
            case_result=_sample_case_result(),
            existing_test_ids=["TC-GET-user-002"],
            retry_invalid_output=0,
            ollama_options={},
        )

        self.assertFalse(output["used_fallback"])
        self.assertGreaterEqual(len(client.prompts), 1)
        self.assertNotIn('"tests_full"', client.prompts[0])
        self.assertNotIn('"results_full"', client.prompts[0])


if __name__ == "__main__":
    unittest.main()
