import json
import unittest

from llm_eval.failure_assistant import _build_compact_suggestion_prompt_bundle, generate_suggested_test


def _sample_evidence() -> dict:
    return {
        "spec": {"operation": {"method": "GET", "path": "/user"}},
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


def _sample_case_result() -> dict:
    return {
        "test_id": "TC-GET-user-002",
        "outcome": "FAIL",
        "expected_status": 404,
        "actual_status": 200,
        "error_message": "Expected 404 but got 200",
        "response_snippet": '{"id":1,"username":"existing-user"}',
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
    def __init__(self, responses: list[tuple[str | None, str | None]], timeout_seconds: int = 120) -> None:
        self._responses = list(responses)
        self.timeout_seconds = int(timeout_seconds)
        self.calls = 0
        self.prompts: list[str] = []

    def generate(self, **kwargs: object) -> tuple[str | None, str | None]:
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
    def test_invalid_schema_then_timeout_returns_mixed_failure_mode(self) -> None:
        client = _StubOllamaClient(
            responses=[
                (json.dumps({"reason": "Bad shape output", "summary": "not a test case"}), None),
                (None, "ReadTimeout: HTTPConnectionPool(host='localhost', port=11434): Read timed out. (read timeout=120)"),
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

        self.assertTrue(output["used_fallback"])
        self.assertTrue(output["can_apply"])
        self.assertEqual(output["failure_mode"], "mixed")
        self.assertIn("validation_error:invalid_test_case", str(output["llm_error"]))
        self.assertIn("ReadTimeout", str(output["llm_error"]))

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
        self.assertTrue(output["used_fallback"])
        self.assertEqual(output["failure_mode"], "timeout")
        self.assertIn("ReadTimeout", str(output["llm_error"]))

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
