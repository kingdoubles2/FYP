from __future__ import annotations

from typing import Any, Dict, List

from test_generator.models import TestCase, TestStep, InputData, ExpectedResult
from test_generator.sample_data import generate_valid_value


AUTH_HEADER_NAMES = {"api_key", "authorization", "x-api-key", "token", "bearer"}


def _get_auth_headers(endpoint: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return header params that look like authentication."""
    return [
        p for p in endpoint.get("header_params", [])
        if p["name"].lower() in AUTH_HEADER_NAMES
    ]


def _build_valid_params(params: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {p["name"]: generate_valid_value(p["schema"], name=p["name"]) for p in params}


def _render_path(path: str, path_params: Dict[str, Any]) -> str:
    result = path
    for name, value in path_params.items():
        result = result.replace(f"{{{name}}}", str(value))
    return result


def generate_auth_cases(endpoint: Dict[str, Any]) -> List[TestCase]:
    auth_params = _get_auth_headers(endpoint)
    if not auth_params:
        return []

    ref = endpoint["endpoint_id"]
    method = endpoint["method"]
    path = endpoint["path"]

    valid_path = _build_valid_params(endpoint.get("path_params", []))
    valid_query = _build_valid_params(endpoint.get("query_params", []))
    valid_headers = _build_valid_params(endpoint.get("header_params", []))
    valid_body = None
    if endpoint.get("request_schema"):
        valid_body = generate_valid_value(endpoint["request_schema"])

    rendered = _render_path(path, valid_path)
    cases: List[TestCase] = []

    for auth_param in auth_params:
        hname = auth_param["name"]

        # 1. Missing auth header
        headers_no_auth = {k: v for k, v in valid_headers.items() if k != hname}
        cases.append(TestCase(
            test_id="",
            title=f"{ref} - Auth: missing '{hname}' header",
            category="auth",
            requirement_ref=ref,
            method=method,
            path=path,
            priority="high",
            preconditions=[],
            steps=[TestStep(
                step_number=1,
                action=f"Send {method} request to {rendered} without '{hname}' header",
                input_data=InputData(path_params=valid_path, query_params=valid_query, headers=headers_no_auth, body=valid_body),
            )],
            expected_result=ExpectedResult(status_code=401, description=f"Rejected: missing authentication header '{hname}'"),
        ))

        # 2. Invalid auth value
        headers_bad = dict(valid_headers)
        headers_bad[hname] = "invalid_key_12345"
        cases.append(TestCase(
            test_id="",
            title=f"{ref} - Auth: invalid '{hname}' value",
            category="auth",
            requirement_ref=ref,
            method=method,
            path=path,
            priority="high",
            preconditions=[],
            steps=[TestStep(
                step_number=1,
                action=f"Send {method} request to {rendered} with {hname}='invalid_key_12345'",
                input_data=InputData(path_params=valid_path, query_params=valid_query, headers=headers_bad, body=valid_body),
            )],
            expected_result=ExpectedResult(status_code=401, description=f"Rejected: invalid value for '{hname}'"),
        ))

        # 3. Empty auth value
        headers_empty = dict(valid_headers)
        headers_empty[hname] = ""
        cases.append(TestCase(
            test_id="",
            title=f"{ref} - Auth: empty '{hname}' header",
            category="auth",
            requirement_ref=ref,
            method=method,
            path=path,
            priority="high",
            preconditions=[],
            steps=[TestStep(
                step_number=1,
                action=f"Send {method} request to {rendered} with {hname}='' (empty)",
                input_data=InputData(path_params=valid_path, query_params=valid_query, headers=headers_empty, body=valid_body),
            )],
            expected_result=ExpectedResult(status_code=401, description=f"Rejected: empty authentication header '{hname}'"),
        ))

    return cases
