from __future__ import annotations

from typing import Any, Dict, List, Optional

from test_generator.models import TestCase, TestStep, InputData, ExpectedResult
from test_generator.sample_data import generate_valid_value, should_autofill_header_param


# Legacy heuristic: header names that look like authentication
AUTH_HEADER_NAMES = {"api_key", "authorization", "x-api-key", "token", "bearer"}


def _get_auth_headers_legacy(endpoint: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return header params that look like authentication (legacy heuristic)."""
    return [
        p for p in endpoint.get("header_params", [])
        if p["name"].lower() in AUTH_HEADER_NAMES
    ]


def _resolve_effective_security(
    endpoint: Dict[str, Any],
    ir_data: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Resolve which auth headers/params to test from structured security info.

    Returns a list of dicts, each with:
      - "name": the header name to test (e.g. "X-API-Key", "Authorization")
      - "type": "apiKey" or "http"
    """
    schemes_list = ir_data.get("security_schemes", [])
    if not schemes_list:
        return []

    # Build lookup: scheme_id -> scheme dict
    scheme_map: Dict[str, Dict[str, Any]] = {}
    for s in schemes_list:
        scheme_map[s["scheme_id"]] = s

    # Per-operation security overrides top-level
    security_reqs = endpoint.get("security")
    if security_reqs is None:
        security_reqs = ir_data.get("security")
    if not security_reqs:
        return []

    auth_targets: List[Dict[str, str]] = []
    seen_names: set = set()

    for req in security_reqs:
        if not isinstance(req, dict):
            continue
        for scheme_id in req:
            scheme = scheme_map.get(scheme_id)
            if not scheme:
                continue
            stype = scheme.get("type", "")

            if stype == "apiKey" and scheme.get("location") == "header":
                hname = scheme.get("name", "")
                if hname and hname not in seen_names:
                    auth_targets.append({"name": hname, "type": "apiKey"})
                    seen_names.add(hname)

            elif stype == "http":
                hname = "Authorization"
                if hname not in seen_names:
                    auth_targets.append({"name": hname, "type": "http"})
                    seen_names.add(hname)

    return auth_targets


def _build_valid_params(params: List[Dict[str, Any]], *, is_header: bool = False) -> Dict[str, Any]:
    selected = params
    if is_header:
        selected = [p for p in params if should_autofill_header_param(p)]
    return {p["name"]: generate_valid_value(p["schema"], name=p["name"]) for p in selected}


def _render_path(path: str, path_params: Dict[str, Any]) -> str:
    result = path
    for name, value in path_params.items():
        result = result.replace(f"{{{name}}}", str(value))
    return result


def generate_auth_cases(
    endpoint: Dict[str, Any],
    *,
    ir_data: Optional[Dict[str, Any]] = None,
) -> List[TestCase]:
    # Resolve which auth headers to test
    auth_targets: List[Dict[str, str]] = []

    if ir_data and ir_data.get("security_schemes"):
        auth_targets = _resolve_effective_security(endpoint, ir_data)

    # Fallback to legacy heuristic if no structured security
    if not auth_targets:
        for p in _get_auth_headers_legacy(endpoint):
            auth_targets.append({"name": p["name"], "type": "apiKey"})

    if not auth_targets:
        return []

    ref = endpoint["endpoint_id"]
    method = endpoint["method"]
    path = endpoint["path"]

    valid_path = _build_valid_params(endpoint.get("path_params", []))
    valid_query = _build_valid_params(endpoint.get("query_params", []))
    valid_headers = _build_valid_params(endpoint.get("header_params", []), is_header=True)
    valid_body = None
    if endpoint.get("request_schema"):
        valid_body = generate_valid_value(endpoint["request_schema"])

    rendered = _render_path(path, valid_path)
    cases: List[TestCase] = []

    for target in auth_targets:
        hname = target["name"]

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
