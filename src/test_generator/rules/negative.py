from __future__ import annotations

from typing import Any, Dict, List

from test_generator.models import TestCase, TestStep, InputData, ExpectedResult
from test_generator.sample_data import (
    generate_valid_value,
    generate_wrong_type_value,
    generate_invalid_format_value,
    pick_error_status,
)


def _build_valid_params(params: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {p["name"]: generate_valid_value(p["schema"], name=p["name"]) for p in params}


def _render_path(path: str, path_params: Dict[str, Any]) -> str:
    result = path
    for name, value in path_params.items():
        result = result.replace(f"{{{name}}}", str(value))
    return result


def _error_status_kw(response_schemas: Dict[str, Any]) -> Dict[str, Any]:
    """Return ExpectedResult kwargs using pick_error_status for tolerance."""
    return pick_error_status(response_schemas, ["400", "422"])


# ---------------------------------------------------------------------------
# Wrong-type cases
# ---------------------------------------------------------------------------

def _wrong_type_param_cases(
    endpoint: Dict[str, Any],
    valid_path: Dict[str, Any],
    valid_query: Dict[str, Any],
    valid_headers: Dict[str, Any],
    valid_body: Any,
    err_kw: Dict[str, Any],
) -> List[TestCase]:
    cases: List[TestCase] = []
    ref = endpoint["endpoint_id"]
    method = endpoint["method"]
    path = endpoint["path"]

    # Path params
    for param in endpoint.get("path_params", []):
        wrong = generate_wrong_type_value(param["schema"])
        pp = dict(valid_path)
        pp[param["name"]] = wrong
        cases.append(TestCase(
            test_id="",
            title=f"{ref} - Negative type: {param['name']}={wrong!r}",
            category="negative_type",
            requirement_ref=ref,
            method=method,
            path=path,
            priority="medium",
            preconditions=[],
            steps=[TestStep(
                step_number=1,
                action=f"Send {method} request to {_render_path(path, pp)} with {param['name']} as wrong type",
                input_data=InputData(path_params=pp, query_params=valid_query, headers=valid_headers, body=valid_body),
            )],
            expected_result=ExpectedResult(description=f"Rejected: {param['name']} has wrong type", **err_kw),
        ))

    # Query params (required ones first)
    for param in endpoint.get("query_params", []):
        if not param.get("required"):
            continue
        wrong = generate_wrong_type_value(param["schema"])
        qp = dict(valid_query)
        qp[param["name"]] = wrong
        cases.append(TestCase(
            test_id="",
            title=f"{ref} - Negative type: query {param['name']}={wrong!r}",
            category="negative_type",
            requirement_ref=ref,
            method=method,
            path=path,
            priority="medium",
            preconditions=[],
            steps=[TestStep(
                step_number=1,
                action=f"Send {method} request with query param {param['name']} as wrong type",
                input_data=InputData(path_params=valid_path, query_params=qp, headers=valid_headers, body=valid_body),
            )],
            expected_result=ExpectedResult(description=f"Rejected: query param {param['name']} has wrong type", **err_kw),
        ))

    return cases


# ---------------------------------------------------------------------------
# Missing-required cases
# ---------------------------------------------------------------------------

def _missing_required_cases(
    endpoint: Dict[str, Any],
    valid_path: Dict[str, Any],
    valid_query: Dict[str, Any],
    valid_headers: Dict[str, Any],
    valid_body: Any,
    err_kw: Dict[str, Any],
) -> List[TestCase]:
    cases: List[TestCase] = []
    ref = endpoint["endpoint_id"]
    method = endpoint["method"]
    path = endpoint["path"]

    # Missing required query params
    for param in endpoint.get("query_params", []):
        if not param.get("required"):
            continue
        qp = dict(valid_query)
        qp.pop(param["name"], None)
        cases.append(TestCase(
            test_id="",
            title=f"{ref} - Negative missing: required query param '{param['name']}' omitted",
            category="negative_missing",
            requirement_ref=ref,
            method=method,
            path=path,
            priority="medium",
            preconditions=[],
            steps=[TestStep(
                step_number=1,
                action=f"Send {method} request to {_render_path(path, valid_path)} without required query param '{param['name']}'",
                input_data=InputData(path_params=valid_path, query_params=qp, headers=valid_headers, body=valid_body),
            )],
            expected_result=ExpectedResult(description=f"Rejected: required query param '{param['name']}' is missing", **err_kw),
        ))

    # Missing required body fields
    req_schema = endpoint.get("request_schema")
    if req_schema and (req_schema.get("type") == "object" or "properties" in req_schema):
        required_fields = req_schema.get("required", [])
        for field_name in required_fields:
            if not isinstance(valid_body, dict):
                continue
            body_copy = dict(valid_body)
            body_copy.pop(field_name, None)
            cases.append(TestCase(
                test_id="",
                title=f"{ref} - Negative missing: required body field '{field_name}' omitted",
                category="negative_missing",
                requirement_ref=ref,
                method=method,
                path=path,
                priority="medium",
                preconditions=[],
                steps=[TestStep(
                    step_number=1,
                    action=f"Send {method} request with body missing required field '{field_name}'",
                    input_data=InputData(path_params=valid_path, query_params=valid_query, headers=valid_headers, body=body_copy),
                )],
                expected_result=ExpectedResult(description=f"Rejected: required body field '{field_name}' is missing", **err_kw),
            ))

    return cases


# ---------------------------------------------------------------------------
# Invalid-value cases (bad enum, bad format)
# ---------------------------------------------------------------------------

def _invalid_value_cases(
    endpoint: Dict[str, Any],
    valid_path: Dict[str, Any],
    valid_query: Dict[str, Any],
    valid_headers: Dict[str, Any],
    valid_body: Any,
    err_kw: Dict[str, Any],
) -> List[TestCase]:
    cases: List[TestCase] = []
    ref = endpoint["endpoint_id"]
    method = endpoint["method"]
    path = endpoint["path"]

    # Enum violations in query params
    for param in endpoint.get("query_params", []):
        enum_vals = param["schema"].get("enum")
        if not enum_vals:
            # Check items.enum for array-type params
            items = param["schema"].get("items", {})
            enum_vals = items.get("enum") if isinstance(items, dict) else None
        if enum_vals:
            bad = "invalid_enum_value_xyz"
            qp = dict(valid_query)
            qp[param["name"]] = bad
            cases.append(TestCase(
                test_id="",
                title=f"{ref} - Negative invalid: query {param['name']}='{bad}'",
                category="negative_invalid",
                requirement_ref=ref,
                method=method,
                path=path,
                priority="medium",
                preconditions=[],
                steps=[TestStep(
                    step_number=1,
                    action=f"Send {method} request with {param['name']}='{bad}' (not in enum)",
                    input_data=InputData(path_params=valid_path, query_params=qp, headers=valid_headers, body=valid_body),
                )],
                expected_result=ExpectedResult(description=f"Rejected: '{bad}' is not a valid value for {param['name']}", **err_kw),
            ))

    # Enum violations in body fields
    req_schema = endpoint.get("request_schema")
    if req_schema and (req_schema.get("type") == "object" or "properties" in req_schema):
        for field_name, field_schema in req_schema.get("properties", {}).items():
            if field_schema.get("enum") and isinstance(valid_body, dict):
                bad = "invalid_enum_value_xyz"
                body_copy = dict(valid_body)
                body_copy[field_name] = bad
                cases.append(TestCase(
                    test_id="",
                    title=f"{ref} - Negative invalid: body field '{field_name}'='{bad}'",
                    category="negative_invalid",
                    requirement_ref=ref,
                    method=method,
                    path=path,
                    priority="medium",
                    preconditions=[],
                    steps=[TestStep(
                        step_number=1,
                        action=f"Send {method} request with body field '{field_name}'='{bad}' (not in enum)",
                        input_data=InputData(path_params=valid_path, query_params=valid_query, headers=valid_headers, body=body_copy),
                    )],
                    expected_result=ExpectedResult(description=f"Rejected: '{bad}' is not a valid enum value for '{field_name}'", **err_kw),
                ))

    # Format violations in query params
    for param in endpoint.get("query_params", []):
        bad_fmt = generate_invalid_format_value(param["schema"])
        if bad_fmt is not None:
            qp = dict(valid_query)
            qp[param["name"]] = bad_fmt
            cases.append(TestCase(
                test_id="",
                title=f"{ref} - Negative invalid: query {param['name']}='{bad_fmt}' (bad format)",
                category="negative_invalid",
                requirement_ref=ref,
                method=method,
                path=path,
                priority="medium",
                preconditions=[],
                steps=[TestStep(
                    step_number=1,
                    action=f"Send {method} request with malformed {param['name']}='{bad_fmt}'",
                    input_data=InputData(path_params=valid_path, query_params=qp, headers=valid_headers, body=valid_body),
                )],
                expected_result=ExpectedResult(description=f"Rejected: '{bad_fmt}' is not a valid format for {param['name']}", **err_kw),
            ))

    return cases


# ---------------------------------------------------------------------------
# Fallback to guarantee >= 2 negative cases
# ---------------------------------------------------------------------------

def _fallback_negative_case(
    endpoint: Dict[str, Any],
    valid_path: Dict[str, Any],
    valid_query: Dict[str, Any],
    valid_headers: Dict[str, Any],
    valid_body: Any,
    err_kw: Dict[str, Any],
) -> TestCase:
    """Non-existent resource test -- a universal negative case."""
    ref = endpoint["endpoint_id"]
    method = endpoint["method"]
    path = endpoint["path"]

    pp = dict(valid_path)
    for param in endpoint.get("path_params", []):
        if param["schema"].get("type") in ("integer", "number"):
            pp[param["name"]] = 9999999

    return TestCase(
        test_id="",
        title=f"{ref} - Negative: request non-existent resource",
        category="negative_invalid",
        requirement_ref=ref,
        method=method,
        path=path,
        priority="medium",
        preconditions=["The requested resource does not exist"],
        steps=[TestStep(
            step_number=1,
            action=f"Send {method} request to {_render_path(path, pp)} for a non-existent resource",
            input_data=InputData(path_params=pp, query_params=valid_query, headers=valid_headers, body=valid_body),
        )],
        expected_result=ExpectedResult(status_code=404, description="Resource not found"),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_negative_cases(endpoint: Dict[str, Any]) -> List[TestCase]:
    valid_path = _build_valid_params(endpoint.get("path_params", []))
    valid_query = _build_valid_params(endpoint.get("query_params", []))
    valid_headers = _build_valid_params(endpoint.get("header_params", []))
    valid_body = None
    if endpoint.get("request_schema"):
        valid_body = generate_valid_value(endpoint["request_schema"])

    err_kw = _error_status_kw(endpoint.get("response_schemas", {}))

    cases: List[TestCase] = []
    cases.extend(_wrong_type_param_cases(endpoint, valid_path, valid_query, valid_headers, valid_body, err_kw))
    cases.extend(_missing_required_cases(endpoint, valid_path, valid_query, valid_headers, valid_body, err_kw))
    cases.extend(_invalid_value_cases(endpoint, valid_path, valid_query, valid_headers, valid_body, err_kw))

    # Guarantee at least 2 negative cases
    if len(cases) < 2:
        cases.append(_fallback_negative_case(endpoint, valid_path, valid_query, valid_headers, valid_body, err_kw))
    if len(cases) < 2:
        # Second fallback: send completely empty body when one is expected
        ref = endpoint["endpoint_id"]
        if endpoint.get("request_schema"):
            cases.append(TestCase(
                test_id="",
                title=f"{ref} - Negative: send empty body when body is required",
                category="negative_missing",
                requirement_ref=ref,
                method=endpoint["method"],
                path=endpoint["path"],
                priority="medium",
                preconditions=[],
                steps=[TestStep(
                    step_number=1,
                    action=f"Send {endpoint['method']} request with empty body",
                    input_data=InputData(path_params=valid_path, query_params=valid_query, headers=valid_headers, body={}),
                )],
                expected_result=ExpectedResult(description="Rejected: request body is empty or missing required fields", **err_kw),
            ))
        else:
            # Minimal endpoint -- add an unsupported method test
            cases.append(TestCase(
                test_id="",
                title=f"{ref} - Negative: unsupported HTTP method",
                category="negative_invalid",
                requirement_ref=ref,
                method="PATCH",
                path=endpoint["path"],
                priority="low",
                preconditions=[],
                steps=[TestStep(
                    step_number=1,
                    action=f"Send PATCH request to {_render_path(endpoint['path'], valid_path)} (unsupported method)",
                    input_data=InputData(path_params=valid_path, query_params={}, headers=valid_headers, body=None),
                )],
                expected_result=ExpectedResult(status_code=405, description="Method not allowed"),
            ))

    return cases
