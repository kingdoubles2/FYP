from __future__ import annotations

from typing import Any, Dict, List, Optional

from test_generator.models import TestCase, TestStep, InputData, ExpectedResult
from test_generator.sample_data import (
    generate_valid_value,
    generate_wrong_type_value,
    generate_invalid_format_value,
    pick_error_status,
    should_autofill_header_param,
)


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


def _error_status_kw(response_schemas: Dict[str, Any]) -> Dict[str, Any]:
    """Return validation-style error expectation, preferring declared spec codes."""
    # Real-world APIs often return 404 before detailed validation errors when resource lookup fails first.
    base = pick_error_status(response_schemas, ["404", "422", "400", "409", "403"])
    return _allow_runtime_404(base)


def _allow_runtime_404(status_kw: Dict[str, Any]) -> Dict[str, Any]:
    """Allow 404 as runtime-tolerant fallback even when not declared in spec."""
    if "status_code_any_of" in status_kw:
        vals = list(status_kw["status_code_any_of"])
        if 404 not in vals:
            vals.append(404)
        return {"status_code_any_of": sorted(set(vals))}

    if "status_code" in status_kw:
        code = int(status_kw["status_code"])
        if code == 404:
            return {"status_code": 404}
        return {"status_code_any_of": sorted({code, 404})}

    return {"status_code": 404}


def _resource_not_found_kw(response_schemas: Dict[str, Any]) -> Dict[str, Any]:
    """Return not-found style error expectation, preferring declared spec codes."""
    return pick_error_status(response_schemas, ["404", "410", "422", "400", "403"])


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

    # Do not generate path "wrong type" cases.
    # Path params are interpolated into URL strings before request dispatch,
    # so numeric/boolean values become strings and most APIs return 404/not-found
    # (or route mismatch) instead of a deterministic 400 type-validation error.

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
) -> Optional[TestCase]:
    """Non-existent resource test for endpoints with path params.

    Returns None when the endpoint has no path params (cannot represent a different resource).
    """
    ref = endpoint["endpoint_id"]
    method = endpoint["method"]
    path = endpoint["path"]

    if not endpoint.get("path_params"):
        return None

    pp = dict(valid_path)
    for param in endpoint.get("path_params", []):
        ptype = param["schema"].get("type")
        if ptype in ("integer", "number"):
            pp[param["name"]] = 9999999
        else:
            pp[param["name"]] = "nonexistent_resource_xyz"

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
        expected_result=ExpectedResult(description="Resource not found", **err_kw),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_negative_cases(endpoint: Dict[str, Any]) -> List[TestCase]:
    valid_path = _build_valid_params(endpoint.get("path_params", []))
    valid_query = _build_valid_params(endpoint.get("query_params", []))
    valid_headers = _build_valid_params(endpoint.get("header_params", []), is_header=True)
    valid_body = None
    if endpoint.get("request_schema"):
        valid_body = generate_valid_value(
            endpoint["request_schema"],
            skip_example=True,
            use_realistic=True,
            required_only=True,
        )

    err_kw = _error_status_kw(endpoint.get("response_schemas", {}))

    cases: List[TestCase] = []
    cases.extend(_wrong_type_param_cases(endpoint, valid_path, valid_query, valid_headers, valid_body, err_kw))
    cases.extend(_missing_required_cases(endpoint, valid_path, valid_query, valid_headers, valid_body, err_kw))
    cases.extend(_invalid_value_cases(endpoint, valid_path, valid_query, valid_headers, valid_body, err_kw))

    # Prefer semantically valid fallback cases that still align with declared spec responses.
    if len(cases) < 2:
        nf_kw = _resource_not_found_kw(endpoint.get("response_schemas", {}))
        fallback = _fallback_negative_case(endpoint, valid_path, valid_query, valid_headers, valid_body, nf_kw)
        if fallback is not None:
            cases.append(fallback)

    if len(cases) < 2:
        # Second fallback: send completely empty body when one is expected
        ref = endpoint["endpoint_id"]
        if endpoint.get("request_schema") and endpoint.get("request_body_required", False):
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

    return cases
