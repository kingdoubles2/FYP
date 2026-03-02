from __future__ import annotations

from typing import Any, Dict, List

from test_generator.models import TestCase, TestStep, InputData, ExpectedResult
from test_generator.sample_data import generate_valid_value


def _build_valid_params(params: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a valid value for every parameter in the list."""
    return {p["name"]: generate_valid_value(p["schema"], name=p["name"]) for p in params}


def _first_success_code(response_schemas: Dict[str, Any], method: str) -> int:
    """Return the most appropriate 2xx status code.

    Prefers 201 for POST, 204 for DELETE when available, otherwise first 2xx.
    """
    codes = sorted(response_schemas)
    two_xx = [c for c in codes if c.startswith("2")]

    # Prefer semantically correct codes for the method
    if method == "POST" and "201" in two_xx:
        return 201
    if method == "DELETE" and "204" in two_xx:
        return 204

    for code in two_xx:
        return int(code)
    return 200


def _describe_response(schema: Any, status_code: int) -> str:
    """Build a short human-readable description of the response shape."""
    if status_code == 204:
        return "No Content"
    if schema is None:
        return "Response may have no body"
    if isinstance(schema, dict):
        stype = schema.get("type", "")
        # Implicit object: has properties but no explicit type
        if stype == "" and "properties" in schema:
            stype = "object"
        if stype == "object":
            props = list(schema.get("properties", {}).keys())
            if props:
                return f"Returns object with fields: {', '.join(props[:6])}"
            return "Returns an object"
        if stype == "array":
            return "Returns an array"
        if stype:
            return f"Returns a {stype} value"
    return "Response body present"


def _render_path(path: str, path_params: Dict[str, Any]) -> str:
    """Substitute path parameters into the template: /pet/{petId} -> /pet/10."""
    result = path
    for name, value in path_params.items():
        result = result.replace(f"{{{name}}}", str(value))
    return result


def generate_happy_path_cases(endpoint: Dict[str, Any]) -> List[TestCase]:
    method = endpoint["method"]
    path = endpoint["path"]
    ref = endpoint["endpoint_id"]

    path_vals = _build_valid_params(endpoint.get("path_params", []))
    query_vals = _build_valid_params(endpoint.get("query_params", []))
    header_vals = _build_valid_params(endpoint.get("header_params", []))

    body = None
    if endpoint.get("request_schema"):
        body = generate_valid_value(endpoint["request_schema"])

    rendered = _render_path(path, path_vals)
    status = _first_success_code(endpoint.get("response_schemas", {}), method)
    resp_schema = endpoint.get("response_schemas", {}).get(str(status))

    preconditions: List[str] = []
    if path_vals:
        preconditions.append(
            "Resource exists for " + ", ".join(f"{k}={v}" for k, v in path_vals.items())
        )
    if header_vals:
        preconditions.append("Valid credentials / headers are available")

    action_parts = [f"Send {method} request to {rendered}"]
    if query_vals:
        action_parts.append(f"with query params {query_vals}")
    if header_vals:
        action_parts.append(f"with headers {list(header_vals.keys())}")
    if body is not None:
        action_parts.append("with valid request body")

    case = TestCase(
        test_id="",
        title=f"{ref} - Happy path: valid request returns {status}",
        category="happy_path",
        requirement_ref=ref,
        method=method,
        path=path,
        priority="high",
        preconditions=preconditions,
        steps=[
            TestStep(
                step_number=1,
                action=" ".join(action_parts),
                input_data=InputData(
                    path_params=path_vals,
                    query_params=query_vals,
                    headers=header_vals,
                    body=body,
                ),
            )
        ],
        expected_result=ExpectedResult(
            status_code=status,
            description=_describe_response(resp_schema, status),
        ),
    )

    cases = [case]

    # Extra happy-path for required enum params (up to 2 extra values)
    for param in endpoint.get("query_params", []):
        enum_vals = param["schema"].get("enum", [])
        if param["required"] and len(enum_vals) > 1:
            for extra_val in enum_vals[1:3]:
                extra_query = dict(query_vals)
                extra_query[param["name"]] = extra_val
                extra_rendered = rendered
                cases.append(
                    TestCase(
                        test_id="",
                        title=f"{ref} - Happy path: {param['name']}={extra_val}",
                        category="happy_path",
                        requirement_ref=ref,
                        method=method,
                        path=path,
                        priority="high",
                        preconditions=preconditions,
                        steps=[
                            TestStep(
                                step_number=1,
                                action=f"Send {method} request to {extra_rendered} with {param['name']}={extra_val}",
                                input_data=InputData(
                                    path_params=path_vals,
                                    query_params=extra_query,
                                    headers=header_vals,
                                    body=body,
                                ),
                            )
                        ],
                        expected_result=ExpectedResult(
                            status_code=status,
                            description=_describe_response(resp_schema, status),
                        ),
                    )
                )

    return cases
