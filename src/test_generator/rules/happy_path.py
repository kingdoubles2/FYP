from __future__ import annotations

from typing import Any, Dict, List

from test_generator.models import TestCase, TestStep, InputData, ExpectedResult
from test_generator.sample_data import generate_valid_value, should_autofill_header_param


def _deep_overlay(base: Any, overlay: Any) -> Any:
    """Overlay example data on top of generated defaults.

    Dicts merge recursively so required defaults remain when example omits them.
    Non-dict values (including arrays) are replaced by overlay.
    """
    if isinstance(base, dict) and isinstance(overlay, dict):
        merged = dict(base)
        for key, value in overlay.items():
            if key in merged:
                merged[key] = _deep_overlay(merged[key], value)
            else:
                merged[key] = value
        return merged
    return overlay


def _fill_required_fields_from_schema(schema: Dict[str, Any], body_example: Any) -> Any:
    """Keep example payload shape while filling missing required fields.

    This is used for example-based happy paths because some specs provide
    illustrative examples that omit formally required fields.
    """
    required_defaults = generate_valid_value(
        schema,
        skip_example=True,
        use_realistic=True,
        required_only=True,
    )
    return _deep_overlay(required_defaults, body_example)


def _build_valid_params(
    params: List[Dict[str, Any]],
    *,
    required_only: bool = False,
    skip_example: bool = False,
    use_realistic: bool = False,
    is_header: bool = False,
) -> Dict[str, Any]:
    """Generate a valid value for every parameter in the list.

    When *required_only* is True, optional parameters are omitted.
    When *skip_example* is True, schema ``example`` values are ignored.
    When *use_realistic* is True, prefers realistic values over generic placeholders.
    """
    selected = params
    if is_header:
        selected = [p for p in selected if should_autofill_header_param(p)]
    if required_only:
        selected = [p for p in selected if p.get("required", False)]
    return {
        p["name"]: generate_valid_value(
            p["schema"], name=p["name"], skip_example=skip_example, use_realistic=use_realistic,
        )
        for p in selected
    }


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


def _has_examples(endpoint: Dict[str, Any]) -> bool:
    """Return True if the endpoint has any example values we can use."""
    for group in ("path_params", "query_params", "header_params"):
        for p in endpoint.get(group, []):
            if "example" in p.get("schema", {}):
                return True
    schema = endpoint.get("request_schema")
    if isinstance(schema, dict):
        if "example" in schema:
            return True
        # Check property-level examples (inline in schema properties)
        for _prop, prop_schema in schema.get("properties", {}).items():
            if isinstance(prop_schema, dict) and "example" in prop_schema:
                return True
    return False


def generate_happy_path_cases(endpoint: Dict[str, Any]) -> List[TestCase]:
    method = endpoint["method"]
    path = endpoint["path"]
    ref = endpoint["endpoint_id"]

    status = _first_success_code(endpoint.get("response_schemas", {}), method)
    resp_schema = endpoint.get("response_schemas", {}).get(str(status))

    cases: List[TestCase] = []

    # ----- Case 1: Minimal valid request (required params only, no examples) -----
    min_path = _build_valid_params(
        endpoint.get("path_params", []), required_only=True, skip_example=True, use_realistic=True,
    )
    min_query = _build_valid_params(
        endpoint.get("query_params", []), required_only=True, skip_example=True, use_realistic=True,
    )
    min_headers = _build_valid_params(
        endpoint.get("header_params", []), required_only=True, skip_example=True, use_realistic=True, is_header=True,
    )
    min_body = None
    req_schema = endpoint.get("request_schema")
    body_required = endpoint.get("request_body_required", False)
    if req_schema and (body_required or method in {"POST", "PUT", "PATCH"}):
        min_body = generate_valid_value(
            req_schema, skip_example=True, use_realistic=True, required_only=True,
        )
        # Many APIs reject null/absent payload when a request schema exists.
        if method in {"POST", "PUT", "PATCH"} and min_body is None:
            min_body = [] if req_schema.get("type") == "array" else {}

    min_rendered = _render_path(path, min_path)
    min_preconditions: List[str] = []
    if min_path:
        min_preconditions.append(
            "Resource exists for " + ", ".join(f"{k}={v}" for k, v in min_path.items())
        )
    if min_headers:
        min_preconditions.append("Valid credentials / headers are available")

    min_action_parts = [f"Send {method} request to {min_rendered}"]
    if min_query:
        min_action_parts.append(f"with query params {min_query}")
    if min_headers:
        min_action_parts.append(f"with headers {list(min_headers.keys())}")
    if min_body is not None:
        min_action_parts.append("with valid request body")

    cases.append(TestCase(
        test_id="",
        title=f"{ref} - Happy path: minimal valid request returns {status}",
        category="happy_path",
        requirement_ref=ref,
        method=method,
        path=path,
        priority="high",
        preconditions=min_preconditions,
        steps=[TestStep(
            step_number=1,
            action=" ".join(min_action_parts),
            input_data=InputData(
                path_params=min_path,
                query_params=min_query,
                headers=min_headers,
                body=min_body,
            ),
        )],
        expected_result=ExpectedResult(
            status_code=status,
            description=_describe_response(resp_schema, status),
        ),
    ))

    # ----- Case 2: Example-based request (only if spec provides examples) -----
    if _has_examples(endpoint):
        ex_path = _build_valid_params(endpoint.get("path_params", []))
        ex_query = _build_valid_params(endpoint.get("query_params", []))
        ex_headers = _build_valid_params(endpoint.get("header_params", []), is_header=True)
        ex_body = None
        if endpoint.get("request_schema"):
            ex_body = generate_valid_value(endpoint["request_schema"])
            ex_body = _fill_required_fields_from_schema(endpoint["request_schema"], ex_body)

        ex_rendered = _render_path(path, ex_path)
        ex_preconditions: List[str] = []
        if ex_path:
            ex_preconditions.append(
                "Resource exists for " + ", ".join(f"{k}={v}" for k, v in ex_path.items())
            )
        if ex_headers:
            ex_preconditions.append("Valid credentials / headers are available")

        ex_action_parts = [f"Send {method} request to {ex_rendered}"]
        if ex_query:
            ex_action_parts.append(f"with query params {ex_query}")
        if ex_headers:
            ex_action_parts.append(f"with headers {list(ex_headers.keys())}")
        if ex_body is not None:
            ex_action_parts.append("with example request body")

        cases.append(TestCase(
            test_id="",
            title=f"{ref} - Happy path: example-based request returns {status}",
            category="happy_path",
            requirement_ref=ref,
            method=method,
            path=path,
            priority="medium",
            preconditions=ex_preconditions,
            steps=[TestStep(
                step_number=1,
                action=" ".join(ex_action_parts),
                input_data=InputData(
                    path_params=ex_path,
                    query_params=ex_query,
                    headers=ex_headers,
                    body=ex_body,
                ),
            )],
            expected_result=ExpectedResult(
                status_code=status,
                description=_describe_response(resp_schema, status),
            ),
        ))

    # ----- Extra happy-path cases for required enum params -----
    for param in endpoint.get("query_params", []):
        enum_vals = param["schema"].get("enum", [])
        if param["required"] and len(enum_vals) > 1:
            for extra_val in enum_vals[1:3]:
                extra_query = dict(min_query)
                extra_query[param["name"]] = extra_val
                cases.append(
                    TestCase(
                        test_id="",
                        title=f"{ref} - Happy path: {param['name']}={extra_val}",
                        category="happy_path",
                        requirement_ref=ref,
                        method=method,
                        path=path,
                        priority="high",
                        preconditions=min_preconditions,
                        steps=[
                            TestStep(
                                step_number=1,
                                action=f"Send {method} request to {min_rendered} with {param['name']}={extra_val}",
                                input_data=InputData(
                                    path_params=min_path,
                                    query_params=extra_query,
                                    headers=min_headers,
                                    body=min_body,
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
