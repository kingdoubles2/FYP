from __future__ import annotations

from typing import Any, Dict, List, Optional

from test_generator.models import TestCase, TestStep, InputData, ExpectedResult
from test_generator.sample_data import generate_valid_value


def _build_valid_params(params: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {p["name"]: generate_valid_value(p["schema"], name=p["name"]) for p in params}


def _render_path(path: str, path_params: Dict[str, Any]) -> str:
    result = path
    for name, value in path_params.items():
        result = result.replace(f"{{{name}}}", str(value))
    return result


def _schema_type_summary(schema: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    """Summarise a response schema into {field: type} for expected_result."""
    if not schema or not isinstance(schema, dict):
        return None
    stype = schema.get("type", "")
    if stype != "object" and "properties" not in schema:
        return None
    props = schema.get("properties", {})
    if not props:
        return None
    return {k: v.get("type", "any") for k, v in props.items()}


# ---------------------------------------------------------------------------
# Only generate 404 cases — the single non-invented error-status scenario.
# 400/422 are covered by negative.py; 401/403 by auth.py; 500 is server-side.
# ---------------------------------------------------------------------------

def _cases_for_404(
    endpoint: Dict[str, Any],
    valid_path: Dict[str, Any],
    valid_query: Dict[str, Any],
    valid_headers: Dict[str, Any],
    valid_body: Any,
    resp_schema: Optional[Dict[str, Any]],
) -> List[TestCase]:
    """Test each path param individually as non-existent resource."""
    path_params = endpoint.get("path_params", [])
    if not path_params:
        return []

    ref = endpoint["endpoint_id"]
    method = endpoint["method"]
    path = endpoint["path"]
    body_contains = _schema_type_summary(resp_schema)
    cases: List[TestCase] = []

    for target_param in path_params:
        pp = dict(valid_path)
        ptype = target_param["schema"].get("type", "")
        if ptype in ("integer", "number"):
            pp[target_param["name"]] = 9999999
        else:
            pp[target_param["name"]] = "nonexistent_resource_xyz"

        cases.append(TestCase(
            test_id="",
            title=f"{ref} - Error 404: non-existent {target_param['name']}",
            category="error_status",
            requirement_ref=ref,
            method=method,
            path=path,
            priority="medium",
            preconditions=[f"No resource exists for {target_param['name']}={pp[target_param['name']]}"],
            steps=[TestStep(
                step_number=1,
                action=f"Send {method} request to {_render_path(path, pp)} with non-existent {target_param['name']}",
                input_data=InputData(path_params=pp, query_params=valid_query, headers=valid_headers, body=valid_body),
            )],
            expected_result=ExpectedResult(
                status_code=404,
                description=f"Resource not found for {target_param['name']}",
                response_body_contains=body_contains,
            ),
        ))

    return cases


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_error_status_cases(endpoint: Dict[str, Any]) -> List[TestCase]:
    resp_schemas = endpoint.get("response_schemas", {})

    # Only generate 404 cases when the IR actually declares a 404 response
    if "404" not in resp_schemas:
        return []

    valid_path = _build_valid_params(endpoint.get("path_params", []))
    valid_query = _build_valid_params(endpoint.get("query_params", []))
    valid_headers = _build_valid_params(endpoint.get("header_params", []))
    valid_body = None
    if endpoint.get("request_schema"):
        valid_body = generate_valid_value(endpoint["request_schema"])

    schema_404 = resp_schemas.get("404")
    return _cases_for_404(endpoint, valid_path, valid_query, valid_headers, valid_body, schema_404)
