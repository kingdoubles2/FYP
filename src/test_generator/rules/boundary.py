from __future__ import annotations

from typing import Any, Dict, List, Tuple

from test_generator.models import TestCase, TestStep, InputData, ExpectedResult
from test_generator.sample_data import generate_valid_value, pick_error_status


def _build_valid_params(params: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {p["name"]: generate_valid_value(p["schema"], name=p["name"]) for p in params}


def _render_path(path: str, path_params: Dict[str, Any]) -> str:
    result = path
    for name, value in path_params.items():
        result = result.replace(f"{{{name}}}", str(value))
    return result


def _success_status_kw(response_schemas: Dict[str, Any], method: str) -> Dict[str, Any]:
    """Pick success expectation from declared 2xx responses (spec-first)."""
    codes = [c for c in response_schemas.keys() if isinstance(c, str) and c.startswith("2")]
    if not codes:
        return {"status_code": 200}

    # Prefer common semantically expected codes per method when declared.
    preferred: List[str] = []
    if method == "POST":
        preferred.append("201")
    if method == "DELETE":
        preferred.append("204")
    preferred.append("200")

    ordered = [c for c in preferred if c in codes]
    ordered.extend(sorted(c for c in codes if c not in ordered))

    if len(ordered) == 1:
        return {"status_code": int(ordered[0])}
    return {"status_code_any_of": [int(c) for c in ordered]}


# ---------------------------------------------------------------------------
# Constraint detection — only generate boundaries when IR has explicit limits
# ---------------------------------------------------------------------------

_INT_CONSTRAINTS = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}
_NUM_CONSTRAINTS = _INT_CONSTRAINTS
_STR_CONSTRAINTS = {"minLength", "maxLength", "pattern"}
_ARR_CONSTRAINTS = {"minItems", "maxItems"}


def _has_explicit_constraints(schema: Dict[str, Any]) -> bool:
    stype = schema.get("type", "")
    if stype == "integer":
        return bool(_INT_CONSTRAINTS & schema.keys()) or schema.get("format") in ("int32", "int64")
    if stype == "number":
        return bool(_NUM_CONSTRAINTS & schema.keys())
    if stype == "string":
        return bool(_STR_CONSTRAINTS & schema.keys()) or schema.get("format") is not None
    if stype == "array":
        return bool(_ARR_CONSTRAINTS & schema.keys())
    return False


def _boundary_pairs(schema: Dict[str, Any]) -> List[Tuple[str, Any, bool]]:
    """Return (label, value, is_invalid) triples derived from explicit IR constraints."""
    stype = schema.get("type", "")
    pairs: List[Tuple[str, Any, bool]] = []

    if stype == "integer":
        fmt = schema.get("format", "")
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        ex_min = schema.get("exclusiveMinimum", False)
        ex_max = schema.get("exclusiveMaximum", False)

        if minimum is not None:
            at = (minimum + 1) if ex_min else minimum
            below = (minimum) if ex_min else (minimum - 1)
            pairs.append((f"at_minimum({at})", at, False))
            pairs.append((f"below_minimum({below})", below, True))
        if maximum is not None:
            at = (maximum - 1) if ex_max else maximum
            above = (maximum) if ex_max else (maximum + 1)
            pairs.append((f"at_maximum({at})", at, False))
            pairs.append((f"above_maximum({above})", above, True))
        # Format-driven boundary (only if no explicit min/max)
        if minimum is None and maximum is None:
            if fmt == "int32":
                pairs.append(("max_int32(2147483647)", 2147483647, False))
            elif fmt == "int64":
                pairs.append(("max_int64(9223372036854775807)", 9223372036854775807, False))

    elif stype == "number":
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        ex_min = schema.get("exclusiveMinimum", False)
        ex_max = schema.get("exclusiveMaximum", False)

        if minimum is not None:
            below = (minimum + 0.001) if ex_min else (minimum - 0.001)
            pairs.append((f"below_minimum({below})", below, True))
        if maximum is not None:
            above = (maximum - 0.001) if ex_max else (maximum + 0.001)
            pairs.append((f"above_maximum({above})", above, True))

    elif stype == "string":
        min_len = schema.get("minLength")
        max_len = schema.get("maxLength")

        if min_len is not None and min_len > 0:
            pairs.append((f"below_minLength({min_len - 1})", "a" * (min_len - 1), True))
            pairs.append((f"at_minLength({min_len})", "a" * min_len, False))
        if max_len is not None:
            pairs.append((f"above_maxLength({max_len + 1})", "a" * (max_len + 1), True))
            pairs.append((f"at_maxLength({max_len})", "a" * max_len, False))

    elif stype == "array":
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        items_schema = schema.get("items", {})

        if min_items is not None and min_items > 0:
            below_count = min_items - 1
            pairs.append((
                f"below_minItems({below_count})",
                [generate_valid_value(items_schema) for _ in range(below_count)],
                True,
            ))
        if max_items is not None:
            above_count = max_items + 1
            pairs.append((
                f"above_maxItems({above_count})",
                [generate_valid_value(items_schema) for _ in range(above_count)],
                True,
            ))

    return pairs


# ---------------------------------------------------------------------------
# Nullable
# ---------------------------------------------------------------------------

def _nullable_cases(
    endpoint: Dict[str, Any],
    valid_path: Dict[str, Any],
    valid_query: Dict[str, Any],
    valid_headers: Dict[str, Any],
    valid_body: Any,
) -> List[TestCase]:
    ref = endpoint["endpoint_id"]
    method = endpoint["method"]
    path = endpoint["path"]
    req_schema = endpoint.get("request_schema")
    cases: List[TestCase] = []

    if not req_schema or not isinstance(valid_body, dict):
        return cases

    props = req_schema.get("properties", {})
    for field_name, field_schema in props.items():
        if field_schema.get("nullable"):
            body_copy = dict(valid_body)
            body_copy[field_name] = None
            cases.append(TestCase(
                test_id="",
                title=f"{ref} - Boundary: body '{field_name}'=null (nullable)",
                category="boundary",
                requirement_ref=ref,
                method=method,
                path=path,
                priority="low",
                preconditions=[],
                steps=[TestStep(
                    step_number=1,
                    action=f"Send {method} request with nullable field '{field_name}'=null",
                    input_data=InputData(
                        path_params=valid_path, query_params=valid_query,
                        headers=valid_headers, body=body_copy,
                    ),
                )],
                expected_result=ExpectedResult(
                    status_code=200,
                    description=f"Nullable field '{field_name}' set to null should be accepted",
                ),
            ))
            break  # One nullable case is enough

    return cases


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_boundary_cases(endpoint: Dict[str, Any]) -> List[TestCase]:
    ref = endpoint["endpoint_id"]
    method = endpoint["method"]
    path = endpoint["path"]
    resp = endpoint.get("response_schemas", {})

    valid_path = _build_valid_params(endpoint.get("path_params", []))
    valid_query = _build_valid_params(endpoint.get("query_params", []))
    valid_headers = _build_valid_params(endpoint.get("header_params", []))
    valid_body = None
    if endpoint.get("request_schema"):
        valid_body = generate_valid_value(endpoint["request_schema"])

    cases: List[TestCase] = []

    # Collect all constrained params and body fields
    all_targets: List[Dict[str, Any]] = []  # (location, name, schema, source_list)

    for param in endpoint.get("path_params", []):
        if _has_explicit_constraints(param["schema"]):
            all_targets.append({"loc": "path", "name": param["name"], "schema": param["schema"]})
    for param in endpoint.get("query_params", []):
        if _has_explicit_constraints(param["schema"]):
            all_targets.append({"loc": "query", "name": param["name"], "schema": param["schema"]})

    req_schema = endpoint.get("request_schema")
    body_type = None
    if req_schema:
        body_type = req_schema.get("type")
        if body_type is None and "properties" in req_schema:
            body_type = "object"
    if req_schema and body_type == "object" and isinstance(valid_body, dict):
        for field_name, field_schema in req_schema.get("properties", {}).items():
            if _has_explicit_constraints(field_schema):
                all_targets.append({"loc": "body", "name": field_name, "schema": field_schema})

    # Generate boundary pairs for each constrained target
    for target in all_targets:
        for label, bval, is_invalid in _boundary_pairs(target["schema"]):
            if target["loc"] == "path":
                pp = dict(valid_path)
                pp[target["name"]] = bval
                rendered = _render_path(path, pp)
                input_data = InputData(path_params=pp, query_params=valid_query, headers=valid_headers, body=valid_body)
                action = f"Send {method} request to {rendered} with {target['name']}={bval!r}"
            elif target["loc"] == "query":
                qp = dict(valid_query)
                qp[target["name"]] = bval
                input_data = InputData(path_params=valid_path, query_params=qp, headers=valid_headers, body=valid_body)
                action = f"Send {method} request with query {target['name']}={bval!r}"
            else:  # body
                body_copy = dict(valid_body)
                body_copy[target["name"]] = bval
                input_data = InputData(path_params=valid_path, query_params=valid_query, headers=valid_headers, body=body_copy)
                action = f"Send {method} request with body '{target['name']}'={bval!r}"

            if is_invalid:
                status_kw = pick_error_status(resp, ["422", "400", "409", "403", "404"])
                desc = f"Boundary violation: {target['name']} {label} should be rejected"
            else:
                status_kw = _success_status_kw(resp, method)
                desc = f"Boundary valid: {target['name']} {label} should be accepted"

            cases.append(TestCase(
                test_id="",
                title=f"{ref} - Boundary: {target['loc']} {target['name']} {label}",
                category="boundary",
                requirement_ref=ref,
                method=method,
                path=path,
                priority="low",
                preconditions=[],
                steps=[TestStep(step_number=1, action=action, input_data=input_data)],
                expected_result=ExpectedResult(description=desc, **status_kw),
            ))

        if len(cases) >= 5:
            break  # Cap early

    # Nullable fields
    cases.extend(_nullable_cases(endpoint, valid_path, valid_query, valid_headers, valid_body))

    return cases[:5]
