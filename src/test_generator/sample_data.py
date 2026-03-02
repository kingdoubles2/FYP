from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Deterministic lookup tables
# ---------------------------------------------------------------------------

VALID_DEFAULTS: Dict[Tuple[str, Optional[str]], Any] = {
    ("integer", None):       10,
    ("integer", "int32"):    42,
    ("integer", "int64"):    100000,
    ("number", None):        3.14,
    ("number", "float"):     1.5,
    ("number", "double"):    52.52,
    ("string", None):        "test_string",
    ("string", "date-time"): "2026-01-15T10:30:00Z",
    ("string", "date"):      "2026-01-15",
    ("string", "email"):     "user@example.com",
    ("string", "uri"):       "https://example.com/resource",
    ("string", "uuid"):      "550e8400-e29b-41d4-a716-446655440000",
    ("string", "byte"):      "dGVzdA==",
    ("string", "binary"):    "<binary_data>",
    ("string", "password"):  "P@ssw0rd123",
    ("boolean", None):       True,
}

WRONG_TYPE_MAP: Dict[str, Any] = {
    "integer": "not_a_number",
    "number":  "not_a_number",
    "string":  12345,
    "boolean": "not_a_boolean",
    "array":   "not_an_array",
    "object":  "not_an_object",
}

INVALID_FORMAT_MAP: Dict[str, str] = {
    "date-time": "not-a-date",
    "date":      "not-a-date",
    "email":     "not-an-email",
    "uri":       "not a uri",
    "uuid":      "not-a-uuid",
}

# Name-aware realistic defaults for common field names (case-insensitive).
# Used when a string field has no example/default/enum in the IR.
FIELD_NAME_HINTS: Dict[str, str] = {
    "timezone":     "UTC",
    "country":      "IE",
    "country_code": "IE",
    "language":     "en",
    "locale":       "en_US",
    "currency":     "USD",
}


# ---------------------------------------------------------------------------
# Constraint-aware boundary values
# ---------------------------------------------------------------------------

def _boundary_values_for_integer(schema: Dict[str, Any]) -> List[Tuple[str, Any]]:
    fmt = schema.get("format", "")
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    exclusive_min = schema.get("exclusiveMinimum", False)
    exclusive_max = schema.get("exclusiveMaximum", False)

    values: List[Tuple[str, Any]] = [("zero", 0), ("negative", -1)]

    if minimum is not None:
        # Just below the minimum
        below = (minimum + 1) if exclusive_min else (minimum - 1)
        values.append(("below_minimum", below))
        at = (minimum + 1) if exclusive_min else minimum
        values.append(("at_minimum", at))
    if maximum is not None:
        above = (maximum - 1) if exclusive_max else (maximum + 1)
        values.append(("above_maximum", above))
        at = (maximum - 1) if exclusive_max else maximum
        values.append(("at_maximum", at))

    if minimum is None and maximum is None:
        if fmt == "int64":
            values.append(("max_int64", 9223372036854775807))
        else:
            values.append(("max_int32", 2147483647))

    return values


def _boundary_values_for_number(schema: Dict[str, Any]) -> List[Tuple[str, Any]]:
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    exclusive_min = schema.get("exclusiveMinimum", False)
    exclusive_max = schema.get("exclusiveMaximum", False)

    values: List[Tuple[str, Any]] = [("zero", 0.0), ("negative", -1.0)]

    if minimum is not None:
        below = (minimum + 0.001) if exclusive_min else (minimum - 0.001)
        values.append(("below_minimum", below))
    if maximum is not None:
        above = (maximum - 0.001) if exclusive_max else (maximum + 0.001)
        values.append(("above_maximum", above))

    if minimum is None and maximum is None:
        values.append(("very_large", 999999999.99))

    return values


def _boundary_values_for_string(schema: Dict[str, Any]) -> List[Tuple[str, Any]]:
    min_len = schema.get("minLength")
    max_len = schema.get("maxLength")

    values: List[Tuple[str, Any]] = [("empty_string", ""), ("single_char", "a")]

    if min_len is not None and min_len > 0:
        values.append(("below_minLength", "a" * (min_len - 1)))
    if max_len is not None:
        values.append(("above_maxLength", "a" * (max_len + 1)))
        values.append(("at_maxLength", "a" * max_len))

    if max_len is None:
        values.append(("long_string", "a" * 256))

    return values


STATIC_BOUNDARY_VALUES: Dict[str, List[Tuple[str, Any]]] = {
    "array": [("empty_array", [])],
}


# ---------------------------------------------------------------------------
# Schema composition helpers
# ---------------------------------------------------------------------------

def _merge_allof(schemas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge a list of schemas from an allOf clause into one."""
    merged: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    for sub in schemas:
        merged["properties"].update(sub.get("properties", {}))
        merged["required"].extend(sub.get("required", []))
        for key in ("type", "format", "description"):
            if key in sub and key not in merged:
                merged[key] = sub[key]
    if not merged["required"]:
        del merged["required"]
    return merged


# ---------------------------------------------------------------------------
# Value generators
# ---------------------------------------------------------------------------

def generate_valid_value(schema: Dict[str, Any], *, name: Optional[str] = None) -> Any:
    """Return a single deterministic valid value for *schema*.

    Priority: example > default > enum[0] > composition > recursive build >
              name hint > type/format table.
    """
    if "example" in schema:
        return schema["example"]
    if "default" in schema:
        return schema["default"]
    if "enum" in schema:
        return schema["enum"][0]

    # Schema composition: oneOf / anyOf / allOf
    if "allOf" in schema:
        return generate_valid_value(_merge_allof(schema["allOf"]))
    if "oneOf" in schema:
        return generate_valid_value(schema["oneOf"][0])
    if "anyOf" in schema:
        return generate_valid_value(schema["anyOf"][0])

    schema_type = schema.get("type")
    schema_format = schema.get("format")

    # Implicit object: has properties but no explicit type
    if schema_type is None and "properties" in schema:
        schema_type = "object"

    if schema_type == "object":
        return generate_valid_object(schema)

    if schema_type == "array":
        items_schema = schema.get("items", {})
        return [generate_valid_value(items_schema)]

    # Respect numeric constraints
    if schema_type in ("integer", "number"):
        return _constrained_numeric(schema, schema_type, schema_format)

    # Respect string length constraints
    if schema_type == "string":
        # Check name hint for realistic defaults
        if name is not None:
            hint = FIELD_NAME_HINTS.get(name.lower())
            if hint is not None:
                return hint
        return _constrained_string(schema, schema_format)

    value = VALID_DEFAULTS.get((schema_type, schema_format))
    if value is not None:
        return value
    value = VALID_DEFAULTS.get((schema_type, None))
    if value is not None:
        return value

    return "unknown"


def _constrained_numeric(
    schema: Dict[str, Any], schema_type: str, schema_format: Optional[str]
) -> Any:
    """Generate a numeric value that respects minimum/maximum constraints."""
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    exclusive_min = schema.get("exclusiveMinimum", False)
    exclusive_max = schema.get("exclusiveMaximum", False)

    # Start from the lookup table default
    default = VALID_DEFAULTS.get((schema_type, schema_format))
    if default is None:
        default = VALID_DEFAULTS.get((schema_type, None), 10)

    value = default

    # Clamp to constraints
    if minimum is not None:
        low = (minimum + 1) if exclusive_min else minimum
        if value < low:
            value = low
    if maximum is not None:
        high = (maximum - 1) if exclusive_max else maximum
        if value > high:
            value = high

    return value


def _constrained_string(schema: Dict[str, Any], schema_format: Optional[str]) -> str:
    """Generate a string that respects minLength/maxLength constraints."""
    lookup = VALID_DEFAULTS.get(("string", schema_format))
    if lookup is None:
        lookup = VALID_DEFAULTS[("string", None)]

    min_len = schema.get("minLength")
    max_len = schema.get("maxLength")

    result = str(lookup)

    if min_len is not None and len(result) < min_len:
        result = result + "a" * (min_len - len(result))
    if max_len is not None and len(result) > max_len:
        result = result[:max_len]

    return result


def generate_valid_object(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Build a complete valid object from an object schema.

    Excludes ``readOnly`` fields (they are server-generated, not sent in requests).
    """
    properties = schema.get("properties", {})
    result: Dict[str, Any] = {}

    for prop_name, prop_schema in properties.items():
        if prop_schema.get("readOnly"):
            continue
        result[prop_name] = generate_valid_value(prop_schema, name=prop_name)

    if not properties and "additionalProperties" in schema:
        ap = schema["additionalProperties"]
        if isinstance(ap, dict):
            result["sample_key"] = generate_valid_value(ap)

    return result


def generate_wrong_type_value(schema: Dict[str, Any]) -> Any:
    """Return a value whose type is wrong for *schema*."""
    schema_type = schema.get("type", "string")
    return WRONG_TYPE_MAP.get(schema_type, "wrong_value")


def generate_boundary_values(schema: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Return ``[(label, value), ...]`` boundary pairs, respecting constraints."""
    schema_type = schema.get("type", "")

    if schema_type == "integer":
        return _boundary_values_for_integer(schema)
    if schema_type == "number":
        return _boundary_values_for_number(schema)
    if schema_type == "string":
        return _boundary_values_for_string(schema)

    return list(STATIC_BOUNDARY_VALUES.get(schema_type, []))


def generate_invalid_format_value(schema: Dict[str, Any]) -> Optional[str]:
    """Return a malformed value if the schema has a known format, else None."""
    fmt = schema.get("format")
    if fmt and fmt in INVALID_FORMAT_MAP:
        return INVALID_FORMAT_MAP[fmt]
    return None


# ---------------------------------------------------------------------------
# Status-code helpers for rules
# ---------------------------------------------------------------------------

ERROR_CODE_SET = {"400", "401", "403", "404", "422", "500"}


def pick_error_status(
    response_schemas: Dict[str, Any],
    candidates: List[str],
) -> Dict[str, Any]:
    """Given candidate error codes, return the right expected-result kwargs.

    If exactly one candidate is declared in the IR, return ``{status_code: N}``.
    If multiple candidates are declared, return ``{status_code_any_of: [N, ...]}``.
    If none declared, fall back to the first candidate as exact.

    Parameters
    ----------
    response_schemas : dict
        The endpoint's ``response_schemas`` from the IR.
    candidates : list[str]
        Error codes that could plausibly be returned (e.g. ``["400", "422"]``).
    """
    declared = [c for c in candidates if c in response_schemas]
    if not declared:
        # Nothing declared — use first candidate as best guess
        return {"status_code": int(candidates[0])}
    if len(declared) == 1:
        return {"status_code": int(declared[0])}
    return {"status_code_any_of": sorted(int(c) for c in declared)}
