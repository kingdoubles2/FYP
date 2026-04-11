from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Deterministic lookup tables
# ---------------------------------------------------------------------------

VALID_DEFAULTS: Dict[Tuple[str, Optional[str]], Any] = {
    ("integer", None):       1,
    ("integer", "int32"):    1,
    ("integer", "int64"):    1,
    ("number", None):        1.0,
    ("number", "float"):     1.0,
    ("number", "double"):    1.0,
    ("string", None):        "standard-text",
    ("string", "date-time"): "2026-01-15T10:30:00Z",
    ("string", "date"):      "2026-01-15",
    ("string", "email"):     "jane.doe@example.com",
    ("string", "uri"):       "https://example.com/resource",
    ("string", "url"):       "https://example.com/resource",
    ("string", "uuid"):      "123e4567-e89b-12d3-a456-426614174000",
    ("string", "byte"):      "dGVzdA==",
    ("string", "binary"):    "binary-content",
    ("string", "password"):  "SecurePass123!",
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

# Name-aware deterministic values for common fields (case-insensitive).
SEMANTIC_STRING_VALUES: Dict[str, str] = {
    "username": "jane_doe",
    "user": "jane_doe",
    "firstname": "Jane",
    "lastname": "Doe",
    "fullname": "Jane Doe",
    "displayname": "Jane Doe",
    "nickname": "jane",
    "givenname": "Jane",
    "surname": "Doe",
    "addressline1": "12 River Street",
    "addressline2": "Suite 5",
    "adminarea1": "Leinster",
    "adminarea2": "Dublin",
    "email": "jane.doe@example.com",
    "emailaddress": "buyer@example.com",
    "contactemail": "contact@example.com",
    "supportemail": "support@example.com",
    "phone": "+353871234567",
    "phonenumber": "871234567",
    "nationalnumber": "871234567",
    "password": "SecurePass123!",
    "sessionid": "sess_10001",
    "requestid": "req_10001",
    "traceid": "trace_10001",
    "correlationid": "corr_10001",
    "title": "Senior Engineer",
    "description": "Primary account profile",
    "summary": "Primary account profile",
    "notes": "Customer requested fast delivery",
    "address": "12 River Street",
    "street": "12 River Street",
    "street1": "12 River Street",
    "street2": "Suite 5",
    "city": "Dublin",
    "state": "Leinster",
    "region": "Leinster",
    "zipcode": "D02X285",
    "postalcode": "D02X285",
    "postcode": "D02X285",
    "country": "Ireland",
    "countrycode": "IE",
    "statecode": "L",
    "continent": "Europe",
    "company": "Acme Ltd",
    "organization": "Acme Ltd",
    "department": "Engineering",
    "role": "admin",
    "permission": "read_write",
    "source": "web",
    "channel": "online",
    "slug": "jane-doe",
    "filename": "report.pdf",
    "filepath": "/reports/report.pdf",
    "filetype": "pdf",
    "mimetype": "application/json",
    "contenttype": "application/json",
    "additionalmetadata": "profile image",
    "metadata": "key-value metadata",
    "timezone": "UTC",
    "currency": "USD",
    "currencycode": "USD",
    "language": "en",
    "locale": "en_US",
    "status": "active",
    "priority": "normal",
    "environment": "sandbox",
    "platform": "web",
    "version": "v1",
    "tag": "general",
    "category": "general",
    "type": "standard",
    "code": "CODE-10001",
    "message": "Operation completed successfully",
    "reason": "valid_request",
    # Commerce / payment (also common in many B2C APIs)
    "merchantid": "MERCHANT12345",
    "invoiceid": "INV-10001",
    "customid": "CUST-10001",
    "referenceid": "REF-10001",
    "payerid": "PAYER12345",
    "transactionid": "TXN-10001",
    "paymentid": "PAY-10001",
    "orderreference": "ORDER-10001",
    "sku": "SKU-001",
    "commoditycode": "53111600",
    "unitofmeasure": "PCS",
    "brandname": "Example Store",
    "softdescriptor": "EXAMPLESTORE",
    "value": "100.00",
    "quantity": "1",
    "expiry": "2028-12",
    "date": "2026-01-15",
    "birthdate": "1995-08-21",
    "ipaddress": "203.0.113.42",
    "consumerip": "203.0.113.42",
    "consumeruseragent": "Mozilla/5.0",
}

SEMANTIC_INTEGER_VALUES: Dict[str, int] = {
    "id": 1001,
    "version": 1,
    "petid": 1001,
    "userid": 1001,
    "requestid": 10001,
    "traceid": 10001,
    "correlationid": 10001,
    "accountid": 2001,
    "itemid": 3001,
    "productid": 4001,
    "categoryid": 1,
    "tagid": 10,
    "orderid": 5001,
    "quantity": 2,
    "age": 30,
    "count": 1,
    "page": 1,
    "perpage": 20,
    "limit": 20,
    "offset": 0,
    "retrycount": 0,
    "attempt": 1,
    "year": 2026,
    "month": 1,
    "day": 15,
    "statuscode": 200,
    "userstatus": 1,
}

SEMANTIC_NUMBER_VALUES: Dict[str, float] = {
    "price": 19.99,
    "amount": 19.99,
    "total": 19.99,
    "subtotal": 17.99,
    "tax": 2.0,
    "discount": 1.0,
    "shipping": 4.99,
    "fee": 0.99,
    "balance": 250.75,
    "rate": 0.05,
    "score": 4.5,
    "percentage": 10.0,
    "weight": 1.2,
    "height": 180.0,
    "width": 80.0,
    "length": 120.0,
    "latitude": 53.3498,
    "longitude": -6.2603,
}

SEMANTIC_BOOLEAN_VALUES: Dict[str, bool] = {
    "active": True,
    "enabled": True,
    "verified": True,
    "success": True,
    "deleted": False,
    "archived": False,
    "disabled": False,
    "paid": True,
    "captured": True,
    "refunded": False,
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


def _normalize_token(token: Optional[str]) -> str:
    if not token:
        return ""
    return "".join(ch for ch in token.lower() if ch.isalnum())


def _normalize_path(path: str) -> str:
    return "/".join(part for part in (_normalize_token(p) for p in path.split("/")) if part)


def _name_parts(token: Optional[str]) -> List[str]:
    if not token:
        return []
    parts = re.split(r"[^a-zA-Z0-9]+|(?<=[a-z])(?=[A-Z])", token)
    return [p.lower() for p in parts if p]


def _token_hints_from_name(norm_name: str) -> Optional[str]:
    if "email" in norm_name:
        return "jane.doe@example.com"
    if "phone" in norm_name or "mobile" in norm_name:
        return "871234567"
    if "address" in norm_name and "line1" in norm_name:
        return "12 River Street"
    if "address" in norm_name and "line2" in norm_name:
        return "Suite 5"
    if "postal" in norm_name or "postcode" in norm_name or "zipcode" in norm_name:
        return "D02X285"
    if "country" in norm_name and "code" in norm_name:
        return "IE"
    if "currency" in norm_name and "code" in norm_name:
        return "USD"
    if "url" in norm_name or "uri" in norm_name:
        return "https://example.com/resource"
    if "name" in norm_name and "given" in norm_name:
        return "Jane"
    if "name" in norm_name and ("sur" in norm_name or "family" in norm_name or "last" in norm_name):
        return "Doe"
    return None


def _pattern_based_string_value(
    *,
    name: Optional[str],
    parent_name: Optional[str],
    path: str,
) -> Optional[str]:
    name_parts = _name_parts(name)
    parent_parts = _name_parts(parent_name)
    path_parts = _name_parts(path.replace("/", "_"))
    parts = set(name_parts + parent_parts + path_parts)

    if "email" in parts:
        return "buyer@example.com"
    if "given" in parts and "name" in parts:
        return "Jane"
    if "surname" in parts or ("last" in parts and "name" in parts) or ("family" in parts and "name" in parts):
        return "Doe"
    if "full" in parts and "name" in parts:
        return "Jane Doe"
    if "phone" in parts and ("number" in parts or "national" in parts):
        return "871234567"
    if "address" in parts and ("line1" in parts or ({"line", "1"} <= parts)):
        return "12 River Street"
    if "address" in parts and ("line2" in parts or ({"line", "2"} <= parts)):
        return "Suite 5"
    if "postal" in parts or "postcode" in parts or "zipcode" in parts:
        return "D02X285"
    if "country" in parts and "code" in parts:
        return "IE"
    if "currency" in parts and "code" in parts:
        return "USD"
    if "url" in parts or "uri" in parts or "callback" in parts:
        return "https://example.com/resource"
    if "merchant" in parts and "customer" in parts and "id" in parts:
        return "MCUST-10001"
    if "customer" in parts and "id" in parts:
        return "CUST-10001"
    if "merchant" in parts and "id" in parts:
        return "MERCHANT12345"
    if "payer" in parts and "id" in parts:
        return "PAYER12345"
    if "invoice" in parts and "id" in parts:
        return "INV-10001"
    if "reference" in parts and "id" in parts:
        return "REF-10001"
    if "customer" in parts and "id" in parts:
        return "CUST-10001"

    return None


def _realistic_value_from_semantics(
    schema_type: Optional[str],
    *,
    name: Optional[str],
    parent_name: Optional[str],
    path: str,
) -> Optional[Any]:
    norm_name = _normalize_token(name)
    norm_parent = _normalize_token(parent_name)
    norm_path = _normalize_path(path)

    if schema_type == "string":
        token_hint = _token_hints_from_name(norm_name)
        if token_hint is not None:
            return token_hint

        pattern_value = _pattern_based_string_value(
            name=name,
            parent_name=parent_name,
            path=path,
        )
        if pattern_value is not None:
            return pattern_value

        if norm_name in {"id", "resourceid"}:
            if "order" in norm_path or norm_parent == "order":
                return "ORDER-10001"
            if "invoice" in norm_path or norm_parent == "invoice":
                return "INV-10001"
            if "payment" in norm_path or norm_parent == "payment":
                return "PAY-10001"
            if "user" in norm_path or norm_parent == "user":
                return "USER-10001"
            return "RES-10001"
        if norm_name.endswith("id") and norm_name in SEMANTIC_STRING_VALUES:
            return SEMANTIC_STRING_VALUES[norm_name]
        if norm_name.endswith("id"):
            return f"{norm_name.upper()}-10001"
        if norm_name in {"createdat", "updatedat", "timestamp"}:
            return "2026-01-15T10:30:00Z"
        if norm_name in {"createddate", "updateddate", "duedate"}:
            return "2026-01-15"
        if norm_name in {"number", "securitycode", "cvv", "cvc", "expiry", "date", "birthdate", "value", "quantity"}:
            encoded = _realistic_value_from_string_encoding(name=name, parent_name=parent_name, path=path)
            if encoded is not None:
                return encoded
        if norm_name == "name" and norm_parent in {"tag", "tags", "label", "labels"}:
            return "friendly"
        if norm_name == "name" and norm_parent == "category":
            return "Dogs"
        if norm_name == "name" and ("tag" in norm_path or "tags" in norm_path):
            return "friendly"
        if norm_name == "name" and ("pet" in norm_path or norm_parent == "pet"):
            return "Buddy"
        if norm_name == "name" and ("user" in norm_path or norm_parent == "user"):
            return "Jane Doe"
        if norm_name == "name":
            return "Sample Name"
        if norm_name == "photourls" or "photourl" in norm_path:
            return "https://example.com/photos/buddy.jpg"
        if "avatar" in norm_name or "image" in norm_name or "photo" in norm_name:
            return "https://example.com/images/profile.jpg"
        if "url" in norm_name or "uri" in norm_name:
            return "https://example.com/resource"
        if norm_name in SEMANTIC_STRING_VALUES:
            return SEMANTIC_STRING_VALUES[norm_name]

    if schema_type == "integer":
        if norm_name == "id":
            if "category" in norm_path or norm_parent == "category":
                return 1
            if "tag" in norm_path or norm_parent == "tag":
                return 10
            if "order" in norm_path or norm_parent == "order":
                return 5001
        if norm_name in SEMANTIC_INTEGER_VALUES:
            return SEMANTIC_INTEGER_VALUES[norm_name]
        if norm_name.endswith("id"):
            return 1001

    if schema_type == "number":
        if norm_name in SEMANTIC_NUMBER_VALUES:
            return SEMANTIC_NUMBER_VALUES[norm_name]

    if schema_type == "boolean":
        if norm_name.startswith("is") or norm_name.startswith("has") or norm_name.startswith("can"):
            return True
        if norm_name in SEMANTIC_BOOLEAN_VALUES:
            return SEMANTIC_BOOLEAN_VALUES[norm_name]
        if norm_name in {"complete", "active", "enabled", "verified", "success"}:
            return True

    return None


def _realistic_value_from_string_encoding(
    *,
    name: Optional[str],
    parent_name: Optional[str],
    path: str,
) -> Optional[str]:
    """Deterministic values for string fields that encode numeric/date concepts."""
    norm_name = _normalize_token(name)
    norm_parent = _normalize_token(parent_name)
    norm_path = _normalize_path(path)

    if norm_name == "value":
        return "100.00"
    if norm_name in {"amount", "price", "total", "subtotal", "tax", "discount"}:
        return "19.99"
    if norm_name == "quantity":
        return "1"
    if norm_name in {"securitycode", "cvv", "cvc"}:
        return "123"
    if norm_name == "expiry":
        return "2028-12"
    if norm_name == "number":
        if "card" in norm_path or norm_parent == "card":
            return "4111111111111111"
        if "phone" in norm_path or norm_parent == "phone":
            return "871234567"
        return "100001"
    if norm_name == "date":
        if "birth" in norm_path or norm_parent == "person":
            return "1995-08-21"
        return "2026-01-15"
    if norm_name in {"startdate", "enddate", "duedate"}:
        return "2026-01-15"
    if norm_name in {"startdatetime", "enddatetime", "timestamp"}:
        return "2026-01-15T10:30:00Z"
    if norm_name in {"tokenrequestorid", "merchantcustomerid", "billingagreementid", "vaultid"}:
        return "ID-10001"
    if norm_name == "birthdate":
        return "1995-08-21"
    return None


def _realistic_value_from_format(schema: Dict[str, Any], schema_type: Optional[str]) -> Optional[Any]:
    fmt = str(schema.get("format", "")).lower()
    if schema_type != "string" or not fmt:
        return None
    if fmt == "email":
        return "jane.doe@example.com"
    if fmt == "date":
        return "2026-01-15"
    if fmt == "date-time":
        return "2026-01-15T10:30:00Z"
    if fmt in {"uri", "url"}:
        return "https://example.com/resource"
    if fmt == "uuid":
        return "123e4567-e89b-12d3-a456-426614174000"
    if fmt == "byte":
        return "dGVzdA=="
    if fmt == "password":
        return "SecurePass123!"
    if fmt == "binary":
        return "binary-content"
    return None


def _apply_numeric_constraints(schema: Dict[str, Any], value: float, *, as_integer: bool) -> Any:
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    exclusive_min = schema.get("exclusiveMinimum", False)
    exclusive_max = schema.get("exclusiveMaximum", False)

    if minimum is not None:
        low = (minimum + 1) if exclusive_min else minimum
        if value < low:
            value = low
    if maximum is not None:
        high = (maximum - 1) if exclusive_max else maximum
        if value > high:
            value = high

    if as_integer:
        return int(value)
    return float(value)


def _short_semantic_value(norm_name: str, max_len: int) -> Optional[str]:
    candidates: Dict[str, List[str]] = {
        "date": ["2026-01-15", "20260115", "202601", "2026"],
        "createdat": ["2026-01-15T10:30:00Z", "20260115", "202601", "2026"],
        "updatedat": ["2026-01-15T10:30:00Z", "20260115", "202601", "2026"],
        "timestamp": ["2026-01-15T10:30:00Z", "20260115", "202601", "2026"],
        "expiry": ["2028-12", "1228", "2028"],
        "quantity": ["1"],
        "value": ["100.00", "100", "10"],
        "amount": ["19.99", "19", "10"],
        "securitycode": ["123"],
        "cvv": ["123"],
        "cvc": ["123"],
    }
    if norm_name in candidates:
        for c in candidates[norm_name]:
            if len(c) <= max_len:
                return c
    return None


def _apply_string_constraints(schema: Dict[str, Any], value: str, *, name: Optional[str] = None) -> str:
    min_len = schema.get("minLength")
    max_len = schema.get("maxLength")

    def _is_valid_length(s: str) -> bool:
        if min_len is not None and len(s) < min_len:
            return False
        if max_len is not None and len(s) > max_len:
            return False
        return True

    def _semantic_candidates(src: str) -> List[str]:
        candidates: List[str] = []

        # Date-like values: degrade into still meaningful compact forms.
        if "T" in src and "-" in src:
            candidates.extend(["2026-01-15", "20260115", "202601", "2026"])
        if src.count("-") == 2 and src[:4].isdigit():
            candidates.extend(["2026-01-15", "20260115", "202601", "2026"])
        if src.count("-") == 1 and src[:4].isdigit():
            candidates.extend(["2028-12", "1228", "2028"])

        # Numeric-like values should remain numeric under short maxLength.
        if any(ch.isdigit() for ch in src):
            candidates.extend(["4111111111111111", "123456789012", "12345678", "123456", "1234", "123"])

        # Generic deterministic short-code candidates.
        candidates.extend(["ABC123", "CODE1", "OK1", "A1"])
        return candidates

    result = str(value)

    if max_len is not None:
        short = _short_semantic_value(_normalize_token(name), max_len)
        if short is not None and _is_valid_length(short):
            result = short

    # Prefer semantic alternatives before blunt truncation.
    if max_len is not None and len(result) > max_len:
        for cand in _semantic_candidates(result):
            if _is_valid_length(cand):
                result = cand
                break

    if min_len is not None and len(result) < min_len:
        result = result + ("x" * (min_len - len(result)))
    if max_len is not None and len(result) > max_len:
        result = result[:max_len]
    return result


def _default_value_from_type_with_constraints(
    schema: Dict[str, Any],
    schema_type: Optional[str],
) -> Any:
    if schema_type == "string":
        return _apply_string_constraints(schema, "standard-text")
    if schema_type == "integer":
        return _apply_numeric_constraints(schema, 1, as_integer=True)
    if schema_type == "number":
        return _apply_numeric_constraints(schema, 1.0, as_integer=False)
    if schema_type == "boolean":
        return True
    return _apply_string_constraints(schema, "standard-text")


# ---------------------------------------------------------------------------
# Value generators
# ---------------------------------------------------------------------------

def generate_valid_value(
    schema: Dict[str, Any],
    *,
    name: Optional[str] = None,
    parent_name: Optional[str] = None,
    path: str = "",
    skip_example: bool = False,
    use_realistic: bool = True,
) -> Any:
    """Return a single deterministic valid value for *schema*.

    Priority: example > enum[0] > semantics(name/parent/path) > format >
              type default > recursive structure.

    When *skip_example* is True the ``example`` key in the schema is ignored,
    producing a safe fallback value suitable for the minimal happy-path test.
    
    When *use_realistic* is True, prefers field-name-based hints over generic
    placeholder values, producing more realistic test data for the first happy path.
    """
    if not skip_example and "example" in schema:
        return schema["example"]
    if "enum" in schema:
        return schema["enum"][0]
    # Schema composition: oneOf / anyOf / allOf
    if "allOf" in schema:
        return generate_valid_value(
            _merge_allof(schema["allOf"]),
            name=name,
            parent_name=parent_name,
            path=path,
            skip_example=skip_example,
            use_realistic=use_realistic,
        )
    if "oneOf" in schema:
        return generate_valid_value(
            schema["oneOf"][0],
            name=name,
            parent_name=parent_name,
            path=path,
            skip_example=skip_example,
            use_realistic=use_realistic,
        )
    if "anyOf" in schema:
        return generate_valid_value(
            schema["anyOf"][0],
            name=name,
            parent_name=parent_name,
            path=path,
            skip_example=skip_example,
            use_realistic=use_realistic,
        )

    schema_type = schema.get("type")
    schema_format = schema.get("format")

    # Implicit object: has properties but no explicit type
    if schema_type is None and "properties" in schema:
        schema_type = "object"

    if schema_type == "object":
        return generate_valid_object(
            schema,
            name=name,
            parent_name=parent_name,
            path=path,
            skip_example=skip_example,
            use_realistic=use_realistic,
        )

    if schema_type == "array":
        items_schema = schema.get("items", {})
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        count = 1
        if isinstance(min_items, int):
            count = max(1, min_items)
        if isinstance(max_items, int):
            count = min(count, max_items)
        item_path = f"{path}[]" if path else "[]"
        return [
            generate_valid_value(
                items_schema,
                name=name,
                parent_name=parent_name,
                path=item_path,
                skip_example=skip_example,
                use_realistic=use_realistic,
            )
            for _ in range(max(0, count))
        ]

    if use_realistic:
        semantic = _realistic_value_from_semantics(
            schema_type,
            name=name,
            parent_name=parent_name,
            path=path,
        )
        if semantic is not None:
            if schema_type == "string":
                return _apply_string_constraints(schema, str(semantic), name=name)
            if schema_type == "integer":
                return _apply_numeric_constraints(schema, float(semantic), as_integer=True)
            if schema_type == "number":
                return _apply_numeric_constraints(schema, float(semantic), as_integer=False)
            return semantic

        if schema_type == "string":
            encoded = _realistic_value_from_string_encoding(
                name=name,
                parent_name=parent_name,
                path=path,
            )
            if encoded is not None:
                return _apply_string_constraints(schema, encoded, name=name)

        formatted = _realistic_value_from_format(schema, schema_type)
        if formatted is not None:
            return _apply_string_constraints(schema, str(formatted), name=name)

    if schema_type == "string" and schema_format in {"date-time", "date", "email", "uri", "url", "uuid", "byte", "binary", "password"}:
        return _apply_string_constraints(schema, str(VALID_DEFAULTS[("string", schema_format)]), name=name)

    return _default_value_from_type_with_constraints(schema, schema_type)


def generate_valid_object(
    schema: Dict[str, Any],
    *,
    name: Optional[str] = None,
    parent_name: Optional[str] = None,
    path: str = "",
    skip_example: bool = False,
    use_realistic: bool = True,
) -> Dict[str, Any]:
    """Build a complete valid object from an object schema.

    Excludes ``readOnly`` fields (they are server-generated, not sent in requests).
    """
    properties = schema.get("properties", {})
    result: Dict[str, Any] = {}

    for prop_name, prop_schema in properties.items():
        if prop_schema.get("readOnly"):
            continue
        child_path = f"{path}/{prop_name}" if path else prop_name
        result[prop_name] = generate_valid_value(
            prop_schema,
            name=prop_name,
            parent_name=name,
            path=child_path,
            skip_example=skip_example,
            use_realistic=use_realistic,
        )

    if not properties and "additionalProperties" in schema:
        ap = schema["additionalProperties"]
        if isinstance(ap, dict):
            child_path = f"{path}/metadata_key" if path else "metadata_key"
            result["metadata_key"] = generate_valid_value(
                ap,
                name="metadata_key",
                parent_name=name,
                path=child_path,
                skip_example=skip_example,
                use_realistic=use_realistic,
            )

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
