from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import yaml

# Allow running as a script: python src/llm_eval/run_llm_eval.py
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_eval.ollama_client import OllamaClient
from spec_parser.parser import parse_openapi
from test_generator.generator import generate_test_cases
from test_runner.run_test import resolve_base_url, run_suite


SENSITIVE_KEYS = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "apikey",
    "token",
}

VALID_CASE_SCOPES = {"fail", "fail_and_noteworthy", "all"}
DEFAULT_CASE_SCOPE = "fail"

META_BANNED_TERMS = [
    "spec_path",
    "ir_path",
    "tests_path",
    "results_path",
    "metadata",
    "artifact",
    "json object",
    "framework",
    "evidence_index",
]

ACTION_WORDS_RE = re.compile(
    r"\b(check|verify|confirm|inspect|review|re-run|compare|validate|ensure|trace)\b",
    flags=re.IGNORECASE,
)


def slugify(text: str) -> str:
    safe = []
    for ch in str(text):
        if ch.isalnum() or ch in (".", "_", "-"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "item"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    write_text(path, json.dumps(data, indent=2, ensure_ascii=True))


def load_config(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping.")
    return raw


def load_spec_document(spec_text: str) -> Dict[str, Any]:
    try:
        doc = json.loads(spec_text)
    except json.JSONDecodeError:
        doc = yaml.safe_load(spec_text)
    if not isinstance(doc, dict):
        raise ValueError("Spec content must parse into a mapping object.")
    return doc


def split_csv_args(values: Optional[List[str]]) -> Optional[List[str]]:
    if not values:
        return None
    items: List[str] = []
    for value in values:
        parts = [p.strip() for p in value.split(",") if p.strip()]
        items.extend(parts)
    return items or None


def coerce_list(value: Any, label: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    raise ValueError(f"{label} must be a list.")


def redact_sensitive(obj: Any) -> Any:
    if isinstance(obj, dict):
        redacted: Dict[str, Any] = {}
        for key, value in obj.items():
            key_str = str(key)
            if key_str.lower() in SENSITIVE_KEYS:
                redacted[key_str] = "<redacted>"
            else:
                redacted[key_str] = redact_sensitive(value)
        return redacted
    if isinstance(obj, list):
        return [redact_sensitive(item) for item in obj]
    return obj


def sanitize_auth_config(auth_cfg: Any) -> Any:
    if not isinstance(auth_cfg, dict):
        return auth_cfg
    cleaned: Dict[str, Any] = {}
    for key, value in auth_cfg.items():
        if isinstance(value, dict):
            cleaned_entry = dict(value)
            raw_value = cleaned_entry.get("value")
            if isinstance(raw_value, str) and not raw_value.startswith("env:"):
                cleaned_entry["value"] = "<redacted>"
            cleaned[key] = cleaned_entry
        elif isinstance(value, str):
            cleaned[key] = value if value.startswith("env:") else "<redacted>"
        else:
            cleaned[key] = value
    return cleaned


def sanitize_config_for_meta(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(cfg)
    if "auth_by_base_url" in cleaned:
        cleaned["auth_by_base_url"] = sanitize_auth_config(cleaned["auth_by_base_url"])
    return cleaned


def normalize_case_scope(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in VALID_CASE_SCOPES:
        return raw
    if raw == "every_fail":
        return "fail"
    if raw == "suite_summary":
        return "fail"
    return DEFAULT_CASE_SCOPE


def count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", str(text)))


def truncate_chars(text: Any, limit: int) -> str:
    raw = str(text or "")
    if limit <= 0:
        return raw
    if len(raw) <= limit:
        return raw
    return raw[:limit].rstrip() + "..."


def truncate_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return text.strip()
    words = re.findall(r"\S+", text.strip())
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip() + "..."


def contains_meta_terms(text: str) -> int:
    lower = (text or "").lower()
    return sum(1 for term in META_BANNED_TERMS if term in lower)


def resolve_auth_headers(base_url: str, auth_cfg: Any) -> Tuple[Dict[str, str], Dict[str, Any]]:
    meta = {
        "mode": "none",
        "provided": False,
        "source": "none",
        "header": None,
    }
    if not isinstance(auth_cfg, dict):
        return {}, meta

    parsed = urlparse(base_url)
    host = parsed.hostname or base_url
    entry = None
    if host in auth_cfg:
        entry = auth_cfg[host]
    elif base_url in auth_cfg:
        entry = auth_cfg[base_url]
    if not entry:
        return {}, meta

    def resolve_value(raw: Any) -> Optional[str]:
        if raw is None:
            return None
        if isinstance(raw, str) and raw.startswith("env:"):
            return os.environ.get(raw[4:])
        if isinstance(raw, str) and raw.startswith("bearer:"):
            return raw[len("bearer:"):]
        if isinstance(raw, str):
            return raw
        return None

    if isinstance(entry, dict):
        auth_type = str(entry.get("type", "bearer")).lower()
        value = resolve_value(entry.get("value"))
        if not value:
            return {}, meta
        if auth_type == "api_key":
            header = str(entry.get("header") or "X-API-Key")
            meta.update({"mode": "api_key", "provided": True, "source": "config", "header": header})
            return {header: value}, meta
        meta.update({"mode": "bearer", "provided": True, "source": "config", "header": "Authorization"})
        return {"Authorization": f"Bearer {value}"}, meta

    if isinstance(entry, str):
        value = resolve_value(entry)
        if not value:
            return {}, meta
        meta.update({"mode": "bearer", "provided": True, "source": "config", "header": "Authorization"})
        return {"Authorization": f"Bearer {value}"}, meta

    return {}, meta


def parse_pointer_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def resolve_local_ref(ref: str, spec_doc: Dict[str, Any]) -> Optional[Any]:
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    node: Any = spec_doc
    for raw_token in ref[2:].split("/"):
        token = parse_pointer_token(raw_token)
        if not isinstance(node, dict) or token not in node:
            return None
        node = node[token]
    return node


def deref_dict(value: Any, spec_doc: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    if "$ref" not in value:
        return value
    target = resolve_local_ref(str(value.get("$ref")), spec_doc)
    if not isinstance(target, dict):
        return value
    merged = dict(target)
    for key, val in value.items():
        if key != "$ref":
            merged[key] = val
    return merged


def get_operation(spec_doc: Dict[str, Any], method: str, path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    paths = spec_doc.get("paths")
    if not isinstance(paths, dict):
        return {}, {}
    path_item = deref_dict(paths.get(path), spec_doc)
    if not path_item:
        return {}, {}
    operation = deref_dict(path_item.get(str(method).lower()), spec_doc)
    if not operation:
        return path_item, {}
    return path_item, operation


def parse_security_requirements(operation: Dict[str, Any], spec_doc: Dict[str, Any]) -> List[str]:
    raw_security = operation.get("security")
    if raw_security is None:
        raw_security = spec_doc.get("security")
    if not isinstance(raw_security, list):
        return []
    names: List[str] = []
    seen = set()
    for req in raw_security:
        if not isinstance(req, dict):
            continue
        for name in req.keys():
            key = str(name)
            if key not in seen:
                seen.add(key)
                names.append(key)
    return names


def compact_list(items: Any, max_items: int) -> List[Any]:
    if not isinstance(items, list):
        return []
    if max_items <= 0 or len(items) <= max_items:
        return items
    return items[:max_items]


def compact_schema(schema: Any, max_items: int = 8, max_depth: int = 2) -> Dict[str, Any]:
    if not isinstance(schema, dict):
        return {}
    keys = [
        "type",
        "format",
        "enum",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "nullable",
        "default",
        "example",
    ]
    out: Dict[str, Any] = {}
    for key in keys:
        if key not in schema:
            continue
        value = schema[key]
        if isinstance(value, list):
            out[key] = compact_list(value, max_items)
            if len(value) > len(out[key]):
                out[f"{key}_truncated"] = True
                out[f"{key}_total"] = len(value)
        else:
            out[key] = value

    required = schema.get("required")
    if isinstance(required, list):
        out["required"] = compact_list(required, max_items)
        if len(required) > len(out["required"]):
            out["required_total"] = len(required)

    if max_depth > 0 and isinstance(schema.get("items"), dict):
        out["items"] = compact_schema(schema["items"], max_items=max_items, max_depth=max_depth - 1)

    if max_depth > 0 and isinstance(schema.get("properties"), dict):
        properties = schema["properties"]
        prop_out: Dict[str, Any] = {}
        for index, (name, prop_schema) in enumerate(properties.items()):
            if max_items > 0 and index >= max_items:
                break
            prop_out[str(name)] = compact_schema(prop_schema, max_items=max_items, max_depth=max_depth - 1)
        out["properties"] = prop_out
        if max_items > 0 and len(properties) > max_items:
            out["properties_total"] = len(properties)
            out["properties_truncated"] = True

    return out


def determine_intent(category: Optional[str]) -> str:
    cat = str(category or "").lower()
    if "auth" in cat:
        return "auth"
    if "negative" in cat or "error" in cat:
        return "negative"
    if "boundary" in cat:
        return "boundary"
    if "fail" in cat:
        return "fail-path"
    return "positive"


def infer_generator_rule(category: Optional[str], title: Optional[str]) -> str:
    cat = str(category or "").lower()
    ttl = str(title or "").lower()
    if cat == "auth":
        return "auth.generate_auth_cases"
    if cat.startswith("negative_type"):
        return "negative._wrong_type_param_cases"
    if cat.startswith("negative_missing"):
        return "negative._missing_required_cases"
    if cat.startswith("negative_invalid"):
        return "negative._invalid_value_cases"
    if cat.startswith("negative"):
        return "negative.generate_negative_cases"
    if cat.startswith("boundary"):
        return "boundary.generate_boundary_cases"
    if cat.startswith("error_status"):
        return "error_status.generate_error_status_cases"
    if cat.startswith("happy"):
        return "happy_path.generate_happy_path_cases"
    if "unsupported method" in ttl:
        return "negative.generate_negative_cases"
    return "unknown"


def summarize_request_constraints(
    endpoint_ir: Dict[str, Any],
    *,
    body_required: bool,
    max_items: int,
) -> Dict[str, Any]:
    path_params = endpoint_ir.get("path_params") or []
    query_params = endpoint_ir.get("query_params") or []

    required_path = [str(p.get("name")) for p in path_params if p.get("required")]
    required_query = [str(p.get("name")) for p in query_params if p.get("required")]

    query_rules: Dict[str, Any] = {}
    for index, param in enumerate(query_params):
        if max_items > 0 and index >= max_items:
            break
        name = str(param.get("name") or f"query_{index}")
        schema = compact_schema(param.get("schema") or {}, max_items=max_items, max_depth=1)
        if param.get("required"):
            schema["required"] = True
        query_rules[name] = schema

    request_schema = endpoint_ir.get("request_schema") or {}
    body_required_fields: List[str] = []
    if isinstance(request_schema, dict):
        body_required_fields = [str(v) for v in compact_list(request_schema.get("required"), max_items)]

    return {
        "required_path_params": compact_list(required_path, max_items),
        "required_query_params": compact_list(required_query, max_items),
        "query_param_rules": query_rules,
        "body_required": bool(body_required),
        "body_required_fields": body_required_fields,
        "body_schema_rules": compact_schema(request_schema, max_items=max_items, max_depth=1),
    }


def summarize_response_contract(endpoint_ir: Dict[str, Any], expected_result: Dict[str, Any]) -> Dict[str, Any]:
    statuses: List[int] = []
    response_schemas = endpoint_ir.get("response_schemas") or {}
    if isinstance(response_schemas, dict):
        for code in response_schemas.keys():
            try:
                statuses.append(int(str(code)))
            except Exception:
                continue
    statuses = sorted(set(statuses))

    primary_expected: List[int] = []
    expected_any = expected_result.get("status_code_any_of")
    expected_single = expected_result.get("status_code")
    if isinstance(expected_any, list):
        primary_expected = sorted({int(v) for v in expected_any if str(v).isdigit()})
    elif expected_single is not None and str(expected_single).isdigit():
        primary_expected = [int(expected_single)]

    return {
        "declared_statuses": statuses,
        "primary_expected_for_test": primary_expected,
    }


def build_assertion_failures(tc: Dict[str, Any], res: Dict[str, Any]) -> List[Dict[str, Any]]:
    expected_result = tc.get("expected_result") or {}
    expected_any = expected_result.get("status_code_any_of")
    expected_single = expected_result.get("status_code")
    actual_status = res.get("actual_status")
    errors: List[Dict[str, Any]] = []

    if res.get("error_message"):
        errors.append(
            {
                "type": "status_code_or_request_error",
                "expected": expected_any if expected_any is not None else expected_single,
                "actual_status": actual_status,
                "actual_error": str(res.get("error_message") or ""),
            }
        )

    if actual_status is not None:
        if isinstance(expected_any, list):
            if actual_status not in expected_any:
                errors.append(
                    {
                        "type": "status_mismatch",
                        "expected": expected_any,
                        "actual_status": actual_status,
                        "actual_error": "",
                    }
                )
        elif expected_single is not None and actual_status != expected_single:
            errors.append(
                {
                    "type": "status_mismatch",
                    "expected": expected_single,
                    "actual_status": actual_status,
                    "actual_error": "",
                }
            )

    if not errors and str(res.get("outcome", "")).upper() == "FAIL":
        errors.append(
            {
                "type": "unknown_failure",
                "expected": expected_any if expected_any is not None else expected_single,
                "actual_status": actual_status,
                "actual_error": str(res.get("error_message") or ""),
            }
        )

    return errors


def summarize_endpoint_results(results: List[Dict[str, Any]], method: str, path: str) -> Dict[str, int]:
    summary = {"pass": 0, "fail": 0, "skip": 0}
    for result in results:
        if result.get("method") != method or result.get("path") != path:
            continue
        outcome = str(result.get("outcome") or "").upper()
        if outcome == "PASS":
            summary["pass"] += 1
        elif outcome == "FAIL":
            summary["fail"] += 1
        elif outcome == "SKIP":
            summary["skip"] += 1
    return summary


def find_endpoint_ir(ir_data: Dict[str, Any], method: str, path: str) -> Dict[str, Any]:
    for endpoint in ir_data.get("endpoints", []):
        if endpoint.get("method") == method and endpoint.get("path") == path:
            return endpoint
    return {}


def classify_failure_signal(evidence: Dict[str, Any]) -> Dict[str, Any]:
    execution = evidence.get("execution") or {}
    test_context = evidence.get("test_context") or {}
    error = str(execution.get("assertion_failures", [{}])[0].get("actual_error", "")).lower()
    category = str(test_context.get("category") or "").lower()
    expected = test_context.get("expected_outcome") or {}
    expected_status = expected.get("status_code")

    if any(token in error for token in ("timeout", "ssl", "connection", "dns", "eof", "max retries")):
        return {"signal": "transport", "keywords": ["transport", "network", "tls", "ssl", "connection", "timeout"]}
    if any(token in error for token in ("401", "403", "unauthorized", "invalid token", "bearer", "api key")):
        return {"signal": "auth", "keywords": ["auth", "token", "bearer", "header", "credential"]}
    if "negative_type" in category:
        return {"signal": "schema_type", "keywords": ["type", "schema", "invalid", "validation", "field"]}
    if "negative_invalid" in category:
        return {"signal": "schema_value", "keywords": ["enum", "format", "invalid", "schema", "parameter"]}
    if "negative_missing" in category:
        return {"signal": "missing_required", "keywords": ["required", "missing", "parameter", "body", "field"]}
    if expected_status in (400, 422):
        return {"signal": "validation", "keywords": ["validation", "schema", "invalid", "request"]}
    return {"signal": "status_mismatch", "keywords": ["status", "endpoint", "contract", "behavior"]}


def infer_expected_negative_alignment(evidence: Dict[str, Any]) -> bool:
    test_context = evidence.get("test_context") or {}
    execution = evidence.get("execution") or {}
    intent = str(test_context.get("intent") or "")
    if intent not in {"negative", "auth", "fail-path"}:
        return False
    actual_status = execution.get("response_received", {}).get("status")
    if isinstance(actual_status, int) and 400 <= actual_status <= 499:
        return True
    return False


def deterministic_fallback_contract(evidence: Dict[str, Any]) -> Dict[str, Any]:
    signal = classify_failure_signal(evidence)
    execution = evidence.get("execution") or {}
    assertion_failures = execution.get("assertion_failures") or []
    first_failure = assertion_failures[0] if assertion_failures else {}
    actual_error = str(first_failure.get("actual_error") or "")
    actual_status = first_failure.get("actual_status")
    expected = first_failure.get("expected")

    if signal["signal"] == "transport":
        cause = "The failure was likely caused by a transport-level error before the API could return a contract response."
        why = f"Runner captured request error '{truncate_chars(actual_error, 120)}' with no usable response status."
        check_next = "Re-run this case and verify TLS/network reachability from the runner environment."
    elif signal["signal"] == "auth":
        cause = "The request likely failed due to missing or invalid authentication material for this operation."
        why = f"Failure details indicate auth-related rejection ({truncate_chars(actual_error, 90)})."
        check_next = "Verify the generated headers and configured bearer/API key values for this target base URL."
    elif signal["signal"] in {"schema_type", "schema_value", "missing_required", "validation"}:
        cause = "The generated input likely violated request validation rules for this operation."
        why = "The case category and expected validation outcome indicate schema/parameter constraints were intentionally exercised."
        check_next = "Compare generated parameter/body values against the endpoint constraints in the evidence payload."
    else:
        cause = "The observed status code did not match the expected contract outcome for this case."
        why = f"Expected {expected} but observed {actual_status}, indicating behavior mismatch for this specific test intent."
        check_next = "Inspect endpoint-specific server behavior and whether this negative-case expectation is still correct."

    return {
        "cause": cause,
        "why_likely": why,
        "check_next": check_next,
        "confidence": "likely",
        "expected_negative_behavior": infer_expected_negative_alignment(evidence),
        "evidence_quotes": [],
    }


def build_case_evidence_v2(
    *,
    selection_reason: str,
    spec_id: str,
    spec_path: Path,
    spec_hash: str,
    spec_doc: Dict[str, Any],
    ir_data: Dict[str, Any],
    tc: Dict[str, Any],
    res: Dict[str, Any],
    results: List[Dict[str, Any]],
    auth_meta: Dict[str, Any],
    timeout_seconds: int,
    max_items: int,
    snippet_chars: int,
    max_similar_failures: int,
) -> Dict[str, Any]:
    method = str(tc.get("method") or "")
    path = str(tc.get("path") or "")
    test_id = str(tc.get("test_id") or "")
    category = str(tc.get("category") or "")
    intent = determine_intent(category)
    steps = tc.get("steps") or []
    first_step = steps[0] if steps else {}
    input_data = first_step.get("input_data") or {}
    expected_result = tc.get("expected_result") or {}

    endpoint_ir = find_endpoint_ir(ir_data, method, path)
    path_item, operation = get_operation(spec_doc, method, path)
    request_body = deref_dict(operation.get("requestBody"), spec_doc)
    body_required = bool(request_body.get("required", False))

    operation_summary = str(operation.get("summary") or "")
    operation_description = str(operation.get("description") or "")
    security_requirements = parse_security_requirements(operation, spec_doc)

    request_constraints = summarize_request_constraints(
        endpoint_ir,
        body_required=body_required,
        max_items=max_items,
    )
    response_contract = summarize_response_contract(endpoint_ir, expected_result)

    assertion_failures = build_assertion_failures(tc, res)

    response_snippet = truncate_chars(
        res.get("response_snippet") or res.get("response_body") or "",
        snippet_chars,
    )
    request_body_value = res.get("request_body")
    request_body_snippet = truncate_chars(request_body_value, snippet_chars) if request_body_value is not None else ""

    same_endpoint_results = summarize_endpoint_results(results, method, path)
    similar_failures: List[str] = []
    for item in results:
        if item.get("method") != method or item.get("path") != path:
            continue
        if item.get("test_id") == test_id:
            continue
        if str(item.get("outcome") or "").upper() == "FAIL":
            similar_failures.append(str(item.get("test_id") or ""))
        if max_similar_failures > 0 and len(similar_failures) >= max_similar_failures:
            break

    missing_evidence = ["server_logs"]
    if res.get("error_message"):
        missing_evidence.append("upstream_service_logs")

    endpoint_ir_compact = {
        "endpoint_id": endpoint_ir.get("endpoint_id"),
        "operation_id": endpoint_ir.get("operation_id"),
        "method": endpoint_ir.get("method"),
        "path": endpoint_ir.get("path"),
        "path_params": compact_list(endpoint_ir.get("path_params") or [], max_items),
        "query_params": compact_list(endpoint_ir.get("query_params") or [], max_items),
        "header_params": compact_list(endpoint_ir.get("header_params") or [], max_items),
        "request_schema": compact_schema(endpoint_ir.get("request_schema") or {}, max_items=max_items, max_depth=1),
        "response_schemas": {
            str(code): compact_schema(schema or {}, max_items=max_items, max_depth=1)
            for code, schema in (endpoint_ir.get("response_schemas") or {}).items()
        },
    }

    evidence = {
        "schema_version": "llm_case_evidence_v2",
        "case_id": f"{spec_id}::{test_id}",
        "selection_reason": selection_reason,
        "spec": {
            "spec_id": spec_id,
            "source_file": str(spec_path),
            "spec_hash": spec_hash,
            "operation": {
                "method": method,
                "path": path,
                "operation_id": endpoint_ir.get("operation_id"),
                "summary": operation_summary,
                "description": operation_description,
            },
            "security_requirements": security_requirements,
            "request_constraints": request_constraints,
            "response_contract": response_contract,
        },
        "ir_context": {
            "endpoint_ir": endpoint_ir_compact,
            "parser_warnings": [],
            "unsupported_spec_warnings": [],
        },
        "test_context": {
            "test_id": test_id,
            "title": tc.get("title"),
            "category": category,
            "intent": intent,
            "generator_rule": infer_generator_rule(category, tc.get("title")),
            "generator_notes": [],
            "expected_outcome": expected_result,
            "generated_input": redact_sensitive(
                {
                    "path_params": input_data.get("path_params") or {},
                    "query_params": input_data.get("query_params") or {},
                    "headers": input_data.get("headers") or {},
                    "body": input_data.get("body"),
                }
            ),
        },
        "execution": {
            "outcome": str(res.get("outcome") or "").upper(),
            "assertion_failures": assertion_failures,
            "request_sent": redact_sensitive(
                {
                    "final_url": res.get("final_url"),
                    "path_params": res.get("request_path_params") or input_data.get("path_params") or {},
                    "query_params": res.get("request_query_params") or input_data.get("query_params") or {},
                    "headers": res.get("request_headers") or input_data.get("headers") or {},
                    "body": request_body_snippet if request_body_value is not None else input_data.get("body"),
                }
            ),
            "response_received": {
                "status": res.get("actual_status"),
                "headers": redact_sensitive(res.get("response_headers") or {}),
                "body_snippet": response_snippet,
            },
            "runner_logs": [],
            "duration_ms": res.get("duration_ms"),
        },
        "related_context": {
            "same_endpoint_results_summary": same_endpoint_results,
            "similar_failures": compact_list(similar_failures, max_similar_failures),
        },
        "runtime_context": {
            "auth": auth_meta,
            "runner_timeout_seconds": timeout_seconds,
            "path_item_has_servers_override": bool(path_item.get("servers")),
        },
        "missing_evidence": missing_evidence,
    }
    return evidence


def select_results_with_reasons(results: List[Dict[str, Any]], case_scope: str) -> List[Tuple[Dict[str, Any], str]]:
    selected: List[Tuple[Dict[str, Any], str]] = []
    scope = normalize_case_scope(case_scope)

    if scope == "all":
        return [(result, "all") for result in results]

    if scope == "fail":
        return [(result, "fail") for result in results if str(result.get("outcome") or "").upper() == "FAIL"]

    # fail_and_noteworthy
    seen = set()
    for result in results:
        outcome = str(result.get("outcome") or "").upper()
        test_id = str(result.get("test_id") or "")
        if outcome == "FAIL":
            selected.append((result, "fail"))
            seen.add(test_id)

    for result in results:
        test_id = str(result.get("test_id") or "")
        if test_id in seen:
            continue
        outcome = str(result.get("outcome") or "").upper()
        status = result.get("actual_status")
        noteworthy = False
        reason = "noteworthy"
        if outcome == "SKIP":
            noteworthy = True
            reason = "skip"
        elif result.get("error_message"):
            noteworthy = True
            reason = "request_error"
        elif isinstance(status, int) and status >= 500:
            noteworthy = True
            reason = "server_5xx"
        if noteworthy:
            selected.append((result, reason))
            seen.add(test_id)
    return selected


def unique_spec_id(stem: str, seen: Dict[str, int]) -> str:
    base = slugify(stem)
    count = seen.get(base, 0)
    seen[base] = count + 1
    if count == 0:
        return base
    return f"{base}_{count + 1:02d}"


def run_pipeline_once(
    *,
    specs: List[str],
    config: Dict[str, Any],
    run_dir: Path,
    case_scope: str,
    max_cases_per_spec: int,
    max_items_per_section: int,
    snippet_chars: int,
    max_similar_failures: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pipeline_root = run_dir / "pipeline"
    evidence_root = run_dir / "evidence"
    pipeline_root.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=True)

    auth_cfg = config.get("auth_by_base_url") or {}
    runner_cfg = config.get("runner") or {}
    timeout_seconds = int(runner_cfg.get("timeout_seconds", 12))

    spec_records: List[Dict[str, Any]] = []
    case_records: List[Dict[str, Any]] = []
    seen_spec_ids: Dict[str, int] = {}

    print(f"\n{'=' * 72}")
    print("PIPELINE: parser -> generator -> runner")
    print(f"{'=' * 72}")

    for raw_spec in specs:
        spec_path = Path(raw_spec)
        spec_id = unique_spec_id(spec_path.stem, seen_spec_ids)
        spec_dir = pipeline_root / spec_id
        spec_dir.mkdir(parents=True, exist_ok=True)

        spec_record: Dict[str, Any] = {
            "spec_id": spec_id,
            "spec_path": str(spec_path),
            "spec_hash": None,
            "parse_ok": False,
            "generate_ok": False,
            "run_ok": False,
            "errors": [],
            "summary": None,
            "artifact_files": {},
            "selected_case_ids": [],
        }

        print(f"\nSpec: {spec_path}")
        try:
            spec_text = read_text(spec_path)
            spec_record["spec_hash"] = sha256_text(spec_text)
            spec_copy_ext = spec_path.suffix if spec_path.suffix else ".txt"
            spec_copy = spec_dir / f"spec_raw{spec_copy_ext}"
            write_text(spec_copy, spec_text)
            spec_record["artifact_files"]["spec_raw"] = str(spec_copy)
        except Exception as exc:  # noqa: BLE001
            msg = f"read_error: {exc}"
            print(f"  ERROR: {msg}")
            spec_record["errors"].append(msg)
            spec_records.append(spec_record)
            continue

        try:
            spec_doc = load_spec_document(spec_text)
            parsed = parse_openapi(spec_text)
            spec_record["parse_ok"] = True
            print(f"  Parsed OK: {parsed.title} ({len(parsed.endpoints)} endpoints)")
        except Exception as exc:  # noqa: BLE001
            msg = f"parse_error: {exc}"
            print(f"  ERROR: {msg}")
            spec_record["errors"].append(msg)
            spec_records.append(spec_record)
            continue

        ir_data = parsed.to_dict()
        ir_path = spec_dir / "ir.json"
        write_json(ir_path, ir_data)
        spec_record["artifact_files"]["ir"] = str(ir_path)

        try:
            suite = generate_test_cases(ir_data)
            suite_data = suite.to_dict()
            spec_record["generate_ok"] = True
            print(f"  Generated tests: {len(suite_data.get('test_cases', []))}")
        except Exception as exc:  # noqa: BLE001
            msg = f"generate_error: {exc}"
            print(f"  ERROR: {msg}")
            spec_record["errors"].append(msg)
            spec_records.append(spec_record)
            continue

        tests_path = spec_dir / "tests.json"
        write_json(tests_path, suite_data)
        spec_record["artifact_files"]["tests"] = str(tests_path)

        base_url = resolve_base_url(None, suite_data.get("base_url"))
        if not base_url:
            msg = "run_error: missing base_url in generated suite"
            print(f"  ERROR: {msg}")
            spec_record["errors"].append(msg)
            spec_record["summary"] = {
                "total": len(suite_data.get("test_cases", [])),
                "passed": 0,
                "failed": 0,
                "skipped": len(suite_data.get("test_cases", [])),
            }
            spec_records.append(spec_record)
            continue

        auth_headers, auth_meta = resolve_auth_headers(base_url, auth_cfg)
        print(f"  Running tests against: {base_url}")

        try:
            results, summary = run_suite(
                suite_data=suite_data,
                base_url=base_url,
                auth_headers=auth_headers,
                timeout=timeout_seconds,
            )
            spec_record["run_ok"] = True
            spec_record["summary"] = summary
            print(
                f"  Summary: {summary.get('passed', 0)} passed, "
                f"{summary.get('failed', 0)} failed, "
                f"{summary.get('skipped', 0)} skipped / {summary.get('total', 0)} total"
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"run_error: {exc}"
            print(f"  ERROR: {msg}")
            spec_record["errors"].append(msg)
            spec_records.append(spec_record)
            continue

        results_path = spec_dir / "results.json"
        results_payload = {
            "api_title": suite_data.get("api_title"),
            "api_version": suite_data.get("api_version"),
            "base_url": base_url,
            "summary": summary,
            "results": results,
        }
        write_json(results_path, results_payload)
        spec_record["artifact_files"]["results"] = str(results_path)

        selected = select_results_with_reasons(results, case_scope)
        if max_cases_per_spec > 0 and len(selected) > max_cases_per_spec:
            print(f"  Evidence: limiting cases to {max_cases_per_spec} of {len(selected)} selected")
            selected = selected[:max_cases_per_spec]

        by_test_id = {str(tc.get("test_id")): tc for tc in suite_data.get("test_cases", [])}
        evidence_spec_dir = evidence_root / spec_id
        evidence_spec_dir.mkdir(parents=True, exist_ok=True)

        for result, reason in selected:
            test_id = str(result.get("test_id") or "")
            test_case = by_test_id.get(test_id)
            if not test_case:
                continue

            evidence = build_case_evidence_v2(
                selection_reason=reason,
                spec_id=spec_id,
                spec_path=spec_path,
                spec_hash=str(spec_record.get("spec_hash") or ""),
                spec_doc=spec_doc,
                ir_data=ir_data,
                tc=test_case,
                res=result,
                results=results,
                auth_meta=auth_meta,
                timeout_seconds=timeout_seconds,
                max_items=max_items_per_section,
                snippet_chars=snippet_chars,
                max_similar_failures=max_similar_failures,
            )

            case_slug = slugify(test_id)
            evidence_path = evidence_spec_dir / f"{case_slug}.evidence.json"
            write_json(evidence_path, evidence)

            case_records.append(
                {
                    "case_id": evidence.get("case_id"),
                    "spec_id": spec_id,
                    "spec_path": str(spec_path),
                    "test_id": test_id,
                    "title": test_case.get("title"),
                    "outcome": str(result.get("outcome") or "").upper(),
                    "category": test_case.get("category"),
                    "intent": determine_intent(test_case.get("category")),
                    "selection_reason": reason,
                    "evidence_file": str(evidence_path),
                }
            )
            spec_record["selected_case_ids"].append(evidence.get("case_id"))

        spec_records.append(spec_record)

    return spec_records, case_records


def load_prompt_templates() -> Tuple[str, str]:
    prompts_dir = Path(__file__).resolve().parent / "prompts"
    system_prompt = read_text(prompts_dir / "failure_system_prompt.txt")
    user_prompt = read_text(prompts_dir / "failure_user_prompt.txt")
    return system_prompt, user_prompt


def format_actual_outcome(evidence: Dict[str, Any]) -> str:
    execution = evidence.get("execution") or {}
    response = execution.get("response_received") or {}
    assertion_failures = execution.get("assertion_failures") or []
    error = ""
    if assertion_failures:
        error = str(assertion_failures[0].get("actual_error") or "")
    status = response.get("status")
    outcome = execution.get("outcome")
    if error:
        return f"outcome={outcome}, status={status}, error={truncate_chars(error, 120)}"
    return f"outcome={outcome}, status={status}"


def build_prompt_evidence_view(evidence: Dict[str, Any], max_items: int) -> Dict[str, Any]:
    spec = evidence.get("spec") or {}
    test_context = evidence.get("test_context") or {}
    execution = evidence.get("execution") or {}
    ir_context = evidence.get("ir_context") or {}
    related = evidence.get("related_context") or {}

    return {
        "spec": {
            "operation": spec.get("operation"),
            "security_requirements": compact_list(spec.get("security_requirements"), max_items),
            "request_constraints": spec.get("request_constraints"),
            "response_contract": spec.get("response_contract"),
        },
        "test_context": {
            "test_id": test_context.get("test_id"),
            "title": test_context.get("title"),
            "category": test_context.get("category"),
            "intent": test_context.get("intent"),
            "generator_rule": test_context.get("generator_rule"),
            "expected_outcome": test_context.get("expected_outcome"),
            "generated_input": test_context.get("generated_input"),
        },
        "execution": {
            "outcome": execution.get("outcome"),
            "assertion_failures": compact_list(execution.get("assertion_failures"), max_items),
            "request_sent": execution.get("request_sent"),
            "response_received": execution.get("response_received"),
            "duration_ms": execution.get("duration_ms"),
        },
        "related_context": related,
        "warnings": {
            "parser_warnings": compact_list(ir_context.get("parser_warnings"), max_items),
            "unsupported_spec_warnings": compact_list(ir_context.get("unsupported_spec_warnings"), max_items),
        },
        "missing_evidence": compact_list(evidence.get("missing_evidence"), max_items),
    }


def build_user_prompt(user_template: str, evidence: Dict[str, Any], evidence_view: Dict[str, Any]) -> str:
    test_context = evidence.get("test_context") or {}
    spec = evidence.get("spec") or {}
    operation = spec.get("operation") or {}
    expected = json.dumps(test_context.get("expected_outcome") or {}, ensure_ascii=True)
    evidence_json = json.dumps(evidence_view, indent=2, ensure_ascii=True)

    return user_template.format(
        case_id=str(evidence.get("case_id") or ""),
        intent=str(test_context.get("intent") or ""),
        category=str(test_context.get("category") or ""),
        method=str(operation.get("method") or ""),
        path=str(operation.get("path") or ""),
        expected_outcome=expected,
        actual_outcome=format_actual_outcome(evidence),
        evidence_json=evidence_json,
    )


def parse_llm_json_response(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    text = (raw or "").strip()
    if not text:
        return None, "empty response"
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, None
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None, "response is not valid JSON object"
    try:
        obj = json.loads(match.group(0))
    except Exception as exc:  # noqa: BLE001
        return None, f"response JSON parse error: {exc}"
    if not isinstance(obj, dict):
        return None, "response JSON root must be an object"
    return obj, None


def normalize_contract(obj: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []

    cause = str(obj.get("cause") or "").strip()
    why_likely = str(obj.get("why_likely") or "").strip()
    check_next = str(obj.get("check_next") or "").strip()
    confidence = str(obj.get("confidence") or "").strip().lower()
    expected_negative_behavior = obj.get("expected_negative_behavior")
    evidence_quotes = obj.get("evidence_quotes")

    if not cause:
        errors.append("missing cause")
    if not why_likely:
        errors.append("missing why_likely")
    if not check_next:
        errors.append("missing check_next")
    if confidence not in {"confirmed", "likely"}:
        errors.append("confidence must be 'confirmed' or 'likely'")
        confidence = "likely"

    if not isinstance(expected_negative_behavior, bool):
        if isinstance(expected_negative_behavior, str):
            expected_negative_behavior = expected_negative_behavior.strip().lower() in {"true", "1", "yes"}
        else:
            expected_negative_behavior = False
            errors.append("expected_negative_behavior must be boolean")

    quotes_out: List[str] = []
    if isinstance(evidence_quotes, list):
        for quote in evidence_quotes:
            q = str(quote or "").strip()
            if q:
                quotes_out.append(q)
    elif evidence_quotes is None:
        quotes_out = []
    else:
        errors.append("evidence_quotes must be a list")

    return (
        {
            "cause": cause,
            "why_likely": why_likely,
            "check_next": check_next,
            "confidence": confidence,
            "expected_negative_behavior": bool(expected_negative_behavior),
            "evidence_quotes": quotes_out[:3],
        },
        errors,
    )


def render_explanation(contract: Dict[str, Any], word_target: int, word_max: int) -> str:
    confidence = str(contract.get("confidence") or "likely").strip().lower()
    label = "Confirmed cause" if confidence == "confirmed" else "Likely cause"
    parts = [
        f"{label}: {contract.get('cause', '').strip()}",
        f"Why likely: {contract.get('why_likely', '').strip()}",
    ]
    if contract.get("expected_negative_behavior"):
        parts.append("This appears aligned with negative/fail-path behavior, so it may not indicate a real defect.")
    parts.append(f"Check next: {contract.get('check_next', '').strip()}")
    text = " ".join(p for p in parts if p.strip())
    text = re.sub(r"\s+", " ", text).strip()

    words = count_words(text)
    if word_max > 0 and words > word_max:
        text = truncate_words(text, word_max)
    if word_target > 0 and count_words(text) < max(20, int(word_target * 0.6)):
        text = text.rstrip(".") + "."
    return text


def validate_contract_output(contract: Dict[str, Any], explanation: str, word_max: int) -> List[str]:
    errors: List[str] = []
    full_text = " ".join(
        [
            str(contract.get("cause") or ""),
            str(contract.get("why_likely") or ""),
            str(contract.get("check_next") or ""),
        ]
    )
    if contains_meta_terms(full_text) > 0:
        errors.append("contains framework/metadata terms")
    if word_max > 0 and count_words(explanation) > word_max:
        errors.append("rendered explanation exceeds word_max")
    for quote in contract.get("evidence_quotes") or []:
        if len(str(quote)) > 160:
            errors.append("evidence quote too long")
            break
    return errors


def compute_auto_scores(
    *,
    evidence: Dict[str, Any],
    prompt_evidence_text: str,
    contract: Dict[str, Any],
    explanation: str,
    word_target: int,
    word_max: int,
) -> Dict[str, Any]:
    full_text = " ".join(
        [
            str(contract.get("cause") or ""),
            str(contract.get("why_likely") or ""),
            str(contract.get("check_next") or ""),
        ]
    )
    quotes = contract.get("evidence_quotes") or []
    if quotes:
        matched = sum(
            1 for quote in quotes
            if str(quote).strip().lower() and str(quote).strip().lower() in prompt_evidence_text.lower()
        )
        groundedness = round(matched / len(quotes), 4)
    else:
        groundedness = 0.25

    signal = classify_failure_signal(evidence)
    signal_keywords = signal.get("keywords") or []
    lowered = full_text.lower()
    correctness_proxy = 0.5
    if signal_keywords:
        correctness_proxy = 1.0 if any(keyword in lowered for keyword in signal_keywords) else 0.0

    meta_hits = contains_meta_terms(full_text)
    relevance = max(0.0, round(1.0 - (0.25 * meta_hits), 4))

    actionability = 1.0 if ACTION_WORDS_RE.search(str(contract.get("check_next") or "")) else 0.0

    words = count_words(explanation)
    if word_max > 0 and words > word_max:
        conciseness = 0.0
    else:
        distance = abs(words - word_target) if word_target > 0 else 0
        conciseness = max(0.4, 1.0 - (distance / max(word_target, 1)) * 0.6)
        conciseness = round(conciseness, 4)

    intent = str((evidence.get("test_context") or {}).get("intent") or "")
    expected_negative = bool(contract.get("expected_negative_behavior"))
    if intent in {"negative", "auth", "fail-path"}:
        contextual_awareness = 1.0 if expected_negative else 0.5
    else:
        contextual_awareness = 1.0 if not expected_negative else 0.6

    non_grounded_quotes = 0
    for quote in quotes:
        q = str(quote).strip().lower()
        if q and q not in prompt_evidence_text.lower():
            non_grounded_quotes += 1
    hallucination_penalty = min(1.0, round((meta_hits * 0.25) + (non_grounded_quotes * 0.2), 4))

    positive = [
        groundedness,
        correctness_proxy,
        relevance,
        actionability,
        conciseness,
        contextual_awareness,
    ]
    overall = max(0.0, min(1.0, round((sum(positive) / len(positive)) - (0.3 * hallucination_penalty), 4)))

    return {
        "groundedness": groundedness,
        "correctness_proxy": correctness_proxy,
        "relevance": relevance,
        "actionability": actionability,
        "conciseness": conciseness,
        "contextual_awareness": contextual_awareness,
        "hallucination_penalty": hallucination_penalty,
        "overall": overall,
        "words": words,
    }


def invoke_case_llm(
    *,
    client: OllamaClient,
    model: str,
    system_prompt: str,
    user_template: str,
    evidence: Dict[str, Any],
    max_items_per_section: int,
    word_target: int,
    word_max: int,
    retry_invalid_output: int,
    ollama_options: Dict[str, Any],
) -> Dict[str, Any]:
    evidence_view = build_prompt_evidence_view(evidence, max_items=max_items_per_section)
    prompt_evidence_text = json.dumps(evidence_view, ensure_ascii=True)
    base_user_prompt = build_user_prompt(user_template, evidence, evidence_view)

    attempts: List[Dict[str, Any]] = []
    retries = max(0, int(retry_invalid_output))
    remaining = retries + 1
    user_prompt = base_user_prompt

    while remaining > 0:
        remaining -= 1
        response, llm_error = client.generate(
            model=model,
            prompt=user_prompt,
            system=system_prompt,
            format_json=True,
            options=ollama_options,
        )

        attempt_record: Dict[str, Any] = {
            "prompt": user_prompt,
            "raw_response": response or "",
            "llm_error": llm_error,
            "parse_error": None,
            "validation_errors": [],
        }

        if llm_error:
            attempt_record["parse_error"] = llm_error
            attempts.append(attempt_record)
            if remaining > 0:
                user_prompt = (
                    base_user_prompt
                    + "\n\nPrevious attempt failed to execute. Return valid JSON only and keep it concise."
                )
                continue
            fallback_contract = deterministic_fallback_contract(evidence)
            fallback_explanation = render_explanation(fallback_contract, word_target=word_target, word_max=word_max)
            fallback_scores = compute_auto_scores(
                evidence=evidence,
                prompt_evidence_text=prompt_evidence_text,
                contract=fallback_contract,
                explanation=fallback_explanation,
                word_target=word_target,
                word_max=word_max,
            )
            return {
                "ok": False,
                "llm_error": llm_error,
                "attempts": attempts,
                "contract": fallback_contract,
                "explanation": fallback_explanation,
                "auto_scores": fallback_scores,
                "prompt_evidence": evidence_view,
                "prompt_evidence_text": prompt_evidence_text,
                "used_fallback": True,
                "validation_errors": ["llm_error"],
            }

        parsed, parse_err = parse_llm_json_response(response or "")
        if parse_err:
            attempt_record["parse_error"] = parse_err
            attempts.append(attempt_record)
            if remaining > 0:
                user_prompt = (
                    base_user_prompt
                    + "\n\nPrevious output was not valid JSON. Return strict JSON only with the required keys."
                )
                continue
            fallback_contract = deterministic_fallback_contract(evidence)
            fallback_explanation = render_explanation(fallback_contract, word_target=word_target, word_max=word_max)
            fallback_scores = compute_auto_scores(
                evidence=evidence,
                prompt_evidence_text=prompt_evidence_text,
                contract=fallback_contract,
                explanation=fallback_explanation,
                word_target=word_target,
                word_max=word_max,
            )
            return {
                "ok": False,
                "llm_error": parse_err,
                "attempts": attempts,
                "contract": fallback_contract,
                "explanation": fallback_explanation,
                "auto_scores": fallback_scores,
                "prompt_evidence": evidence_view,
                "prompt_evidence_text": prompt_evidence_text,
                "used_fallback": True,
                "validation_errors": ["parse_error"],
            }

        contract, normalize_errors = normalize_contract(parsed or {})
        explanation = render_explanation(contract, word_target=word_target, word_max=word_max)
        validation_errors = normalize_errors + validate_contract_output(contract, explanation, word_max=word_max)
        attempt_record["validation_errors"] = validation_errors
        attempts.append(attempt_record)

        if validation_errors and remaining > 0:
            user_prompt = (
                base_user_prompt
                + "\n\nPrevious output had issues: "
                + "; ".join(validation_errors)
                + ". Return corrected JSON only."
            )
            continue

        if validation_errors and remaining <= 0:
            contract = deterministic_fallback_contract(evidence)
            explanation = render_explanation(contract, word_target=word_target, word_max=word_max)
            used_fallback = True
            llm_error_out = "; ".join(validation_errors)
        else:
            used_fallback = False
            llm_error_out = None

        auto_scores = compute_auto_scores(
            evidence=evidence,
            prompt_evidence_text=prompt_evidence_text,
            contract=contract,
            explanation=explanation,
            word_target=word_target,
            word_max=word_max,
        )
        return {
            "ok": not validation_errors,
            "llm_error": llm_error_out,
            "attempts": attempts,
            "contract": contract,
            "explanation": explanation,
            "auto_scores": auto_scores,
            "prompt_evidence": evidence_view,
            "prompt_evidence_text": prompt_evidence_text,
            "used_fallback": used_fallback,
            "validation_errors": validation_errors,
        }

    fallback_contract = deterministic_fallback_contract(evidence)
    fallback_explanation = render_explanation(fallback_contract, word_target=word_target, word_max=word_max)
    fallback_scores = compute_auto_scores(
        evidence=evidence,
        prompt_evidence_text=prompt_evidence_text,
        contract=fallback_contract,
        explanation=fallback_explanation,
        word_target=word_target,
        word_max=word_max,
    )
    return {
        "ok": False,
        "llm_error": "internal_error",
        "attempts": attempts,
        "contract": fallback_contract,
        "explanation": fallback_explanation,
        "auto_scores": fallback_scores,
        "prompt_evidence": evidence_view,
        "prompt_evidence_text": prompt_evidence_text,
        "used_fallback": True,
        "validation_errors": ["internal_error"],
    }


def avg_metric(case_rows: List[Dict[str, Any]], key: str) -> float:
    if not case_rows:
        return 0.0
    return round(sum(float((row.get("auto_scores") or {}).get(key, 0.0)) for row in case_rows) / len(case_rows), 4)


def evaluate_models(
    *,
    run_dir: Path,
    models: List[str],
    case_records: List[Dict[str, Any]],
    spec_records: List[Dict[str, Any]],
    system_prompt: str,
    user_prompt_template: str,
    client: OllamaClient,
    max_items_per_section: int,
    word_target: int,
    word_max: int,
    retry_invalid_output: int,
    ollama_options: Dict[str, Any],
) -> List[Dict[str, Any]]:
    llm_root = run_dir / "llm"
    llm_root.mkdir(parents=True, exist_ok=True)

    model_results: List[Dict[str, Any]] = []

    for model in models:
        print(f"\n{'=' * 72}")
        print(f"MODEL: {model}")
        print(f"{'=' * 72}")
        model_slug = slugify(model)
        model_root = llm_root / model_slug
        model_root.mkdir(parents=True, exist_ok=True)

        model_cases: List[Dict[str, Any]] = []
        total_cases = len(case_records)

        for idx, case in enumerate(case_records, start=1):
            spec_id = case["spec_id"]
            test_id = case["test_id"]
            case_slug = slugify(test_id)
            spec_dir = model_root / spec_id
            spec_dir.mkdir(parents=True, exist_ok=True)

            evidence_path = Path(case["evidence_file"])
            try:
                evidence = json.loads(read_text(evidence_path))
            except Exception as exc:  # noqa: BLE001
                explanation = f"LLM_ERROR: unable to load evidence ({exc})"
                response_txt = spec_dir / f"{case_slug}.response.txt"
                response_json = spec_dir / f"{case_slug}.response.json"
                prompt_txt = spec_dir / f"{case_slug}.prompt.txt"
                write_text(response_txt, explanation)
                write_text(prompt_txt, "")
                write_json(
                    response_json,
                    {
                        "case_id": case.get("case_id"),
                        "model": model,
                        "error": str(exc),
                        "explanation": explanation,
                    },
                )
                model_cases.append(
                    {
                        **case,
                        "model": model,
                        "explanation": explanation,
                        "confidence": "likely",
                        "expected_negative_behavior": False,
                        "auto_scores": {
                            "groundedness": 0.0,
                            "correctness_proxy": 0.0,
                            "relevance": 0.0,
                            "actionability": 0.0,
                            "conciseness": 0.0,
                            "contextual_awareness": 0.0,
                            "hallucination_penalty": 1.0,
                            "overall": 0.0,
                            "words": count_words(explanation),
                        },
                        "llm_error": str(exc),
                        "validation_errors": ["evidence_load_error"],
                        "files": {
                            "response_json": str(response_json),
                            "response_txt": str(response_txt),
                            "prompt_txt": str(prompt_txt),
                        },
                    }
                )
                continue

            print(f"  LLM case {idx}/{total_cases}: {spec_id}::{test_id}")
            llm_out = invoke_case_llm(
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_template=user_prompt_template,
                evidence=evidence,
                max_items_per_section=max_items_per_section,
                word_target=word_target,
                word_max=word_max,
                retry_invalid_output=retry_invalid_output,
                ollama_options=ollama_options,
            )

            response_txt = spec_dir / f"{case_slug}.response.txt"
            response_json = spec_dir / f"{case_slug}.response.json"
            prompt_txt = spec_dir / f"{case_slug}.prompt.txt"

            first_prompt = ""
            if llm_out.get("attempts"):
                first_prompt = str(llm_out["attempts"][0].get("prompt") or "")
            write_text(prompt_txt, first_prompt)
            write_text(response_txt, str(llm_out.get("explanation") or ""))
            write_json(
                response_json,
                {
                    "case_id": case.get("case_id"),
                    "model": model,
                    "llm_error": llm_out.get("llm_error"),
                    "used_fallback": llm_out.get("used_fallback"),
                    "validation_errors": llm_out.get("validation_errors") or [],
                    "contract": llm_out.get("contract") or {},
                    "explanation": llm_out.get("explanation"),
                    "auto_scores": llm_out.get("auto_scores") or {},
                    "prompt_evidence": llm_out.get("prompt_evidence") or {},
                    "attempts": llm_out.get("attempts") or [],
                },
            )

            contract = llm_out.get("contract") or {}
            model_cases.append(
                {
                    **case,
                    "model": model,
                    "explanation": llm_out.get("explanation"),
                    "confidence": contract.get("confidence"),
                    "expected_negative_behavior": contract.get("expected_negative_behavior"),
                    "evidence_quotes": contract.get("evidence_quotes") or [],
                    "auto_scores": llm_out.get("auto_scores") or {},
                    "llm_error": llm_out.get("llm_error"),
                    "validation_errors": llm_out.get("validation_errors") or [],
                    "files": {
                        "response_json": str(response_json),
                        "response_txt": str(response_txt),
                        "prompt_txt": str(prompt_txt),
                    },
                }
            )

        by_spec: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for case_row in model_cases:
            by_spec[str(case_row.get("spec_id"))].append(case_row)

        spec_rows: List[Dict[str, Any]] = []
        for spec_record in spec_records:
            spec_id = spec_record["spec_id"]
            rows = sorted(by_spec.get(spec_id, []), key=lambda item: str(item.get("test_id")))
            spec_rows.append(
                {
                    "spec_id": spec_id,
                    "spec_path": spec_record.get("spec_path"),
                    "pipeline_summary": {
                        "parse_ok": spec_record.get("parse_ok"),
                        "generate_ok": spec_record.get("generate_ok"),
                        "run_ok": spec_record.get("run_ok"),
                        "summary": spec_record.get("summary"),
                        "errors": spec_record.get("errors") or [],
                    },
                    "test_cases": rows,
                }
            )

        aggregate_scores = {
            "groundedness": avg_metric(model_cases, "groundedness"),
            "correctness_proxy": avg_metric(model_cases, "correctness_proxy"),
            "relevance": avg_metric(model_cases, "relevance"),
            "actionability": avg_metric(model_cases, "actionability"),
            "conciseness": avg_metric(model_cases, "conciseness"),
            "contextual_awareness": avg_metric(model_cases, "contextual_awareness"),
            "hallucination_penalty": avg_metric(model_cases, "hallucination_penalty"),
            "overall": avg_metric(model_cases, "overall"),
        }

        model_results.append(
            {
                "model": model,
                "model_slug": model_slug,
                "specs": spec_rows,
                "cases": model_cases,
                "aggregate_scores": aggregate_scores,
                "counts": {
                    "cases": len(model_cases),
                    "llm_errors": sum(1 for row in model_cases if row.get("llm_error")),
                    "fallback_used": sum(1 for row in model_cases if row.get("validation_errors")),
                },
            }
        )

    return model_results


def relpath(path_str: str, base_dir: Path) -> str:
    try:
        return str(Path(path_str).relative_to(base_dir))
    except Exception:
        return path_str


def write_benchmark_files(run_dir: Path, model_results: List[Dict[str, Any]]) -> None:
    benchmark_dir = run_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    auto_payload = {
        "run_id": run_dir.name,
        "models": [],
    }
    comparison_rows: List[Dict[str, Any]] = []

    for model_result in model_results:
        model_name = model_result.get("model")
        model_cases = model_result.get("cases") or []
        auto_payload["models"].append(
            {
                "model": model_name,
                "aggregate_scores": model_result.get("aggregate_scores") or {},
                "counts": model_result.get("counts") or {},
                "cases": [
                    {
                        "case_id": row.get("case_id"),
                        "spec_id": row.get("spec_id"),
                        "test_id": row.get("test_id"),
                        "title": row.get("title"),
                        "auto_scores": row.get("auto_scores") or {},
                        "llm_error": row.get("llm_error"),
                    }
                    for row in model_cases
                ],
            }
        )
        for row in model_cases:
            comparison_rows.append(
                {
                    "model": model_name,
                    "spec_id": row.get("spec_id"),
                    "test_id": row.get("test_id"),
                    "case_id": row.get("case_id"),
                    "explanation": row.get("explanation"),
                    "auto_scores": row.get("auto_scores") or {},
                    "human_scores": None,
                    "notes": "",
                }
            )

    write_json(benchmark_dir / "auto_scores.json", auto_payload)

    with (benchmark_dir / "comparison.jsonl").open("w", encoding="utf-8") as f:
        for row in comparison_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    csv_path = benchmark_dir / "human_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "spec_id",
                "test_id",
                "case_id",
                "explanation",
                "correctness_1_5",
                "relevance_1_5",
                "actionability_1_5",
                "conciseness_1_5",
                "context_awareness_1_5",
                "wrongly_flags_negative_as_bug",
                "notes",
            ]
        )
        for row in comparison_rows:
            writer.writerow(
                [
                    row.get("model"),
                    row.get("spec_id"),
                    row.get("test_id"),
                    row.get("case_id"),
                    row.get("explanation"),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )


def build_index_data(
    *,
    run_dir: Path,
    run_meta: Dict[str, Any],
    spec_records: List[Dict[str, Any]],
    case_records: List[Dict[str, Any]],
    model_results: List[Dict[str, Any]],
    case_scope: str,
) -> Dict[str, Any]:
    model_entries: List[Dict[str, Any]] = []
    best_model = None
    best_score = -1.0

    for result in model_results:
        specs_out = []
        for spec_row in result.get("specs") or []:
            test_cases_out = []
            for case in spec_row.get("test_cases") or []:
                files = case.get("files") or {}
                test_cases_out.append(
                    {
                        "case_id": case.get("case_id"),
                        "test_id": case.get("test_id"),
                        "title": case.get("title"),
                        "outcome": case.get("outcome"),
                        "selection_reason": case.get("selection_reason"),
                        "category": case.get("category"),
                        "intent": case.get("intent"),
                        "explanation": case.get("explanation"),
                        "confidence": case.get("confidence"),
                        "expected_negative_behavior": case.get("expected_negative_behavior"),
                        "auto_scores": case.get("auto_scores") or {},
                        "llm_error": case.get("llm_error"),
                        "validation_errors": case.get("validation_errors") or [],
                        "files": {
                            "response_json": relpath(str(files.get("response_json") or ""), run_dir),
                            "response_txt": relpath(str(files.get("response_txt") or ""), run_dir),
                            "prompt_txt": relpath(str(files.get("prompt_txt") or ""), run_dir),
                            "evidence": relpath(str(case.get("evidence_file") or ""), run_dir),
                        },
                    }
                )

            specs_out.append(
                {
                    "spec_id": spec_row.get("spec_id"),
                    "spec_path": spec_row.get("spec_path"),
                    "pipeline_summary": spec_row.get("pipeline_summary") or {},
                    "test_cases": test_cases_out,
                }
            )

        aggregate = result.get("aggregate_scores") or {}
        overall = float(aggregate.get("overall", 0.0))
        if overall > best_score:
            best_model = result.get("model")
            best_score = overall

        model_entries.append(
            {
                "model": result.get("model"),
                "counts": result.get("counts") or {},
                "aggregate_scores": aggregate,
                "specs": specs_out,
            }
        )

    totals = {"total": 0, "passed": 0, "failed": 0, "skipped": 0}
    for spec in spec_records:
        summary = spec.get("summary") or {}
        totals["total"] += int(summary.get("total", 0))
        totals["passed"] += int(summary.get("passed", 0))
        totals["failed"] += int(summary.get("failed", 0))
        totals["skipped"] += int(summary.get("skipped", 0))

    return {
        "run_id": run_meta.get("run_id"),
        "started_at": run_meta.get("started_at"),
        "finished_at": run_meta.get("finished_at"),
        "case_scope": case_scope,
        "models_requested": run_meta.get("models") or [],
        "specs_requested": run_meta.get("specs") or [],
        "totals": totals,
        "pipeline_specs": spec_records,
        "selected_cases": [
            {
                **case,
                "evidence_file": relpath(str(case.get("evidence_file") or ""), run_dir),
            }
            for case in case_records
        ],
        "models": model_entries,
        "best_model": best_model,
    }


def write_index_files(run_dir: Path, index_data: Dict[str, Any]) -> None:
    write_json(run_dir / "index.json", index_data)

    lines = [
        "# LLM Evaluation Index",
        "",
        f"Run ID: {index_data.get('run_id')}",
        f"Started: {index_data.get('started_at')}",
        f"Finished: {index_data.get('finished_at')}",
        f"Case Scope: {index_data.get('case_scope')}",
        "",
        "Totals:",
        f"- Total: {index_data.get('totals', {}).get('total', 0)}",
        f"- Passed: {index_data.get('totals', {}).get('passed', 0)}",
        f"- Failed: {index_data.get('totals', {}).get('failed', 0)}",
        f"- Skipped: {index_data.get('totals', {}).get('skipped', 0)}",
        "",
    ]

    for model in index_data.get("models", []):
        lines.append(f"## Model: {model.get('model')}")
        agg = model.get("aggregate_scores") or {}
        lines.append(
            "Scores: "
            f"overall={agg.get('overall', 0)} "
            f"groundedness={agg.get('groundedness', 0)} "
            f"correctness_proxy={agg.get('correctness_proxy', 0)} "
            f"relevance={agg.get('relevance', 0)} "
            f"actionability={agg.get('actionability', 0)} "
            f"conciseness={agg.get('conciseness', 0)} "
            f"contextual_awareness={agg.get('contextual_awareness', 0)}"
        )
        lines.append("")
        for spec in model.get("specs", []):
            lines.append(f"### Spec: {spec.get('spec_id')}")
            cases = spec.get("test_cases") or []
            if not cases:
                lines.append("(No selected cases)")
                lines.append("")
                continue
            for case in cases:
                lines.append(f"- {case.get('test_id')} ({case.get('selection_reason')}): {case.get('explanation')}")
            lines.append("")

    write_text(run_dir / "index.md", "\n".join(lines).rstrip() + "\n")


def parse_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    llm_cfg = config.get("llm") or {}
    evidence_cfg = config.get("evidence") or {}
    ollama_cfg = config.get("ollama") or {}
    runner_cfg = config.get("runner") or {}

    case_scope = normalize_case_scope(config.get("case_scope") or config.get("llm_scope"))
    timeout_seconds = int(llm_cfg.get("timeout_seconds") or ollama_cfg.get("timeout_seconds", 120))
    word_target = int(llm_cfg.get("output_word_target", 55))
    word_max = int(llm_cfg.get("output_word_max", 100))
    retry_invalid_output = int(llm_cfg.get("retry_invalid_output", 1))
    max_cases_per_spec = int(llm_cfg.get("max_cases_per_spec", config.get("llm_max_failures_per_spec", 25)))

    max_items_per_section = int(evidence_cfg.get("max_items_per_section", 12))
    snippet_chars = int(evidence_cfg.get("snippet_chars", 300))
    max_similar_failures = int(evidence_cfg.get("max_similar_failures", 5))
    runner_timeout = int(runner_cfg.get("timeout_seconds", 12))

    return {
        "case_scope": case_scope,
        "llm_timeout_seconds": timeout_seconds,
        "word_target": word_target,
        "word_max": word_max,
        "retry_invalid_output": retry_invalid_output,
        "max_cases_per_spec": max_cases_per_spec,
        "max_items_per_section": max_items_per_section,
        "snippet_chars": snippet_chars,
        "max_similar_failures": max_similar_failures,
        "runner_timeout": runner_timeout,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run grounded per-test LLM evaluation on ContractGuard specs.")
    parser.add_argument(
        "--config",
        default="config/llm_eval.yaml",
        help="Path to YAML config (default: config/llm_eval.yaml)",
    )
    parser.add_argument("--models", action="append", help="Comma-separated model list override")
    parser.add_argument("--specs", action="append", help="Comma-separated spec path list override")
    parser.add_argument("--out", default=None, help="Override output directory")
    parser.add_argument("--timeout", type=int, default=None, help="Override runner timeout seconds")
    parser.add_argument("--ollama-url", default=None, help="Override Ollama base URL")
    parser.add_argument("--llm-timeout", type=int, default=None, help="Override LLM timeout seconds")
    parser.add_argument("--case-scope", default=None, help="fail|fail_and_noteworthy|all")
    parser.add_argument("--word-target", type=int, default=None, help="Target words for final explanation")
    parser.add_argument("--word-max", type=int, default=None, help="Hard max words for final explanation")
    parser.add_argument("--retry-invalid-output", type=int, default=None, help="Retry count for invalid model output")
    parser.add_argument(
        "--llm-max-failures-per-spec",
        type=int,
        default=None,
        help="Alias for max cases per spec",
    )
    parser.add_argument(
        "--llm-max-chars",
        type=int,
        default=None,
        help="Deprecated legacy option (ignored). Use --word-max instead.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        return 1

    try:
        config = load_config(config_path)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Failed to load config: {exc}", file=sys.stderr)
        return 1

    model_override = split_csv_args(args.models)
    spec_override = split_csv_args(args.specs)
    if model_override:
        config["models"] = model_override
    if spec_override:
        config["specs"] = spec_override
    if args.out:
        config["output_dir"] = args.out
    if args.timeout is not None:
        runner_cfg = config.get("runner") or {}
        runner_cfg["timeout_seconds"] = int(args.timeout)
        config["runner"] = runner_cfg
    if args.ollama_url:
        ollama_cfg = config.get("ollama") or {}
        ollama_cfg["base_url"] = args.ollama_url
        config["ollama"] = ollama_cfg
    if args.llm_timeout is not None:
        llm_cfg = config.get("llm") or {}
        llm_cfg["timeout_seconds"] = int(args.llm_timeout)
        config["llm"] = llm_cfg
    if args.case_scope:
        config["case_scope"] = normalize_case_scope(args.case_scope)
    if args.word_target is not None:
        llm_cfg = config.get("llm") or {}
        llm_cfg["output_word_target"] = int(args.word_target)
        config["llm"] = llm_cfg
    if args.word_max is not None:
        llm_cfg = config.get("llm") or {}
        llm_cfg["output_word_max"] = int(args.word_max)
        config["llm"] = llm_cfg
    if args.retry_invalid_output is not None:
        llm_cfg = config.get("llm") or {}
        llm_cfg["retry_invalid_output"] = int(args.retry_invalid_output)
        config["llm"] = llm_cfg
    if args.llm_max_failures_per_spec is not None:
        llm_cfg = config.get("llm") or {}
        llm_cfg["max_cases_per_spec"] = int(args.llm_max_failures_per_spec)
        config["llm"] = llm_cfg
    if args.llm_max_chars is not None:
        print("WARN: --llm-max-chars is deprecated and ignored. Use --word-max.")

    models = coerce_list(config.get("models"), "models")
    specs = coerce_list(config.get("specs"), "specs")
    if not models or not specs:
        print("ERROR: Config must include non-empty 'models' and 'specs'.", file=sys.stderr)
        return 1

    settings = parse_settings(config)
    case_scope = settings["case_scope"]
    max_cases_per_spec = settings["max_cases_per_spec"]
    max_items_per_section = settings["max_items_per_section"]
    snippet_chars = settings["snippet_chars"]
    max_similar_failures = settings["max_similar_failures"]

    output_dir = Path(config.get("output_dir") or "build/llm_runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    runner_cfg = config.get("runner") or {}
    runner_cfg["timeout_seconds"] = int(settings["runner_timeout"])
    config["runner"] = runner_cfg

    print(f"Run directory: {run_dir}")
    print(f"Models: {', '.join(models)}")
    print(f"Specs: {len(specs)}")
    print(f"Case scope: {case_scope}")

    run_started = datetime.now(timezone.utc).isoformat()

    spec_records, case_records = run_pipeline_once(
        specs=specs,
        config=config,
        run_dir=run_dir,
        case_scope=case_scope,
        max_cases_per_spec=max_cases_per_spec,
        max_items_per_section=max_items_per_section,
        snippet_chars=snippet_chars,
        max_similar_failures=max_similar_failures,
    )

    print(f"\nSelected cases for LLM: {len(case_records)}")

    ollama_cfg = config.get("ollama") or {}
    ollama_options = ollama_cfg.get("options") or {}
    if not isinstance(ollama_options, dict):
        ollama_options = {}

    client = OllamaClient(
        base_url=str(ollama_cfg.get("base_url") or "http://localhost:11434"),
        timeout_seconds=int(settings["llm_timeout_seconds"]),
    )
    system_prompt, user_prompt_template = load_prompt_templates()

    model_results = evaluate_models(
        run_dir=run_dir,
        models=models,
        case_records=case_records,
        spec_records=spec_records,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        client=client,
        max_items_per_section=max_items_per_section,
        word_target=int(settings["word_target"]),
        word_max=int(settings["word_max"]),
        retry_invalid_output=int(settings["retry_invalid_output"]),
        ollama_options=ollama_options,
    )

    write_benchmark_files(run_dir, model_results)

    run_finished = datetime.now(timezone.utc).isoformat()
    run_meta = {
        "run_id": run_id,
        "started_at": run_started,
        "finished_at": run_finished,
        "models": models,
        "specs": specs,
        "case_scope": case_scope,
        "settings": settings,
        "output_dir": str(output_dir),
        "config": sanitize_config_for_meta(config),
    }
    write_json(run_dir / "run_meta.json", run_meta)

    index_data = build_index_data(
        run_dir=run_dir,
        run_meta=run_meta,
        spec_records=spec_records,
        case_records=case_records,
        model_results=model_results,
        case_scope=case_scope,
    )
    write_index_files(run_dir, index_data)

    print(f"Run complete. Results in: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
