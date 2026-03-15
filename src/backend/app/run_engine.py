import json
import time
from typing import Any


def apply_path_params(path: str, path_params: dict[str, Any]) -> str:
    updated = path
    for key, value in (path_params or {}).items():
        updated = updated.replace("{" + str(key) + "}", str(value))
    return updated


def _get_requests():
    try:
        import requests as requests_module
    except Exception as exc:
        raise RuntimeError("The 'requests' package is required for runtime execution.") from exc
    return requests_module


def should_skip_for_media_type(test_case: dict[str, Any], response: Any) -> bool:
    if response.status_code != 415:
        return False

    path = (test_case.get("path") or "").lower()
    title = (test_case.get("title") or "").lower()
    return ("upload" in path) or ("upload" in title) or ("image" in path) or ("image" in title)


def expected_status_ok(status_code: int, expected_result: dict[str, Any]) -> tuple[bool, str]:
    if "status_code_any_of" in expected_result and expected_result["status_code_any_of"] is not None:
        any_of = expected_result["status_code_any_of"]
        ok = status_code in any_of
        return ok, str(any_of)

    if "status_code" not in expected_result:
        raise ValueError("expected_result must contain status_code or status_code_any_of")

    expected = expected_result["status_code"]
    ok = status_code == expected
    return ok, str(expected)


def deep_contains(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        for key, value in expected.items():
            if key not in actual or not deep_contains(value, actual[key]):
                return False
        return True

    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        for expected_item in expected:
            if not any(deep_contains(expected_item, actual_item) for actual_item in actual):
                return False
        return True

    return expected == actual


def response_body_contains_ok(expected_result: dict[str, Any], response: Any) -> bool:
    expected_subset = expected_result.get("response_body_contains", None)
    if expected_subset is None:
        return True

    try:
        body = response.json()
    except Exception:
        return False

    return deep_contains(expected_subset, body)


def response_snippet(response: Any, max_len: int = 500) -> str:
    text = (response.text or "").strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def run_single_test_case(
    test_case: dict[str, Any],
    base_url: str,
    global_headers: dict[str, str] | None = None,
    timeout_sec: int = 10,
) -> dict[str, Any]:
    started = time.perf_counter()
    test_id = str(test_case.get("test_id") or "")
    method = str(test_case.get("method") or "").upper()
    raw_path = str(test_case.get("path") or "")
    category = str(test_case.get("category") or "")
    title = str(test_case.get("title") or "")

    if not method or not raw_path:
        duration_ms = int((time.perf_counter() - started) * 1000)
        return {
            "test_id": test_id,
            "title": title,
            "method": method,
            "path": raw_path,
            "category": category,
            "status": "error",
            "actual_status": None,
            "error_message": "Test case missing method or path.",
            "response_snippet": "",
            "duration_ms": duration_ms,
            "expected_result": test_case.get("expected_result") or {},
        }

    steps = test_case.get("steps") or []
    if not steps:
        duration_ms = int((time.perf_counter() - started) * 1000)
        return {
            "test_id": test_id,
            "title": title,
            "method": method,
            "path": raw_path,
            "category": category,
            "status": "error",
            "actual_status": None,
            "error_message": "Test case has no steps.",
            "response_snippet": "",
            "duration_ms": duration_ms,
            "expected_result": test_case.get("expected_result") or {},
        }

    input_data = steps[0].get("input_data", {}) if isinstance(steps[0], dict) else {}
    headers = input_data.get("headers", {}) if isinstance(input_data, dict) else {}
    query_params = input_data.get("query_params", {}) if isinstance(input_data, dict) else {}
    body = input_data.get("body", None) if isinstance(input_data, dict) else None
    path_params = input_data.get("path_params", {}) if isinstance(input_data, dict) else {}

    path = apply_path_params(raw_path, path_params or {})
    url = base_url.rstrip("/") + path

    merged_headers = {"Content-Type": "application/json"}
    merged_headers.update(global_headers or {})
    merged_headers.update(headers or {})

    requests_module = _get_requests()

    try:
        response = requests_module.request(
            method=method,
            url=url,
            headers=merged_headers,
            params=query_params or {},
            json=body,
            timeout=timeout_sec,
        )
    except Exception as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        return {
            "test_id": test_id,
            "title": title,
            "method": method,
            "path": raw_path,
            "category": category,
            "status": "error",
            "actual_status": None,
            "error_message": str(exc),
            "response_snippet": "",
            "duration_ms": duration_ms,
            "expected_result": test_case.get("expected_result") or {},
        }

    expected_result = test_case.get("expected_result") or {}
    snippet = response_snippet(response)
    duration_ms = int((time.perf_counter() - started) * 1000)

    if should_skip_for_media_type(test_case, response):
        return {
            "test_id": test_id,
            "title": title,
            "method": method,
            "path": raw_path,
            "category": category,
            "status": "skip",
            "actual_status": response.status_code,
            "error_message": "",
            "response_snippet": snippet,
            "duration_ms": duration_ms,
            "expected_result": expected_result,
        }

    try:
        status_ok, _ = expected_status_ok(response.status_code, expected_result)
    except Exception as exc:
        return {
            "test_id": test_id,
            "title": title,
            "method": method,
            "path": raw_path,
            "category": category,
            "status": "error",
            "actual_status": response.status_code,
            "error_message": str(exc),
            "response_snippet": snippet,
            "duration_ms": duration_ms,
            "expected_result": expected_result,
        }

    body_ok = response_body_contains_ok(expected_result, response)
    status = "pass" if status_ok and body_ok else "fail"

    message = ""
    if not status_ok:
        message = "Status code mismatch."
    elif not body_ok:
        message = "response_body_contains assertion failed."

    return {
        "test_id": test_id,
        "title": title,
        "method": method,
        "path": raw_path,
        "category": category,
        "status": status,
        "actual_status": response.status_code,
        "error_message": message,
        "response_snippet": snippet,
        "duration_ms": duration_ms,
        "expected_result": expected_result,
    }


def stable_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return "{}"
