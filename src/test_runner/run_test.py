import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests

#DEFAULT_BASE_URL = "https://petstore3.swagger.io/api/v3"
DEFAULT_BASE_URL = "https://api.open-meteo.com"  # for testing with a different API


def load_tests(json_path: Path) -> Dict[str, Any]:
    if not json_path.exists():
        raise FileNotFoundError(f"Test file not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_first_step_input(tc: Dict[str, Any]) -> Dict[str, Any]:
    steps = tc.get("steps") or []
    if not steps:
        raise ValueError(f"{tc.get('test_id','<no id>')} has no steps.")
    input_data = steps[0].get("input_data")
    if input_data is None:
        raise ValueError(f"{tc.get('test_id','<no id>')} first step missing input_data.")
    return input_data


def apply_path_params(path: str, path_params: Dict[str, Any]) -> str:
    """
    Replace /resource/{id} placeholders using path_params.
    Example: path="/pet/{petId}", path_params={"petId": 10} -> "/pet/10"
    """
    out = path
    for k, v in (path_params or {}).items():
        out = out.replace("{" + str(k) + "}", str(v))
    return out


def expected_status_ok(status_code: int, expected_result: Dict[str, Any]) -> (bool, str):
    if "status_code_any_of" in expected_result and expected_result["status_code_any_of"] is not None:
        any_of = expected_result["status_code_any_of"]
        ok = status_code in any_of
        return ok, str(any_of)

    if "status_code" not in expected_result:
        raise ValueError("expected_result must contain status_code or status_code_any_of")

    expected = expected_result["status_code"]
    ok = status_code == expected
    return ok, str(expected)


def should_skip_for_media_type(tc: Dict[str, Any], resp: requests.Response) -> bool:
    """
    Minimal 'skip' policy to avoid noise:
    If server returns 415 and this is an upload/image style endpoint,
    mark as SKIP instead of FAIL (runner doesn't implement multipart).
    """
    if resp.status_code != 415:
        return False

    path = (tc.get("path") or "").lower()
    title = (tc.get("title") or "").lower()
    return ("upload" in path) or ("upload" in title) or ("image" in path) or ("image" in title)


def run_test_case(tc: Dict[str, Any], base_url: str, timeout_sec: int = 10) -> str:
    """
    Returns: "PASS" | "FAIL" | "SKIP"
    """
    test_id = tc.get("test_id", "<no id>")
    method = tc.get("method")
    raw_path = tc.get("path")

    if not method or not raw_path:
        raise ValueError(f"{test_id} missing method/path")

    input_data = extract_first_step_input(tc)

    headers: Dict[str, str] = input_data.get("headers", {}) or {}
    query_params: Dict[str, Any] = input_data.get("query_params", {}) or {}
    body = input_data.get("body", None)
    path_params: Dict[str, Any] = input_data.get("path_params", {}) or {}

    # Apply path params like {petId}
    path = apply_path_params(raw_path, path_params)
    url = base_url.rstrip("/") + path

    # Default JSON headers (good enough for most endpoints; multipart not supported here)
    request_headers = {"Content-Type": "application/json", **headers}

    try:
        resp = requests.request(
            method=method,
            url=url,
            headers=request_headers,
            params=query_params,
            json=body,
            timeout=timeout_sec,
        )
    except Exception as e:
        print(f"💥 {test_id} ERROR: {e}")
        return "FAIL"

    # Skip known unsupported 415 upload cases (minimal)
    if should_skip_for_media_type(tc, resp):
        print(f"⏭️  {test_id} SKIPPED (415 Unsupported Media Type - multipart not implemented)")
        return "SKIP"

    exp = tc.get("expected_result", {}) or {}
    try:
        ok, expected_display = expected_status_ok(resp.status_code, exp)
    except Exception as e:
        print(f"💥 {test_id} ERROR in expected_result: {e}")
        return "FAIL"

    if ok:
        print(f"✅ {test_id} PASSED ({resp.status_code})")
        return "PASS"

    snippet = (resp.text or "").strip()
    if len(snippet) > 300:
        snippet = snippet[:300] + "..."

    print(f"❌ {test_id} FAILED (expected {expected_display}, got {resp.status_code})")
    if snippet:
        print(f"   Response: {snippet}")
    return "FAIL"


def parse_args(argv: List[str]) -> Dict[str, Any]:
    """
    Minimal CLI:
      python3 run_test.py <test_json_path> [base_url]
    """
    if len(argv) < 2:
        return {
            "test_path": Path("../test_cases/user1/open_meteo_tests.json"),
            "base_url": DEFAULT_BASE_URL,
        }

    test_path = Path(argv[1])
    base_url = argv[2] if len(argv) >= 3 else DEFAULT_BASE_URL
    return {"test_path": test_path, "base_url": base_url}


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    test_path: Path = args["test_path"]
    base_url: str = args["base_url"]

    data = load_tests(test_path)
    test_cases = data.get("test_cases", [])

    if not isinstance(test_cases, list) or not test_cases:
        print("No test_cases found in JSON.")
        return 2

    print(f"Running {len(test_cases)} tests from {test_path} against {base_url}")

    total = 0
    passed = 0
    failed = 0
    skipped = 0

    for tc in test_cases:
        total += 1
        result = run_test_case(tc, base_url=base_url)
        if result == "PASS":
            passed += 1
        elif result == "SKIP":
            skipped += 1
        else:
            failed += 1

    print("\n===== SUMMARY =====")
    print(f"Total:   {total}")
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Skipped: {skipped}")

    # exit code 0 if no failures, 1 otherwise
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))