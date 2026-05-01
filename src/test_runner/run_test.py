"""
ContractGuard Test Runner

Loads generated test case JSON files, executes HTTP requests against a
target API, and produces structured PASS / FAIL / SKIP results.

Usage:
    python run_test.py <path_to_test_file_or_folder> \
        [--base-url https://api.example.com] \
        [--timeout 10] \
        [--json-output result.json] \
        [--bearer-token <token>] \
        [--api-key <key>] \
        [--api-key-header X-API-Key]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

SENSITIVE_HEADERS = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "apikey",
    "token",
}


def redact_headers(headers: Dict[str, Any]) -> Dict[str, Any]:
    redacted: Dict[str, Any] = {}
    for key, value in (headers or {}).items():
        if str(key).lower() in SENSITIVE_HEADERS:
            redacted[key] = "<redacted>"
        else:
            redacted[key] = value
    return redacted


# ---------------------------------------------------------------------------
# Auth resolution
# ---------------------------------------------------------------------------

def resolve_auth_headers(args: argparse.Namespace) -> Dict[str, str]:
    """Build auth headers from CLI args or environment variables.

    Priority: CLI args > environment variables > no auth.
    """
    headers: Dict[str, str] = {}

    # Bearer token
    bearer = args.bearer_token or os.environ.get("CONTRACTGUARD_BEARER_TOKEN")
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
        return headers  # bearer takes precedence; don't mix with API key

    # API key
    api_key = args.api_key or os.environ.get("CONTRACTGUARD_API_KEY")
    if api_key:
        header_name = (
            args.api_key_header
            or os.environ.get("CONTRACTGUARD_API_KEY_HEADER")
            or "X-API-Key"
        )
        headers[header_name] = api_key

    return headers


# ---------------------------------------------------------------------------
# Base URL resolution
# ---------------------------------------------------------------------------

def _is_full_url(url: str) -> bool:
    """Return True if the URL has a scheme (http/https)."""
    return url.startswith("http://") or url.startswith("https://")


def resolve_base_url(
    cli_url: Optional[str],
    json_url: Optional[str],
) -> Optional[str]:
    """Resolve the base URL per suite.

    Priority:
    1. --base-url CLI argument
    2. CONTRACTGUARD_BASE_URL environment variable
    3. Auto-extracted base_url from the test JSON (if it's a full URL)
    4. None (caller should skip this suite)
    """
    # CLI has top priority so CI/runtime secrets can override spec servers.
    if cli_url and _is_full_url(cli_url):
        return cli_url.rstrip("/")

    # Environment variable fallback
    env_url = os.environ.get("CONTRACTGUARD_BASE_URL")
    if env_url and _is_full_url(env_url):
        return env_url.rstrip("/")

    # Suite JSON fallback
    if json_url and _is_full_url(json_url):
        return json_url.rstrip("/")

    return None


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def collect_test_files(path: Path) -> List[Path]:
    """Return a list of test JSON files from a file or directory path."""
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("*.json"))
        if not files:
            print(f"ERROR: No JSON files found in {path}", file=sys.stderr)
            sys.exit(1)
        return files
    print(f"ERROR: Path not found: {path}", file=sys.stderr)
    sys.exit(1)


def load_test_suite(json_path: Path) -> Dict[str, Any]:
    """Load and return a test suite JSON file."""
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

def apply_path_params(path: str, path_params: Dict[str, Any]) -> str:
    """Replace {param} placeholders in the URL path."""
    for k, v in (path_params or {}).items():
        path = path.replace("{" + str(k) + "}", str(v))
    return path


def extract_first_step(tc: Dict[str, Any]) -> Dict[str, Any]:
    """Extract input_data from the first step of a test case.

    Designed for single-step execution now; multi-step is a future extension.
    """
    steps = tc.get("steps") or []
    if not steps:
        raise ValueError(f"{tc.get('test_id', '<no id>')} has no steps")
    input_data = steps[0].get("input_data")
    if input_data is None:
        raise ValueError(f"{tc.get('test_id', '<no id>')} step 1 missing input_data")
    return input_data


# ---------------------------------------------------------------------------
# Status checking
# ---------------------------------------------------------------------------

def check_expected_status(
    actual: int,
    expected_result: Dict[str, Any],
) -> tuple[bool, str]:
    """Check if the actual status code matches expectations.

    Returns (passed, expected_display_string).
    """
    any_of = expected_result.get("status_code_any_of")
    if any_of is not None:
        return actual in any_of, str(any_of)

    expected = expected_result.get("status_code")
    if expected is None:
        raise ValueError("expected_result has neither status_code nor status_code_any_of")

    return actual == expected, str(expected)


def should_skip(tc: Dict[str, Any], resp: requests.Response) -> bool:
    """Return True if this test should be marked SKIP rather than FAIL.

    Currently: 415 on upload/image endpoints (runner doesn't do multipart).
    """
    if resp.status_code != 415:
        return False
    path = (tc.get("path") or "").lower()
    title = (tc.get("title") or "").lower()
    keywords = ("upload", "image", "multipart")
    return any(kw in path or kw in title for kw in keywords)


# ---------------------------------------------------------------------------
# Single test execution
# ---------------------------------------------------------------------------

def run_one(
    tc: Dict[str, Any],
    base_url: str,
    auth_headers: Dict[str, str],
    timeout: int,
) -> Dict[str, Any]:
    """Execute a single test case and return a structured result dict."""
    test_id = tc.get("test_id", "<no id>")
    title = tc.get("title", "")
    method = tc.get("method", "GET")
    raw_path = tc.get("path", "/")

    result: Dict[str, Any] = {
        "test_id": test_id,
        "title": title,
        "method": method,
        "path": raw_path,
        "final_url": "",
        "expected_status": None,
        "expected_status_any_of": None,
        "actual_status": None,
        "outcome": "FAIL",
        "response_snippet": "",
        "response_body": "",
        "response_headers": {},
        "request_path_params": {},
        "request_query_params": {},
        "request_headers": {},
        "request_body": None,
        "error_message": "",
        "duration_ms": 0,
    }

    exp = tc.get("expected_result", {}) or {}
    result["expected_status"] = exp.get("status_code")
    result["expected_status_any_of"] = exp.get("status_code_any_of")
    is_auth_case = str(tc.get("category") or "").lower() == "auth"

    # Extract input data from first step
    try:
        input_data = extract_first_step(tc)
    except ValueError as e:
        result["error_message"] = str(e)
        return result

    path_params = input_data.get("path_params") or {}
    query_params = input_data.get("query_params") or {}
    headers = input_data.get("headers") or {}
    body = input_data.get("body")

    path = apply_path_params(raw_path, path_params)
    url = base_url.rstrip("/") + path
    result["final_url"] = url
    result["request_path_params"] = dict(path_params)
    result["request_query_params"] = dict(query_params)

    # Merge headers:
    # - apply global auth only for non-auth test categories
    # - always let test-specified headers override
    request_headers: Dict[str, str] = {}
    if not is_auth_case:
        request_headers.update(auth_headers)
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    request_headers.update(headers)
    result["request_headers"] = redact_headers(request_headers)
    result["request_body"] = body

    # Execute the request
    start = time.perf_counter()
    try:
        resp = requests.request(
            method=method,
            url=url,
            headers=request_headers,
            params=query_params,
            json=body if body is not None else None,
            timeout=timeout,
        )
    except requests.RequestException as e:
        elapsed = (time.perf_counter() - start) * 1000
        result["duration_ms"] = round(elapsed, 1)
        result["error_message"] = str(e)
        return result

    elapsed = (time.perf_counter() - start) * 1000
    result["duration_ms"] = round(elapsed, 1)
    result["actual_status"] = resp.status_code
    result["response_headers"] = dict(resp.headers)
    result["response_body"] = (resp.text or "").strip()

    # Response snippet (truncated)
    snippet = result["response_body"]
    if len(snippet) > 500:
        snippet = snippet[:500] + "..."
    result["response_snippet"] = snippet

    # Skip check (415 on upload-style endpoints)
    if should_skip(tc, resp):
        result["outcome"] = "SKIP"
        return result

    # Status check
    try:
        passed, _ = check_expected_status(resp.status_code, exp)
    except ValueError as e:
        result["error_message"] = str(e)
        return result

    result["outcome"] = "PASS" if passed else "FAIL"
    return result


# ---------------------------------------------------------------------------
# Suite execution
# ---------------------------------------------------------------------------

def run_suite(
    suite_data: Dict[str, Any],
    base_url: str,
    auth_headers: Dict[str, str],
    timeout: int,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Run all test cases in a suite. Returns (results, summary)."""
    test_cases = suite_data.get("test_cases", [])
    results: List[Dict[str, Any]] = []
    passed = failed = skipped = 0
    consecutive_401 = 0

    for tc in test_cases:
        result = run_one(tc, base_url, auth_headers, timeout)
        results.append(result)

        outcome = result["outcome"]
        label = result["test_id"]
        status = result.get("actual_status", "???")

        if outcome == "PASS":
            passed += 1
            consecutive_401 = 0
            print(f"  PASS  {label} ({status})")
        elif outcome == "SKIP":
            skipped += 1
            print(f"  SKIP  {label} ({status})")
        else:
            failed += 1
            exp_display = result["expected_status_any_of"] or result["expected_status"]
            print(f"  FAIL  {label} (expected {exp_display}, got {status})")
            if result["error_message"]:
                print(f"        Error: {result['error_message']}")
            if result["response_snippet"]:
                print(f"        Response: {result['response_snippet']}")

            # Detect auth wall: if first 3 requests all return 401 and
            # no auth was provided, stop early and tell the user.
            if result["actual_status"] == 401:
                consecutive_401 += 1
            else:
                consecutive_401 = 0

            if consecutive_401 >= 3 and not auth_headers:
                remaining = len(test_cases) - len(results)
                skipped += remaining
                print(f"\n  ** Server returned 401 Unauthorized - authentication required.")
                print(f"  ** Skipping remaining {remaining} tests.")
                print(f"  ** Re-run with authentication:")
                print(f"       --bearer-token <token>")
                print(f"       --api-key <key> [--api-key-header <header>]")
                print(f"  ** Or set environment variables:")
                print(f"       CONTRACTGUARD_BEARER_TOKEN")
                print(f"       CONTRACTGUARD_API_KEY\n")

                # Emit explicit SKIP results for all tests we didn't execute,
                # so UI/JSON consumers can reflect skipped status per test case.
                for pending_tc in test_cases[len(results):]:
                    pending_exp = pending_tc.get("expected_result", {}) or {}
                    skip_result = {
                        "test_id": pending_tc.get("test_id", "<no id>"),
                        "title": pending_tc.get("title", ""),
                        "method": pending_tc.get("method", "GET"),
                        "path": pending_tc.get("path", "/"),
                        "final_url": "",
                        "expected_status": pending_exp.get("status_code"),
                        "expected_status_any_of": pending_exp.get("status_code_any_of"),
                        "actual_status": None,
                        "outcome": "SKIP",
                        "response_snippet": "",
                        "response_body": "",
                        "response_headers": {},
                        "request_path_params": {},
                        "request_query_params": {},
                        "request_headers": {},
                        "request_body": None,
                        "error_message": "Skipped after repeated 401 Unauthorized without authentication.",
                        "duration_ms": 0,
                    }
                    results.append(skip_result)
                    print(f"  SKIP  {skip_result['test_id']} (auth wall)")

                break

    summary = {
        "total": len(test_cases),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
    }
    return results, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ContractGuard Test Runner - execute generated API test cases",
    )
    p.add_argument(
        "path",
        help="Path to a test JSON file or a directory containing test JSON files",
    )
    p.add_argument("--base-url", default=None, help="Target API base URL")
    p.add_argument("--timeout", type=int, default=10, help="Request timeout in seconds (default: 10)")
    p.add_argument("--json-output", default=None, help="Write structured results to a JSON file")
    p.add_argument("--bearer-token", default=None, help="Bearer token for Authorization header")
    p.add_argument("--api-key", default=None, help="API key value")
    p.add_argument("--api-key-header", default=None, help="Header name for API key (default: X-API-Key)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    test_files = collect_test_files(Path(args.path))
    auth_headers = resolve_auth_headers(args)

    all_results: List[Dict[str, Any]] = []
    total_passed = total_failed = total_skipped = total_count = 0

    for json_path in test_files:
        suite_data = load_test_suite(json_path)
        test_cases = suite_data.get("test_cases", [])
        if not test_cases:
            continue

        # Resolve base URL per suite (each JSON may have its own base_url)
        base_url = resolve_base_url(args.base_url, suite_data.get("base_url"))
        api_title = suite_data.get("api_title", json_path.stem)

        if not base_url:
            print(f"\n{'=' * 60}")
            print(f"  {api_title}  -- SKIPPED (no base URL)")
            print(f"  Provide --base-url or set CONTRACTGUARD_BASE_URL")
            print(f"{'=' * 60}")
            total_count += len(test_cases)
            total_skipped += len(test_cases)
            continue

        print(f"\n{'=' * 60}")
        print(f"  {api_title}  ({len(test_cases)} tests)")
        print(f"  Base URL: {base_url}")
        print(f"{'=' * 60}")

        results, summary = run_suite(suite_data, base_url, auth_headers, args.timeout)

        print(f"  {'-' * 56}")
        print(f"  {api_title}: {summary['passed']} passed, "
              f"{summary['failed']} failed, {summary['skipped']} skipped "
              f"/ {summary['total']} total")

        all_results.append({
            "source_file": str(json_path),
            "api_title": api_title,
            "base_url": base_url,
            "results": results,
            "summary": summary,
        })

        total_count += summary["total"]
        total_passed += summary["passed"]
        total_failed += summary["failed"]
        total_skipped += summary["skipped"]

    # Final summary
    print(f"\n{'=' * 60}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total:   {total_count}")
    print(f"  Passed:  {total_passed}")
    print(f"  Failed:  {total_failed}")
    print(f"  Skipped: {total_skipped}")
    print(f"{'=' * 60}")

    # Write JSON output if requested
    if args.json_output:
        output = {
            "summary": {
                "total": total_count,
                "passed": total_passed,
                "failed": total_failed,
                "skipped": total_skipped,
            },
            "suites": all_results,
        }
        Path(args.json_output).write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nResults written to {args.json_output}")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
