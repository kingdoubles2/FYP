from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PLACEHOLDER_PATTERN = re.compile(r"\{([^{}]+)\}")
ENV_PREFIX = "CONTRACTGUARD_MAP_"
RUNTIME_MAPPING_JSON_ENV = "CONTRACTGUARD_RUNTIME_MAPPING_JSON"
GENERATOR_PLACEHOLDER_VALUES = {
    "standard-text",
    "standard_text",
    "jane_doe",
    "john_doe",
    "sample",
    "sample-text",
    "sample_text",
    "example",
    "example-text",
    "example_text",
    "string",
    "text",
}


def collect_test_files(path: Path) -> List[Path]:
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


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_param_map(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    normalized: Dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key).strip().lower()
        if key_text:
            normalized[key_text] = item
    return normalized


def normalize_mapping(raw: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    normalized: Dict[str, Dict[str, Dict[str, Any]]] = {
        "globals": {},
        "suites": {},
        "tests": {},
    }

    globals_raw = raw.get("globals")
    if isinstance(globals_raw, dict):
        normalized["globals"] = normalize_param_map(globals_raw)

    suites_raw = raw.get("suites")
    if isinstance(suites_raw, dict):
        suites_normalized: Dict[str, Dict[str, Any]] = {}
        for suite_key, suite_map in suites_raw.items():
            key_text = str(suite_key).strip()
            if not key_text:
                continue
            suites_normalized[key_text] = normalize_param_map(suite_map)
        normalized["suites"] = suites_normalized

    tests_raw = raw.get("tests")
    if isinstance(tests_raw, dict):
        tests_normalized: Dict[str, Dict[str, Any]] = {}
        for test_key, test_map in tests_raw.items():
            key_text = str(test_key).strip()
            if not key_text:
                continue
            tests_normalized[key_text] = normalize_param_map(test_map)
        normalized["tests"] = tests_normalized

    return normalized


def merge_mapping_sections(
    base: Dict[str, Dict[str, Dict[str, Any]]],
    overlay: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    merged = {
        "globals": dict(base.get("globals") or {}),
        "suites": dict(base.get("suites") or {}),
        "tests": dict(base.get("tests") or {}),
    }

    globals_overlay = overlay.get("globals") if isinstance(overlay.get("globals"), dict) else {}
    merged["globals"].update(globals_overlay)

    for section in ("suites", "tests"):
        section_overlay = overlay.get(section) if isinstance(overlay.get(section), dict) else {}
        for key, value in section_overlay.items():
            existing = merged[section].get(key)
            if isinstance(existing, dict) and isinstance(value, dict):
                merged[section][key] = {**existing, **value}
            else:
                merged[section][key] = value

    return merged


def load_mapping(mapping_path: Path | None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    base_raw: Dict[str, Any] = {
        "globals": {},
        "suites": {},
        "tests": {},
    }

    if mapping_path is not None:
        if mapping_path.exists():
            base_raw = load_json(mapping_path)
        else:
            print(f"WARNING: mapping file not found, using env-only mapping: {mapping_path}")

    mapping = normalize_mapping(base_raw)

    secret_mapping_raw = os.environ.get(RUNTIME_MAPPING_JSON_ENV, "").strip()
    if secret_mapping_raw:
        try:
            parsed = json.loads(secret_mapping_raw)
            if isinstance(parsed, dict):
                mapping = merge_mapping_sections(mapping, normalize_mapping(parsed))
            else:
                print(f"WARNING: {RUNTIME_MAPPING_JSON_ENV} is not a JSON object; ignoring")
        except Exception as exc:
            print(f"WARNING: failed to parse {RUNTIME_MAPPING_JSON_ENV}: {exc}")

    return mapping


def env_mapping_overrides() -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for name, value in os.environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        map_key = name[len(ENV_PREFIX):].strip().lower()
        if map_key and value.strip():
            overrides[map_key] = value.strip()
    return overrides


def is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def is_placeholder_value(value: Any, key: str) -> bool:
    if is_missing_value(value):
        return True
    if not isinstance(value, str):
        return False

    normalized = value.strip().lower()
    key_norm = key.strip().lower()
    candidates = {
        key_norm,
        "{" + key_norm + "}",
        "<" + key_norm + ">",
        ":" + key_norm,
        "placeholder",
        "replace_me",
        "changeme",
        "todo",
    }
    if normalized in candidates:
        return True
    if normalized in GENERATOR_PLACEHOLDER_VALUES:
        return True
    return False


def to_mapping_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def resolve_mapping_value(
    key: str,
    *,
    env_overrides: Dict[str, str],
    test_map: Dict[str, Any],
    suite_map: Dict[str, Any],
    global_map: Dict[str, Any],
) -> Any:
    key_norm = key.lower()
    if key_norm in env_overrides:
        return env_overrides[key_norm]
    if key_norm in test_map and not is_missing_value(test_map.get(key_norm)):
        return test_map.get(key_norm)
    if key_norm in suite_map and not is_missing_value(suite_map.get(key_norm)):
        return suite_map.get(key_norm)
    if key_norm in global_map and not is_missing_value(global_map.get(key_norm)):
        return global_map.get(key_norm)
    return None


def extract_required_path_keys(path_template: str) -> List[str]:
    if not isinstance(path_template, str):
        return []
    return [match.group(1).strip() for match in PLACEHOLDER_PATTERN.finditer(path_template) if match.group(1).strip()]


def first_step_input_data(test_case: Dict[str, Any]) -> Dict[str, Any] | None:
    steps = test_case.get("steps")
    if not isinstance(steps, list) or not steps:
        return None
    first = steps[0]
    if not isinstance(first, dict):
        return None
    payload = first.get("input_data")
    if not isinstance(payload, dict):
        return None
    return payload


def normalize_case_path_params(input_data: Dict[str, Any]) -> Dict[str, Any]:
    current = input_data.get("path_params")
    if isinstance(current, dict):
        return dict(current)
    return {}


def process_suite(
    suite_data: Dict[str, Any],
    *,
    suite_file_name: str,
    mapping: Dict[str, Dict[str, Dict[str, Any]]],
    env_overrides: Dict[str, str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    api_title = str(suite_data.get("api_title") or "").strip()
    tests = suite_data.get("test_cases")
    if not isinstance(tests, list):
        tests = []

    globals_map = to_mapping_dict(mapping.get("globals"))
    suites_map = to_mapping_dict(mapping.get("suites"))
    tests_map = to_mapping_dict(mapping.get("tests"))

    suite_map = to_mapping_dict(suites_map.get(api_title))
    if not suite_map:
        suite_map = to_mapping_dict(suites_map.get(suite_file_name))

    runnable_cases: List[Dict[str, Any]] = []
    skipped_cases: List[Dict[str, Any]] = []

    for test_case in tests:
        if not isinstance(test_case, dict):
            continue

        test_id = str(test_case.get("test_id") or "").strip()
        path_template = str(test_case.get("path") or "")
        required_keys = extract_required_path_keys(path_template)

        payload = first_step_input_data(test_case)
        if payload is None:
            skipped_cases.append(
                {
                    "test_id": test_id,
                    "reason": "MISSING_INPUT_DATA",
                    "missing_keys": [],
                }
            )
            continue

        case_map = to_mapping_dict(tests_map.get(test_id))
        path_params = normalize_case_path_params(payload)

        unresolved_keys: List[str] = []
        for key in required_keys:
            current_value = path_params.get(key)
            if is_placeholder_value(current_value, key):
                replacement = resolve_mapping_value(
                    key,
                    env_overrides=env_overrides,
                    test_map=case_map,
                    suite_map=suite_map,
                    global_map=globals_map,
                )
                if not is_missing_value(replacement):
                    path_params[key] = replacement
                    current_value = replacement

            if is_placeholder_value(current_value, key):
                unresolved_keys.append(key)

        if unresolved_keys:
            skipped_cases.append(
                {
                    "test_id": test_id,
                    "reason": "MISSING_MAPPING",
                    "missing_keys": unresolved_keys,
                }
            )
            continue

        payload["path_params"] = path_params
        runnable_cases.append(test_case)

    updated = dict(suite_data)
    updated["test_cases"] = runnable_cases
    return updated, skipped_cases


def build_skip_report(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_skipped = sum(int(item.get("skipped_count") or 0) for item in entries)
    return {
        "summary": {
            "suite_count": len(entries),
            "total_skipped": total_skipped,
        },
        "suites": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inject runtime mapping into generated tests and skip unresolved cases.",
    )
    parser.add_argument("input", help="Input test JSON file or directory")
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Output directory for runtime-ready test JSON files",
    )
    parser.add_argument(
        "--mapping",
        default="config/runtime_mapping.json",
        help="Path to mapping JSON file (default: config/runtime_mapping.json)",
    )
    parser.add_argument(
        "--skip-report",
        default="build/results/runtime-skip-report.json",
        help="Output JSON file for skipped-case report",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    mapping_path = Path(args.mapping) if args.mapping else None
    skip_report_path = Path(args.skip_report)

    mapping = load_mapping(mapping_path)
    env_overrides = env_mapping_overrides()

    files = collect_test_files(input_path)
    suite_reports: List[Dict[str, Any]] = []

    for file_path in files:
        suite_data = load_json(file_path)
        updated_suite, skipped_cases = process_suite(
            suite_data,
            suite_file_name=file_path.name,
            mapping=mapping,
            env_overrides=env_overrides,
        )

        out_path = output_dir / file_path.name
        save_json(out_path, updated_suite)

        skipped_count = len(skipped_cases)
        runnable_count = len(updated_suite.get("test_cases") or [])
        api_title = str(updated_suite.get("api_title") or file_path.stem)

        print(
            f"  {file_path} -> {out_path} "
            f"(runnable={runnable_count}, skipped={skipped_count})"
        )

        suite_reports.append(
            {
                "source_file": str(file_path),
                "output_file": str(out_path),
                "api_title": api_title,
                "runnable_count": runnable_count,
                "skipped_count": skipped_count,
                "skipped_cases": skipped_cases,
            }
        )

    save_json(skip_report_path, build_skip_report(suite_reports))
    print(f"\nSkip report written to {skip_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
