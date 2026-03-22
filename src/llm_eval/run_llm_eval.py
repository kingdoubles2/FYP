from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import yaml

# Allow running as a script: python src/llm_eval/run_llm_eval.py
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spec_parser.parser import parse_openapi
from test_generator.generator import generate_test_cases
from test_runner.run_test import resolve_base_url, run_suite
from llm_eval.ollama_client import OllamaClient


SENSITIVE_KEYS = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "apikey",
    "token",
}


def slugify(text: str) -> str:
    safe = []
    for ch in text:
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
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def load_prompt_template() -> str:
    template_path = Path(__file__).resolve().parent / "prompts" / "issue_prompt.txt"
    return read_text(template_path)


def load_config(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping.")
    return raw


def split_csv_args(values: Optional[List[str]]) -> Optional[List[str]]:
    if not values:
        return None
    items: List[str] = []
    for v in values:
        parts = [p.strip() for p in v.split(",") if p.strip()]
        items.extend(parts)
    return items or None


def coerce_list(value: Any, label: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    raise ValueError(f"{label} must be a list.")


def redact_sensitive(obj: Any) -> Any:
    if isinstance(obj, dict):
        redacted: Dict[str, Any] = {}
        for k, v in obj.items():
            key = str(k)
            if key.lower() in SENSITIVE_KEYS:
                redacted[key] = "<redacted>"
            else:
                redacted[key] = redact_sensitive(v)
        return redacted
    if isinstance(obj, list):
        return [redact_sensitive(v) for v in obj]
    return obj


def sanitize_auth_config(auth_cfg: Any) -> Any:
    if not isinstance(auth_cfg, dict):
        return auth_cfg
    cleaned: Dict[str, Any] = {}
    for k, v in auth_cfg.items():
        if isinstance(v, dict):
            cleaned_entry = dict(v)
            val = cleaned_entry.get("value")
            if isinstance(val, str) and not val.startswith("env:"):
                cleaned_entry["value"] = "<redacted>"
            cleaned[k] = cleaned_entry
        elif isinstance(v, str):
            cleaned[k] = v if v.startswith("env:") else "<redacted>"
        else:
            cleaned[k] = v
    return cleaned


def sanitize_config_for_meta(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(cfg)
    if "auth_by_base_url" in cleaned:
        cleaned["auth_by_base_url"] = sanitize_auth_config(cleaned["auth_by_base_url"])
    return cleaned


def resolve_auth_headers(base_url: str, auth_cfg: Any) -> Dict[str, str]:
    if not isinstance(auth_cfg, dict):
        return {}

    parsed = urlparse(base_url)
    host = parsed.hostname or base_url
    entry = None

    if host in auth_cfg:
        entry = auth_cfg[host]
    elif base_url in auth_cfg:
        entry = auth_cfg[base_url]

    if not entry:
        return {}

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
            return {}
        if auth_type == "api_key":
            header = str(entry.get("header") or "X-API-Key")
            return {header: value}
        return {"Authorization": f"Bearer {value}"}

    if isinstance(entry, str):
        value = resolve_value(entry)
        if not value:
            return {}
        return {"Authorization": f"Bearer {value}"}

    return {}


def build_prompt(template: str, context: Dict[str, Any], include_extra_tests: bool) -> str:
    directive = "YES" if include_extra_tests else "NO"
    context_json = json.dumps(context, indent=2, ensure_ascii=True)
    return (
        f"{template}\n"
        f"Include Suggested extra tests section: {directive}\n\n"
        f"```json\n{context_json}\n```\n"
    )


def issue_record(
    issues_dir: Path,
    prompt_template: str,
    client: OllamaClient,
    model: str,
    issue_id: str,
    context: Dict[str, Any],
    include_extra_tests: bool,
    announce: bool = True,
) -> Dict[str, Any]:
    ctx_path = issues_dir / f"{issue_id}_context.json"
    prompt_path = issues_dir / f"{issue_id}_prompt.txt"
    response_path = issues_dir / f"{issue_id}_response.md"

    write_json(ctx_path, context)
    prompt = build_prompt(prompt_template, context, include_extra_tests)
    write_text(prompt_path, prompt)

    if announce:
        print(f"    LLM: generating explanation for {issue_id} ...")
    response, err = client.generate(model=model, prompt=prompt)
    if announce:
        status = "ok" if not err else "error"
        print(f"    LLM: {issue_id} done ({status})")
    if err:
        response_text = f"LLM_ERROR: {err}"
    else:
        response_text = response or ""

    write_text(response_path, response_text)
    return {
        "issue_id": issue_id,
        "context_file": str(ctx_path),
        "prompt_file": str(prompt_path),
        "response_file": str(response_path),
        "llm_error": err,
    }


def make_issue_id(prefix: str, counter: int) -> str:
    return f"{slugify(prefix)}_{counter:03d}"


def collect_failed_contexts(
    spec_path: Path,
    base_url: str,
    suite_data: Dict[str, Any],
    results: List[Dict[str, Any]],
    limit: int = 25,
) -> Dict[str, Any]:
    by_id = {tc.get("test_id"): tc for tc in suite_data.get("test_cases", [])}
    failed = [r for r in results if r.get("outcome") == "FAIL"]
    trimmed = failed[:limit]
    items = []
    for r in trimmed:
        tc = by_id.get(r.get("test_id"), {})
        steps = tc.get("steps") or []
        input_data = {}
        if steps:
            input_data = steps[0].get("input_data") or {}
        items.append({
            "test_id": r.get("test_id"),
            "title": tc.get("title"),
            "category": tc.get("category"),
            "method": tc.get("method"),
            "path": tc.get("path"),
            "expected_result": tc.get("expected_result"),
            "input_data": redact_sensitive(input_data),
            "actual_status": r.get("actual_status"),
            "expected_status": r.get("expected_status"),
            "expected_status_any_of": r.get("expected_status_any_of"),
            "response_snippet": r.get("response_snippet"),
            "error_message": r.get("error_message"),
            "final_url": r.get("final_url"),
        })

    return {
        "issue_type": "suite_summary",
        "spec_path": str(spec_path),
        "base_url": base_url,
        "failed_count": len(failed),
        "included_failures": len(items),
        "truncated": len(failed) > len(items),
        "failures": items,
    }


def run_model(
    model: str,
    specs: List[str],
    config: Dict[str, Any],
    run_dir: Path,
    client: OllamaClient,
    prompt_template: str,
) -> Dict[str, Any]:
    model_slug = slugify(model)
    model_dir = run_dir / model_slug
    artifacts_dir = model_dir / "artifacts"
    issues_dir = model_dir / "issues"
    ir_dir = artifacts_dir / "ir"
    tests_dir = artifacts_dir / "tests"
    results_dir = artifacts_dir / "results"
    for d in (ir_dir, tests_dir, results_dir, issues_dir):
        d.mkdir(parents=True, exist_ok=True)

    llm_scope = str(config.get("llm_scope", "every_fail")).lower()
    include_extra_tests = bool(config.get("include_extra_tests", True))
    auth_cfg = config.get("auth_by_base_url") or {}
    runner_cfg = config.get("runner") or {}
    timeout_seconds = int(runner_cfg.get("timeout_seconds", 10))

    issue_counter = 1
    spec_summaries: List[Dict[str, Any]] = []
    issues_index: List[Dict[str, Any]] = []

    print(f"\n{'=' * 72}")
    print(f"MODEL: {model}")
    print(f"{'=' * 72}")

    for spec in specs:
        spec_path = Path(spec)
        print(f"\nSpec: {spec_path}")
        spec_record: Dict[str, Any] = {
            "spec_path": str(spec_path),
            "spec_hash": None,
            "parse_ok": False,
            "generate_ok": False,
            "run_ok": False,
            "summary": None,
            "issues": [],
        }

        try:
            spec_text = read_text(spec_path)
        except Exception as exc:  # noqa: BLE001
            issue_id = make_issue_id(f"{spec_path.stem}_read_error", issue_counter)
            issue_counter += 1
            context = {
                "issue_type": "read_error",
                "spec_path": str(spec_path),
                "error_message": str(exc),
            }
            print(f"  ERROR: unable to read spec ({exc})")
            issue = issue_record(
                issues_dir,
                prompt_template,
                client,
                model,
                issue_id,
                context,
                include_extra_tests,
            )
            spec_record["issues"].append(issue_id)
            issues_index.append(issue)
            spec_summaries.append(spec_record)
            continue

        spec_record["spec_hash"] = sha256_text(spec_text)

        try:
            parsed = parse_openapi(spec_text)
            spec_record["parse_ok"] = True
            print(f"  Parsed OK: {parsed.title} ({len(parsed.endpoints)} endpoints)")
        except Exception as exc:  # noqa: BLE001
            issue_id = make_issue_id(f"{spec_path.stem}_parse_error", issue_counter)
            issue_counter += 1
            context = {
                "issue_type": "parse_error",
                "spec_path": str(spec_path),
                "error_message": str(exc),
                "spec_text": spec_text,
            }
            print(f"  ERROR: parse failed ({exc})")
            issue = issue_record(
                issues_dir,
                prompt_template,
                client,
                model,
                issue_id,
                context,
                include_extra_tests,
            )
            spec_record["issues"].append(issue_id)
            issues_index.append(issue)
            spec_summaries.append(spec_record)
            continue

        ir_data = parsed.to_dict()
        ir_path = ir_dir / f"{spec_path.stem}_ir.json"
        write_json(ir_path, ir_data)

        try:
            suite = generate_test_cases(ir_data)
            spec_record["generate_ok"] = True
            print(f"  Generated tests: {len(suite.test_cases)}")
        except Exception as exc:  # noqa: BLE001
            issue_id = make_issue_id(f"{spec_path.stem}_generate_error", issue_counter)
            issue_counter += 1
            context = {
                "issue_type": "generate_error",
                "spec_path": str(spec_path),
                "error_message": str(exc),
                "ir_json": ir_data,
            }
            print(f"  ERROR: test generation failed ({exc})")
            issue = issue_record(
                issues_dir,
                prompt_template,
                client,
                model,
                issue_id,
                context,
                include_extra_tests,
            )
            spec_record["issues"].append(issue_id)
            issues_index.append(issue)
            spec_summaries.append(spec_record)
            continue

        suite_data = suite.to_dict()
        tests_path = tests_dir / f"{spec_path.stem}_tests.json"
        write_json(tests_path, suite_data)

        base_url = resolve_base_url(None, suite_data.get("base_url"))
        if not base_url:
            issue_id = make_issue_id(f"{spec_path.stem}_missing_base_url", issue_counter)
            issue_counter += 1
            context = {
                "issue_type": "missing_base_url",
                "spec_path": str(spec_path),
                "error_message": "No valid base_url found in generated test suite.",
            }
            print("  ERROR: missing base_url (suite skipped)")
            issue = issue_record(
                issues_dir,
                prompt_template,
                client,
                model,
                issue_id,
                context,
                include_extra_tests,
            )
            spec_record["issues"].append(issue_id)
            issues_index.append(issue)
            spec_record["summary"] = {
                "total": len(suite_data.get("test_cases", [])),
                "passed": 0,
                "failed": 0,
                "skipped": len(suite_data.get("test_cases", [])),
            }
            spec_summaries.append(spec_record)
            continue

        auth_headers = resolve_auth_headers(base_url, auth_cfg)
        print(f"  Running tests against: {base_url}")

        try:
            results, summary = run_suite(
                suite_data=suite_data,
                base_url=base_url,
                auth_headers=auth_headers,
                timeout=timeout_seconds,
            )
            spec_record["run_ok"] = True
        except Exception as exc:  # noqa: BLE001
            issue_id = make_issue_id(f"{spec_path.stem}_runner_error", issue_counter)
            issue_counter += 1
            context = {
                "issue_type": "runner_error",
                "spec_path": str(spec_path),
                "base_url": base_url,
                "error_message": str(exc),
            }
            print(f"  ERROR: runner failed ({exc})")
            issue = issue_record(
                issues_dir,
                prompt_template,
                client,
                model,
                issue_id,
                context,
                include_extra_tests,
            )
            spec_record["issues"].append(issue_id)
            issues_index.append(issue)
            spec_summaries.append(spec_record)
            continue

        results_path = results_dir / f"{spec_path.stem}_results.json"
        results_payload = {
            "api_title": suite_data.get("api_title"),
            "api_version": suite_data.get("api_version"),
            "base_url": base_url,
            "summary": summary,
            "results": results,
        }
        write_json(results_path, results_payload)
        spec_record["summary"] = summary
        print(
            f"  Summary: {summary.get('passed', 0)} passed, "
            f"{summary.get('failed', 0)} failed, "
            f"{summary.get('skipped', 0)} skipped / {summary.get('total', 0)} total"
        )

        if llm_scope == "every_fail":
            by_id = {tc.get("test_id"): tc for tc in suite_data.get("test_cases", [])}
            for res in results:
                if res.get("outcome") != "FAIL":
                    continue
                tc = by_id.get(res.get("test_id"), {})
                steps = tc.get("steps") or []
                input_data = steps[0].get("input_data") if steps else {}
                context = {
                    "issue_type": "test_fail",
                    "spec_path": str(spec_path),
                    "base_url": base_url,
                    "test_id": res.get("test_id"),
                    "title": tc.get("title"),
                    "category": tc.get("category"),
                    "method": tc.get("method"),
                    "path": tc.get("path"),
                    "expected_result": tc.get("expected_result"),
                    "input_data": redact_sensitive(input_data or {}),
                    "actual_status": res.get("actual_status"),
                    "expected_status": res.get("expected_status"),
                    "expected_status_any_of": res.get("expected_status_any_of"),
                    "response_snippet": res.get("response_snippet"),
                    "error_message": res.get("error_message"),
                    "final_url": res.get("final_url"),
                }
                issue_id = make_issue_id(f"{spec_path.stem}_{res.get('test_id')}", issue_counter)
                issue_counter += 1
                issue = issue_record(
                    issues_dir,
                    prompt_template,
                    client,
                    model,
                    issue_id,
                    context,
                    include_extra_tests,
                )
                spec_record["issues"].append(issue_id)
                issues_index.append(issue)

        elif llm_scope == "suite_summary" and summary.get("failed", 0) > 0:
            context = collect_failed_contexts(spec_path, base_url, suite_data, results)
            issue_id = make_issue_id(f"{spec_path.stem}_suite_summary", issue_counter)
            issue_counter += 1
            issue = issue_record(
                issues_dir,
                prompt_template,
                client,
                model,
                issue_id,
                context,
                include_extra_tests,
            )
            spec_record["issues"].append(issue_id)
            issues_index.append(issue)

        spec_summaries.append(spec_record)

    totals = {"total": 0, "passed": 0, "failed": 0, "skipped": 0}
    for spec_sum in spec_summaries:
        summary = spec_sum.get("summary") or {}
        totals["total"] += int(summary.get("total", 0))
        totals["passed"] += int(summary.get("passed", 0))
        totals["failed"] += int(summary.get("failed", 0))
        totals["skipped"] += int(summary.get("skipped", 0))

    summary_json = {
        "model": model,
        "totals": totals,
        "specs": spec_summaries,
        "issues": [i["issue_id"] for i in issues_index],
    }
    write_json(model_dir / "summary.json", summary_json)

    summary_md_lines = [
        "# LLM Eval Summary",
        f"Model: {model}",
        "",
        "Totals:",
        f"Total: {totals['total']}",
        f"Passed: {totals['passed']}",
        f"Failed: {totals['failed']}",
        f"Skipped: {totals['skipped']}",
        "",
        "Specs:",
    ]
    for spec_sum in spec_summaries:
        summ = spec_sum.get("summary") or {}
        line = (
            f"{spec_sum['spec_path']} | "
            f"parse_ok={spec_sum['parse_ok']} "
            f"generate_ok={spec_sum['generate_ok']} "
            f"run_ok={spec_sum['run_ok']} "
            f"passed={summ.get('passed', 0)} "
            f"failed={summ.get('failed', 0)} "
            f"skipped={summ.get('skipped', 0)} "
            f"issues={len(spec_sum.get('issues', []))}"
        )
        summary_md_lines.append(line)

    write_text(model_dir / "summary.md", "\n".join(summary_md_lines) + "\n")
    print(
        f"\nModel totals: {totals['passed']} passed, {totals['failed']} failed, "
        f"{totals['skipped']} skipped / {totals['total']} total"
    )

    return {
        "model": model,
        "model_dir": str(model_dir),
        "totals": totals,
        "issues": issues_index,
        "specs": spec_summaries,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run LLM-backed evaluation on ContractGuard specs.")
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
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=None,
        help="Override Ollama timeout seconds",
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
        ollama_cfg = config.get("ollama") or {}
        ollama_cfg["timeout_seconds"] = int(args.llm_timeout)
        config["ollama"] = ollama_cfg

    models = coerce_list(config.get("models"), "models")
    specs = coerce_list(config.get("specs"), "specs")
    if not models or not specs:
        print("ERROR: Config must include non-empty 'models' and 'specs'.", file=sys.stderr)
        return 1

    output_dir = Path(config.get("output_dir") or "build/llm_runs")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    print(f"Models: {', '.join(models)}")
    print(f"Specs: {len(specs)}")

    ollama_cfg = config.get("ollama") or {}
    client = OllamaClient(
        base_url=str(ollama_cfg.get("base_url") or "http://localhost:11434"),
        timeout_seconds=int(ollama_cfg.get("timeout_seconds", 120)),
    )
    prompt_template = load_prompt_template()

    run_start = datetime.now(timezone.utc).isoformat()
    model_results = []
    for model in models:
        model_results.append(run_model(model, specs, config, run_dir, client, prompt_template))
    run_end = datetime.now(timezone.utc).isoformat()

    run_meta = {
        "run_id": timestamp,
        "started_at": run_start,
        "finished_at": run_end,
        "models": models,
        "specs": specs,
        "output_dir": str(output_dir),
        "config": sanitize_config_for_meta(config),
        "model_results": [
            {
                "model": r["model"],
                "model_dir": r["model_dir"],
                "totals": r["totals"],
            }
            for r in model_results
        ],
    }
    write_json(run_dir / "run_meta.json", run_meta)

    print(f"Run complete. Results in: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
