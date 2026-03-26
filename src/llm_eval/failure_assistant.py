from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from llm_eval import run_llm_eval as core
from llm_eval.ollama_client import OllamaClient


def _safe_int(value: Any, default: int, *, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return "{}"


def safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def sha256_text(text: str) -> str:
    return core.sha256_text(text)


def parse_spec_document(spec_text: str) -> Dict[str, Any]:
    return core.load_spec_document(spec_text)


def load_backend_llm_settings(config_path: Optional[str] = None) -> Dict[str, Any]:
    config_override = config_path or os.getenv("CONTRACTGUARD_LLM_CONFIG")
    path = Path(config_override or "config/llm_eval.yaml")
    config = _load_yaml_mapping(path) if path.exists() else {}

    llm_cfg = config.get("llm") or {}
    ollama_cfg = config.get("ollama") or {}
    evidence_cfg = config.get("evidence") or {}
    models = config.get("models") if isinstance(config.get("models"), list) else []

    model = os.getenv("CONTRACTGUARD_LLM_MODEL") or (str(models[0]) if models else "kimi-k2.5:cloud")
    base_url = (
        os.getenv("CONTRACTGUARD_OLLAMA_BASE_URL")
        or os.getenv("OLLAMA_BASE_URL")
        or str(ollama_cfg.get("base_url") or "http://localhost:11434")
    )
    timeout_seconds = _safe_int(
        os.getenv("CONTRACTGUARD_LLM_TIMEOUT_SECONDS", llm_cfg.get("timeout_seconds", 120)),
        120,
        minimum=5,
        maximum=600,
    )
    word_target = _safe_int(
        os.getenv("CONTRACTGUARD_LLM_WORD_TARGET", llm_cfg.get("output_word_target", 65)),
        65,
        minimum=30,
        maximum=220,
    )
    default_word_max = max(55, _safe_int(llm_cfg.get("output_word_max", word_target + 20), word_target + 20))
    word_max = _safe_int(
        os.getenv("CONTRACTGUARD_LLM_WORD_MAX", default_word_max),
        default_word_max,
        minimum=max(50, word_target),
        maximum=320,
    )
    default_word_min = min(
        word_max,
        max(20, _safe_int(llm_cfg.get("output_word_min", max(35, word_target - 12)), max(35, word_target - 12))),
    )
    word_min = _safe_int(
        os.getenv("CONTRACTGUARD_LLM_WORD_MIN", default_word_min),
        default_word_min,
        minimum=20,
        maximum=word_max,
    )
    word_target = min(max(word_target, word_min), word_max)
    retry_invalid_output = _safe_int(
        os.getenv("CONTRACTGUARD_LLM_RETRY_INVALID", llm_cfg.get("retry_invalid_output", 1)),
        1,
        minimum=0,
        maximum=5,
    )
    max_items_per_section = _safe_int(
        os.getenv("CONTRACTGUARD_LLM_MAX_ITEMS", evidence_cfg.get("max_items_per_section", 12)),
        12,
        minimum=2,
        maximum=60,
    )
    snippet_chars = _safe_int(
        os.getenv("CONTRACTGUARD_LLM_SNIPPET_CHARS", evidence_cfg.get("snippet_chars", 300)),
        300,
        minimum=60,
        maximum=3000,
    )
    max_similar_failures = _safe_int(
        os.getenv("CONTRACTGUARD_LLM_MAX_SIMILAR_FAILURES", evidence_cfg.get("max_similar_failures", 5)),
        5,
        minimum=0,
        maximum=50,
    )
    max_full_context_items = _safe_int(
        os.getenv("CONTRACTGUARD_LLM_MAX_FULL_CONTEXT_ITEMS", 80),
        80,
        minimum=10,
        maximum=500,
    )
    ollama_options = ollama_cfg.get("options") if isinstance(ollama_cfg.get("options"), dict) else {}

    return {
        "model": str(model),
        "base_url": str(base_url),
        "timeout_seconds": timeout_seconds,
        "word_target": word_target,
        "word_min": min(word_min, word_max),
        "word_max": word_max,
        "retry_invalid_output": retry_invalid_output,
        "max_items_per_section": max_items_per_section,
        "snippet_chars": snippet_chars,
        "max_similar_failures": max_similar_failures,
        "max_full_context_items": max_full_context_items,
        "ollama_options": ollama_options,
    }


def build_case_evidence(
    *,
    selection_reason: str,
    spec_id: str,
    spec_path: Path,
    spec_hash: str,
    spec_doc: Dict[str, Any],
    ir_data: Dict[str, Any],
    test_case: Dict[str, Any],
    case_result: Dict[str, Any],
    all_results: List[Dict[str, Any]],
    auth_meta: Dict[str, Any],
    timeout_seconds: int,
    max_items: int,
    snippet_chars: int,
    max_similar_failures: int,
) -> Dict[str, Any]:
    return core.build_case_evidence_v2(
        selection_reason=selection_reason,
        spec_id=spec_id,
        spec_path=spec_path,
        spec_hash=spec_hash,
        spec_doc=spec_doc,
        ir_data=ir_data,
        tc=test_case,
        res=case_result,
        results=all_results,
        auth_meta=auth_meta,
        timeout_seconds=timeout_seconds,
        max_items=max_items,
        snippet_chars=snippet_chars,
        max_similar_failures=max_similar_failures,
    )


def build_pipeline_context(
    *,
    suite_data: Dict[str, Any],
    all_results: List[Dict[str, Any]],
    test_case: Dict[str, Any],
    case_result: Dict[str, Any],
    spec_doc: Dict[str, Any],
    ir_data: Dict[str, Any],
    max_full_items: int,
) -> Dict[str, Any]:
    method = str(test_case.get("method") or "")
    path = str(test_case.get("path") or "")
    suite_cases = list(suite_data.get("test_cases") or [])
    endpoint_cases = [case for case in suite_cases if case.get("method") == method and case.get("path") == path]
    endpoint_results = [row for row in all_results if row.get("method") == method and row.get("path") == path]
    path_item, operation = core.get_operation(spec_doc, method, path)
    endpoint_ir = core.find_endpoint_ir(ir_data, method, path)

    return {
        "test_case_full": core.redact_sensitive(deepcopy(test_case)),
        "tests_for_endpoint": core.redact_sensitive(deepcopy(endpoint_cases)),
        "tests_full": core.redact_sensitive(deepcopy(suite_cases[:max_full_items])),
        "tests_full_total": len(suite_cases),
        "tests_full_truncated": len(suite_cases) > max_full_items,
        "results_for_endpoint": core.redact_sensitive(deepcopy(endpoint_results)),
        "results_full": core.redact_sensitive(deepcopy(all_results[:max_full_items])),
        "results_full_total": len(all_results),
        "results_full_truncated": len(all_results) > max_full_items,
        "ir_full": {
            "api_title": ir_data.get("title"),
            "api_version": ir_data.get("version"),
            "base_url": ir_data.get("base_url"),
            "endpoint_count": len(ir_data.get("endpoints") or []),
            "endpoint_for_case": core.redact_sensitive(deepcopy(endpoint_ir)),
        },
        "spec_full": {
            "info": core.redact_sensitive(deepcopy(spec_doc.get("info") or {})),
            "servers": core.redact_sensitive(deepcopy(spec_doc.get("servers") or [])),
            "path_item": core.redact_sensitive(deepcopy(path_item)),
            "operation": core.redact_sensitive(deepcopy(operation)),
        },
        "execution": {
            "result_full": core.redact_sensitive(deepcopy(case_result)),
            "same_endpoint_summary": core.summarize_endpoint_results(all_results, method, path),
        },
    }


def build_assistant_prompt_bundle(
    *,
    evidence: Dict[str, Any],
    pipeline_context: Dict[str, Any],
    max_items_per_section: int,
) -> Dict[str, Any]:
    return {
        "case_evidence": core.build_prompt_evidence_view(evidence, max_items=max_items_per_section),
        "pipeline_context": pipeline_context,
    }


def classify_failure_signal(evidence: Dict[str, Any]) -> str:
    signal_info = core.classify_failure_signal(evidence)
    signal = str(signal_info.get("signal") or "status_mismatch")
    if signal in {"transport", "auth", "schema_type", "schema_value", "missing_required", "validation"}:
        return signal

    execution = evidence.get("execution") or {}
    response = execution.get("response_received") or {}
    assertion_failures = execution.get("assertion_failures") or []
    status = response.get("status")
    if status is None and assertion_failures:
        first = assertion_failures[0] if isinstance(assertion_failures[0], dict) else {}
        status = first.get("actual_status")
    status_int = int(status) if str(status).isdigit() else None
    reason = _extract_primary_response_reason(evidence).lower()
    auth_tokens = (
        "unauthorized",
        "requires authentication",
        "authentication required",
        "bad credentials",
        "invalid token",
        "token",
        "bearer",
        "api key",
        "forbidden",
    )

    if status_int in {401, 403}:
        return "auth"
    if any(token in reason for token in auth_tokens):
        return "auth"
    if status_int in {400, 422}:
        return "validation"
    return signal


def is_external_failure(evidence: Dict[str, Any]) -> bool:
    return classify_failure_signal(evidence) in {"transport", "auth"}


def _truncate_words(text: str, max_words: int) -> str:
    return core.truncate_words(str(text or ""), max_words)


def _word_count(text: str) -> int:
    return core.count_words(str(text or ""))


def _extract_expected_actual_tokens(evidence: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    execution = evidence.get("execution") or {}
    assertion_failures = execution.get("assertion_failures") or []
    expected_tokens: List[str] = []
    actual_tokens: List[str] = []
    if assertion_failures:
        first = assertion_failures[0] if isinstance(assertion_failures[0], dict) else {}
        expected = first.get("expected")
        actual = first.get("actual_status")
        if isinstance(expected, list):
            expected_tokens.extend([str(item) for item in expected if item is not None])
        elif expected is not None:
            expected_tokens.append(str(expected))
        if actual is not None:
            actual_tokens.append(str(actual))
        err = str(first.get("actual_error") or "").strip()
        if err:
            actual_tokens.append(err)

    response = execution.get("response_received") or {}
    status = response.get("status")
    if status is not None:
        actual_tokens.append(str(status))
    snippet = str(response.get("body_snippet") or "").strip()
    if snippet:
        actual_tokens.append(snippet)

    return expected_tokens, actual_tokens


def _extract_input_tokens(evidence: Dict[str, Any]) -> List[str]:
    test_context = evidence.get("test_context") or {}
    generated_input = test_context.get("generated_input") or {}
    tokens: List[str] = []
    for section in ("path_params", "query_params", "headers"):
        section_obj = generated_input.get(section) or {}
        if isinstance(section_obj, dict):
            for key, value in section_obj.items():
                if value is None:
                    continue
                value_text = str(value).strip()
                if value_text and value_text.lower() != "<redacted>":
                    tokens.append(f"{key}={value_text}")
    body = generated_input.get("body")
    if isinstance(body, dict):
        for key, value in list(body.items())[:5]:
            if value is None:
                continue
            value_text = str(value).strip()
            if value_text:
                tokens.append(f"{key}={value_text}")
    elif body is not None:
        body_text = str(body).strip()
        if body_text:
            tokens.append(body_text)
    return tokens


def _extract_primary_response_reason(evidence: Dict[str, Any]) -> str:
    execution = evidence.get("execution") or {}
    snippet = str((execution.get("response_received") or {}).get("body_snippet") or "").strip()
    if snippet:
        parsed = safe_json_loads(snippet)
        if isinstance(parsed, dict):
            for key in ("message", "detail", "error", "reason", "title", "description"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    return core.truncate_chars(value.strip(), 170)
        return core.truncate_chars(snippet, 170)

    assertion_failures = execution.get("assertion_failures") or []
    if assertion_failures:
        first = assertion_failures[0] if isinstance(assertion_failures[0], dict) else {}
        actual_error = str(first.get("actual_error") or "").strip()
        if actual_error:
            return core.truncate_chars(actual_error, 170)
    return ""


def _request_has_auth_material(evidence: Dict[str, Any]) -> bool:
    execution = evidence.get("execution") or {}
    request_sent = execution.get("request_sent") or {}
    headers = request_sent.get("headers") or {}
    if not isinstance(headers, dict):
        return False
    auth_keys = {"authorization", "x-api-key", "api-key", "x_auth_token", "x-auth-token"}
    return any(str(key).strip().lower() in auth_keys for key in headers.keys())


def _signal_keywords(signal: str) -> List[str]:
    keyword_map = {
        "transport": ["transport", "network", "tls", "ssl", "connection", "timeout"],
        "auth": ["auth", "token", "bearer", "api key", "unauthorized", "forbidden"],
        "schema_type": ["type", "schema", "invalid", "validation", "field"],
        "schema_value": ["enum", "format", "invalid", "schema", "parameter"],
        "missing_required": ["required", "missing", "parameter", "body", "field"],
        "validation": ["validation", "schema", "invalid", "request", "constraint"],
        "status_mismatch": ["status", "endpoint", "contract", "behavior"],
    }
    return keyword_map.get(str(signal), keyword_map["status_mismatch"])


def _get_test_id(evidence: Dict[str, Any]) -> str:
    return str((evidence.get("test_context") or {}).get("test_id") or "unknown")


def _status_strings(evidence: Dict[str, Any]) -> Tuple[str, str]:
    expected_tokens, actual_tokens = _extract_expected_actual_tokens(evidence)
    expected_text = ", ".join(expected_tokens[:2]) if expected_tokens else "contract-defined status"
    actual_text = actual_tokens[0] if actual_tokens else "no concrete status"
    return expected_text, actual_text


def _build_diagnostic_triplet(evidence: Dict[str, Any], signal: str) -> Tuple[str, str, str]:
    operation = (evidence.get("spec") or {}).get("operation") or {}
    method = str(operation.get("method") or "REQUEST").upper()
    path = str(operation.get("path") or "/")
    expected_text, actual_text = _status_strings(evidence)
    reason = _extract_primary_response_reason(evidence)
    reason_clause = f" The response reason was '{reason}'." if reason else ""
    has_auth = _request_has_auth_material(evidence)

    if signal == "auth":
        if has_auth:
            cause = "Authentication credentials were provided but rejected by the API."
            why = (
                f"{method} {path} expected {expected_text} but returned {actual_text}.{reason_clause} "
                "This pattern usually means an invalid, expired, or insufficient-scope token."
            )
            check_next = (
                "Verify token validity/scope for this endpoint and regenerate credentials if needed, then rerun the same case."
            )
        else:
            cause = "The request was sent without required authentication credentials."
            why = (
                f"{method} {path} expected {expected_text} but returned {actual_text}.{reason_clause} "
                "Request headers show no Authorization or API-key material."
            )
            check_next = (
                "Add a valid Authorization bearer token (or required API-key header) in run auth settings and rerun this test."
            )
        return cause, re.sub(r"\s+", " ", why).strip(), check_next

    if signal == "transport":
        cause = "The request likely failed before endpoint business logic due to transport/connectivity issues."
        why = f"{method} {path} did not complete a normal contract response path.{reason_clause}"
        check_next = "Validate runner network/TLS reachability to the target host, then rerun the identical payload."
        return cause, re.sub(r"\s+", " ", why).strip(), check_next

    if signal in {"schema_type", "schema_value", "missing_required", "validation"}:
        cause = "The request appears to violate endpoint validation constraints for this contract."
        why = (
            f"{method} {path} expected {expected_text} but observed {actual_text}.{reason_clause} "
            "This aligns with request-shape/value validation handling."
        )
        check_next = "Compare the generated input against schema constraints and update either input generation or expected status accordingly."
        return cause, re.sub(r"\s+", " ", why).strip(), check_next

    cause = "The expected contract outcome does not match current endpoint behavior."
    why = f"{method} {path} expected {expected_text} but observed {actual_text}.{reason_clause}"
    check_next = "Confirm the endpoint's intended status behavior for this scenario and align test expectation with the authoritative contract."
    return cause, re.sub(r"\s+", " ", why).strip(), check_next


def _format_structured_explanation(*, test_id: str, cause: str, why_likely: str, check_next: str) -> str:
    return (
        f"{test_id} (fail): "
        f"Likely cause: {str(cause).strip()} "
        f"Why likely: {str(why_likely).strip()} "
        f"Check next: {str(check_next).strip()}"
    )


def _build_grounding_tail(evidence: Dict[str, Any]) -> str:
    expected_tokens, actual_tokens = _extract_expected_actual_tokens(evidence)
    operation = (evidence.get("spec") or {}).get("operation") or {}
    method = str(operation.get("method") or "REQUEST").upper()
    path = str(operation.get("path") or "/")
    expected_text = ", ".join(expected_tokens[:2]) if expected_tokens else "the contract status"
    actual_text = actual_tokens[0] if actual_tokens else "an unknown status/error"
    reason = _extract_primary_response_reason(evidence)
    if reason:
        return (
            f"Observed {method} {path} expected {expected_text} but got {actual_text}, "
            f"with runner evidence '{reason}'."
        )
    return f"Observed {method} {path} expected {expected_text} but got {actual_text}."


def _contains_direct_reference(text: str, evidence: Dict[str, Any]) -> bool:
    lowered = str(text or "").lower()
    if not lowered:
        return False

    expected_tokens, actual_tokens = _extract_expected_actual_tokens(evidence)
    input_tokens = _extract_input_tokens(evidence)
    spec_op = (evidence.get("spec") or {}).get("operation") or {}
    method = str(spec_op.get("method") or "").lower()
    path = str(spec_op.get("path") or "").lower()
    reason = _extract_primary_response_reason(evidence).lower()
    signal_keywords = [str(keyword).strip().lower() for keyword in _signal_keywords(classify_failure_signal(evidence))]

    status_hit = any(str(token).lower() in lowered for token in (expected_tokens + actual_tokens) if token)
    input_hit = any(str(token).lower() in lowered for token in input_tokens if token)
    op_hit = bool(method and re.search(rf"\b{re.escape(method)}\b", lowered)) or bool(path and path in lowered)

    reason_hit = False
    if reason:
        reason_tokens = [part for part in re.split(r"[^a-z0-9]+", reason) if len(part) >= 5]
        reason_hit = reason in lowered or any(token in lowered for token in reason_tokens[:6])

    signal_hit = any(keyword in lowered for keyword in signal_keywords[:6])
    grounding_hits = sum([status_hit, input_hit, op_hit, reason_hit, signal_hit])

    if status_hit and (input_hit or op_hit or reason_hit or signal_hit):
        return True
    return grounding_hits >= 3


def _build_default_explanation(evidence: Dict[str, Any], word_min: int, word_max: int) -> str:
    signal = classify_failure_signal(evidence)
    test_id = _get_test_id(evidence)
    cause, why_likely, check_next = _build_diagnostic_triplet(evidence, signal)
    draft = _format_structured_explanation(
        test_id=test_id,
        cause=cause,
        why_likely=why_likely,
        check_next=check_next,
    )
    text = re.sub(r"\s+", " ", draft).strip()
    if word_max > 0 and _word_count(text) > word_max:
        text = _truncate_words(text, word_max)
    structured = "likely cause:" in text.lower() and "why likely:" in text.lower() and "check next:" in text.lower()
    if word_min > 0 and _word_count(text) < word_min and not structured:
        grounding_tail = _build_grounding_tail(evidence)
        text = f"{text.rstrip('.')} {grounding_tail}"
        if word_max > 0 and _word_count(text) > word_max:
            text = _truncate_words(text, word_max)
    return text


def _normalize_explanation_text(
    raw_text: str,
    *,
    evidence: Dict[str, Any],
    word_min: int,
    word_max: int,
) -> str:
    text = re.sub(r"\s+", " ", str(raw_text or "")).strip()
    if text.startswith('"') and text.endswith('"') and len(text) > 2:
        text = text[1:-1].strip()
    if not text:
        return _build_default_explanation(evidence, word_min=word_min, word_max=word_max)
    structured = (
        "likely cause:" in text.lower()
        and "why likely:" in text.lower()
        and "check next:" in text.lower()
    )

    if word_max > 0 and _word_count(text) > word_max:
        text = _truncate_words(text, word_max)

    if word_min > 0 and _word_count(text) < word_min and not structured:
        grounding_tail = _build_grounding_tail(evidence)
        if grounding_tail and grounding_tail.lower() not in text.lower():
            text = f"{text.rstrip('.')} {grounding_tail}"
            if word_max > 0 and _word_count(text) > word_max:
                text = _truncate_words(text, word_max)

    if _word_count(text) < max(18, int(word_min * 0.55)) and not structured:
        fallback = _build_default_explanation(evidence, word_min=word_min, word_max=word_max)
        merged = f"{text.rstrip('.')} {fallback}".strip()
        text = _truncate_words(merged, word_max) if word_max > 0 else merged

    return text


def generate_failure_explanation(
    *,
    client: OllamaClient,
    model: str,
    evidence: Dict[str, Any],
    prompt_bundle: Dict[str, Any],
    word_min: int,
    word_max: int,
    retry_invalid_output: int,
    ollama_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    signal = classify_failure_signal(evidence)
    external = signal in {"transport", "auth"}
    min_words = max(18, int(word_min or 0))
    max_words = max(min_words + 5, int(word_max or 0))
    test_id = _get_test_id(evidence)
    system_prompt = (
        "You are the ContractGuard failure assistant. "
        "Return strict JSON only with keys: cause, why_likely, check_next, confidence. "
        f"The combined response should read naturally at about {min_words}-{max_words} words when rendered. "
        "Ground every claim in supplied evidence. "
        "In why_likely, explicitly mention the failing HTTP method/path and expected vs observed status or error reason. "
        "Avoid generic placeholder debugging advice."
    )
    base_user_prompt = (
        "Analyze this failed test and diagnose the most likely failure cause.\n"
        "Output JSON:\n"
        "{\n"
        '  "cause": "one-sentence likely cause",\n'
        '  "why_likely": "one or two evidence-grounded sentences",\n'
        '  "check_next": "one concrete next action",\n'
        '  "confidence": "confirmed|likely"\n'
        "}\n\n"
        "Do not output markdown.\n"
        "Evidence bundle:\n"
        f"{json.dumps(prompt_bundle, ensure_ascii=True, indent=2)}"
    )

    attempts: List[str] = []
    retries = max(0, int(retry_invalid_output))
    user_prompt = base_user_prompt

    for _ in range(retries + 1):
        response, llm_error = client.generate(
            model=model,
            prompt=user_prompt,
            system=system_prompt,
            format_json=True,
            options=ollama_options or {},
        )
        if llm_error:
            attempts.append(f"llm_error:{llm_error}")
            user_prompt = (
                base_user_prompt
                + "\n\nPrevious attempt failed. Return strict JSON only with the required keys."
            )
            continue

        parsed, parse_error = core.parse_llm_json_response(response or "")
        if parse_error or not isinstance(parsed, dict):
            attempts.append(f"parse_error:{parse_error or 'invalid_json'}")
            user_prompt = (
                base_user_prompt
                + "\n\nPrevious attempt was not valid JSON. Return strict JSON only."
            )
            continue

        cause_raw = str(parsed.get("cause") or "").strip()
        why_raw = str(parsed.get("why_likely") or parsed.get("why") or "").strip()
        check_raw = str(parsed.get("check_next") or parsed.get("next_step") or "").strip()
        explanation_raw = str(parsed.get("explanation") or "").strip()
        confidence = str(parsed.get("confidence") or "likely").strip().lower()
        if confidence not in {"confirmed", "likely"}:
            confidence = "likely"

        if cause_raw and why_raw and check_raw:
            explanation_raw = _format_structured_explanation(
                test_id=test_id,
                cause=cause_raw,
                why_likely=why_raw,
                check_next=check_raw,
            )
        elif explanation_raw:
            normalized_lower = explanation_raw.lower()
            if not (
                "likely cause:" in normalized_lower
                and "why likely:" in normalized_lower
                and "check next:" in normalized_lower
            ):
                det_cause, _, det_check = _build_diagnostic_triplet(evidence, signal)
                explanation_raw = _format_structured_explanation(
                    test_id=test_id,
                    cause=det_cause,
                    why_likely=explanation_raw,
                    check_next=det_check,
                )

        explanation = _normalize_explanation_text(
            explanation_raw,
            evidence=evidence,
            word_min=word_min,
            word_max=word_max,
        )

        if explanation and not _contains_direct_reference(explanation, evidence):
            explanation = _normalize_explanation_text(
                f"{explanation.rstrip('.')} {_build_grounding_tail(evidence)}",
                evidence=evidence,
                word_min=word_min,
                word_max=word_max,
            )

        words = _word_count(explanation)
        has_grounding = _contains_direct_reference(explanation, evidence)
        strict_min = max(18, int(word_min or 0))
        relaxed_min = max(18, int(strict_min * 0.7))
        if has_grounding and words <= max_words and words >= strict_min:
            warning = (
                "This failure pattern looks external (authentication/network/upstream). "
                "Validate external dependencies before treating it as backend logic."
                if external
                else ""
            )
            return {
                "mode": "explanation",
                "explanation": explanation,
                "confidence": confidence,
                "signal": signal,
                "external_failure": external,
                "warning_external": bool(warning),
                "warning": warning,
                "word_count": words,
                "model": model,
                "used_fallback": False,
                "llm_error": None,
            }

        if has_grounding and words <= max_words and words >= relaxed_min:
            warning = (
                "This failure pattern looks external (authentication/network/upstream). "
                "Validate external dependencies before treating it as backend logic."
                if external
                else ""
            )
            return {
                "mode": "explanation",
                "explanation": explanation,
                "confidence": confidence,
                "signal": signal,
                "external_failure": external,
                "warning_external": bool(warning),
                "warning": warning,
                "word_count": words,
                "model": model,
                "used_fallback": False,
                "llm_error": None,
            }

        attempts.append(f"validation_error:word_or_grounding(words={words}, grounded={has_grounding})")
        user_prompt = (
            base_user_prompt
            + "\n\nPrevious output failed grounding/length checks. Return corrected JSON with cause, why_likely, check_next and explicit method/path plus expected-vs-actual evidence."
        )

    fallback = _build_default_explanation(evidence, word_min=word_min, word_max=word_max)
    warning = (
        "This failure pattern looks external (authentication/network/upstream). "
        "Validate external dependencies before treating it as backend logic."
        if external
        else ""
    )
    return {
        "mode": "explanation",
        "explanation": fallback,
        "confidence": "likely",
        "signal": signal,
        "external_failure": external,
        "warning_external": bool(warning),
        "warning": warning,
        "word_count": _word_count(fallback),
        "model": model,
        "used_fallback": True,
        "llm_error": "; ".join(attempts) if attempts else "fallback_used",
    }


def _next_unique_test_id(base_id: str, existing_test_ids: Sequence[str]) -> str:
    clean_base = str(base_id or "TC-LLM-FOLLOWUP").strip() or "TC-LLM-FOLLOWUP"
    seen = {str(item) for item in existing_test_ids}
    if clean_base not in seen:
        return clean_base
    index = 1
    while True:
        candidate = f"{clean_base}-LLM-{index:02d}"
        if candidate not in seen:
            return candidate
        index += 1


def _normalize_expected_result(
    expected: Any,
    *,
    original_expected: Dict[str, Any],
    observed_status: Optional[int],
) -> Dict[str, Any]:
    default_expected: Dict[str, Any] = {}
    if isinstance(original_expected, dict):
        default_expected = deepcopy(original_expected)
    if observed_status is not None:
        default_expected["status_code"] = int(observed_status)
        default_expected.pop("status_code_any_of", None)
    if "description" not in default_expected:
        default_expected["description"] = "Follow-up case suggested by LLM assistant."

    if not isinstance(expected, dict):
        return default_expected

    normalized = deepcopy(expected)
    status_any = normalized.get("status_code_any_of")
    status_single = normalized.get("status_code")
    if isinstance(status_any, list):
        normalized["status_code_any_of"] = [int(item) for item in status_any if str(item).isdigit()]
        if not normalized["status_code_any_of"] and observed_status is not None:
            normalized["status_code"] = int(observed_status)
            normalized.pop("status_code_any_of", None)
    elif status_single is not None and str(status_single).isdigit():
        normalized["status_code"] = int(status_single)
    elif observed_status is not None:
        normalized["status_code"] = int(observed_status)
        normalized.pop("status_code_any_of", None)
    elif isinstance(default_expected.get("status_code_any_of"), list):
        normalized["status_code_any_of"] = default_expected["status_code_any_of"]
    elif default_expected.get("status_code") is not None:
        normalized["status_code"] = default_expected["status_code"]

    if "description" not in normalized:
        normalized["description"] = str(default_expected.get("description") or "Follow-up case suggested by LLM assistant.")
    if (
        "response_body_contains" not in normalized
        and isinstance(default_expected.get("response_body_contains"), dict)
    ):
        normalized["response_body_contains"] = default_expected["response_body_contains"]
    return normalized


def _normalize_step(step: Any, fallback_step: Dict[str, Any]) -> Dict[str, Any]:
    default_step = deepcopy(fallback_step)
    if not isinstance(default_step, dict):
        default_step = {"step_number": 1, "action": "Execute request", "input_data": {}}
    if not isinstance(step, dict):
        return default_step

    normalized = deepcopy(step)
    step_number = normalized.get("step_number")
    if not isinstance(step_number, int) or step_number <= 0:
        normalized["step_number"] = int(default_step.get("step_number") or 1)
    if not isinstance(normalized.get("action"), str) or not normalized.get("action"):
        normalized["action"] = str(default_step.get("action") or "Execute request")
    input_data = normalized.get("input_data")
    if not isinstance(input_data, dict):
        normalized["input_data"] = deepcopy(default_step.get("input_data") or {})
    return normalized


def _normalize_suggested_case(
    candidate: Any,
    *,
    original_test_case: Dict[str, Any],
    observed_status: Optional[int],
    existing_test_ids: Sequence[str],
) -> Optional[Dict[str, Any]]:
    if not isinstance(candidate, dict):
        return None

    base_id = str(candidate.get("test_id") or f"{original_test_case.get('test_id', 'TC-CASE')}-FOLLOWUP")
    test_id = _next_unique_test_id(base_id, existing_test_ids)
    original_steps = list(original_test_case.get("steps") or [])
    fallback_step = original_steps[0] if original_steps else {"step_number": 1, "action": "Execute request", "input_data": {}}
    suggested_steps = list(candidate.get("steps") or [])
    first_step = _normalize_step(suggested_steps[0] if suggested_steps else None, fallback_step)

    return {
        "test_id": test_id,
        "title": str(candidate.get("title") or f"LLM follow-up for {original_test_case.get('test_id', 'failed case')}"),
        "category": str(candidate.get("category") or "llm_followup"),
        "requirement_ref": str(candidate.get("requirement_ref") or original_test_case.get("requirement_ref") or "llm_assistant"),
        "method": str(candidate.get("method") or original_test_case.get("method") or "GET").upper(),
        "path": str(candidate.get("path") or original_test_case.get("path") or "/"),
        "priority": str(candidate.get("priority") or original_test_case.get("priority") or "medium"),
        "preconditions": [
            str(item)
            for item in (candidate.get("preconditions") if isinstance(candidate.get("preconditions"), list) else original_test_case.get("preconditions") or [])
        ],
        "steps": [first_step],
        "expected_result": _normalize_expected_result(
            candidate.get("expected_result"),
            original_expected=original_test_case.get("expected_result") or {},
            observed_status=observed_status,
        ),
    }


def _default_suggested_case(
    *,
    original_test_case: Dict[str, Any],
    case_result: Dict[str, Any],
    existing_test_ids: Sequence[str],
) -> Dict[str, Any]:
    observed_status = case_result.get("actual_status")
    if not isinstance(observed_status, int):
        observed_status = None
    base_id = f"{original_test_case.get('test_id', 'TC-CASE')}-FOLLOWUP"
    test_id = _next_unique_test_id(base_id, existing_test_ids)
    original_steps = list(original_test_case.get("steps") or [])
    fallback_step = original_steps[0] if original_steps else {"step_number": 1, "action": "Execute request", "input_data": {}}
    expected_result = _normalize_expected_result(
        None,
        original_expected=original_test_case.get("expected_result") or {},
        observed_status=observed_status,
    )
    expected_status_text = expected_result.get("status_code") or expected_result.get("status_code_any_of")
    return {
        "test_id": test_id,
        "title": f"LLM follow-up for {original_test_case.get('test_id', 'failed case')}",
        "category": "llm_followup",
        "requirement_ref": str(original_test_case.get("requirement_ref") or "llm_assistant"),
        "method": str(original_test_case.get("method") or "GET").upper(),
        "path": str(original_test_case.get("path") or "/"),
        "priority": "medium",
        "preconditions": [
            *[str(item) for item in (original_test_case.get("preconditions") or [])],
            "Generated from failed-case evidence to verify the observed behavior path.",
        ],
        "steps": [_normalize_step(None, fallback_step)],
        "expected_result": {
            **expected_result,
            "description": f"Validate the observed behavior path currently returning {expected_status_text}.",
        },
    }


def generate_suggested_test(
    *,
    client: OllamaClient,
    model: str,
    evidence: Dict[str, Any],
    prompt_bundle: Dict[str, Any],
    original_test_case: Dict[str, Any],
    case_result: Dict[str, Any],
    existing_test_ids: Sequence[str],
    retry_invalid_output: int,
    ollama_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    signal = classify_failure_signal(evidence)
    external = signal in {"transport", "auth"}
    if external:
        warning = (
            "This failure appears external (authentication/network/upstream). "
            "No extra test is suggested until backend-contract ownership is confirmed."
        )
        return {
            "mode": "suggest_test",
            "reason": warning,
            "signal": signal,
            "external_failure": True,
            "warning_external": True,
            "warning": warning,
            "can_apply": False,
            "suggested_test_case": None,
            "model": model,
            "used_fallback": False,
            "llm_error": None,
        }

    system_prompt = (
        "You are the ContractGuard test assistant. "
        "Return strict JSON only with keys: reason, suggested_test_case. "
        "suggested_test_case must be exactly one ContractGuard-compatible test case object."
    )
    base_user_prompt = (
        "Generate exactly one follow-up test suggestion for this failed case.\n"
        "Rules:\n"
        "- Keep the suggestion grounded in the supplied evidence.\n"
        "- Prefer same endpoint unless evidence strongly indicates otherwise.\n"
        "- Include concrete expected status.\n"
        "- Return JSON with keys reason and suggested_test_case.\n\n"
        "Evidence bundle:\n"
        f"{json.dumps(prompt_bundle, ensure_ascii=True, indent=2)}"
    )

    retries = max(0, int(retry_invalid_output))
    attempts: List[str] = []
    user_prompt = base_user_prompt
    observed_status = case_result.get("actual_status")
    observed_status = int(observed_status) if str(observed_status).isdigit() else None

    for _ in range(retries + 1):
        response, llm_error = client.generate(
            model=model,
            prompt=user_prompt,
            system=system_prompt,
            format_json=True,
            options=ollama_options or {},
        )
        if llm_error:
            attempts.append(f"llm_error:{llm_error}")
            user_prompt = base_user_prompt + "\n\nPrevious attempt failed. Return strict JSON only."
            continue

        parsed, parse_error = core.parse_llm_json_response(response or "")
        if parse_error or not isinstance(parsed, dict):
            attempts.append(f"parse_error:{parse_error or 'invalid_json'}")
            user_prompt = base_user_prompt + "\n\nPrevious output was invalid. Return strict JSON only."
            continue

        reason = str(parsed.get("reason") or "").strip()
        suggested_case = _normalize_suggested_case(
            parsed.get("suggested_test_case"),
            original_test_case=original_test_case,
            observed_status=observed_status,
            existing_test_ids=existing_test_ids,
        )
        if suggested_case:
            if not reason:
                reason = (
                    "Follow-up suggestion generated from this failed case to verify whether the observed endpoint behavior "
                    "is intentional for the same request shape."
                )
            reason = _truncate_words(reason, 35)
            return {
                "mode": "suggest_test",
                "reason": reason,
                "signal": signal,
                "external_failure": False,
                "warning_external": False,
                "warning": "",
                "can_apply": True,
                "suggested_test_case": suggested_case,
                "model": model,
                "used_fallback": False,
                "llm_error": None,
            }

        attempts.append("validation_error:invalid_test_case")
        user_prompt = (
            base_user_prompt
            + "\n\nPrevious output did not contain a valid ContractGuard test case object. Return corrected strict JSON."
        )

    fallback_case = _default_suggested_case(
        original_test_case=original_test_case,
        case_result=case_result,
        existing_test_ids=existing_test_ids,
    )
    return {
        "mode": "suggest_test",
        "reason": (
            "LLM output was invalid, so this fallback follow-up test captures the observed behavior for the same "
            "endpoint and inputs to support a quick rerun."
        ),
        "signal": signal,
        "external_failure": False,
        "warning_external": False,
        "warning": "",
        "can_apply": True,
        "suggested_test_case": fallback_case,
        "model": model,
        "used_fallback": True,
        "llm_error": "; ".join(attempts) if attempts else "fallback_used",
    }
