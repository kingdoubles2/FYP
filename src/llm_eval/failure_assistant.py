from __future__ import annotations

import json
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from llm_eval import run_llm_eval as core
from llm_eval.llm_client_types import (
    LLMGenerateClient,
    build_llm_error_meta,
    is_rate_limit_or_quota_kind,
    normalize_llm_error_meta,
)
from test_generator.sample_data import generate_valid_value, generate_wrong_type_value

INPUT_RELATED_SIGNALS = frozenset({"schema_type", "schema_value", "missing_required", "validation"})
SUGGESTION_TIMEOUT_SECONDS_MAX = 120
FOLLOWUP_MODE_REPAIR_VALID = "repair_valid"
FOLLOWUP_MODE_PRESERVE_NEGATIVE = "preserve_negative"
GENERIC_FOLLOWUP_CATEGORIES = frozenset(
    {
        "llm_followup",
        "llm_follow_up",
        "followup",
        "follow_up",
        "assistant_followup",
        "auto_followup",
    }
)
EXPLICIT_STATUS_CHANGE_PHRASES = (
    "change expected status",
    "change the expected status",
    "expected status should be changed",
    "set expected status to",
    "switch expected status",
    "update expected status to",
    "expectation/status should be changed",
)
ONE_SENTENCE_HINT_RE = re.compile(r"\b(?:one|1|single)\s+sentence\b", flags=re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
EXTERNAL_FAILURE_WARNING = (
    "This failure appears external (authentication/network/upstream). "
    "Validate external dependencies first; AI may defer follow-up test suggestions until ownership is confirmed."
)
EXTERNAL_SOFT_DEFER_REASON = (
    "This failure appears external (authentication/network/upstream). "
    "AI is deferring a follow-up test suggestion until dependency ownership is confirmed."
)
NON_INPUT_SOFT_DEFER_REASON = (
    "This failure does not appear input-related. "
    "AI is deferring a follow-up test suggestion for now."
)
GENERIC_SOFT_DEFER_REASON = (
    "AI could not produce a reliable follow-up test suggestion for this case, "
    "so no extra test is suggested right now."
)


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


def load_backend_model_catalog(config_path: Optional[str] = None) -> Dict[str, Any]:
    config_override = config_path or os.getenv("CONTRACTGUARD_LLM_CONFIG")
    path = Path(config_override or "config/llm_eval.yaml")
    config = _load_yaml_mapping(path) if path.exists() else {}
    ollama_cfg = config.get("ollama") or {}
    raw_models = config.get("models") if isinstance(config.get("models"), list) else []

    model_candidates: List[str] = []
    env_model = str(os.getenv("CONTRACTGUARD_LLM_MODEL") or "").strip()
    if env_model:
        model_candidates.append(env_model)
    for item in raw_models:
        model = str(item or "").strip()
        if model:
            model_candidates.append(model)

    deduped: List[str] = []
    seen: set[str] = set()
    for candidate in model_candidates:
        if candidate in seen:
            continue
        deduped.append(candidate)
        seen.add(candidate)

    if not deduped:
        deduped = ["qwen3-coder:latest"]

    base_url = (
        os.getenv("CONTRACTGUARD_OLLAMA_BASE_URL")
        or os.getenv("OLLAMA_BASE_URL")
        or str(ollama_cfg.get("base_url") or "http://localhost:11434")
    )
    return {
        "default_model": deduped[0],
        "models": deduped,
        "ollama_base_url": str(base_url),
    }


def load_backend_llm_settings(config_path: Optional[str] = None) -> Dict[str, Any]:
    config_override = config_path or os.getenv("CONTRACTGUARD_LLM_CONFIG")
    path = Path(config_override or "config/llm_eval.yaml")
    env_model = str(os.getenv("CONTRACTGUARD_LLM_MODEL") or "").strip()
    if not path.exists() and not env_model:
        raise ValueError(
            f"LLM config file not found at '{path}'. Set CONTRACTGUARD_LLM_CONFIG or CONTRACTGUARD_LLM_MODEL."
        )

    config = _load_yaml_mapping(path) if path.exists() else {}

    llm_cfg = config.get("llm") or {}
    ollama_cfg = config.get("ollama") or {}
    evidence_cfg = config.get("evidence") or {}
    models = config.get("models") if isinstance(config.get("models"), list) else []

    model = env_model or (str(models[0]).strip() if models else "")
    if not model:
        raise ValueError(
            "No LLM model configured. Set CONTRACTGUARD_LLM_MODEL or provide a non-empty first entry in models."
        )
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


def load_default_failure_system_prompt() -> str:
    system_prompt, _ = core.load_prompt_templates()
    prompt_text = str(system_prompt or "")
    if not prompt_text.strip():
        raise ValueError("Default failure system prompt is empty.")
    return prompt_text


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


def is_input_related_signal(signal: Any) -> bool:
    return str(signal or "").strip().lower() in INPUT_RELATED_SIGNALS


def _normalize_suggestion_signal(signal: Any) -> str:
    return str(signal or "status_mismatch").strip().lower() or "status_mismatch"


def build_failure_response_policy(signal: Any) -> Dict[str, Any]:
    normalized_signal = _normalize_suggestion_signal(signal)
    external = normalized_signal in {"transport", "auth"}
    input_related = is_input_related_signal(normalized_signal)
    can_generate_followup = bool(input_related and not external)
    if external:
        default_defer_reason = EXTERNAL_SOFT_DEFER_REASON
    elif not input_related:
        default_defer_reason = NON_INPUT_SOFT_DEFER_REASON
    else:
        default_defer_reason = GENERIC_SOFT_DEFER_REASON
    warning = EXTERNAL_FAILURE_WARNING if external else ""
    return {
        "signal": normalized_signal,
        "external": external,
        "input_related": input_related,
        "can_generate_followup": can_generate_followup,
        "warning": warning,
        "default_defer_reason": default_defer_reason,
    }


def normalize_suggestion_explanation_context(
    raw: Any,
    *,
    fallback_signal: str = "",
) -> Dict[str, Any]:
    raw_context = raw if isinstance(raw, dict) else {}
    contract = raw_context.get("contract") if isinstance(raw_context.get("contract"), dict) else {}

    signal = str(raw_context.get("signal") or fallback_signal or "").strip().lower()
    if not signal and fallback_signal:
        signal = str(fallback_signal).strip().lower()

    normalized: Dict[str, Any] = {"signal": signal}
    field_map = (
        ("likely_cause", ("likely_cause", "cause")),
        ("why_likely", ("why_likely",)),
        ("check_next", ("check_next",)),
        ("explanation", ("explanation", "explanation_text")),
    )
    for target_field, source_fields in field_map:
        value = ""
        for source_field in source_fields:
            candidate = raw_context.get(source_field)
            if not isinstance(candidate, str) or not candidate.strip():
                candidate = contract.get(source_field)
            if isinstance(candidate, str) and candidate.strip():
                value = candidate.strip()
                break
        if value:
            normalized[target_field] = core.truncate_chars(value, 360)
    return normalized


def build_suggestion_skip_payload(
    *,
    model: str,
    signal: str,
    reason: str = "",
    llm_error: Optional[str] = None,
    failure_mode: str = "none",
    used_fallback: bool = False,
    eligible_for_generation: Optional[bool] = None,
    attempts: Optional[Sequence[str]] = None,
    failure_kind: str = "",
    status_code: Optional[int] = None,
    error_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    policy = build_failure_response_policy(signal)
    normalized_signal = str(policy.get("signal") or "status_mismatch")
    external = bool(policy.get("external"))
    warning = str(policy.get("warning") or "")
    final_reason = str(reason or policy.get("default_defer_reason") or "No suggested test was generated.").strip()
    if not final_reason:
        final_reason = "No suggested test was generated."
    normalized_attempts = [str(item).strip() for item in (attempts or []) if str(item).strip()]
    return {
        "mode": "suggest_test",
        "reason": final_reason,
        "signal": normalized_signal,
        "external_failure": external,
        "warning_external": bool(warning),
        "warning": warning,
        "can_apply": False,
        "suggested_test_case": None,
        "model": model,
        "used_fallback": bool(used_fallback),
        "llm_error": str(llm_error or "").strip() or None,
        "failure_mode": str(failure_mode or "none"),
        "attempts": normalized_attempts,
        "skipped": True,
        "skip_reason": final_reason,
        "eligible_for_generation": (
            bool(policy.get("can_generate_followup"))
            if eligible_for_generation is None
            else bool(eligible_for_generation)
        ),
        **({"failure_kind": str(failure_kind).strip()} if str(failure_kind).strip() else {}),
        **({"status_code": status_code} if status_code is not None else {}),
        **({"error_meta": dict(error_meta)} if isinstance(error_meta, dict) else {}),
    }


def _truncate_words(text: str, max_words: int) -> str:
    return core.truncate_words(str(text or ""), max_words)


def _to_single_sentence(text: Any) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact:
        return ""
    first = SENTENCE_SPLIT_RE.split(compact, maxsplit=1)[0].strip()
    if first and first[-1] not in ".!?":
        first = f"{first}."
    return first


def _apply_explanation_style_from_system_prompt(
    *,
    explanation: str,
    contract: Dict[str, Any],
    system_prompt: str,
) -> str:
    prompt_text = str(system_prompt or "")
    if not ONE_SENTENCE_HINT_RE.search(prompt_text):
        return explanation
    cause_sentence = _to_single_sentence(contract.get("cause"))
    return cause_sentence or _to_single_sentence(explanation)


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


def _is_numeric_like_string(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    return bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?", text, flags=re.IGNORECASE))


def _build_executed_input_snapshot(evidence: Dict[str, Any]) -> Dict[str, Any]:
    execution = evidence.get("execution") if isinstance(evidence.get("execution"), dict) else {}
    request_sent = execution.get("request_sent") if isinstance(execution.get("request_sent"), dict) else {}
    test_context = evidence.get("test_context") if isinstance(evidence.get("test_context"), dict) else {}
    generated_input = test_context.get("generated_input") if isinstance(test_context.get("generated_input"), dict) else {}

    path_params = _coerce_mapping(request_sent.get("path_params"))
    if not path_params:
        path_params = _coerce_mapping(generated_input.get("path_params"))

    query_params = _coerce_mapping(request_sent.get("query_params"))
    if not query_params:
        query_params = _coerce_mapping(generated_input.get("query_params"))

    headers = _coerce_mapping(request_sent.get("headers"))
    if not headers:
        headers = _coerce_mapping(generated_input.get("headers"))

    body = deepcopy(request_sent.get("body"))
    if body is None:
        body = deepcopy(generated_input.get("body"))

    return {
        "path_params": path_params,
        "query_params": query_params,
        "headers": headers,
        "body": body,
    }


def _extract_negative_type_hint_from_title(title: str) -> tuple[str, Any]:
    text = str(title or "").strip()
    if not text:
        return "", None

    patterns = (
        r"negative\s+type:\s*(?:query|path|header|body(?:\s+field)?)\s+([A-Za-z0-9_\-]+)\s*=\s*['\"]([^'\"]+)['\"]",
        r"negative\s+type:\s*(?:query|path|header|body(?:\s+field)?)\s+([A-Za-z0-9_\-]+)\s*=\s*([^\s,;]+)",
        r"negative\s+type:\s*([A-Za-z0-9_\-]+)\s*=\s*['\"]([^'\"]+)['\"]",
        r"negative\s+type:\s*([A-Za-z0-9_\-]+)\s*=\s*([^\s,;]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        field = str(match.group(1) or "").strip()
        raw_value = str(match.group(2) or "").strip()
        if field and raw_value:
            return field, _parse_reason_scalar(raw_value)
    return "", None


def _extract_negative_intent_context(
    evidence: Dict[str, Any],
    *,
    original_test_case: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    test_context = evidence.get("test_context") if isinstance(evidence.get("test_context"), dict) else {}
    original_case = original_test_case if isinstance(original_test_case, dict) else {}
    category = str(test_context.get("category") or original_case.get("category") or "").strip()
    title = str(test_context.get("title") or original_case.get("title") or "").strip()
    raw_intent = str(test_context.get("intent") or "").strip().lower()
    intent = raw_intent or core.determine_intent(category)
    normalized_intent = str(intent or "").strip().lower()
    category_lower = category.lower()
    title_lower = title.lower()
    is_negative_intent = normalized_intent in {"negative", "auth", "fail-path"} or any(
        token in category_lower for token in ("negative", "auth", "fail")
    )
    is_negative_type = "negative_type" in category_lower or "negative type" in title_lower
    return {
        "category": category,
        "intent": normalized_intent,
        "title": title,
        "is_negative_intent": bool(is_negative_intent),
        "is_negative_type": bool(is_negative_type),
    }


def _values_differ_loose(left: Any, right: Any) -> bool:
    if left is None and right is None:
        return False
    if left is None or right is None:
        return True
    return str(left).strip() != str(right).strip()


def _find_negative_type_target_field(
    *,
    evidence: Dict[str, Any],
    executed_input: Dict[str, Any],
    hint_field: str,
) -> tuple[str, Any]:
    if hint_field:
        found, value, key = _lookup_field_value(executed_input, hint_field)
        if found:
            return key or hint_field, value

    for section_name in ("query_params", "path_params", "body"):
        section = executed_input.get(section_name)
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            schema = _resolve_field_schema(evidence, str(key))
            schema_type = str((schema or {}).get("type") or "").strip().lower()
            if schema_type in {"number", "integer"} and isinstance(value, str):
                return str(key), value
    return "", None


def analyze_negative_setup_drift(
    evidence: Dict[str, Any],
    *,
    original_test_case: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    context = _extract_negative_intent_context(evidence, original_test_case=original_test_case)
    executed_input = _build_executed_input_snapshot(evidence)
    hint_field, hint_value = _extract_negative_type_hint_from_title(str(context.get("title") or ""))

    target_field, executed_value = _find_negative_type_target_field(
        evidence=evidence,
        executed_input=executed_input,
        hint_field=hint_field,
    )
    schema = _resolve_field_schema(evidence, target_field) if target_field else {}
    schema_type = str((schema or {}).get("type") or "").strip().lower()
    is_numeric_schema = schema_type in {"number", "integer"}
    numeric_like_string = bool(is_numeric_schema and _is_numeric_like_string(executed_value))

    rule_numeric_string = bool(context.get("is_negative_type") and numeric_like_string)
    rule_title_hint_mismatch = bool(
        context.get("is_negative_intent")
        and hint_field
        and target_field
        and _field_names_match(hint_field, target_field)
        and _values_differ_loose(executed_value, hint_value)
    )
    drift_detected = bool(rule_numeric_string or rule_title_hint_mismatch)

    return {
        "executed_input": executed_input,
        "negative_intent_context": context,
        "drift_detected": drift_detected,
        "target_field": str(target_field or ""),
        "executed_value": deepcopy(executed_value),
        "expected_invalid_hint": deepcopy(hint_value),
        "numeric_like_string": numeric_like_string,
        "schema_type": schema_type,
        "drift_reasons": {
            "numeric_string_for_numeric_schema": rule_numeric_string,
            "title_invalid_hint_mismatch": rule_title_hint_mismatch,
        },
    }


def _drift_prompt_context(drift_analysis: Dict[str, Any]) -> Dict[str, Any]:
    context = drift_analysis.get("negative_intent_context") if isinstance(drift_analysis.get("negative_intent_context"), dict) else {}
    return {
        "drift_detected": bool(drift_analysis.get("drift_detected")),
        "target_field": str(drift_analysis.get("target_field") or ""),
        "executed_value": drift_analysis.get("executed_value"),
        "expected_invalid_hint": drift_analysis.get("expected_invalid_hint"),
        "is_negative_intent": bool(context.get("is_negative_intent")),
        "is_negative_type": bool(context.get("is_negative_type")),
    }


def _build_negative_drift_override_contract(evidence: Dict[str, Any], drift_analysis: Dict[str, Any]) -> Dict[str, Any]:
    field = str(drift_analysis.get("target_field") or "input field")
    executed_value = drift_analysis.get("executed_value")
    expected_hint = drift_analysis.get("expected_invalid_hint")
    expected_text, actual_text = _status_strings(evidence)
    value_text = _format_inline_value(executed_value)
    expected_hint_text = _format_inline_value(expected_hint) if expected_hint is not None else "an intentionally invalid value"
    cause = (
        f"Negative-case setup drift: {field} was executed as {value_text}, which is effectively valid for this negative-type check."
    )
    why_likely = (
        f"The case expected {expected_text} but observed {actual_text}; title intent indicates invalid {field}={expected_hint_text}, "
        f"while executed request sent {field}={value_text}."
    )
    check_next = (
        f"Restore intentionally invalid {field} (for example {expected_hint_text}) or update test expectation/category if valid input is intended."
    )
    evidence_quotes = [
        f"{field}={value_text}",
        f"expected={expected_text}",
        f"actual={actual_text}",
    ]
    return {
        "cause": re.sub(r"\s+", " ", cause).strip(),
        "why_likely": re.sub(r"\s+", " ", why_likely).strip(),
        "check_next": re.sub(r"\s+", " ", check_next).strip(),
        "confidence": "confirmed",
        "expected_negative_behavior": False,
        "evidence_quotes": evidence_quotes,
    }


def _apply_negative_drift_override_to_contract(
    contract: Dict[str, Any],
    *,
    evidence: Dict[str, Any],
    drift_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    if not bool(drift_analysis.get("drift_detected")):
        return contract
    override_contract = _build_negative_drift_override_contract(evidence, drift_analysis)
    return override_contract


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


def _normalize_llm_error(raw_error: Any) -> dict[str, Any]:
    meta = normalize_llm_error_meta(raw_error)
    if meta:
        return dict(meta)
    fallback = build_llm_error_meta(message=str(raw_error or "LLM request failed."), raw_error=str(raw_error or ""))
    return dict(fallback)


def _llm_error_message(raw_error: Any) -> tuple[str, dict[str, Any]]:
    meta = _normalize_llm_error(raw_error)
    message = str(meta.get("message") or str(raw_error or "LLM request failed.")).strip() or "LLM request failed."
    meta["message"] = message
    return message, meta


def generate_failure_explanation(
    *,
    client: LLMGenerateClient,
    model: str,
    evidence: Dict[str, Any],
    prompt_bundle: Dict[str, Any],
    system_prompt_override: Optional[str] = None,
    word_target: int,
    word_max: int,
    retry_invalid_output: int,
    max_items_per_section: int,
    ollama_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    _ = prompt_bundle
    signal = classify_failure_signal(evidence)
    drift_analysis = analyze_negative_setup_drift(evidence)
    response_policy = build_failure_response_policy(signal)
    external = bool(response_policy.get("external"))
    warning = str(response_policy.get("warning") or "")
    evidence_view = core.build_prompt_evidence_view(evidence, max_items=max(2, int(max_items_per_section)))
    prompt_evidence_text = json.dumps(evidence_view, ensure_ascii=True)

    def _coerce_prompt_snapshot(raw_snapshot: Any) -> Dict[str, str]:
        snapshot = raw_snapshot if isinstance(raw_snapshot, dict) else {}
        return {
            "system": str(snapshot.get("system") or ""),
            "user": str(snapshot.get("user") or ""),
        }

    def _failed_payload(
        *,
        llm_error: str,
        attempts: List[Dict[str, Any]],
        prompt_snapshot: Optional[Dict[str, Any]] = None,
        validation_errors: Optional[List[str]] = None,
        failure_kind: str = "other",
        status_code: Optional[int] = None,
        error_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "ok": False,
            "mode": "explanation",
            "case_id": str(evidence.get("case_id") or ""),
            "model": model,
            "llm_error": llm_error,
            "used_fallback": False,
            "validation_errors": list(validation_errors or []),
            "contract": {},
            "explanation": "",
            "auto_scores": {},
            "prompt_evidence": evidence_view,
            "prompt_evidence_text": prompt_evidence_text,
            "prompt_snapshot": _coerce_prompt_snapshot(prompt_snapshot),
            "attempts": attempts,
            "confidence": None,
            "signal": signal,
            "external_failure": external,
            "warning_external": bool(warning),
            "warning": warning,
            "word_count": 0,
            "failure_kind": str(failure_kind or "other"),
            "status_code": status_code,
            "error_meta": dict(error_meta or {}),
        }

    prompt_snapshot: Dict[str, str] = {"system": "", "user": ""}
    try:
        default_system_prompt, user_template = core.load_prompt_templates()
    except Exception as exc:
        return _failed_payload(
            llm_error=f"prompt_template_error: {exc}",
            attempts=[],
            prompt_snapshot=prompt_snapshot,
            validation_errors=["prompt_template_error"],
        )

    override_text = str(system_prompt_override or "")
    system_prompt = override_text if override_text.strip() else str(default_system_prompt or "")
    if not system_prompt.strip():
        return _failed_payload(
            llm_error="prompt_template_error: failure system prompt is empty.",
            attempts=[],
            prompt_snapshot=prompt_snapshot,
            validation_errors=["prompt_template_error"],
        )

    base_user_prompt = core.build_user_prompt(user_template, evidence, evidence_view)
    drift_context_preview = _drift_prompt_context(drift_analysis)
    base_user_prompt = (
        base_user_prompt
        + "\n\nGrounding policy:\n"
        + "- Treat execution.request_sent as the authoritative executed input when it conflicts with title/category intent.\n"
        + "- If a negative case appears manually edited away from its intended invalid input, diagnose setup drift before backend-defect hypotheses.\n"
        + "Drift context:\n"
        + f"{json.dumps(drift_context_preview, ensure_ascii=True)}"
    )
    prompt_snapshot = {"system": system_prompt, "user": base_user_prompt}
    attempts: List[Dict[str, Any]] = []
    retries = max(0, int(retry_invalid_output))
    remaining = retries + 1
    user_prompt = base_user_prompt
    last_error = "internal_error"
    last_validation_errors: List[str] = ["internal_error"]
    last_error_meta: Optional[Dict[str, Any]] = None

    while remaining > 0:
        remaining -= 1
        response, llm_error = client.generate(
            model=model,
            prompt=user_prompt,
            system=system_prompt,
            format_json=True,
            options=ollama_options or {},
        )

        attempt_record: Dict[str, Any] = {
            "prompt": user_prompt,
            "raw_response": response or "",
            "llm_error": None,
            "parse_error": None,
            "validation_errors": [],
            "error_meta": {},
        }

        if llm_error:
            llm_error_text, error_meta = _llm_error_message(llm_error)
            attempt_record["llm_error"] = llm_error_text
            attempt_record["parse_error"] = llm_error_text
            attempt_record["error_meta"] = error_meta
            attempts.append(attempt_record)
            last_error = llm_error_text
            last_validation_errors = ["llm_error"]
            last_error_meta = error_meta
            if is_rate_limit_or_quota_kind(error_meta.get("kind")):
                return _failed_payload(
                    llm_error=llm_error_text,
                    attempts=attempts,
                    prompt_snapshot=prompt_snapshot,
                    validation_errors=last_validation_errors,
                    failure_kind=str(error_meta.get("kind") or "other"),
                    status_code=error_meta.get("status_code"),
                    error_meta=error_meta,
                )
            if remaining > 0:
                user_prompt = (
                    base_user_prompt
                    + "\n\nPrevious attempt failed to execute. Return valid JSON only and keep it concise."
                )
            continue

        parsed, parse_error = core.parse_llm_json_response(response or "")
        if parse_error or not isinstance(parsed, dict):
            attempt_record["parse_error"] = parse_error or "invalid_json"
            attempts.append(attempt_record)
            last_error = str(parse_error or "invalid_json")
            last_validation_errors = ["parse_error"]
            last_error_meta = {
                "kind": "invalid_response",
                "message": last_error,
                "status_code": None,
                "provider_error_code": "",
                "provider_error_type": "",
                "retryable": False,
                "raw_error": last_error,
            }
            if remaining > 0:
                user_prompt = (
                    base_user_prompt
                    + "\n\nPrevious output was not valid JSON. Return strict JSON only with the required keys."
                )
            continue

        contract, normalize_errors = core.normalize_contract(parsed or {})
        contract = _apply_negative_drift_override_to_contract(
            contract,
            evidence=evidence,
            drift_analysis=drift_analysis,
        )
        explanation = core.render_explanation(contract, word_target=int(word_target), word_max=int(word_max))
        explanation = _apply_explanation_style_from_system_prompt(
            explanation=explanation,
            contract=contract,
            system_prompt=system_prompt,
        )
        validation_errors = normalize_errors + core.validate_contract_output(contract, explanation, word_max=int(word_max))
        attempt_record["validation_errors"] = validation_errors
        attempts.append(attempt_record)

        if validation_errors:
            last_error = "; ".join(validation_errors)
            last_validation_errors = validation_errors
            last_error_meta = {
                "kind": "invalid_response",
                "message": last_error,
                "status_code": None,
                "provider_error_code": "",
                "provider_error_type": "",
                "retryable": False,
                "raw_error": last_error,
            }
            if remaining > 0:
                user_prompt = (
                    base_user_prompt
                    + "\n\nPrevious output had issues: "
                    + "; ".join(validation_errors)
                    + ". Return corrected JSON only."
                )
                continue
            return _failed_payload(
                llm_error=last_error,
                attempts=attempts,
                prompt_snapshot=prompt_snapshot,
                validation_errors=validation_errors,
                failure_kind="invalid_response",
                status_code=None,
                error_meta=last_error_meta,
            )

        auto_scores = core.compute_auto_scores(
            evidence=evidence,
            prompt_evidence_text=prompt_evidence_text,
            contract=contract,
            explanation=explanation,
            word_target=int(word_target),
            word_max=int(word_max),
        )
        return {
            "ok": True,
            "mode": "explanation",
            "case_id": str(evidence.get("case_id") or ""),
            "model": model,
            "llm_error": None,
            "used_fallback": False,
            "validation_errors": [],
            "contract": contract,
            "explanation": explanation,
            "auto_scores": auto_scores,
            "prompt_evidence": evidence_view,
            "prompt_evidence_text": prompt_evidence_text,
            "prompt_snapshot": _coerce_prompt_snapshot(prompt_snapshot),
            "attempts": attempts,
            "confidence": contract.get("confidence"),
            "signal": signal,
            "external_failure": external,
            "warning_external": bool(warning),
            "warning": warning,
            "word_count": core.count_words(explanation),
        }

    return _failed_payload(
        llm_error=last_error,
        attempts=attempts,
        prompt_snapshot=prompt_snapshot,
        validation_errors=last_validation_errors,
        failure_kind=str((last_error_meta or {}).get("kind") or "other"),
        status_code=(last_error_meta or {}).get("status_code"),
        error_meta=last_error_meta,
    )


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


def _coerce_status_code(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return int(value)
    if str(value).isdigit():
        return int(str(value))
    return None


def _collect_expected_status_codes(expected_result: Any) -> List[int]:
    if not isinstance(expected_result, dict):
        return []
    status_any = expected_result.get("status_code_any_of")
    if isinstance(status_any, list):
        return [int(item) for item in status_any if str(item).isdigit()]
    status_single = _coerce_status_code(expected_result.get("status_code"))
    return [status_single] if status_single is not None else []


def _resolve_followup_mode(
    *,
    evidence: Dict[str, Any],
    original_test_case: Dict[str, Any],
) -> str:
    test_context = evidence.get("test_context") if isinstance(evidence.get("test_context"), dict) else {}
    raw_intent = str(test_context.get("intent") or "").strip().lower()
    category = str(
        original_test_case.get("category")
        or test_context.get("category")
        or ""
    ).strip()
    intent = raw_intent or core.determine_intent(category)
    normalized_intent = str(intent).strip().lower()
    negative_like_intent = normalized_intent in {"negative", "auth", "fail-path"}

    expected_codes = _collect_expected_status_codes(original_test_case.get("expected_result"))
    has_success_expectation = any(200 <= code <= 299 for code in expected_codes)
    has_error_expectation = any(code >= 400 for code in expected_codes)

    if has_success_expectation and not has_error_expectation and not negative_like_intent:
        return FOLLOWUP_MODE_REPAIR_VALID
    return FOLLOWUP_MODE_PRESERVE_NEGATIVE


def _collect_explanation_context_text(explanation_context: Optional[Dict[str, Any]]) -> str:
    context = explanation_context if isinstance(explanation_context, dict) else {}
    chunks: List[str] = []
    for key in ("likely_cause", "why_likely", "check_next", "explanation"):
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            chunks.append(value.strip())
    return re.sub(r"\s+", " ", " ".join(chunks)).strip()


def _contains_explicit_status_change_request(explanation_context: Optional[Dict[str, Any]]) -> bool:
    lowered = _collect_explanation_context_text(explanation_context).lower()
    if not lowered:
        return False
    return any(phrase in lowered for phrase in EXPLICIT_STATUS_CHANGE_PHRASES)


def _extract_status_code_from_explicit_request(explanation_context: Optional[Dict[str, Any]]) -> Optional[int]:
    text = _collect_explanation_context_text(explanation_context)
    if not text:
        return None
    patterns = (
        r"(?:change|set|switch|update)\s+(?:the\s+)?expected status(?:\s*code)?(?:\s+to|\s+as|\s*=)\s*(\d{3})\b",
        r"expected status should be\s*(\d{3})\b",
        r"expectation/status should be changed(?:\s+to)?\s*(\d{3})\b",
    )
    lowered = text.lower()
    for pattern in patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if not match:
            continue
        code = _coerce_status_code(match.group(1))
        if code is not None:
            return code
    return None


def _resolve_expected_status_override(
    *,
    explanation_context: Optional[Dict[str, Any]],
    case_result: Dict[str, Any],
) -> Optional[int]:
    if not _contains_explicit_status_change_request(explanation_context):
        return None
    explicit_status = _extract_status_code_from_explicit_request(explanation_context)
    if explicit_status is not None:
        return explicit_status
    return _coerce_status_code(case_result.get("actual_status"))


def _normalize_category_slug(category: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(category or "").strip().lower()).strip("_")


def _resolve_suggested_category(candidate_category: Any, original_category: Any) -> str:
    candidate = str(candidate_category or "").strip()
    original = str(original_category or "").strip()
    if candidate and _normalize_category_slug(candidate) not in GENERIC_FOLLOWUP_CATEGORIES:
        return candidate
    if original:
        return original
    return candidate or "llm_followup"


def _expected_status_text(expected_result: Dict[str, Any]) -> str:
    expected_status = expected_result.get("status_code")
    expected_any = expected_result.get("status_code_any_of") if isinstance(expected_result.get("status_code_any_of"), list) else []
    if isinstance(expected_status, int):
        return str(expected_status)
    if expected_any:
        tokens = [str(item) for item in expected_any if str(item).strip()]
        if tokens:
            return "/".join(tokens)
    return "expected status"


def _normalize_expected_result(
    expected: Any,
    *,
    original_expected: Dict[str, Any],
    status_override: Optional[int],
) -> Dict[str, Any]:
    original = deepcopy(original_expected) if isinstance(original_expected, dict) else {}
    candidate = deepcopy(expected) if isinstance(expected, dict) else {}
    normalized: Dict[str, Any] = {}

    if status_override is not None:
        normalized["status_code"] = int(status_override)
    else:
        original_any = original.get("status_code_any_of")
        original_single = _coerce_status_code(original.get("status_code"))
        if isinstance(original_any, list):
            parsed_any = [int(item) for item in original_any if str(item).isdigit()]
            if parsed_any:
                normalized["status_code_any_of"] = parsed_any
        elif original_single is not None:
            normalized["status_code"] = original_single
        else:
            candidate_any = candidate.get("status_code_any_of")
            candidate_single = _coerce_status_code(candidate.get("status_code"))
            if isinstance(candidate_any, list):
                parsed_any = [int(item) for item in candidate_any if str(item).isdigit()]
                if parsed_any:
                    normalized["status_code_any_of"] = parsed_any
            elif candidate_single is not None:
                normalized["status_code"] = candidate_single

    description = candidate.get("description") if isinstance(candidate.get("description"), str) else ""
    if not description:
        fallback_description = original.get("description")
        if isinstance(fallback_description, str) and fallback_description.strip():
            description = fallback_description.strip()
    if not description:
        description = "Follow-up case suggested by LLM assistant."
    normalized["description"] = description

    response_body_contains = candidate.get("response_body_contains")
    if isinstance(response_body_contains, dict):
        normalized["response_body_contains"] = deepcopy(response_body_contains)
    elif isinstance(original.get("response_body_contains"), dict):
        normalized["response_body_contains"] = deepcopy(original["response_body_contains"])
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
    status_override: Optional[int],
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
        "category": _resolve_suggested_category(candidate.get("category"), original_test_case.get("category")),
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
            status_override=status_override,
        ),
    }


def _default_suggested_case(
    *,
    original_test_case: Dict[str, Any],
    status_override: Optional[int],
    followup_mode: str,
    existing_test_ids: Sequence[str],
) -> Dict[str, Any]:
    base_id = f"{original_test_case.get('test_id', 'TC-CASE')}-FOLLOWUP"
    test_id = _next_unique_test_id(base_id, existing_test_ids)
    original_steps = list(original_test_case.get("steps") or [])
    fallback_step = original_steps[0] if original_steps else {"step_number": 1, "action": "Execute request", "input_data": {}}
    expected_result = _normalize_expected_result(
        None,
        original_expected=original_test_case.get("expected_result") or {},
        status_override=status_override,
    )
    expected_status_text = _expected_status_text(expected_result)
    mode_hint = (
        "repair the failing field with a valid value"
        if followup_mode == FOLLOWUP_MODE_REPAIR_VALID
        else "preserve the negative-invalid focus on the failing field"
    )
    return {
        "test_id": test_id,
        "title": f"LLM follow-up for {original_test_case.get('test_id', 'failed case')}",
        "category": _resolve_suggested_category(None, original_test_case.get("category")),
        "requirement_ref": str(original_test_case.get("requirement_ref") or "llm_assistant"),
        "method": str(original_test_case.get("method") or "GET").upper(),
        "path": str(original_test_case.get("path") or "/"),
        "priority": "medium",
        "preconditions": [
            *[str(item) for item in (original_test_case.get("preconditions") or [])],
            f"Generated from failed-case evidence to {mode_hint}.",
        ],
        "steps": [_normalize_step(None, fallback_step)],
        "expected_result": {
            **expected_result,
            "description": f"Validate root-cause handling while keeping expected outcome {expected_status_text}.",
        },
    }


def _build_compact_suggestion_prompt_bundle(prompt_bundle: Dict[str, Any]) -> Dict[str, Any]:
    bundle = prompt_bundle if isinstance(prompt_bundle, dict) else {}
    case_evidence = bundle.get("case_evidence") if isinstance(bundle.get("case_evidence"), dict) else {}
    pipeline_context = bundle.get("pipeline_context") if isinstance(bundle.get("pipeline_context"), dict) else {}

    execution = pipeline_context.get("execution") if isinstance(pipeline_context.get("execution"), dict) else {}
    result_full = execution.get("result_full") if isinstance(execution.get("result_full"), dict) else {}
    same_endpoint_summary = (
        execution.get("same_endpoint_summary")
        if isinstance(execution.get("same_endpoint_summary"), dict)
        else {}
    )

    ir_full = pipeline_context.get("ir_full") if isinstance(pipeline_context.get("ir_full"), dict) else {}
    spec_full = pipeline_context.get("spec_full") if isinstance(pipeline_context.get("spec_full"), dict) else {}
    tests_for_endpoint = (
        pipeline_context.get("tests_for_endpoint")
        if isinstance(pipeline_context.get("tests_for_endpoint"), list)
        else []
    )
    results_for_endpoint = (
        pipeline_context.get("results_for_endpoint")
        if isinstance(pipeline_context.get("results_for_endpoint"), list)
        else []
    )

    compact_execution = {
        "result_summary": {
            "test_id": str(result_full.get("test_id") or ""),
            "outcome": str(result_full.get("outcome") or ""),
            "expected_status": result_full.get("expected_status"),
            "expected_status_any_of": result_full.get("expected_status_any_of"),
            "actual_status": result_full.get("actual_status"),
            "error_message": core.truncate_chars(result_full.get("error_message") or "", 220),
            "response_snippet": core.truncate_chars(result_full.get("response_snippet") or "", 220),
        },
        "same_endpoint_summary": same_endpoint_summary,
    }

    return {
        "case_evidence": case_evidence,
        "pipeline_context": {
            "test_case_full": pipeline_context.get("test_case_full") if isinstance(pipeline_context.get("test_case_full"), dict) else {},
            "tests_for_endpoint": tests_for_endpoint,
            "results_for_endpoint": results_for_endpoint,
            "execution": compact_execution,
            "ir_full": {
                "endpoint_for_case": ir_full.get("endpoint_for_case") if isinstance(ir_full.get("endpoint_for_case"), dict) else {},
                "api_title": ir_full.get("api_title"),
                "api_version": ir_full.get("api_version"),
                "base_url": ir_full.get("base_url"),
            },
            "spec_full": {
                "operation": spec_full.get("operation") if isinstance(spec_full.get("operation"), dict) else {},
                "path_item": spec_full.get("path_item") if isinstance(spec_full.get("path_item"), dict) else {},
            },
        },
    }


def _looks_like_test_case_object(candidate: Any) -> bool:
    if not isinstance(candidate, dict):
        return False
    shape_keys = {"method", "path", "steps", "expected_result", "test_id", "title", "category"}
    return any(key in candidate for key in shape_keys)


def _extract_suggested_case_candidate(parsed: Dict[str, Any]) -> Any:
    if not isinstance(parsed, dict):
        return None
    for key in ("suggested_test_case", "suggested_test", "test_case"):
        candidate = parsed.get(key)
        if isinstance(candidate, dict):
            return candidate
    if _looks_like_test_case_object(parsed):
        return parsed
    return None


def _is_timeout_llm_error(error_text: str) -> bool:
    lowered = str(error_text or "").lower()
    return any(token in lowered for token in ("timeout", "timed out", "readtimeout", "connecttimeout"))


def _classify_suggestion_failure_mode(attempts: Sequence[str]) -> str:
    attempt_texts = [str(item or "") for item in attempts]
    has_timeout = any(text.lower().startswith("llm_error:") and _is_timeout_llm_error(text) for text in attempt_texts)
    has_invalid_schema = any(
        text.lower().startswith("validation_error:invalid_test_case")
        or text.lower().startswith("parse_error:")
        for text in attempt_texts
    )
    if has_timeout and has_invalid_schema:
        return "mixed"
    if has_timeout:
        return "timeout"
    if has_invalid_schema:
        return "invalid_schema"
    return "mixed" if attempt_texts else "none"


def _attempt_near_timeout(elapsed_seconds: float, timeout_seconds: int) -> bool:
    if timeout_seconds <= 0:
        return False
    threshold = max(1.0, float(timeout_seconds) * 0.9)
    return elapsed_seconds >= threshold


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    return deepcopy(value) if isinstance(value, dict) else {}


AUTH_HEADER_KEYS = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "apikey",
    "token",
    "x_auth_token",
    "x-auth-token",
    "bearer",
}
REDACTED_SENTINEL = "<redacted>"


def _is_auth_header_key(key: Any) -> bool:
    return str(key or "").strip().lower() in AUTH_HEADER_KEYS


def _is_redacted_header_value(value: Any) -> bool:
    return str(value or "").strip().lower() == REDACTED_SENTINEL


def _sanitize_headers_for_followup(headers: Any) -> tuple[Dict[str, Any], bool]:
    source = _coerce_mapping(headers)
    if not source:
        return {}, False

    sanitized: Dict[str, Any] = {}
    saw_redacted_auth = False
    saw_retained_header = False
    for key, value in source.items():
        key_text = str(key or "").strip()
        if not key_text:
            continue
        if _is_auth_header_key(key_text) and _is_redacted_header_value(value):
            saw_redacted_auth = True
            continue
        sanitized[key_text] = deepcopy(value)
        saw_retained_header = True
    return sanitized, bool(saw_redacted_auth and not saw_retained_header)


def _normalize_field_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _parse_reason_scalar(value: str) -> Any:
    text = str(value or "").strip()
    if not text:
        return ""
    trimmed = text.rstrip(".,;:)")
    if (
        (trimmed.startswith('"') and trimmed.endswith('"'))
        or (trimmed.startswith("'") and trimmed.endswith("'"))
    ) and len(trimmed) >= 2:
        trimmed = trimmed[1:-1].strip()
    lowered = trimmed.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    if re.fullmatch(r"[+-]?\d+", trimmed):
        try:
            return int(trimmed)
        except Exception:
            return trimmed
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?", trimmed, flags=re.IGNORECASE):
        try:
            parsed = float(trimmed)
            if parsed.is_integer():
                return int(parsed)
            return parsed
        except Exception:
            return trimmed
    return trimmed


def _extract_reason_field_value(
    evidence: Dict[str, Any],
    explanation_context: Dict[str, Any],
) -> tuple[str, Any]:
    reason_candidates: List[str] = []
    primary_reason = _extract_primary_response_reason(evidence)
    if isinstance(primary_reason, str) and primary_reason.strip():
        reason_candidates.append(primary_reason.strip())
    for key in ("likely_cause", "why_likely", "check_next", "explanation"):
        value = explanation_context.get(key)
        if isinstance(value, str) and value.strip():
            reason_candidates.append(value.strip())

    pattern = re.compile(
        r"\b([A-Za-z][A-Za-z0-9_\- ]{0,48})\s+must\b.*?\bGiven:\s*([^\s,;]+)",
        flags=re.IGNORECASE,
    )
    for reason_text in reason_candidates:
        match = pattern.search(reason_text)
        if not match:
            continue
        raw_field = str(match.group(1) or "").strip()
        if not raw_field:
            continue
        field_tokens = [part for part in re.split(r"[^A-Za-z0-9_]+", raw_field) if part]
        if not field_tokens:
            continue
        field_name = field_tokens[-1]
        raw_value = str(match.group(2) or "").strip()
        return field_name, _parse_reason_scalar(raw_value)
    return "", None


def _apply_reason_field_override(
    input_data: Dict[str, Any],
    field_name: str,
    field_value: Any,
) -> tuple[str, Any]:
    normalized_field = _normalize_field_key(field_name)
    if not normalized_field:
        return "", None

    section_order: List[Dict[str, Any]] = []
    for section_name in ("query_params", "path_params", "body"):
        section_obj = input_data.get(section_name)
        if isinstance(section_obj, dict):
            section_order.append(section_obj)

    for section_obj in section_order:
        for existing_key in list(section_obj.keys()):
            normalized_existing = _normalize_field_key(existing_key)
            if (
                normalized_existing == normalized_field
                or normalized_field in normalized_existing
                or normalized_existing in normalized_field
            ):
                section_obj[existing_key] = deepcopy(field_value)
                return str(existing_key), field_value

    query_params = input_data.get("query_params")
    if isinstance(query_params, dict):
        fallback_key = re.sub(r"[^a-zA-Z0-9_]+", "_", str(field_name).strip().lower()).strip("_")
        fallback_key = fallback_key or str(field_name).strip() or "input_field"
        query_params[fallback_key] = deepcopy(field_value)
        input_data["query_params"] = query_params
        return fallback_key, field_value
    return "", None


def _format_inline_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    except Exception:
        return str(value)


def _build_input_data_from_evidence(
    evidence: Dict[str, Any],
    original_test_case: Dict[str, Any],
    *,
    prefer_request_sent: bool,
) -> Dict[str, Any]:
    test_context = evidence.get("test_context") if isinstance(evidence.get("test_context"), dict) else {}
    generated_input = test_context.get("generated_input") if isinstance(test_context.get("generated_input"), dict) else {}
    execution = evidence.get("execution") if isinstance(evidence.get("execution"), dict) else {}
    request_sent = execution.get("request_sent") if isinstance(execution.get("request_sent"), dict) else {}
    original_steps = list(original_test_case.get("steps") or [])
    original_input = (
        original_steps[0].get("input_data")
        if original_steps and isinstance(original_steps[0], dict) and isinstance(original_steps[0].get("input_data"), dict)
        else {}
    )

    source_order = (
        (request_sent, generated_input, original_input)
        if prefer_request_sent
        else (generated_input, original_input, request_sent)
    )

    path_params: Dict[str, Any] = {}
    for source in source_order:
        path_params = _coerce_mapping(source.get("path_params"))
        if path_params:
            break

    query_params: Dict[str, Any] = {}
    for source in source_order:
        query_params = _coerce_mapping(source.get("query_params"))
        if query_params:
            break

    headers: Dict[str, Any] = {}
    for source in source_order:
        candidate_headers, only_redacted_auth = _sanitize_headers_for_followup(source.get("headers"))
        if only_redacted_auth:
            continue
        if candidate_headers:
            headers = candidate_headers
            break

    body: Any = None
    for source in source_order:
        candidate_body = deepcopy(source.get("body"))
        if candidate_body is not None:
            body = candidate_body
            break

    return {
        "path_params": path_params,
        "query_params": query_params,
        "headers": headers,
        "body": body,
    }


def _build_step_action(method: str, path: str, input_data: Dict[str, Any]) -> str:
    query_params = input_data.get("query_params") if isinstance(input_data.get("query_params"), dict) else {}
    path_params = input_data.get("path_params") if isinstance(input_data.get("path_params"), dict) else {}
    body = input_data.get("body")
    segments = [f"Send {method} request to {path}"]
    if query_params:
        segments.append(f"with query params {json.dumps(query_params, ensure_ascii=True, sort_keys=True)}")
    if path_params:
        segments.append(f"path params {json.dumps(path_params, ensure_ascii=True, sort_keys=True)}")
    if body is not None:
        if isinstance(body, dict):
            segments.append(f"body {json.dumps(body, ensure_ascii=True, sort_keys=True)}")
        else:
            segments.append(f"body {str(body)}")
    return " using ".join(segments)


def _extract_validation_focus(
    explanation_context: Dict[str, Any],
    input_data: Dict[str, Any],
) -> tuple[str, Any]:
    focus_text = " ".join(
        str(explanation_context.get(key) or "")
        for key in ("likely_cause", "why_likely", "check_next", "explanation")
    ).lower()
    query_params = input_data.get("query_params") if isinstance(input_data.get("query_params"), dict) else {}
    for key, value in query_params.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        if key_text.lower() in focus_text:
            return key_text, value
    path_params = input_data.get("path_params") if isinstance(input_data.get("path_params"), dict) else {}
    for key, value in path_params.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        if key_text.lower() in focus_text:
            return key_text, value
    body = input_data.get("body") if isinstance(input_data.get("body"), dict) else {}
    for key, value in body.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        if key_text.lower() in focus_text:
            return key_text, value
    for key in query_params.keys():
        key_text = str(key).strip()
        if "latitude" in key_text.lower():
            return key_text, query_params.get(key)
    for key in query_params.keys():
        key_text = str(key).strip()
        if key_text:
            return key_text, query_params.get(key)
    for key in path_params.keys():
        key_text = str(key).strip()
        if key_text:
            return key_text, path_params.get(key)
    for key in body.keys():
        key_text = str(key).strip()
        if key_text:
            return key_text, body.get(key)
    return "", None


def _field_names_match(left: Any, right: Any) -> bool:
    normalized_left = _normalize_field_key(left)
    normalized_right = _normalize_field_key(right)
    if not normalized_left or not normalized_right:
        return False
    return (
        normalized_left == normalized_right
        or normalized_left in normalized_right
        or normalized_right in normalized_left
    )


def _lookup_field_value(input_data: Dict[str, Any], field_name: str) -> tuple[bool, Any, str]:
    for section_name in ("query_params", "path_params", "body"):
        section_obj = input_data.get(section_name)
        if not isinstance(section_obj, dict):
            continue
        for existing_key, existing_value in section_obj.items():
            if _field_names_match(existing_key, field_name):
                return True, deepcopy(existing_value), str(existing_key)
    return False, None, ""


def _resolve_schema_from_request_constraints(
    request_constraints: Dict[str, Any],
    field_name: str,
) -> Dict[str, Any]:
    query_rules = request_constraints.get("query_param_rules")
    if isinstance(query_rules, dict):
        for rule_name, rule_schema in query_rules.items():
            if _field_names_match(rule_name, field_name) and isinstance(rule_schema, dict):
                return deepcopy(rule_schema)

    body_schema = request_constraints.get("body_schema_rules")
    if isinstance(body_schema, dict):
        properties = body_schema.get("properties")
        if isinstance(properties, dict):
            for prop_name, prop_schema in properties.items():
                if _field_names_match(prop_name, field_name) and isinstance(prop_schema, dict):
                    return deepcopy(prop_schema)
    return {}


def _resolve_schema_from_endpoint_ir(endpoint_ir: Dict[str, Any], field_name: str) -> Dict[str, Any]:
    for section_name in ("query_params", "path_params", "header_params"):
        section_items = endpoint_ir.get(section_name)
        if not isinstance(section_items, list):
            continue
        for item in section_items:
            if not isinstance(item, dict):
                continue
            if not _field_names_match(item.get("name"), field_name):
                continue
            schema = item.get("schema")
            if isinstance(schema, dict):
                return deepcopy(schema)

    request_schema = endpoint_ir.get("request_schema")
    if isinstance(request_schema, dict):
        properties = request_schema.get("properties")
        if isinstance(properties, dict):
            for prop_name, prop_schema in properties.items():
                if _field_names_match(prop_name, field_name) and isinstance(prop_schema, dict):
                    return deepcopy(prop_schema)
    return {}


def _resolve_field_schema(evidence: Dict[str, Any], field_name: str) -> Dict[str, Any]:
    spec = evidence.get("spec") if isinstance(evidence.get("spec"), dict) else {}
    request_constraints = (
        spec.get("request_constraints")
        if isinstance(spec.get("request_constraints"), dict)
        else {}
    )
    from_constraints = _resolve_schema_from_request_constraints(request_constraints, field_name)
    if from_constraints:
        return from_constraints

    ir_context = evidence.get("ir_context") if isinstance(evidence.get("ir_context"), dict) else {}
    endpoint_ir = ir_context.get("endpoint_ir") if isinstance(ir_context.get("endpoint_ir"), dict) else {}
    return _resolve_schema_from_endpoint_ir(endpoint_ir, field_name)


def _resolve_baseline_valid_value(
    *,
    evidence: Dict[str, Any],
    original_test_case: Dict[str, Any],
    field_name: str,
) -> tuple[bool, Any]:
    test_context = evidence.get("test_context") if isinstance(evidence.get("test_context"), dict) else {}
    generated_input = test_context.get("generated_input") if isinstance(test_context.get("generated_input"), dict) else {}
    found, value, _ = _lookup_field_value(generated_input, field_name)
    if found:
        return True, value

    original_steps = list(original_test_case.get("steps") or [])
    if original_steps and isinstance(original_steps[0], dict):
        original_input = original_steps[0].get("input_data") if isinstance(original_steps[0].get("input_data"), dict) else {}
    else:
        original_input = {}
    found, value, _ = _lookup_field_value(original_input, field_name)
    if found:
        return True, value
    return False, None


def _infer_schema_from_value(value: Any) -> Dict[str, Any]:
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, dict):
        return {"type": "object"}
    if isinstance(value, list):
        return {"type": "array"}
    if isinstance(value, str):
        return {"type": "string"}
    return {"type": "string"}


def _resolve_target_field_and_value(
    *,
    evidence: Dict[str, Any],
    explanation_context: Dict[str, Any],
    input_data: Dict[str, Any],
) -> tuple[str, Any]:
    override_field, override_value = _extract_reason_field_value(evidence, explanation_context)
    if override_field:
        return override_field, override_value
    focus_field, focus_value = _extract_validation_focus(explanation_context, input_data)
    if focus_field:
        return focus_field, focus_value
    return "", None


def _resolve_valid_repair_value(
    *,
    evidence: Dict[str, Any],
    original_test_case: Dict[str, Any],
    input_data: Dict[str, Any],
    target_field: str,
    invalid_value: Any,
) -> Any:
    schema = _resolve_field_schema(evidence, target_field)
    if schema:
        try:
            return generate_valid_value(schema, name=target_field)
        except Exception:
            pass

    found_baseline, baseline_value = _resolve_baseline_valid_value(
        evidence=evidence,
        original_test_case=original_test_case,
        field_name=target_field,
    )
    if found_baseline:
        return baseline_value

    found_current, current_value, _ = _lookup_field_value(input_data, target_field)
    if found_current:
        try:
            guessed_schema = _infer_schema_from_value(current_value)
            return generate_valid_value(guessed_schema, name=target_field)
        except Exception:
            pass

    if invalid_value is not None:
        try:
            guessed_schema = _infer_schema_from_value(invalid_value)
            return generate_valid_value(guessed_schema, name=target_field)
        except Exception:
            pass
    return "valid_value"


def _resolve_negative_drift_invalid_value(
    *,
    evidence: Dict[str, Any],
    drift_analysis: Dict[str, Any],
) -> Any:
    expected_hint = drift_analysis.get("expected_invalid_hint")
    if expected_hint is not None and str(expected_hint).strip():
        return deepcopy(expected_hint)

    target_field = str(drift_analysis.get("target_field") or "").strip()
    schema = _resolve_field_schema(evidence, target_field) if target_field else {}
    if schema:
        try:
            return generate_wrong_type_value(schema)
        except Exception:
            pass

    executed_value = drift_analysis.get("executed_value")
    if _is_numeric_like_string(executed_value):
        return "not_a_number"
    return "invalid_value"


def _apply_focused_field_repair(
    *,
    evidence: Dict[str, Any],
    original_test_case: Dict[str, Any],
    explanation_context: Dict[str, Any],
    input_data: Dict[str, Any],
) -> tuple[str, Any]:
    target_field, invalid_value = _resolve_target_field_and_value(
        evidence=evidence,
        explanation_context=explanation_context,
        input_data=input_data,
    )
    if not target_field:
        return "", None
    replacement = _resolve_valid_repair_value(
        evidence=evidence,
        original_test_case=original_test_case,
        input_data=input_data,
        target_field=target_field,
        invalid_value=invalid_value,
    )
    return _apply_reason_field_override(input_data, target_field, replacement)


def _apply_followup_policy_to_case(
    *,
    suggested_case: Dict[str, Any],
    followup_mode: str,
    evidence: Dict[str, Any],
    original_test_case: Dict[str, Any],
    explanation_context: Dict[str, Any],
    status_override: Optional[int],
    drift_analysis: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], str, Any]:
    normalized_case = deepcopy(suggested_case)
    drift = drift_analysis if isinstance(drift_analysis, dict) else {}
    drift_detected = bool(drift.get("drift_detected"))
    input_data = _build_input_data_from_evidence(
        evidence,
        original_test_case,
        prefer_request_sent=(followup_mode == FOLLOWUP_MODE_PRESERVE_NEGATIVE),
    )
    applied_focus_field = ""
    applied_focus_value: Any = None
    drift_focus_restored = False

    if followup_mode == FOLLOWUP_MODE_REPAIR_VALID:
        applied_focus_field, applied_focus_value = _apply_focused_field_repair(
            evidence=evidence,
            original_test_case=original_test_case,
            explanation_context=explanation_context,
            input_data=input_data,
        )
    else:
        if drift_detected:
            drift_target_field = str(drift.get("target_field") or "").strip()
            if drift_target_field:
                drift_invalid_value = _resolve_negative_drift_invalid_value(
                    evidence=evidence,
                    drift_analysis=drift,
                )
                applied_focus_field, applied_focus_value = _apply_reason_field_override(
                    input_data,
                    drift_target_field,
                    drift_invalid_value,
                )
                drift_focus_restored = bool(applied_focus_field)

        if not applied_focus_field:
            override_field, override_value = _extract_reason_field_value(evidence, explanation_context)
            if override_field:
                applied_focus_field, applied_focus_value = _apply_reason_field_override(
                    input_data,
                    override_field,
                    override_value,
                )
        if not applied_focus_field:
            applied_focus_field, applied_focus_value = _extract_validation_focus(explanation_context, input_data)

        if drift_detected and not drift_focus_restored and applied_focus_field:
            drift_focus_restored = _field_names_match(applied_focus_field, drift.get("target_field"))

    steps = normalized_case.get("steps") if isinstance(normalized_case.get("steps"), list) else []
    fallback_step = {"step_number": 1, "action": "Execute request", "input_data": {}}
    current_step = steps[0] if steps and isinstance(steps[0], dict) else fallback_step
    normalized_step = _normalize_step(current_step, fallback_step)
    normalized_step["input_data"] = input_data

    method = str(normalized_case.get("method") or original_test_case.get("method") or "GET").upper()
    path = str(normalized_case.get("path") or original_test_case.get("path") or "/")
    normalized_step["action"] = _build_step_action(method, path, input_data)
    normalized_case["steps"] = [normalized_step]

    expected_result = normalized_case.get("expected_result") if isinstance(normalized_case.get("expected_result"), dict) else {}
    if not expected_result:
        expected_result = _normalize_expected_result(
            None,
            original_expected=original_test_case.get("expected_result") or {},
            status_override=status_override,
        )
    expected_status_text = _expected_status_text(expected_result)
    if followup_mode == FOLLOWUP_MODE_REPAIR_VALID:
        if applied_focus_field:
            expected_result["description"] = (
                f"Validate that {method} {path} keeps {expected_status_text} after repairing {applied_focus_field} to a valid value."
            )
        else:
            expected_result["description"] = (
                f"Validate that {method} {path} keeps {expected_status_text} after repairing the root-cause input."
            )
    else:
        if drift_focus_restored and applied_focus_field:
            expected_result["description"] = (
                f"Validate that {method} {path} keeps {expected_status_text} after restoring negative setup drift for {applied_focus_field}."
            )
        elif applied_focus_field:
            expected_result["description"] = (
                f"Validate that {method} {path} keeps {expected_status_text} when {applied_focus_field} remains invalid."
            )
        else:
            expected_result["description"] = (
                f"Validate that {method} {path} keeps {expected_status_text} for the targeted negative input."
            )
    normalized_case["expected_result"] = expected_result
    return normalized_case, applied_focus_field, applied_focus_value


def build_deterministic_suggested_test_payload(
    *,
    model: str,
    evidence: Dict[str, Any],
    original_test_case: Dict[str, Any],
    case_result: Dict[str, Any],
    existing_test_ids: Sequence[str],
    explanation_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    classified_signal = classify_failure_signal(evidence)
    normalized_context = normalize_suggestion_explanation_context(
        explanation_context,
        fallback_signal=classified_signal,
    )
    signal = str(normalized_context.get("signal") or classified_signal or "status_mismatch").strip().lower()
    if not is_input_related_signal(signal):
        return build_suggestion_skip_payload(
            model=model,
            signal=signal,
            reason="",
        )

    drift_analysis = analyze_negative_setup_drift(
        evidence,
        original_test_case=original_test_case,
    )
    followup_mode = _resolve_followup_mode(
        evidence=evidence,
        original_test_case=original_test_case,
    )
    drift_context = (
        drift_analysis.get("negative_intent_context")
        if isinstance(drift_analysis.get("negative_intent_context"), dict)
        else {}
    )
    if bool(drift_analysis.get("drift_detected")) and bool(drift_context.get("is_negative_intent")):
        followup_mode = FOLLOWUP_MODE_PRESERVE_NEGATIVE
    status_override = _resolve_expected_status_override(
        explanation_context=normalized_context,
        case_result=case_result,
    )
    suggested_case = _default_suggested_case(
        original_test_case=original_test_case,
        status_override=status_override,
        followup_mode=followup_mode,
        existing_test_ids=existing_test_ids,
    )
    suggested_case, focus_field, focus_value = _apply_followup_policy_to_case(
        suggested_case=suggested_case,
        followup_mode=followup_mode,
        evidence=evidence,
        original_test_case=original_test_case,
        explanation_context=normalized_context,
        status_override=status_override,
        drift_analysis=drift_analysis,
    )

    method = str(suggested_case.get("method") or original_test_case.get("method") or "GET").upper()
    path = str(suggested_case.get("path") or original_test_case.get("path") or "/")
    expected_result = suggested_case.get("expected_result") if isinstance(suggested_case.get("expected_result"), dict) else {}
    expected_status_text = _expected_status_text(expected_result)
    focus_clause = ""
    if focus_field:
        focus_clause = f" ({focus_field}={_format_inline_value(focus_value)})"
    drift_detected = bool(drift_analysis.get("drift_detected"))
    if followup_mode == FOLLOWUP_MODE_REPAIR_VALID:
        reason = (
            f"Adds a follow-up {method} {path} test addressing the root cause by repairing the focused field"
            f"{focus_clause} and preserving expected status {expected_status_text}."
        )
    else:
        if drift_detected:
            reason = (
                f"Adds a follow-up {method} {path} test restoring negative setup drift with invalid-focused"
                f" input{focus_clause} while keeping expected status {expected_status_text}."
            )
        else:
            reason = (
                f"Adds a follow-up {method} {path} test addressing the root cause by preserving the negative-invalid"
                f" focus{focus_clause} while keeping expected status {expected_status_text}."
            )
    reason = re.sub(r"\s+", " ", reason).strip()

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
        "failure_mode": "none",
        "skipped": False,
        "skip_reason": "",
        "eligible_for_generation": True,
    }


def generate_suggested_test(
    *,
    client: LLMGenerateClient,
    model: str,
    evidence: Dict[str, Any],
    prompt_bundle: Dict[str, Any],
    original_test_case: Dict[str, Any],
    case_result: Dict[str, Any],
    existing_test_ids: Sequence[str],
    retry_invalid_output: int,
    explanation_context: Optional[Dict[str, Any]] = None,
    max_generation_seconds: int = SUGGESTION_TIMEOUT_SECONDS_MAX,
    ollama_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    _ = retry_invalid_output
    classified_signal = classify_failure_signal(evidence)
    normalized_context = normalize_suggestion_explanation_context(
        explanation_context,
        fallback_signal=classified_signal,
    )
    signal = str(normalized_context.get("signal") or classified_signal or "status_mismatch").strip().lower()
    response_policy = build_failure_response_policy(signal)
    allow_followup_generation = bool(response_policy.get("can_generate_followup"))

    drift_analysis = analyze_negative_setup_drift(
        evidence,
        original_test_case=original_test_case,
    )
    followup_mode = _resolve_followup_mode(
        evidence=evidence,
        original_test_case=original_test_case,
    )
    drift_context = (
        drift_analysis.get("negative_intent_context")
        if isinstance(drift_analysis.get("negative_intent_context"), dict)
        else {}
    )
    if bool(drift_analysis.get("drift_detected")) and bool(drift_context.get("is_negative_intent")):
        followup_mode = FOLLOWUP_MODE_PRESERVE_NEGATIVE
    status_override = _resolve_expected_status_override(
        explanation_context=normalized_context,
        case_result=case_result,
    )
    original_expected = _normalize_expected_result(
        None,
        original_expected=original_test_case.get("expected_result") or {},
        status_override=status_override,
    )
    expected_status_text = _expected_status_text(original_expected)
    status_rule = (
        f"Status override is explicitly requested; use expected status {expected_status_text}."
        if status_override is not None
        else f"Preserve the source test expected status {expected_status_text}."
    )
    followup_mode_rule = (
        "Repair mode: correct the root-cause field to a valid value."
        if followup_mode == FOLLOWUP_MODE_REPAIR_VALID
        else "Negative preserve mode: keep the focused invalid field behavior."
    )
    drift_rule = (
        "Negative setup drift is detected. Restore the intended invalid input shape before proposing backend-defect hypotheses."
        if bool(drift_analysis.get("drift_detected"))
        else "No negative setup drift was detected from executed request input."
    )
    capability_rule = (
        "Signal appears non-input or external. Prefer a soft defer: set suggested_test_case to null."
        if not allow_followup_generation
        else "If evidence supports a reliable input-focused follow-up, return one test case; otherwise soft defer with suggested_test_case null."
    )

    context_preview = {
        key: value
        for key, value in (
            ("signal", signal),
            ("policy_allows_actionable_followup", allow_followup_generation),
            ("followup_mode", followup_mode),
            ("status_policy", status_rule),
            ("negative_setup_drift", _drift_prompt_context(drift_analysis)),
            ("likely_cause", normalized_context.get("likely_cause")),
            ("why_likely", normalized_context.get("why_likely")),
            ("check_next", normalized_context.get("check_next")),
            ("explanation", normalized_context.get("explanation")),
        )
        if (isinstance(value, str) and value.strip()) or isinstance(value, dict)
    }
    compact_prompt_bundle = _build_compact_suggestion_prompt_bundle(prompt_bundle)
    system_prompt = (
        "You are the ContractGuard test assistant. "
        "Return strict JSON only with keys: reason, suggested_test_case. "
        "suggested_test_case must be either null (soft defer) or exactly one ContractGuard-compatible test case object."
    )
    base_user_prompt = (
        "Generate exactly one follow-up test suggestion for this failed case.\n"
        "Rules:\n"
        "- Keep the suggestion grounded in the supplied evidence.\n"
        "- Prefer same endpoint unless evidence strongly indicates otherwise.\n"
        f"- {followup_mode_rule}\n"
        f"- {status_rule}\n"
        "- Treat execution.request_sent as the authoritative executed input when it conflicts with title/category intent.\n"
        f"- {drift_rule}\n"
        f"- {capability_rule}\n"
        "- Include concrete expected status in expected_result.\n"
        "- If you cannot confidently help, set suggested_test_case to null and explain briefly in reason.\n"
        "- Return JSON with keys reason and suggested_test_case only.\n\n"
        "Explanation context:\n"
        f"{json.dumps(context_preview, ensure_ascii=True, indent=2)}\n\n"
        "Evidence bundle:\n"
        f"{json.dumps(compact_prompt_bundle, ensure_ascii=True, indent=2)}"
    )

    retries = 0
    attempts: List[str] = []
    user_prompt = base_user_prompt
    current_timeout = max(1, int(getattr(client, "timeout_seconds", 0) or 1))
    timeout_seconds = min(
        current_timeout,
        max(1, int(max_generation_seconds)),
    )
    if hasattr(client, "timeout_seconds"):
        try:
            setattr(client, "timeout_seconds", timeout_seconds)
        except Exception:
            pass

    for attempt_index in range(retries + 1):
        attempt_start = time.monotonic()
        response, llm_error = client.generate(
            model=model,
            prompt=user_prompt,
            system=system_prompt,
            format_json=True,
            options=ollama_options or {},
        )
        attempt_elapsed = time.monotonic() - attempt_start
        remaining = retries - attempt_index
        near_timeout = _attempt_near_timeout(attempt_elapsed, timeout_seconds)
        if llm_error:
            llm_error_text, error_meta = _llm_error_message(llm_error)
            attempts.append(f"llm_error:{llm_error_text}")
            if is_rate_limit_or_quota_kind(error_meta.get("kind")):
                return {
                    "mode": "suggest_test",
                    "reason": "LLM suggested-test generation could not complete due to provider limits.",
                    "signal": signal,
                    "external_failure": bool(response_policy.get("external")),
                    "warning_external": bool(response_policy.get("warning")),
                    "warning": str(response_policy.get("warning") or ""),
                    "can_apply": False,
                    "suggested_test_case": None,
                    "model": model,
                    "used_fallback": False,
                    "llm_error": llm_error_text,
                    "failure_mode": str(error_meta.get("kind") or "other"),
                    "failure_kind": str(error_meta.get("kind") or "other"),
                    "status_code": error_meta.get("status_code"),
                    "error_meta": error_meta,
                    "attempts": attempts,
                    "skipped": False,
                    "skip_reason": "",
                    "eligible_for_generation": allow_followup_generation,
                }
            if _is_timeout_llm_error(llm_error_text):
                break
            if near_timeout and remaining > 0:
                attempts.append("guardrail:near_timeout_abort_retries")
                break
            user_prompt = base_user_prompt + "\n\nPrevious attempt failed. Return strict JSON only."
            continue

        parsed, parse_error = core.parse_llm_json_response(response or "")
        if parse_error or not isinstance(parsed, dict):
            attempts.append(f"parse_error:{parse_error or 'invalid_json'}")
            if near_timeout and remaining > 0:
                attempts.append("guardrail:near_timeout_abort_retries")
                break
            user_prompt = base_user_prompt + "\n\nPrevious output was invalid. Return strict JSON only."
            continue

        reason = str(parsed.get("reason") or "").strip()
        parsed_has_explicit_null = any(
            key in parsed and parsed.get(key) is None
            for key in ("suggested_test_case", "suggested_test", "test_case")
        )
        suggested_case = _normalize_suggested_case(
            _extract_suggested_case_candidate(parsed),
            original_test_case=original_test_case,
            status_override=status_override,
            existing_test_ids=existing_test_ids,
        )
        if suggested_case:
            if not allow_followup_generation:
                attempts.append("policy_defer:non_input_or_external_signal")
                return build_suggestion_skip_payload(
                    model=model,
                    signal=signal,
                    reason="",
                    eligible_for_generation=allow_followup_generation,
                    attempts=attempts,
                )
            suggested_case, focus_field, focus_value = _apply_followup_policy_to_case(
                suggested_case=suggested_case,
                followup_mode=followup_mode,
                evidence=evidence,
                original_test_case=original_test_case,
                explanation_context=normalized_context,
                status_override=status_override,
                drift_analysis=drift_analysis,
            )
            if not reason:
                method = str(suggested_case.get("method") or original_test_case.get("method") or "GET").upper()
                path = str(suggested_case.get("path") or original_test_case.get("path") or "/")
                if followup_mode == FOLLOWUP_MODE_REPAIR_VALID:
                    if focus_field:
                        reason = (
                            f"Follow-up suggestion repairs {focus_field} to a valid value for {method} {path} "
                            f"while preserving expected status {expected_status_text}."
                        )
                    else:
                        reason = (
                            f"Follow-up suggestion repairs the root-cause input for {method} {path} "
                            f"while preserving expected status {expected_status_text}."
                        )
                else:
                    focus_clause = f" on {focus_field}={_format_inline_value(focus_value)}" if focus_field else ""
                    if bool(drift_analysis.get("drift_detected")):
                        reason = (
                            f"Follow-up suggestion restores negative setup drift{focus_clause} for {method} {path} "
                            f"while keeping expected status {expected_status_text}."
                        )
                    else:
                        reason = (
                            f"Follow-up suggestion preserves the negative-invalid focus{focus_clause} for {method} {path} "
                            f"while keeping expected status {expected_status_text}."
                        )
            reason = re.sub(r"\s+", " ", reason).strip()
            warning = str(response_policy.get("warning") or "")
            return {
                "mode": "suggest_test",
                "reason": reason,
                "signal": signal,
                "external_failure": bool(response_policy.get("external")),
                "warning_external": bool(warning),
                "warning": warning,
                "can_apply": True,
                "suggested_test_case": suggested_case,
                "model": model,
                "used_fallback": False,
                "llm_error": None,
                "failure_mode": "none",
                "skipped": False,
                "skip_reason": "",
                "eligible_for_generation": allow_followup_generation,
            }

        if parsed_has_explicit_null or reason:
            return build_suggestion_skip_payload(
                model=model,
                signal=signal,
                reason=reason if allow_followup_generation else "",
                eligible_for_generation=allow_followup_generation,
                attempts=attempts,
            )

        attempts.append("validation_error:invalid_test_case")
        if near_timeout and remaining > 0:
            attempts.append("guardrail:near_timeout_abort_retries")
            break
        user_prompt = (
            base_user_prompt
            + "\n\nPrevious output did not contain a valid ContractGuard test case object. Return corrected strict JSON."
        )

    return build_suggestion_skip_payload(
        model=model,
        signal=signal,
        reason=str(response_policy.get("default_defer_reason") or ""),
        llm_error="; ".join(attempts) if attempts else None,
        failure_mode=_classify_suggestion_failure_mode(attempts),
        used_fallback=False,
        eligible_for_generation=allow_followup_generation,
        attempts=attempts,
    )
