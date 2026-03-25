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
        os.getenv("CONTRACTGUARD_LLM_WORD_TARGET", llm_cfg.get("output_word_target", 75)),
        75,
        minimum=70,
        maximum=80,
    )
    word_max = _safe_int(
        os.getenv("CONTRACTGUARD_LLM_WORD_MAX", 80),
        80,
        minimum=70,
        maximum=120,
    )
    word_min = _safe_int(
        os.getenv("CONTRACTGUARD_LLM_WORD_MIN", 70),
        70,
        minimum=70,
        maximum=word_max,
    )
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
    return str(signal_info.get("signal") or "status_mismatch")


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


def _contains_direct_reference(text: str, evidence: Dict[str, Any]) -> bool:
    lowered = str(text or "").lower()
    if not lowered:
        return False
    expected_tokens, actual_tokens = _extract_expected_actual_tokens(evidence)
    input_tokens = _extract_input_tokens(evidence)
    spec_op = (evidence.get("spec") or {}).get("operation") or {}
    method = str(spec_op.get("method") or "").lower()
    path = str(spec_op.get("path") or "").lower()

    status_hit = any(token.lower() in lowered for token in expected_tokens + actual_tokens if token)
    input_hit = any(token.lower() in lowered for token in input_tokens if token)
    op_hit = bool(method and method in lowered) or bool(path and path in lowered)
    return sum([status_hit, input_hit, op_hit]) >= 2


def _build_default_explanation(evidence: Dict[str, Any], word_min: int, word_max: int) -> str:
    signal = classify_failure_signal(evidence)
    test_context = evidence.get("test_context") or {}
    operation = (evidence.get("spec") or {}).get("operation") or {}
    test_id = str(test_context.get("test_id") or "unknown")
    method = str(operation.get("method") or "REQUEST").upper()
    path = str(operation.get("path") or "/")

    expected_tokens, actual_tokens = _extract_expected_actual_tokens(evidence)
    expected_text = ", ".join(expected_tokens[:2]) if expected_tokens else "contract-defined status"
    actual_text = actual_tokens[0] if actual_tokens else "no concrete status"

    input_tokens = _extract_input_tokens(evidence)
    input_text = ", ".join(input_tokens[:2]) if input_tokens else "no explicit scalar input values"
    response_text = actual_tokens[-1] if len(actual_tokens) > 1 else actual_text
    response_text = core.truncate_chars(response_text, 130)

    cause_map = {
        "transport": "a transport/connectivity problem before normal API handling",
        "auth": "an authentication rejection rather than core business logic",
        "schema_type": "input type/shape validation mismatch against endpoint constraints",
        "schema_value": "invalid input value handling for this endpoint contract",
        "missing_required": "missing-required-field handling diverging from expected contract behavior",
        "validation": "request validation behavior differing from declared constraints",
    }
    cause_text = cause_map.get(signal, "endpoint behavior that diverges from this test's expected contract outcome")

    draft = (
        f"Case {test_id} failed on {method} {path}: expected {expected_text}, but observed {actual_text}. "
        f"The request used concrete input values {input_text}. "
        f"Runner evidence shows \"{response_text}\". "
        f"This pattern most likely indicates {cause_text}. "
        "Re-run with the same payload and compare sibling endpoint results to confirm whether this is backend contract logic or an external dependency effect before changing assertions."
    )
    text = re.sub(r"\s+", " ", draft).strip()
    if word_max > 0 and _word_count(text) > word_max:
        text = _truncate_words(text, word_max)
    if word_min > 0 and _word_count(text) < word_min:
        addendum = (
            "Focus verification on the exact expected-versus-actual status pair and the cited input values."
        )
        text = f"{text.rstrip('.')} {addendum}"
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
    if word_max > 0 and _word_count(text) > word_max:
        text = _truncate_words(text, word_max)
    if word_min > 0 and _word_count(text) < word_min:
        fallback = _build_default_explanation(evidence, word_min=word_min, word_max=word_max)
        if _word_count(text) >= max(40, int(word_min * 0.8)):
            text = f"{text.rstrip('.')} {fallback}"
            if word_max > 0 and _word_count(text) > word_max:
                text = _truncate_words(text, word_max)
        else:
            text = fallback
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
    system_prompt = (
        "You are the ContractGuard failure assistant. "
        "Return strict JSON only with keys: explanation, confidence. "
        f"explanation must be {word_min}-{word_max} words. "
        "It must cite direct observed data (expected vs actual status/error and at least one concrete input value). "
        "Do not define structures or provide generic debugging advice."
    )
    base_user_prompt = (
        "Analyze this failed test and explain why it failed.\n"
        "Output JSON:\n"
        "{\n"
        '  "explanation": "70-80 word concise explanation",\n'
        '  "confidence": "confirmed|likely"\n'
        "}\n\n"
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

        explanation_raw = str(parsed.get("explanation") or "").strip()
        confidence = str(parsed.get("confidence") or "likely").strip().lower()
        if confidence not in {"confirmed", "likely"}:
            confidence = "likely"

        explanation = _normalize_explanation_text(
            explanation_raw,
            evidence=evidence,
            word_min=word_min,
            word_max=word_max,
        )
        words = _word_count(explanation)
        if word_min <= words <= word_max and _contains_direct_reference(explanation, evidence):
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

        attempts.append("validation_error:word_or_grounding")
        user_prompt = (
            base_user_prompt
            + "\n\nPrevious output did not satisfy grounding/length rules. Return corrected strict JSON only."
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
