import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, text

from llm_eval.failure_assistant import (
    SUGGESTION_TIMEOUT_SECONDS_MAX,
    build_deterministic_suggested_test_payload,
    build_suggestion_skip_payload,
    build_assistant_prompt_bundle,
    build_case_evidence,
    build_pipeline_context,
    classify_failure_signal,
    generate_failure_explanation,
    generate_suggested_test,
    is_input_related_signal,
    load_backend_model_catalog,
    load_backend_llm_settings,
    load_default_failure_system_prompt,
    normalize_suggestion_explanation_context,
    parse_spec_document,
    safe_json_dumps,
    safe_json_loads,
    sha256_text,
)
from llm_eval.llm_client_types import is_rate_limit_or_quota_kind, normalize_llm_error_meta
from llm_eval.ollama_client import OllamaClient
from llm_eval.provider_clients import AnthropicClient, OpenAIClient
from spec_parser.parser import parse_openapi
from test_generator.generator import generate_test_cases

from .auth import create_access_token, hash_password, verify_password
from .db import SessionLocal
from .deps import get_current_user
from .init_db import init_db
from .models_db import LLMRunInsight, Spec, SpecArtifact, TestRun, User, UserLLMSettings


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class RunTestsRequest(BaseModel):
    spec_id: Optional[int] = None
    api_title: Optional[str] = None
    api_version: Optional[str] = None
    base_url: Optional[str] = None
    test_cases: list[dict[str, Any]]
    timeout: int = 10
    bearer_token: Optional[str] = None
    api_key: Optional[str] = None
    api_key_header: Optional[str] = None
    merge_into_run_id: Optional[int] = None


class AddLLMModelRequest(BaseModel):
    provider: str
    model: str
    label: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None


class UpdateLLMSettingsRequest(BaseModel):
    active_model_id: Optional[str] = None
    custom_instruction: Optional[str] = None


class DiscoverProviderModelsRequest(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class SuggestTestRequest(BaseModel):
    explanation: Optional[dict[str, Any]] = None


class SpecAssistantThreadMessage(BaseModel):
    role: str
    content: str


class SpecAssistantLatestRunSnapshot(BaseModel):
    run_id: Optional[int] = None
    summary: Optional[dict[str, Any]] = None
    results: list[dict[str, Any]] = Field(default_factory=list)
    baseline_tests: list[dict[str, Any]] = Field(default_factory=list)


class SpecAssistantContextSnapshot(BaseModel):
    parsed_spec: Optional[dict[str, Any]] = None
    generated_tests: list[dict[str, Any]] = Field(default_factory=list)
    latest_run: Optional[SpecAssistantLatestRunSnapshot] = None


class SpecAssistantChatRequest(BaseModel):
    spec_id: int
    message: str
    thread: list[SpecAssistantThreadMessage] = Field(default_factory=list)
    context_snapshot: Optional[SpecAssistantContextSnapshot] = None


MODEL_SOURCE_BUILTIN = "builtin"
MODEL_SOURCE_USER = "user"
PROVIDER_OLLAMA = "ollama"
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
SUPPORTED_PROVIDERS = {PROVIDER_OLLAMA, PROVIDER_OPENAI, PROVIDER_ANTHROPIC}
CHAT_THREAD_ROLE_USER = "user"
CHAT_THREAD_ROLE_ASSISTANT = "assistant"
CHAT_ALLOWED_THREAD_ROLES = {CHAT_THREAD_ROLE_USER, CHAT_THREAD_ROLE_ASSISTANT}
CHAT_MAX_MESSAGE_CHARS = 4000
CHAT_MAX_THREAD_MESSAGES = 12
CHAT_MAX_THREAD_MESSAGE_CHARS = 2000
CHAT_MAX_SPEC_ENDPOINTS = 40
CHAT_MAX_TEST_CASES = 120
CHAT_MAX_RUN_RESULTS_SAMPLE = 60
CHAT_MAX_FAILED_RESULTS = 40
CHAT_REFUSAL_MESSAGE = (
    "I can only help with the selected API spec context (spec details, generated tests, and latest run results). "
    "Please ask about this spec's endpoints, generated test cases, run outcomes, or failure diagnostics."
)


def _validate_credentials(req: RegisterRequest | LoginRequest) -> None:
    email = req.email.strip().lower()
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Please provide a valid email address.")
    if len(req.password.encode("utf-8")) > 72:
        raise HTTPException(status_code=400, detail="Password must be 72 bytes or fewer.")


def _json_load_or_default(raw: Optional[str], default: Any) -> Any:
    parsed = safe_json_loads(raw)
    return parsed if parsed is not None else default


def _coerce_non_negative_int(value: Any) -> int:
    try:
        parsed = int(value)
    except Exception:
        return 0
    return parsed if parsed >= 0 else 0


def _normalize_run_summary_counts(summary: Any) -> dict[str, int]:
    source = summary if isinstance(summary, dict) else {}
    passed = _coerce_non_negative_int(source.get("passed"))
    failed = _coerce_non_negative_int(source.get("failed"))
    skipped = _coerce_non_negative_int(source.get("skipped"))
    provided_total = _coerce_non_negative_int(source.get("total"))
    computed_total = passed + failed + skipped
    total = max(provided_total, computed_total)
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
    }


def _normalize_test_id(value: Any) -> str:
    return str(value or "").strip()


def _collect_unique_test_ids(rows: Any) -> list[str]:
    source_rows = rows if isinstance(rows, list) else []
    ordered_ids: list[str] = []
    seen_ids: set[str] = set()
    for row in source_rows:
        if not isinstance(row, dict):
            continue
        test_id = _normalize_test_id(row.get("test_id"))
        if not test_id or test_id in seen_ids:
            continue
        seen_ids.add(test_id)
        ordered_ids.append(test_id)
    return ordered_ids


def _merge_records_by_test_id(existing_rows: Any, updated_rows: Any) -> list[dict[str, Any]]:
    existing_source = existing_rows if isinstance(existing_rows, list) else []
    updated_source = updated_rows if isinstance(updated_rows, list) else []
    merged_rows: list[dict[str, Any]] = []
    position_by_test_id: dict[str, int] = {}

    for row in existing_source:
        if not isinstance(row, dict):
            continue
        cloned_row = dict(row)
        test_id = _normalize_test_id(cloned_row.get("test_id"))
        if test_id and test_id in position_by_test_id:
            merged_rows[position_by_test_id[test_id]] = cloned_row
            continue
        if test_id:
            position_by_test_id[test_id] = len(merged_rows)
        merged_rows.append(cloned_row)

    for row in updated_source:
        if not isinstance(row, dict):
            continue
        cloned_row = dict(row)
        test_id = _normalize_test_id(cloned_row.get("test_id"))
        if test_id and test_id in position_by_test_id:
            merged_rows[position_by_test_id[test_id]] = cloned_row
            continue
        if test_id:
            position_by_test_id[test_id] = len(merged_rows)
        merged_rows.append(cloned_row)

    return merged_rows


def _summarize_run_results(rows: Any) -> dict[str, int]:
    source_rows = rows if isinstance(rows, list) else []
    summary = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
    }
    for row in source_rows:
        if not isinstance(row, dict):
            continue
        summary["total"] += 1
        outcome = str(row.get("outcome") or "").strip().upper()
        if outcome == "PASS":
            summary["passed"] += 1
        elif outcome == "FAIL":
            summary["failed"] += 1
        elif outcome == "SKIP":
            summary["skipped"] += 1
    return summary


def _merge_suite_snapshot(
    existing_suite: Any,
    latest_suite: dict[str, Any],
    *,
    resolved_base_url: str,
) -> dict[str, Any]:
    existing_source = existing_suite if isinstance(existing_suite, dict) else {}
    merged_suite: dict[str, Any] = dict(existing_source)
    latest_api_title = str(latest_suite.get("api_title") or "").strip()
    latest_api_version = str(latest_suite.get("api_version") or "").strip()
    merged_suite["api_title"] = latest_api_title or str(merged_suite.get("api_title") or "Generated Test Suite")
    merged_suite["api_version"] = latest_api_version or str(merged_suite.get("api_version") or "Unknown")
    merged_suite["base_url"] = str(resolved_base_url or "").strip()
    merged_suite["test_cases"] = _merge_records_by_test_id(
        existing_source.get("test_cases"),
        latest_suite.get("test_cases"),
    )
    return merged_suite


def _normalize_auth_meta(auth_meta: Any) -> dict[str, Any]:
    source = auth_meta if isinstance(auth_meta, dict) else {}
    mode = str(source.get("mode") or "none").strip().lower() or "none"
    provided = bool(source.get("provided"))
    header = str(source.get("header") or "").strip() or None
    return {
        "mode": mode,
        "provided": provided,
        "header": header,
    }


def _derive_run_duration_ms(results: Any) -> int:
    run_results = results if isinstance(results, list) else []
    total = 0
    for row in run_results:
        if not isinstance(row, dict):
            continue
        total += _coerce_non_negative_int(row.get("duration_ms"))
    return total


def _run_matches_state_filter(summary: dict[str, int], state: str) -> bool:
    normalized_state = str(state or "all").strip().lower()
    if normalized_state == "all":
        return True
    if normalized_state == "failed":
        return int(summary.get("failed") or 0) > 0
    if normalized_state == "passed":
        return int(summary.get("total") or 0) > 0 and int(summary.get("failed") or 0) == 0
    return True


def _extract_llm_payload_model(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    candidates = [
        payload.get("model"),
    ]
    explanation_payload = payload.get("explanation") if isinstance(payload.get("explanation"), dict) else {}
    suggestion_payload = payload.get("suggestion") if isinstance(payload.get("suggestion"), dict) else {}
    candidates.append(explanation_payload.get("model"))
    candidates.append(suggestion_payload.get("model"))
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return None


def _upsert_llm_run_insight(
    *,
    db: Any,
    user_id: int,
    run_row: Any,
    test_id: str,
    mode: str,
    payload: Any,
) -> None:
    if not hasattr(db, "query"):
        return
    run_id = getattr(run_row, "id", None)
    spec_id = getattr(run_row, "spec_id", None)
    if run_id is None or spec_id is None:
        return
    try:
        run_id_int = int(run_id)
        spec_id_int = int(spec_id)
    except Exception:
        return

    mode_text = str(mode or "").strip().lower()
    test_id_text = str(test_id or "").strip()
    if not mode_text or not test_id_text:
        return

    normalized_payload: Any
    if isinstance(payload, (dict, list)):
        normalized_payload = payload
    else:
        normalized_payload = {"value": payload}
    model_name = _extract_llm_payload_model(normalized_payload)

    existing_row = (
        db.query(LLMRunInsight)
        .filter(
            LLMRunInsight.user_id == int(user_id),
            LLMRunInsight.run_id == run_id_int,
            LLMRunInsight.test_id == test_id_text,
            LLMRunInsight.mode == mode_text,
        )
        .first()
    )
    if existing_row:
        existing_row.spec_id = spec_id_int
        existing_row.model = model_name
        existing_row.payload_json = safe_json_dumps(normalized_payload)
        return

    db.add(
        LLMRunInsight(
            run_id=run_id_int,
            spec_id=spec_id_int,
            user_id=int(user_id),
            test_id=test_id_text,
            mode=mode_text,
            model=model_name,
            payload_json=safe_json_dumps(normalized_payload),
        )
    )


def _summarize_llm_insights_for_runs(
    db: Any,
    *,
    user_id: int,
    run_ids: list[int],
) -> dict[int, dict[str, Any]]:
    if not run_ids:
        return {}
    normalized_run_ids = [int(run_id) for run_id in run_ids if run_id is not None]
    if not normalized_run_ids:
        return {}

    summary_by_run: dict[int, dict[str, Any]] = {
        run_id: {
            "has_any": False,
            "has_explanations": False,
            "has_suggestions": False,
            "explanation_count": 0,
            "suggestion_count": 0,
            "model_list": [],
            "_model_set": set(),
        }
        for run_id in normalized_run_ids
    }

    grouped_rows = (
        db.query(
            LLMRunInsight.run_id,
            LLMRunInsight.mode,
            LLMRunInsight.model,
            func.count(LLMRunInsight.id).label("insight_count"),
        )
        .filter(
            LLMRunInsight.user_id == int(user_id),
            LLMRunInsight.run_id.in_(normalized_run_ids),
        )
        .group_by(LLMRunInsight.run_id, LLMRunInsight.mode, LLMRunInsight.model)
        .all()
    )

    for row in grouped_rows:
        run_id = int(row.run_id)
        mode = str(row.mode or "").strip().lower()
        model_name = str(row.model or "").strip()
        insight_count = _coerce_non_negative_int(row.insight_count)
        target = summary_by_run.get(run_id)
        if not target:
            continue
        target["has_any"] = True
        if mode == "explanation":
            target["has_explanations"] = True
            target["explanation_count"] += insight_count
        elif mode == "suggest_test":
            target["has_suggestions"] = True
            target["suggestion_count"] += insight_count
        elif mode == "analysis":
            # Analysis endpoint stores both explanation and suggestion in one payload.
            target["has_explanations"] = True
            target["has_suggestions"] = True
            target["explanation_count"] += insight_count
            target["suggestion_count"] += insight_count
        if model_name:
            target["_model_set"].add(model_name)

    for value in summary_by_run.values():
        model_set = value.pop("_model_set", set())
        value["model_list"] = sorted([str(model) for model in model_set if str(model).strip()])

    return summary_by_run


def _is_field_provided(model: BaseModel, field_name: str) -> bool:
    fields_set = getattr(model, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(model, "__fields_set__", set())
    return field_name in fields_set


def _coerce_provider(provider: Any) -> str:
    normalized = str(provider or "").strip().lower()
    if normalized == "claude":
        return PROVIDER_ANTHROPIC
    return normalized


def _mask_api_key(api_key: str) -> str:
    raw = str(api_key or "").strip()
    if not raw:
        return ""
    if len(raw) <= 8:
        return "*" * len(raw)
    return f"{raw[:4]}...{raw[-4:]}"


def _builtin_model_id(provider: str, model: str) -> str:
    return f"builtin:{provider}:{model}"


def _normalize_base_url(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _normalize_user_model_entry(raw: Any) -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    provider = _coerce_provider(raw.get("provider"))
    model = str(raw.get("model") or "").strip()
    if provider not in SUPPORTED_PROVIDERS or not model:
        return None

    entry_id = str(raw.get("id") or f"user:{uuid4().hex}").strip()
    if not entry_id:
        entry_id = f"user:{uuid4().hex}"
    label = str(raw.get("label") or "").strip() or model
    base_url = _normalize_base_url(raw.get("base_url"))
    api_key = str(raw.get("api_key") or "").strip()
    return {
        "id": entry_id,
        "provider": provider,
        "model": model,
        "label": label,
        "source": MODEL_SOURCE_USER,
        "base_url": base_url,
        "api_key": api_key,
    }


def _serialize_model_for_response(entry: dict[str, Any]) -> dict[str, Any]:
    api_key = str(entry.get("api_key") or "").strip()
    return {
        "id": str(entry.get("id") or ""),
        "provider": str(entry.get("provider") or ""),
        "model": str(entry.get("model") or ""),
        "label": str(entry.get("label") or ""),
        "source": str(entry.get("source") or MODEL_SOURCE_USER),
        "base_url": _normalize_base_url(entry.get("base_url")),
        "has_api_key": bool(api_key),
        "api_key_masked": _mask_api_key(api_key) if api_key else "",
    }


def _get_user_llm_settings_row(db: Any, user_id: int) -> Optional[UserLLMSettings]:
    return db.query(UserLLMSettings).filter(UserLLMSettings.user_id == user_id).first()


def _ensure_user_llm_settings_row(
    db: Any,
    user_id: int,
    *,
    default_active_model_id: Optional[str] = None,
) -> UserLLMSettings:
    row = _get_user_llm_settings_row(db, user_id)
    if row:
        return row
    row = UserLLMSettings(
        user_id=user_id,
        active_model_id=default_active_model_id,
        custom_instruction="",
        saved_models_json="[]",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _load_saved_user_models(row: Optional[UserLLMSettings]) -> list[dict[str, Any]]:
    if not row:
        return []
    parsed = _json_load_or_default(row.saved_models_json, [])
    if not isinstance(parsed, list):
        return []
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in parsed:
        item = _normalize_user_model_entry(raw)
        if not item:
            continue
        if item["id"] in seen_ids:
            item["id"] = f"user:{uuid4().hex}"
        seen_ids.add(item["id"])
        normalized.append(item)
    return normalized


def _save_user_models(row: UserLLMSettings, models: list[dict[str, Any]]) -> None:
    row.saved_models_json = safe_json_dumps(models)


def _build_builtin_model_entries(settings: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    catalog = load_backend_model_catalog()
    default_model = str(settings.get("model") or catalog.get("default_model") or "qwen3-coder:latest").strip()
    configured_models = catalog.get("models") if isinstance(catalog.get("models"), list) else []
    candidate_models = [default_model, *[str(item).strip() for item in configured_models]]

    deduped_models: list[str] = []
    seen: set[str] = set()
    for model_name in candidate_models:
        if not model_name or model_name in seen:
            continue
        deduped_models.append(model_name)
        seen.add(model_name)
    if not deduped_models:
        deduped_models = ["qwen3-coder:latest"]

    base_url = str(settings.get("base_url") or catalog.get("ollama_base_url") or "http://localhost:11434").strip()
    entries: list[dict[str, Any]] = []
    for model_name in deduped_models:
        entries.append(
            {
                "id": _builtin_model_id(PROVIDER_OLLAMA, model_name),
                "provider": PROVIDER_OLLAMA,
                "model": model_name,
                "label": model_name,
                "source": MODEL_SOURCE_BUILTIN,
                "base_url": base_url,
                "api_key": "",
            }
        )
    return entries, _builtin_model_id(PROVIDER_OLLAMA, deduped_models[0])


def _resolve_effective_llm_settings(
    db: Any,
    user_id: int,
    *,
    base_settings: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if base_settings:
        settings = base_settings
    else:
        try:
            settings = load_backend_llm_settings()
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=f"LLM settings error: {exc}") from exc
    builtin_models, default_model_id = _build_builtin_model_entries(settings)
    row = _get_user_llm_settings_row(db, user_id)
    user_models = _load_saved_user_models(row)
    all_models = [*builtin_models, *user_models]
    by_id = {str(model.get("id") or ""): model for model in all_models}

    active_model_id = str(row.active_model_id or "").strip() if row else ""
    if not active_model_id or active_model_id not in by_id:
        active_model_id = default_model_id
    active_model = by_id.get(active_model_id) or by_id.get(default_model_id) or (all_models[0] if all_models else None)

    try:
        default_prompt = load_default_failure_system_prompt()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM prompt template error: {exc}") from exc

    saved_custom_instruction = str(row.custom_instruction or "") if row else ""
    custom_instruction = saved_custom_instruction if saved_custom_instruction.strip() else default_prompt
    return {
        "settings": settings,
        "row": row,
        "models": all_models,
        "models_by_id": by_id,
        "default_model_id": default_model_id,
        "active_model_id": active_model_id,
        "active_model": active_model,
        "custom_instruction": custom_instruction,
    }


def _settings_response_payload(resolved: dict[str, Any]) -> dict[str, Any]:
    models = resolved.get("models") if isinstance(resolved.get("models"), list) else []
    return {
        "active_model_id": str(resolved.get("active_model_id") or ""),
        "default_model_id": str(resolved.get("default_model_id") or ""),
        "custom_instruction": str(resolved.get("custom_instruction") or ""),
        "models": [_serialize_model_for_response(model) for model in models if isinstance(model, dict)],
    }


def _build_runtime_from_active_model(resolved: dict[str, Any]) -> dict[str, Any]:
    active_model = resolved.get("active_model") if isinstance(resolved.get("active_model"), dict) else None
    if not active_model:
        raise HTTPException(status_code=500, detail="No active model available for LLM execution.")

    settings = resolved.get("settings") if isinstance(resolved.get("settings"), dict) else {}
    provider = str(active_model.get("provider") or "").strip().lower()
    model_name = str(active_model.get("model") or "").strip()
    if provider not in SUPPORTED_PROVIDERS or not model_name:
        raise HTTPException(status_code=400, detail="Selected model configuration is invalid.")

    timeout_seconds = int(settings.get("timeout_seconds") or 120)
    if provider == PROVIDER_OLLAMA:
        base_url = str(active_model.get("base_url") or settings.get("base_url") or "http://localhost:11434").strip()
        return {
            "provider": provider,
            "model": model_name,
            "client": OllamaClient(base_url=base_url, timeout_seconds=timeout_seconds),
            "options": settings.get("ollama_options") if isinstance(settings.get("ollama_options"), dict) else {},
        }
    if provider == PROVIDER_OPENAI:
        api_key = str(active_model.get("api_key") or "").strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="Selected OpenAI model is missing an API key.")
        base_url = str(active_model.get("base_url") or "https://api.openai.com/v1").strip()
        return {
            "provider": provider,
            "model": model_name,
            "client": OpenAIClient(api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds),
            "options": {},
        }
    api_key = str(active_model.get("api_key") or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="Selected Anthropic model is missing an API key.")
    base_url = str(active_model.get("base_url") or "https://api.anthropic.com/v1").strip()
    return {
        "provider": provider,
        "model": model_name,
        "client": AnthropicClient(api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds),
        "options": {},
    }


def _llm_failure_guidance(failure_kind: str) -> str:
    kind = str(failure_kind or "").strip().lower()
    if kind == "quota":
        return "Provider quota is exhausted. Add credits or switch models in Settings."
    if kind == "rate_limit":
        return "Provider rate limit reached. Wait and retry, or switch models in Settings."
    if kind == "timeout":
        return "Provider timed out. Retry shortly, or switch to another model in Settings."
    if kind == "transport":
        return "Provider request failed due to a transport issue. Retry, or switch models in Settings."
    if kind == "invalid_response":
        return "Provider returned an invalid response shape. Retry, or switch models in Settings."
    return "LLM request failed. Retry, or switch models in Settings."


def _resolve_llm_failure_status(failure_kind: str, status_hint: Any) -> int:
    if is_rate_limit_or_quota_kind(failure_kind):
        return 429
    try:
        parsed = int(status_hint)
    except Exception:
        parsed = 0
    if 400 <= parsed <= 599:
        return parsed
    return 502


def _build_llm_failure_detail(
    *,
    message: str,
    llm_error: Any,
    validation_errors: Any,
    attempt_diagnostics: list[str],
    provider: str,
    model: str,
    failure_kind: str,
    status_hint: Any,
    error_meta: Any,
) -> dict[str, Any]:
    normalized_kind = str(failure_kind or "other").strip().lower() or "other"
    resolved_status = _resolve_llm_failure_status(normalized_kind, status_hint)
    return {
        "message": str(message or "LLM request failed."),
        "llm_error": str(llm_error or ""),
        "validation_errors": validation_errors if isinstance(validation_errors, list) else [],
        "attempt_diagnostics": attempt_diagnostics,
        "provider": str(provider or ""),
        "model": str(model or ""),
        "failure_kind": normalized_kind,
        "status": resolved_status,
        "guidance": _llm_failure_guidance(normalized_kind),
        "error_meta": error_meta if isinstance(error_meta, dict) else {},
    }


def _generate_failure_explanation_or_raise(
    *,
    bundle: dict[str, Any],
    resolved_settings: dict[str, Any],
) -> dict[str, Any]:
    settings = bundle["settings"]
    runtime = _build_runtime_from_active_model(resolved_settings)
    payload = generate_failure_explanation(
        client=runtime["client"],
        model=str(runtime["model"]),
        evidence=bundle["evidence"],
        prompt_bundle=bundle["prompt_bundle"],
        system_prompt_override=str(resolved_settings.get("custom_instruction") or ""),
        word_target=int(settings["word_target"]),
        word_max=int(settings["word_max"]),
        retry_invalid_output=int(settings["retry_invalid_output"]),
        max_items_per_section=int(settings["max_items_per_section"]),
        ollama_options=runtime.get("options") if isinstance(runtime.get("options"), dict) else {},
    )

    if not bool(payload.get("ok")):
        attempts = payload.get("attempts") if isinstance(payload.get("attempts"), list) else []
        diagnostics: list[str] = []
        for idx, attempt in enumerate(attempts, start=1):
            if not isinstance(attempt, dict):
                continue
            parse_error = str(attempt.get("parse_error") or "").strip()
            llm_error = str(attempt.get("llm_error") or "").strip()
            validation_errors = attempt.get("validation_errors") if isinstance(attempt.get("validation_errors"), list) else []
            if parse_error:
                diagnostics.append(f"attempt {idx}: {parse_error}")
            elif llm_error:
                diagnostics.append(f"attempt {idx}: {llm_error}")
            elif validation_errors:
                diagnostics.append(f"attempt {idx}: {'; '.join(str(item) for item in validation_errors)}")
        detail_payload = _build_llm_failure_detail(
            message="LLM explanation failed after retry attempts.",
            llm_error=payload.get("llm_error"),
            validation_errors=payload.get("validation_errors"),
            attempt_diagnostics=diagnostics,
            provider=str(runtime.get("provider") or ""),
            model=str(runtime.get("model") or ""),
            failure_kind=str(payload.get("failure_kind") or "other"),
            status_hint=payload.get("status_code"),
            error_meta=payload.get("error_meta"),
        )
        raise HTTPException(status_code=int(detail_payload["status"]), detail=detail_payload)

    return {
        "runtime": runtime,
        "payload": payload,
    }


def _get_owned_spec(db: Any, user_id: int, spec_id: int) -> Spec:
    spec_row = db.query(Spec).filter(Spec.id == spec_id, Spec.user_id == user_id).first()
    if not spec_row:
        raise HTTPException(status_code=404, detail="Specification not found.")
    return spec_row


def _get_owned_run(db: Any, user_id: int, run_id: int) -> TestRun:
    run_row = db.query(TestRun).filter(TestRun.id == run_id, TestRun.user_id == user_id).first()
    if not run_row:
        raise HTTPException(status_code=404, detail="Run not found.")
    return run_row


def _find_test_case(suite_data: dict[str, Any], test_id: str) -> dict[str, Any]:
    for test_case in suite_data.get("test_cases", []):
        if str(test_case.get("test_id") or "") == test_id:
            return test_case
    raise HTTPException(status_code=404, detail="Test case not found in this run.")


def _find_test_result(results: list[dict[str, Any]], test_id: str) -> dict[str, Any]:
    for row in results:
        if str(row.get("test_id") or "") == test_id:
            return row
    raise HTTPException(status_code=404, detail="Test result not found in this run.")


def _load_run_case_bundle(
    *,
    db: Any,
    user_id: int,
    run_id: int,
    test_id: str,
    selection_reason: str,
) -> dict[str, Any]:
    try:
        settings = load_backend_llm_settings()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"LLM settings error: {exc}") from exc
    run_row = _get_owned_run(db, user_id, run_id)
    spec_row = _get_owned_spec(db, user_id, run_row.spec_id)
    artifact_row = (
        db.query(SpecArtifact)
        .filter(SpecArtifact.spec_id == spec_row.id, SpecArtifact.user_id == user_id)
        .order_by(SpecArtifact.id.desc())
        .first()
    )

    suite_data = _json_load_or_default(run_row.suite_snapshot_json, {})
    results = _json_load_or_default(run_row.results_json, [])
    if not isinstance(suite_data, dict) or not isinstance(results, list):
        raise HTTPException(status_code=500, detail="Stored run payload is invalid.")

    test_case = _find_test_case(suite_data, test_id)
    case_result = _find_test_result(results, test_id)

    spec_raw_text = artifact_row.spec_raw_text if artifact_row and artifact_row.spec_raw_text else ""
    spec_doc: dict[str, Any] = {}
    if spec_raw_text:
        try:
            spec_doc = parse_spec_document(spec_raw_text)
        except Exception:
            spec_doc = {}

    ir_data = _json_load_or_default(artifact_row.parsed_ir_json if artifact_row else None, {})
    if not isinstance(ir_data, dict):
        ir_data = {}

    spec_hash = str(artifact_row.spec_hash or "") if artifact_row else ""
    if not spec_hash and spec_raw_text:
        spec_hash = sha256_text(spec_raw_text)

    auth_meta = _json_load_or_default(run_row.auth_meta_json, {})
    if not isinstance(auth_meta, dict):
        auth_meta = {}

    evidence = build_case_evidence(
        selection_reason=selection_reason,
        spec_id=str(spec_row.id),
        spec_path=Path(spec_row.filename or f"spec_{spec_row.id}.yaml"),
        spec_hash=spec_hash,
        spec_doc=spec_doc,
        ir_data=ir_data,
        test_case=test_case,
        case_result=case_result,
        all_results=results,
        auth_meta=auth_meta,
        timeout_seconds=int(run_row.timeout_seconds or 10),
        max_items=int(settings["max_items_per_section"]),
        snippet_chars=int(settings["snippet_chars"]),
        max_similar_failures=int(settings["max_similar_failures"]),
    )
    pipeline_context = build_pipeline_context(
        suite_data=suite_data,
        all_results=results,
        test_case=test_case,
        case_result=case_result,
        spec_doc=spec_doc,
        ir_data=ir_data,
        max_full_items=int(settings["max_full_context_items"]),
    )
    prompt_bundle = build_assistant_prompt_bundle(
        evidence=evidence,
        pipeline_context=pipeline_context,
        max_items_per_section=int(settings["max_items_per_section"]),
    )

    return {
        "settings": settings,
        "run_row": run_row,
        "spec_row": spec_row,
        "artifact_row": artifact_row,
        "suite_data": suite_data,
        "results": results,
        "test_case": test_case,
        "case_result": case_result,
        "evidence": evidence,
        "prompt_bundle": prompt_bundle,
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_spec_chat_thread(raw_thread: list[SpecAssistantThreadMessage]) -> list[dict[str, str]]:
    if len(raw_thread) > CHAT_MAX_THREAD_MESSAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Thread history must include {CHAT_MAX_THREAD_MESSAGES} messages or fewer.",
        )
    normalized: list[dict[str, str]] = []
    for item in raw_thread:
        role = str(item.role or "").strip().lower()
        if role not in CHAT_ALLOWED_THREAD_ROLES:
            raise HTTPException(status_code=400, detail="Thread message role must be 'user' or 'assistant'.")
        content = str(item.content or "").strip()
        if not content:
            continue
        if len(content) > CHAT_MAX_THREAD_MESSAGE_CHARS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Each thread message must be {CHAT_MAX_THREAD_MESSAGE_CHARS} characters or fewer."
                ),
            )
        normalized.append({"role": role, "content": content})
    return normalized


def _extract_expected_status(test_case: dict[str, Any]) -> str:
    expected = test_case.get("expected_result") if isinstance(test_case.get("expected_result"), dict) else {}
    any_of = expected.get("status_code_any_of")
    if isinstance(any_of, list) and any_of:
        return " / ".join(str(item) for item in any_of[:3])
    status_code = expected.get("status_code")
    return str(status_code) if status_code is not None else ""


def _summarize_spec_endpoints(parsed_spec: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    endpoints = parsed_spec.get("endpoints") if isinstance(parsed_spec.get("endpoints"), list) else []
    total = len(endpoints)
    summarized: list[dict[str, Any]] = []
    for endpoint in endpoints[:CHAT_MAX_SPEC_ENDPOINTS]:
        if not isinstance(endpoint, dict):
            continue
        response_schemas = endpoint.get("response_schemas")
        response_codes = (
            sorted([str(key) for key in response_schemas.keys()])
            if isinstance(response_schemas, dict)
            else []
        )
        summarized.append(
            {
                "method": str(endpoint.get("method") or "").upper(),
                "path": str(endpoint.get("path") or ""),
                "operation_id": str(endpoint.get("operation_id") or ""),
                "response_codes": response_codes[:8],
            }
        )
    return summarized, total


def _summarize_generated_tests(generated_tests: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    category_counts: dict[str, int] = {}
    for test_case in generated_tests:
        if not isinstance(test_case, dict):
            continue
        category = str(test_case.get("category") or "other")
        category_counts[category] = category_counts.get(category, 0) + 1

    summarized: list[dict[str, Any]] = []
    for test_case in generated_tests[:CHAT_MAX_TEST_CASES]:
        if not isinstance(test_case, dict):
            continue
        category = str(test_case.get("category") or "other")
        summarized.append(
            {
                "test_id": str(test_case.get("test_id") or ""),
                "title": str(test_case.get("title") or ""),
                "category": category,
                "method": str(test_case.get("method") or "").upper(),
                "path": str(test_case.get("path") or ""),
                "expected_status": _extract_expected_status(test_case),
            }
        )
    return summarized, category_counts, len(generated_tests)


def _summarize_latest_run(latest_run: Optional[dict[str, Any]]) -> tuple[dict[str, Any], int]:
    if not isinstance(latest_run, dict):
        return {"present": False}, 0
    summary = latest_run.get("summary") if isinstance(latest_run.get("summary"), dict) else {}
    results = latest_run.get("results") if isinstance(latest_run.get("results"), list) else []
    run_id_raw = latest_run.get("run_id")
    try:
        run_id = int(run_id_raw) if run_id_raw is not None else None
    except Exception:
        run_id = None

    failed_cases: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        if str(result.get("outcome") or "").upper() != "FAIL":
            continue
        failed_cases.append(
            {
                "test_id": str(result.get("test_id") or ""),
                "method": str(result.get("method") or "").upper(),
                "path": str(result.get("path") or ""),
                "expected_status": result.get("expected_status_any_of") or result.get("expected_status"),
                "actual_status": result.get("actual_status"),
                "error_message": str(result.get("error_message") or ""),
                "response_snippet": str(result.get("response_snippet") or ""),
            }
        )
        if len(failed_cases) >= CHAT_MAX_FAILED_RESULTS:
            break

    result_sample: list[dict[str, Any]] = []
    for result in results[:CHAT_MAX_RUN_RESULTS_SAMPLE]:
        if not isinstance(result, dict):
            continue
        result_sample.append(
            {
                "test_id": str(result.get("test_id") or ""),
                "outcome": str(result.get("outcome") or ""),
                "actual_status": result.get("actual_status"),
                "expected_status": result.get("expected_status_any_of") or result.get("expected_status"),
            }
        )

    return {
        "present": True,
        "run_id": run_id,
        "summary": summary,
        "failed_cases": failed_cases,
        "result_sample": result_sample,
        "result_total": len(results),
    }, len(failed_cases)


def _build_spec_chat_grounding_bundle(
    *,
    spec_row: Spec,
    parsed_spec: dict[str, Any],
    generated_tests: list[dict[str, Any]],
    latest_run: Optional[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    endpoint_summaries, endpoint_total = _summarize_spec_endpoints(parsed_spec)
    test_summaries, category_counts, test_total = _summarize_generated_tests(generated_tests)
    latest_run_summary, failed_count = _summarize_latest_run(latest_run)

    spec_summary = {
        "id": int(spec_row.id),
        "title": str(parsed_spec.get("title") or spec_row.title or ""),
        "version": str(parsed_spec.get("version") or spec_row.version or ""),
        "base_url": str(parsed_spec.get("base_url") or ""),
        "endpoint_total": endpoint_total,
        "endpoint_sample": endpoint_summaries,
    }
    tests_summary = {
        "total": test_total,
        "category_counts": category_counts,
        "case_sample": test_summaries,
    }
    bundle = {
        "spec": spec_summary,
        "generated_tests": tests_summary,
        "latest_run": latest_run_summary,
    }
    stats = {
        "endpoint_total": endpoint_total,
        "endpoint_included": len(endpoint_summaries),
        "test_count": test_total,
        "test_included": len(test_summaries),
        "has_latest_run": bool(latest_run_summary.get("present")),
        "failed_count": failed_count,
        "run_result_total": int(latest_run_summary.get("result_total") or 0),
    }
    return bundle, stats


def _is_spec_chat_out_of_scope(message: str) -> bool:
    text = str(message or "").strip().lower()
    if not text:
        return False

    scoped_tokens = (
        "spec",
        "openapi",
        "contract",
        "endpoint",
        "api",
        "test",
        "case",
        "run",
        "failure",
        "status",
        "response",
        "request",
        "path",
        "query",
        "header",
        "schema",
        "payload",
        "generated",
        "suite",
        "assertion",
        "auth",
    )
    if any(token in text for token in scoped_tokens):
        return False

    general_tokens = (
        "weather",
        "sports",
        "football",
        "basketball",
        "stock",
        "bitcoin",
        "crypto",
        "news",
        "movie",
        "music",
        "recipe",
        "travel",
        "joke",
        "poem",
        "horoscope",
        "celebrity",
        "politics",
        "translate",
        "essay",
        "workout",
    )
    if any(token in text for token in general_tokens):
        return True

    if text.startswith("who is") or text.startswith("what is") or text.startswith("tell me about"):
        return True
    return False


def _build_spec_chat_system_prompt() -> str:
    return (
        "You are ContractGuard's spec-grounded assistant.\n"
        "Use only the provided grounding context for the selected spec.\n"
        "Never answer general questions unrelated to this spec, generated tests, or run results.\n"
        "If context is missing, say what is missing and ask for a spec-focused question.\n"
        "Do not invent endpoints, statuses, test cases, or execution outcomes.\n"
        "Keep answers concise and actionable."
    )


def _build_spec_chat_user_prompt(
    *,
    message: str,
    thread: list[dict[str, str]],
    grounding_bundle: dict[str, Any],
) -> str:
    history_lines = [f"{item['role']}: {item['content']}" for item in thread]
    history_text = "\n".join(history_lines) if history_lines else "(none)"
    return (
        "Selected spec grounding context (JSON):\n"
        f"{safe_json_dumps(grounding_bundle)}\n\n"
        "Conversation thread (oldest first):\n"
        f"{history_text}\n\n"
        "Current user message:\n"
        f"{message}\n\n"
        "Answer in plain text, grounded only in the context above."
    )


def _build_chat_assistant_payload(
    *,
    spec_id: int,
    content: str,
    meta: dict[str, Any],
) -> dict[str, Any]:
    created_at = _utc_now_iso()
    return {
        "assistantMessage": {
            "id": f"assistant-{uuid4().hex}",
            "role": CHAT_THREAD_ROLE_ASSISTANT,
            "content": str(content or "").strip(),
            "createdAt": created_at,
            "context": {"mode": "spec", "specId": int(spec_id)},
            "meta": meta,
        },
        "meta": meta,
    }


def _reset_table_id_sequences(db: Any, table_names: list[str]) -> None:
    if not table_names:
        return
    bind = db.get_bind() if hasattr(db, "get_bind") else None
    dialect = str(getattr(getattr(bind, "dialect", None), "name", "")).lower()
    if not dialect:
        return

    safe_table_names = [name for name in table_names if isinstance(name, str) and name.strip()]
    if not safe_table_names:
        return

    if dialect == "sqlite":
        sqlite_seq_exists = db.execute(
            text("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'")
        ).first()
        if not sqlite_seq_exists:
            return
        for table_name in safe_table_names:
            max_id = db.execute(text(f"SELECT COALESCE(MAX(id), 0) FROM {table_name}")).scalar()
            try:
                max_id_int = int(max_id or 0)
            except Exception:
                max_id_int = 0
            if max_id_int <= 0:
                db.execute(text("DELETE FROM sqlite_sequence WHERE name = :name"), {"name": table_name})
            else:
                db.execute(
                    text("UPDATE sqlite_sequence SET seq = :seq WHERE name = :name"),
                    {"seq": max_id_int, "name": table_name},
                )
        return

    if dialect.startswith("postgres"):
        for table_name in safe_table_names:
            seq_name = db.execute(
                text("SELECT pg_get_serial_sequence(:table_name, 'id')"),
                {"table_name": table_name},
            ).scalar()
            if not seq_name:
                continue
            max_id = db.execute(text(f"SELECT COALESCE(MAX(id), 0) FROM {table_name}")).scalar()
            try:
                next_value = int(max_id or 0) + 1
            except Exception:
                next_value = 1
            db.execute(
                text("SELECT setval(:seq_name, :next_value, false)"),
                {"seq_name": str(seq_name), "next_value": int(next_value)},
            )


app = FastAPI(title="ContractGuard Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True}


@app.get("/api/llm/settings")
def get_llm_settings(current_user: User = Depends(get_current_user)) -> dict[str, Any]:
    db = SessionLocal()
    try:
        resolved = _resolve_effective_llm_settings(db, current_user.id)
        return _settings_response_payload(resolved)
    finally:
        db.close()


@app.post("/api/llm/settings/providers/{provider}/models")
def discover_provider_models(
    provider: str,
    req: DiscoverProviderModelsRequest,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    _ = current_user
    normalized_provider = _coerce_provider(provider)
    if normalized_provider not in {PROVIDER_OPENAI, PROVIDER_ANTHROPIC}:
        raise HTTPException(status_code=400, detail="Unsupported provider. Use openai or anthropic for discovery.")

    api_key = str(req.api_key or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="An API key is required to list provider models.")

    timeout_seconds = 120
    try:
        settings = load_backend_llm_settings()
        timeout_seconds = max(1, int(settings.get("timeout_seconds") or 120))
    except Exception:
        timeout_seconds = 120

    base_url = _normalize_base_url(req.base_url)
    if normalized_provider == PROVIDER_OPENAI:
        client = OpenAIClient(
            api_key=api_key,
            base_url=str(base_url or "https://api.openai.com/v1"),
            timeout_seconds=timeout_seconds,
        )
    else:
        client = AnthropicClient(
            api_key=api_key,
            base_url=str(base_url or "https://api.anthropic.com/v1"),
            timeout_seconds=timeout_seconds,
        )

    models, error = client.list_models()
    if error:
        raise HTTPException(
            status_code=502,
            detail=f"Unable to load models from provider '{normalized_provider}': {error}",
        )
    return {
        "provider": normalized_provider,
        "models": models or [],
    }


@app.post("/api/llm/settings/models")
def add_llm_model(req: AddLLMModelRequest, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
    provider = _coerce_provider(req.provider)
    model_name = str(req.model or "").strip()
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(status_code=400, detail="Unsupported provider. Use ollama, openai, or anthropic.")
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required.")

    label = str(req.label or "").strip() or model_name
    base_url = _normalize_base_url(req.base_url)
    api_key = str(req.api_key or "").strip()
    if provider in {PROVIDER_OPENAI, PROVIDER_ANTHROPIC} and not api_key:
        raise HTTPException(status_code=400, detail=f"An API key is required for provider '{provider}'.")

    db = SessionLocal()
    try:
        resolved_before = _resolve_effective_llm_settings(db, current_user.id)
        row = _ensure_user_llm_settings_row(
            db,
            current_user.id,
            default_active_model_id=str(resolved_before.get("default_model_id") or ""),
        )
        saved_models = _load_saved_user_models(row)
        new_entry = {
            "id": f"user:{uuid4().hex}",
            "provider": provider,
            "model": model_name,
            "label": label,
            "source": MODEL_SOURCE_USER,
            "base_url": base_url,
            "api_key": api_key,
        }
        saved_models.append(new_entry)
        # Newly added models become active immediately to reduce extra UI clicks.
        row.active_model_id = str(new_entry["id"])
        _save_user_models(row, saved_models)
        db.commit()
        db.refresh(row)

        resolved_after = _resolve_effective_llm_settings(db, current_user.id)
        return _settings_response_payload(resolved_after)
    finally:
        db.close()


@app.patch("/api/llm/settings")
def update_llm_settings(req: UpdateLLMSettingsRequest, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
    db = SessionLocal()
    try:
        resolved_before = _resolve_effective_llm_settings(db, current_user.id)
        row = _ensure_user_llm_settings_row(
            db,
            current_user.id,
            default_active_model_id=str(resolved_before.get("default_model_id") or ""),
        )

        if _is_field_provided(req, "active_model_id"):
            requested_model_id = str(req.active_model_id or "").strip()
            if requested_model_id:
                if requested_model_id not in resolved_before.get("models_by_id", {}):
                    raise HTTPException(status_code=400, detail="Selected model does not exist.")
                row.active_model_id = requested_model_id
            else:
                row.active_model_id = str(resolved_before.get("default_model_id") or "")

        if _is_field_provided(req, "custom_instruction"):
            instruction = str(req.custom_instruction or "")
            if len(instruction) > 12000:
                raise HTTPException(status_code=400, detail="Prompt must be 12000 characters or fewer.")
            row.custom_instruction = instruction

        db.commit()
        db.refresh(row)
        resolved_after = _resolve_effective_llm_settings(db, current_user.id)
        return _settings_response_payload(resolved_after)
    finally:
        db.close()


@app.delete("/api/llm/settings/models/{model_id}")
def delete_llm_model(model_id: str, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
    target_model_id = str(model_id or "").strip()
    if not target_model_id:
        raise HTTPException(status_code=400, detail="Model id is required.")

    db = SessionLocal()
    try:
        resolved_before = _resolve_effective_llm_settings(db, current_user.id)
        row = _ensure_user_llm_settings_row(
            db,
            current_user.id,
            default_active_model_id=str(resolved_before.get("default_model_id") or ""),
        )
        saved_models = _load_saved_user_models(row)
        next_models = [entry for entry in saved_models if str(entry.get("id") or "") != target_model_id]
        if len(next_models) == len(saved_models):
            raise HTTPException(status_code=404, detail="Model not found in user settings.")

        _save_user_models(row, next_models)
        if str(row.active_model_id or "").strip() == target_model_id:
            row.active_model_id = str(resolved_before.get("default_model_id") or "")
        db.commit()
        db.refresh(row)

        resolved_after = _resolve_effective_llm_settings(db, current_user.id)
        return _settings_response_payload(resolved_after)
    finally:
        db.close()


@app.post("/api/chat/spec-assistant")
def chat_with_spec_assistant(
    req: SpecAssistantChatRequest,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    message = str(req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(message) > CHAT_MAX_MESSAGE_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Message must be {CHAT_MAX_MESSAGE_CHARS} characters or fewer.",
        )
    thread = _normalize_spec_chat_thread(req.thread or [])
    started = time.perf_counter()

    db = SessionLocal()
    try:
        spec_row = _get_owned_spec(db, current_user.id, int(req.spec_id))
        context_snapshot = req.context_snapshot or SpecAssistantContextSnapshot()

        artifact_row = (
            db.query(SpecArtifact)
            .filter(SpecArtifact.spec_id == spec_row.id, SpecArtifact.user_id == current_user.id)
            .order_by(SpecArtifact.id.desc())
            .first()
        )

        parsed_spec = context_snapshot.parsed_spec if isinstance(context_snapshot.parsed_spec, dict) else {}
        if not parsed_spec:
            parsed_spec = _json_load_or_default(artifact_row.parsed_ir_json if artifact_row else None, {})
            if not isinstance(parsed_spec, dict):
                parsed_spec = {}

        generated_tests = context_snapshot.generated_tests if isinstance(context_snapshot.generated_tests, list) else []
        if not generated_tests:
            generated_suite = _json_load_or_default(artifact_row.generated_suite_json if artifact_row else None, {})
            if isinstance(generated_suite, dict):
                generated_tests = (
                    generated_suite.get("test_cases") if isinstance(generated_suite.get("test_cases"), list) else []
                )
        generated_tests = [item for item in generated_tests if isinstance(item, dict)]

        latest_run_payload = context_snapshot.latest_run.model_dump() if context_snapshot.latest_run else None
        if not latest_run_payload:
            latest_run_row = (
                db.query(TestRun)
                .filter(TestRun.spec_id == spec_row.id, TestRun.user_id == current_user.id)
                .order_by(TestRun.id.desc())
                .first()
            )
            if latest_run_row:
                run_summary = _json_load_or_default(latest_run_row.summary_json, {})
                if not isinstance(run_summary, dict):
                    run_summary = {}
                run_results = _json_load_or_default(latest_run_row.results_json, [])
                if not isinstance(run_results, list):
                    run_results = []
                run_suite = _json_load_or_default(latest_run_row.suite_snapshot_json, {})
                baseline_tests = run_suite.get("test_cases") if isinstance(run_suite, dict) else []
                if not isinstance(baseline_tests, list):
                    baseline_tests = []
                latest_run_payload = {
                    "run_id": int(latest_run_row.id),
                    "summary": run_summary,
                    "results": run_results,
                    "baseline_tests": baseline_tests,
                }
        if isinstance(latest_run_payload, dict):
            if not isinstance(latest_run_payload.get("summary"), dict):
                latest_run_payload["summary"] = {}
            if not isinstance(latest_run_payload.get("results"), list):
                latest_run_payload["results"] = []
            if not isinstance(latest_run_payload.get("baseline_tests"), list):
                latest_run_payload["baseline_tests"] = []
        else:
            latest_run_payload = None

        grounding_bundle, grounding_stats = _build_spec_chat_grounding_bundle(
            spec_row=spec_row,
            parsed_spec=parsed_spec,
            generated_tests=generated_tests,
            latest_run=latest_run_payload,
        )

        resolved_settings = _resolve_effective_llm_settings(db, current_user.id)
        active_model = resolved_settings.get("active_model") if isinstance(resolved_settings.get("active_model"), dict) else {}
        base_meta = {
            **grounding_stats,
            "thread_count": len(thread),
            "adapter": "spec_assistant",
            "provider": str(active_model.get("provider") or ""),
            "model": str(active_model.get("model") or ""),
            "out_of_scope": False,
        }

        if _is_spec_chat_out_of_scope(message):
            latency_ms = int((time.perf_counter() - started) * 1000)
            refusal_meta = {
                **base_meta,
                "adapter": "spec_assistant_refusal",
                "latencyMs": latency_ms,
                "out_of_scope": True,
            }
            return _build_chat_assistant_payload(
                spec_id=int(spec_row.id),
                content=CHAT_REFUSAL_MESSAGE,
                meta=refusal_meta,
            )

        runtime = _build_runtime_from_active_model(resolved_settings)
        runtime_provider = str(runtime.get("provider") or "")
        runtime_model = str(runtime.get("model") or "")
        system_prompt = _build_spec_chat_system_prompt()
        user_prompt = _build_spec_chat_user_prompt(
            message=message,
            thread=thread,
            grounding_bundle=grounding_bundle,
        )
        response_text, llm_error = runtime["client"].generate(
            model=runtime_model or "qwen3-coder:latest",
            prompt=user_prompt,
            system=system_prompt,
            format_json=False,
            options=runtime.get("options") if isinstance(runtime.get("options"), dict) else {},
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        if llm_error:
            error_meta = normalize_llm_error_meta(llm_error) or {}
            failure_kind = str(error_meta.get("kind") or "other")
            detail_payload = _build_llm_failure_detail(
                message="LLM chat request failed.",
                llm_error=error_meta.get("message") or llm_error,
                validation_errors=[],
                attempt_diagnostics=[],
                provider=runtime_provider,
                model=runtime_model,
                failure_kind=failure_kind,
                status_hint=error_meta.get("status_code"),
                error_meta=error_meta,
            )
            raise HTTPException(status_code=int(detail_payload["status"]), detail=detail_payload)

        assistant_text = str(response_text or "").strip()
        if not assistant_text:
            detail_payload = _build_llm_failure_detail(
                message="LLM chat request returned empty output.",
                llm_error="empty response",
                validation_errors=[],
                attempt_diagnostics=[],
                provider=runtime_provider,
                model=runtime_model,
                failure_kind="invalid_response",
                status_hint=502,
                error_meta={},
            )
            raise HTTPException(status_code=int(detail_payload["status"]), detail=detail_payload)

        success_meta = {
            **base_meta,
            "provider": runtime_provider,
            "model": runtime_model,
            "latencyMs": latency_ms,
        }
        return _build_chat_assistant_payload(
            spec_id=int(spec_row.id),
            content=assistant_text,
            meta=success_meta,
        )
    finally:
        db.close()


@app.post("/api/auth/register")
def register(req: RegisterRequest) -> dict[str, Any]:
    _validate_credentials(req)
    email = req.email.strip().lower()

    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered.")

        user = User(email=email, password_hash=hash_password(req.password))
        db.add(user)
        db.commit()
        db.refresh(user)
        return {"id": user.id, "email": user.email}
    finally:
        db.close()


@app.post("/api/auth/login")
def login(req: LoginRequest) -> dict[str, Any]:
    _validate_credentials(req)
    email = req.email.strip().lower()

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user or not verify_password(req.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        token = create_access_token(subject=str(user.id))
        return {"access_token": token, "token_type": "bearer"}
    finally:
        db.close()


@app.post("/api/specs/parse")
async def parse_spec(file: UploadFile = File(...), current_user: User = Depends(get_current_user)) -> dict[str, Any]:
    try:
        content = await file.read()
        spec_text = content.decode("utf-8", errors="replace")
        parsed = parse_openapi(spec_text)
        suite = generate_test_cases(parsed.to_dict())

        db = SessionLocal()
        try:
            spec_row = Spec(
                user_id=current_user.id,
                filename=file.filename or "uploaded",
                title=parsed.title,
                version=parsed.version,
            )
            db.add(spec_row)
            db.flush()

            artifact_row = SpecArtifact(
                spec_id=spec_row.id,
                user_id=current_user.id,
                spec_hash=sha256_text(spec_text),
                spec_raw_text=spec_text,
                parsed_ir_json=safe_json_dumps(parsed.to_dict()),
                generated_suite_json=safe_json_dumps(suite.to_dict()),
            )
            db.add(artifact_row)
            db.commit()
            db.refresh(spec_row)
        finally:
            db.close()

        return {
            "id": spec_row.id,
            "parsed": parsed.to_dict(),
            "generated_tests": suite.to_dict(),
            "summary": {
                "total_test_cases": len(suite.test_cases),
                "endpoint_count": len(parsed.endpoints),
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/specs/upload-ir")
async def upload_ir(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        raw = await file.read()
        text = raw.decode("utf-8", errors="replace")
        data = json.loads(text)
        if not isinstance(data, dict) or "endpoints" not in data:
            raise HTTPException(status_code=400, detail="This file doesn't look like a ContractGuard IR JSON.")
        return data
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON.") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/tests/run")
def run_generated_tests(req: RunTestsRequest, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
    if req.spec_id is None:
        raise HTTPException(status_code=400, detail="spec_id is required for run persistence.")
    if not req.test_cases:
        raise HTTPException(status_code=400, detail="No test cases were provided.")

    try:
        from test_runner.run_test import resolve_base_url, run_suite
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Test runner unavailable. {exc}") from exc

    resolved_base_url = resolve_base_url(None, req.base_url)
    if not resolved_base_url:
        raise HTTPException(
            status_code=400,
            detail="No valid base_url found. Please provide an absolute http(s) URL in the generated test suite.",
        )

    timeout = max(1, min(int(req.timeout or 10), 120))
    try:
        spec_id = int(req.spec_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="spec_id must be a valid integer.") from exc
    merge_into_run_id: Optional[int] = None
    if req.merge_into_run_id is not None:
        try:
            merge_into_run_id = int(req.merge_into_run_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="merge_into_run_id must be a valid integer.") from exc
        if merge_into_run_id <= 0:
            raise HTTPException(status_code=400, detail="merge_into_run_id must be a positive integer.")

    auth_headers: dict[str, str] = {}
    auth_meta: dict[str, Any] = {"mode": "none", "provided": False, "header": None}
    if req.bearer_token:
        auth_headers["Authorization"] = f"Bearer {req.bearer_token}"
        auth_meta = {"mode": "bearer", "provided": True, "header": "Authorization"}
    elif req.api_key:
        header_name = (req.api_key_header or "X-API-Key").strip() or "X-API-Key"
        auth_headers[header_name] = req.api_key
        auth_meta = {"mode": "api_key", "provided": True, "header": header_name}

    suite_data = {
        "api_title": req.api_title or "Generated Test Suite",
        "api_version": req.api_version or "Unknown",
        "base_url": req.base_url,
        "test_cases": req.test_cases,
    }

    try:
        results, summary = run_suite(
            suite_data=suite_data,
            base_url=resolved_base_url,
            auth_headers=auth_headers,
            timeout=timeout,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to execute test suite. {exc}") from exc

    db = SessionLocal()
    response_run_id: Optional[int] = None
    response_summary: dict[str, int]
    response_results: list[dict[str, Any]]
    try:
        _ = _get_owned_spec(db, current_user.id, spec_id)
        if merge_into_run_id is None:
            run_row = TestRun(
                spec_id=spec_id,
                user_id=current_user.id,
                api_title=str(suite_data["api_title"]),
                api_version=str(suite_data["api_version"]),
                base_url=resolved_base_url,
                suite_snapshot_json=safe_json_dumps(suite_data),
                results_json=safe_json_dumps(results),
                summary_json=safe_json_dumps(summary),
                auth_meta_json=safe_json_dumps(auth_meta),
                timeout_seconds=timeout,
            )
            db.add(run_row)
            db.commit()
            db.refresh(run_row)
            response_run_id = int(run_row.id)
            response_summary = _normalize_run_summary_counts(summary)
            response_results = [row for row in results if isinstance(row, dict)]
        else:
            run_row = _get_owned_run(db, current_user.id, int(merge_into_run_id))
            if int(run_row.spec_id) != int(spec_id):
                raise HTTPException(
                    status_code=400,
                    detail="merge_into_run_id must refer to a run for the same spec_id.",
                )

            stored_suite = _json_load_or_default(run_row.suite_snapshot_json, {})
            stored_results = _json_load_or_default(run_row.results_json, [])
            merged_suite = _merge_suite_snapshot(
                existing_suite=stored_suite,
                latest_suite=suite_data,
                resolved_base_url=resolved_base_url,
            )
            merged_results = _merge_records_by_test_id(stored_results, results)
            merged_summary = _summarize_run_results(merged_results)
            rerun_test_ids = _collect_unique_test_ids(results)

            run_row.api_title = str(merged_suite.get("api_title") or run_row.api_title or "")
            run_row.api_version = str(merged_suite.get("api_version") or run_row.api_version or "")
            run_row.base_url = resolved_base_url
            run_row.suite_snapshot_json = safe_json_dumps(merged_suite)
            run_row.results_json = safe_json_dumps(merged_results)
            run_row.summary_json = safe_json_dumps(merged_summary)
            run_row.auth_meta_json = safe_json_dumps(auth_meta)
            run_row.timeout_seconds = timeout

            if rerun_test_ids:
                db.query(LLMRunInsight).filter(
                    LLMRunInsight.user_id == int(current_user.id),
                    LLMRunInsight.run_id == int(run_row.id),
                    LLMRunInsight.test_id.in_(rerun_test_ids),
                ).delete(synchronize_session=False)

            db.commit()
            db.refresh(run_row)
            response_run_id = int(run_row.id)
            response_summary = merged_summary
            response_results = merged_results
    finally:
        db.close()

    return {
        "run_id": response_run_id,
        "spec_id": spec_id,
        "api_title": suite_data["api_title"],
        "api_version": suite_data["api_version"],
        "base_url": resolved_base_url,
        "summary": response_summary,
        "results": response_results,
    }


@app.get("/api/specs")
def list_specs(current_user: User = Depends(get_current_user), limit: Optional[int] = 50) -> list[dict[str, Any]]:
    safe_limit = max(1, min(limit or 50, 200))
    db = SessionLocal()
    try:
        spec_rows = (
            db.query(Spec)
            .filter(Spec.user_id == current_user.id)
            .order_by(Spec.id.desc())
            .limit(safe_limit)
            .all()
        )
        spec_ids = [int(row.id) for row in spec_rows]
        latest_run_by_spec_id: dict[int, dict[str, Any]] = {}

        if spec_ids:
            latest_run_id_subquery = (
                db.query(
                    TestRun.spec_id.label("spec_id"),
                    func.max(TestRun.id).label("latest_run_id"),
                )
                .filter(
                    TestRun.user_id == current_user.id,
                    TestRun.spec_id.in_(spec_ids),
                )
                .group_by(TestRun.spec_id)
                .subquery()
            )
            latest_run_rows = (
                db.query(TestRun)
                .join(
                    latest_run_id_subquery,
                    TestRun.id == latest_run_id_subquery.c.latest_run_id,
                )
                .all()
            )
            latest_run_by_spec_id = {
                int(run_row.spec_id): {
                    "id": int(run_row.id),
                    "created_at": run_row.created_at.isoformat() if run_row.created_at else None,
                    "summary": _normalize_run_summary_counts(
                        _json_load_or_default(run_row.summary_json, {})
                    ),
                }
                for run_row in latest_run_rows
            }

        return [
            {
                "id": row.id,
                "filename": row.filename,
                "title": row.title,
                "version": row.version,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "latest_run": latest_run_by_spec_id.get(int(row.id)),
            }
            for row in spec_rows
        ]
    finally:
        db.close()


@app.get("/api/specs/{spec_id}/runs/latest")
def get_latest_run_for_spec(spec_id: int, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
    db = SessionLocal()
    try:
        spec_row = _get_owned_spec(db, current_user.id, spec_id)
        artifact_row = (
            db.query(SpecArtifact)
            .filter(SpecArtifact.spec_id == spec_row.id, SpecArtifact.user_id == current_user.id)
            .order_by(SpecArtifact.id.desc())
            .first()
        )
        run_row = (
            db.query(TestRun)
            .filter(TestRun.spec_id == spec_row.id, TestRun.user_id == current_user.id)
            .order_by(TestRun.id.desc())
            .first()
        )

        artifact_payload = {
            "spec_hash": artifact_row.spec_hash if artifact_row else None,
            "parsed": _json_load_or_default(artifact_row.parsed_ir_json if artifact_row else None, None),
            "generated_suite": _json_load_or_default(artifact_row.generated_suite_json if artifact_row else None, None),
        }
        run_payload = None
        if run_row:
            run_payload = {
                "id": run_row.id,
                "created_at": run_row.created_at.isoformat() if run_row.created_at else None,
                "api_title": run_row.api_title,
                "api_version": run_row.api_version,
                "base_url": run_row.base_url,
                "summary": _json_load_or_default(run_row.summary_json, {}),
                "results": _json_load_or_default(run_row.results_json, []),
                "suite_snapshot": _json_load_or_default(run_row.suite_snapshot_json, {}),
            }

        return {
            "spec_id": spec_row.id,
            "artifact": artifact_payload,
            "latest_run": run_payload,
            "llm_outputs": [],
        }
    finally:
        db.close()


@app.post("/api/tests/{run_id}/cases/{test_id}/llm/explanation")
def explain_failed_case(run_id: int, test_id: str, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
    db = SessionLocal()
    try:
        bundle = _load_run_case_bundle(
            db=db,
            user_id=current_user.id,
            run_id=run_id,
            test_id=test_id,
            selection_reason="user_requested_explanation",
        )
        case_result = bundle["case_result"]
        if str(case_result.get("outcome") or "").upper() != "FAIL":
            raise HTTPException(status_code=400, detail="LLM explanation is only available for failed tests.")

        settings = bundle["settings"]
        resolved_settings = _resolve_effective_llm_settings(
            db,
            current_user.id,
            base_settings=settings,
        )
        explanation_output = _generate_failure_explanation_or_raise(
            bundle=bundle,
            resolved_settings=resolved_settings,
        )
        payload = explanation_output["payload"]
        _upsert_llm_run_insight(
            db=db,
            user_id=current_user.id,
            run_row=bundle.get("run_row"),
            test_id=test_id,
            mode="explanation",
            payload=payload,
        )
        if hasattr(db, "commit"):
            db.commit()

        return {
            "run_id": run_id,
            "test_id": test_id,
            "mode": "explanation",
            "payload": payload,
        }
    finally:
        db.close()


@app.post("/api/tests/{run_id}/cases/{test_id}/llm/analyze-failure")
def analyze_failure_with_suggestion(
    run_id: int,
    test_id: str,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    db = SessionLocal()
    try:
        bundle = _load_run_case_bundle(
            db=db,
            user_id=current_user.id,
            run_id=run_id,
            test_id=test_id,
            selection_reason="user_requested_explanation",
        )
        case_result = bundle["case_result"]
        if str(case_result.get("outcome") or "").upper() != "FAIL":
            raise HTTPException(status_code=400, detail="LLM analysis is only available for failed tests.")

        settings = bundle["settings"]
        resolved_settings = _resolve_effective_llm_settings(
            db,
            current_user.id,
            base_settings=settings,
        )
        explanation_output = _generate_failure_explanation_or_raise(
            bundle=bundle,
            resolved_settings=resolved_settings,
        )
        runtime = explanation_output["runtime"]
        explanation_payload = explanation_output["payload"]
        existing_ids = [str(case.get("test_id") or "") for case in (bundle["suite_data"].get("test_cases") or [])]
        suggestion_payload = build_deterministic_suggested_test_payload(
            model=str(runtime.get("model") or settings.get("model") or "qwen3-coder:latest"),
            evidence=bundle["evidence"],
            original_test_case=bundle["test_case"],
            case_result=case_result,
            existing_test_ids=existing_ids,
            explanation_context={
                "signal": explanation_payload.get("signal"),
                "contract": explanation_payload.get("contract"),
                "explanation": explanation_payload.get("explanation"),
            },
        )
        response_payload = {
            "run_id": run_id,
            "test_id": test_id,
            "mode": "analysis",
            "payload": {
                "explanation": explanation_payload,
                "suggestion": suggestion_payload,
            },
        }
        _upsert_llm_run_insight(
            db=db,
            user_id=current_user.id,
            run_row=bundle.get("run_row"),
            test_id=test_id,
            mode="analysis",
            payload=response_payload["payload"],
        )
        if hasattr(db, "commit"):
            db.commit()
        return response_payload
    finally:
        db.close()


@app.post("/api/tests/{run_id}/cases/{test_id}/llm/suggest-test")
def suggest_test_for_failure(
    run_id: int,
    test_id: str,
    req: Optional[SuggestTestRequest] = None,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    db = SessionLocal()
    try:
        bundle = _load_run_case_bundle(
            db=db,
            user_id=current_user.id,
            run_id=run_id,
            test_id=test_id,
            selection_reason="user_requested_suggest_test",
        )
        case_result = bundle["case_result"]
        if str(case_result.get("outcome") or "").upper() != "FAIL":
            raise HTTPException(status_code=400, detail="Extra test suggestions are only available for failed tests.")

        settings = bundle["settings"]
        resolved_settings = _resolve_effective_llm_settings(
            db,
            current_user.id,
            base_settings=settings,
        )
        active_model = resolved_settings.get("active_model") if isinstance(resolved_settings.get("active_model"), dict) else {}
        resolved_model_name = str(active_model.get("model") or settings.get("model") or "").strip() or "qwen3-coder:latest"
        classified_signal = classify_failure_signal(bundle["evidence"])
        raw_explanation_context = req.explanation if req and isinstance(req.explanation, dict) else {}
        normalized_explanation_context = normalize_suggestion_explanation_context(
            raw_explanation_context,
            fallback_signal=classified_signal,
        )
        signal = str(normalized_explanation_context.get("signal") or classified_signal or "status_mismatch").strip().lower()
        if not is_input_related_signal(signal):
            skip_payload = build_suggestion_skip_payload(
                model=resolved_model_name,
                signal=signal,
                reason=(
                    "No suggested test was generated because this failure is not input-related. "
                    "Suggested tests are only generated for schema/validation input failures."
                ),
            )
            response_payload = {
                "run_id": run_id,
                "test_id": test_id,
                "mode": "suggest_test",
                "payload": skip_payload,
            }
            _upsert_llm_run_insight(
                db=db,
                user_id=current_user.id,
                run_row=bundle.get("run_row"),
                test_id=test_id,
                mode="suggest_test",
                payload=skip_payload,
            )
            if hasattr(db, "commit"):
                db.commit()
            return response_payload

        runtime = _build_runtime_from_active_model(resolved_settings)
        existing_ids = [str(case.get("test_id") or "") for case in (bundle["suite_data"].get("test_cases") or [])]
        payload = generate_suggested_test(
            client=runtime["client"],
            model=str(runtime["model"]),
            evidence=bundle["evidence"],
            prompt_bundle=bundle["prompt_bundle"],
            original_test_case=bundle["test_case"],
            case_result=case_result,
            existing_test_ids=existing_ids,
            retry_invalid_output=0,
            explanation_context=normalized_explanation_context,
            max_generation_seconds=SUGGESTION_TIMEOUT_SECONDS_MAX,
            ollama_options=runtime.get("options") if isinstance(runtime.get("options"), dict) else {},
        )
        suggestion_failure_kind = str(payload.get("failure_kind") or "").strip().lower()
        if is_rate_limit_or_quota_kind(suggestion_failure_kind):
            attempts = payload.get("attempts") if isinstance(payload.get("attempts"), list) else []
            diagnostics = [str(item).strip() for item in attempts if str(item).strip()]
            detail_payload = _build_llm_failure_detail(
                message="LLM suggested-test generation failed.",
                llm_error=payload.get("llm_error"),
                validation_errors=[],
                attempt_diagnostics=diagnostics,
                provider=str(runtime.get("provider") or ""),
                model=str(runtime.get("model") or ""),
                failure_kind=suggestion_failure_kind,
                status_hint=payload.get("status_code"),
                error_meta=payload.get("error_meta"),
            )
            raise HTTPException(status_code=int(detail_payload["status"]), detail=detail_payload)
        response_payload = {
            "run_id": run_id,
            "test_id": test_id,
            "mode": "suggest_test",
            "payload": payload,
        }
        _upsert_llm_run_insight(
            db=db,
            user_id=current_user.id,
            run_row=bundle.get("run_row"),
            test_id=test_id,
            mode="suggest_test",
            payload=payload,
        )
        if hasattr(db, "commit"):
            db.commit()
        return response_payload
    finally:
        db.close()


@app.get("/api/logistics/runs")
def list_logistics_runs(
    current_user: User = Depends(get_current_user),
    limit: Optional[int] = 25,
    before_run_id: Optional[int] = None,
    spec_query: Optional[str] = None,
    state: str = "all",
) -> dict[str, Any]:
    safe_limit = max(1, min(limit or 25, 100))
    normalized_state = str(state or "all").strip().lower() or "all"
    if normalized_state not in {"all", "passed", "failed"}:
        raise HTTPException(status_code=400, detail="state must be one of: all, passed, failed.")

    safe_spec_query = str(spec_query or "").strip().lower()
    safe_before_run_id: Optional[int]
    if before_run_id is None:
        safe_before_run_id = None
    else:
        safe_before_run_id = _coerce_non_negative_int(before_run_id)
        if safe_before_run_id <= 0:
            safe_before_run_id = None

    db = SessionLocal()
    try:
        base_query = (
            db.query(TestRun, Spec)
            .join(Spec, Spec.id == TestRun.spec_id)
            .filter(
                TestRun.user_id == current_user.id,
                Spec.user_id == current_user.id,
            )
        )
        if safe_spec_query:
            pattern = f"%{safe_spec_query}%"
            base_query = base_query.filter(
                or_(
                    func.lower(Spec.title).like(pattern),
                    func.lower(Spec.filename).like(pattern),
                    func.lower(Spec.version).like(pattern),
                )
            )

        collected: list[tuple[TestRun, Spec, dict[str, int]]] = []
        cursor = safe_before_run_id
        chunk_size = max(50, min(safe_limit * 4, 400))

        while len(collected) < (safe_limit + 1):
            page_query = base_query
            if cursor is not None:
                page_query = page_query.filter(TestRun.id < int(cursor))
            batch_rows = page_query.order_by(TestRun.id.desc()).limit(chunk_size).all()
            if not batch_rows:
                break

            for run_row, spec_row in batch_rows:
                summary = _normalize_run_summary_counts(_json_load_or_default(run_row.summary_json, {}))
                if not _run_matches_state_filter(summary, normalized_state):
                    continue
                collected.append((run_row, spec_row, summary))
                if len(collected) >= (safe_limit + 1):
                    break

            min_batch_id: Optional[int] = None
            for run_row, _ in batch_rows:
                if run_row.id is None:
                    continue
                run_id = int(run_row.id)
                min_batch_id = run_id if min_batch_id is None else min(min_batch_id, run_id)
            if len(batch_rows) < chunk_size or min_batch_id is None or min_batch_id <= 1:
                break
            cursor = min_batch_id

        has_more = len(collected) > safe_limit
        visible_rows = collected[:safe_limit]
        visible_run_ids = [int(run_row.id) for run_row, _, _ in visible_rows if run_row.id is not None]
        insight_summary_by_run = _summarize_llm_insights_for_runs(
            db,
            user_id=current_user.id,
            run_ids=visible_run_ids,
        )

        items: list[dict[str, Any]] = []
        for run_row, spec_row, summary in visible_rows:
            results_payload = _json_load_or_default(run_row.results_json, [])
            duration_ms = _derive_run_duration_ms(results_payload)
            auth_meta = _normalize_auth_meta(_json_load_or_default(run_row.auth_meta_json, {}))
            ai_summary = insight_summary_by_run.get(int(run_row.id), {
                "has_any": False,
                "has_explanations": False,
                "has_suggestions": False,
                "explanation_count": 0,
                "suggestion_count": 0,
                "model_list": [],
            })
            items.append(
                {
                    "spec": {
                        "id": int(spec_row.id),
                        "title": spec_row.title,
                        "filename": spec_row.filename,
                        "version": spec_row.version,
                        "uploaded_at": spec_row.created_at.isoformat() if spec_row.created_at else None,
                    },
                    "run": {
                        "id": int(run_row.id),
                        "created_at": run_row.created_at.isoformat() if run_row.created_at else None,
                        "summary": summary,
                        "duration_ms": duration_ms,
                        "base_url": run_row.base_url,
                    },
                    "auth": {
                        "provided": bool(auth_meta.get("provided")),
                        "mode": str(auth_meta.get("mode") or "none"),
                    },
                    "ai": ai_summary,
                }
            )

        next_before_run_id = int(visible_rows[-1][0].id) if has_more and visible_rows else None
        return {
            "items": items,
            "has_more": has_more,
            "next_before_run_id": next_before_run_id,
            "limit": safe_limit,
        }
    finally:
        db.close()


@app.get("/api/logistics/runs/{run_id}")
def get_logistics_run_detail(run_id: int, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
    db = SessionLocal()
    try:
        run_row = _get_owned_run(db, current_user.id, run_id)
        spec_row = _get_owned_spec(db, current_user.id, int(run_row.spec_id))

        suite_payload = _json_load_or_default(run_row.suite_snapshot_json, {})
        if not isinstance(suite_payload, dict):
            suite_payload = {}
        test_cases = suite_payload.get("test_cases") if isinstance(suite_payload.get("test_cases"), list) else []
        if not isinstance(test_cases, list):
            test_cases = []

        results_payload = _json_load_or_default(run_row.results_json, [])
        if not isinstance(results_payload, list):
            results_payload = []
        summary_payload = _normalize_run_summary_counts(_json_load_or_default(run_row.summary_json, {}))
        auth_meta = _normalize_auth_meta(_json_load_or_default(run_row.auth_meta_json, {}))
        duration_ms = _derive_run_duration_ms(results_payload)

        llm_rows = (
            db.query(LLMRunInsight)
            .filter(
                LLMRunInsight.user_id == current_user.id,
                LLMRunInsight.run_id == int(run_row.id),
            )
            .order_by(LLMRunInsight.id.desc())
            .all()
        )
        llm_payloads: list[dict[str, Any]] = []
        for row in llm_rows:
            payload = _json_load_or_default(row.payload_json, {})
            if payload is None:
                payload = {}
            llm_payloads.append(
                {
                    "id": int(row.id),
                    "test_id": str(row.test_id),
                    "mode": str(row.mode),
                    "model": str(row.model or ""),
                    "payload": payload,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                }
            )

        return {
            "spec": {
                "id": int(spec_row.id),
                "title": spec_row.title,
                "filename": spec_row.filename,
                "version": spec_row.version,
                "uploaded_at": spec_row.created_at.isoformat() if spec_row.created_at else None,
            },
            "run": {
                "id": int(run_row.id),
                "created_at": run_row.created_at.isoformat() if run_row.created_at else None,
                "api_title": run_row.api_title,
                "api_version": run_row.api_version,
                "base_url": run_row.base_url,
                "summary": summary_payload,
                "duration_ms": duration_ms,
                "timeout_seconds": int(run_row.timeout_seconds or 10),
            },
            "auth": {
                "provided": bool(auth_meta.get("provided")),
                "mode": str(auth_meta.get("mode") or "none"),
                "header": auth_meta.get("header"),
            },
            "suite": {
                "test_cases": test_cases,
            },
            "results": results_payload,
            "llm_insights": llm_payloads,
        }
    finally:
        db.close()


@app.delete("/api/logistics/runs/{run_id}")
def delete_logistics_run(run_id: int, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
    db = SessionLocal()
    try:
        run_row = _get_owned_run(db, current_user.id, run_id)
        deleted_insights = (
            db.query(LLMRunInsight)
            .filter(
                LLMRunInsight.user_id == current_user.id,
                LLMRunInsight.run_id == int(run_row.id),
            )
            .delete(synchronize_session=False)
        )
        db.delete(run_row)
        db.commit()
        return {
            "deleted": 1,
            "run_id": int(run_id),
            "deleted_llm_insights": int(deleted_insights or 0),
        }
    finally:
        db.close()


@app.delete("/api/logistics/runs")
def clear_logistics_runs(current_user: User = Depends(get_current_user)) -> dict[str, Any]:
    db = SessionLocal()
    try:
        run_ids = [
            int(row.id)
            for row in db.query(TestRun.id).filter(TestRun.user_id == current_user.id).all()
            if row.id is not None
        ]
        if not run_ids:
            return {
                "deleted_runs": 0,
                "deleted_llm_insights": 0,
            }

        deleted_insights = (
            db.query(LLMRunInsight)
            .filter(
                LLMRunInsight.user_id == current_user.id,
                LLMRunInsight.run_id.in_(run_ids),
            )
            .delete(synchronize_session=False)
        )
        deleted_runs = (
            db.query(TestRun)
            .filter(TestRun.user_id == current_user.id)
            .delete(synchronize_session=False)
        )
        db.commit()
        return {
            "deleted_runs": int(deleted_runs or 0),
            "deleted_llm_insights": int(deleted_insights or 0),
        }
    finally:
        db.close()


@app.delete("/api/specs")
def clear_specs_history(current_user: User = Depends(get_current_user)) -> dict[str, Any]:
    db = SessionLocal()
    try:
        spec_ids = [
            int(row.id)
            for row in db.query(Spec.id).filter(Spec.user_id == current_user.id).all()
            if row.id is not None
        ]
        if not spec_ids:
            return {"deleted": 0}

        db.query(LLMRunInsight).filter(
            LLMRunInsight.user_id == current_user.id,
            LLMRunInsight.spec_id.in_(spec_ids),
        ).delete(synchronize_session=False)
        db.query(TestRun).filter(
            TestRun.user_id == current_user.id,
            TestRun.spec_id.in_(spec_ids),
        ).delete(synchronize_session=False)
        db.query(SpecArtifact).filter(
            SpecArtifact.user_id == current_user.id,
            SpecArtifact.spec_id.in_(spec_ids),
        ).delete(synchronize_session=False)
        deleted_count = db.query(Spec).filter(Spec.user_id == current_user.id).delete(synchronize_session=False)
        _reset_table_id_sequences(
            db,
            [
                LLMRunInsight.__tablename__,
                TestRun.__tablename__,
                SpecArtifact.__tablename__,
                Spec.__tablename__,
            ],
        )
        db.commit()
        return {"deleted": deleted_count}
    finally:
        db.close()
