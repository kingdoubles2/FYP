from __future__ import annotations

import re
from typing import Any, Optional, Protocol, TypedDict


class LLMErrorMeta(TypedDict, total=False):
    kind: str
    message: str
    status_code: Optional[int]
    provider_error_code: str
    provider_error_type: str
    retryable: bool
    raw_error: str


class LLMGenerateClient(Protocol):
    timeout_seconds: int

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        format_json: bool = False,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[Optional[str], Optional[Any]]:
        ...


_FAILURE_KIND_RATE_LIMIT = "rate_limit"
_FAILURE_KIND_QUOTA = "quota"
_FAILURE_KIND_TIMEOUT = "timeout"
_FAILURE_KIND_TRANSPORT = "transport"
_FAILURE_KIND_INVALID_RESPONSE = "invalid_response"
_FAILURE_KIND_OTHER = "other"

_KNOWN_FAILURE_KINDS = {
    _FAILURE_KIND_RATE_LIMIT,
    _FAILURE_KIND_QUOTA,
    _FAILURE_KIND_TIMEOUT,
    _FAILURE_KIND_TRANSPORT,
    _FAILURE_KIND_INVALID_RESPONSE,
    _FAILURE_KIND_OTHER,
}

_STATUS_FROM_CLIENT_ERROR_RE = re.compile(r"\b(?P<code>\d{3})\s+Client Error\b", flags=re.IGNORECASE)
_STATUS_FROM_GENERIC_RE = re.compile(r"\bstatus(?:_code)?(?:=|:)?\s*(?P<code>\d{3})\b", flags=re.IGNORECASE)

_QUOTA_TOKENS = (
    "insufficient_quota",
    "quota exceeded",
    "quota",
    "billing hard limit",
    "billing_limit",
)
_RATE_LIMIT_TOKENS = (
    "too many requests",
    "rate limit",
    "ratelimit",
    "rate_limit",
)
_TIMEOUT_TOKENS = (
    "timeout",
    "timed out",
    "readtimeout",
    "connecttimeout",
)
_TRANSPORT_TOKENS = (
    "connectionerror",
    "proxyerror",
    "sslerror",
    "connection aborted",
    "connection reset",
    "connection refused",
    "name resolution",
    "dns",
)
_INVALID_RESPONSE_TOKENS = (
    "missing choices",
    "missing text content",
    "missing message content",
    "unexpected openai response payload",
    "unexpected anthropic response payload",
    "invalid_json",
)


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    if parsed < 100 or parsed > 599:
        return None
    return parsed


def _extract_status_code_from_text(message: str) -> Optional[int]:
    text = str(message or "")
    match = _STATUS_FROM_CLIENT_ERROR_RE.search(text)
    if match:
        return _coerce_optional_int(match.group("code"))
    match = _STATUS_FROM_GENERIC_RE.search(text)
    if match:
        return _coerce_optional_int(match.group("code"))
    return None


def classify_llm_failure_kind(
    *,
    status_code: Optional[int],
    provider_error_code: str,
    provider_error_type: str,
    message: str,
) -> str:
    lowered_message = str(message or "").lower()
    lowered_code = str(provider_error_code or "").lower()
    lowered_type = str(provider_error_type or "").lower()

    quota_hit = any(token in lowered_code for token in _QUOTA_TOKENS) or any(token in lowered_message for token in _QUOTA_TOKENS)
    if quota_hit:
        return _FAILURE_KIND_QUOTA

    rate_limit_hit = (
        status_code == 429
        or any(token in lowered_code for token in _RATE_LIMIT_TOKENS)
        or any(token in lowered_type for token in _RATE_LIMIT_TOKENS)
        or any(token in lowered_message for token in _RATE_LIMIT_TOKENS)
    )
    if rate_limit_hit:
        return _FAILURE_KIND_RATE_LIMIT

    if any(token in lowered_message for token in _TIMEOUT_TOKENS):
        return _FAILURE_KIND_TIMEOUT

    if any(token in lowered_message for token in _TRANSPORT_TOKENS):
        return _FAILURE_KIND_TRANSPORT

    if any(token in lowered_message for token in _INVALID_RESPONSE_TOKENS):
        return _FAILURE_KIND_INVALID_RESPONSE

    return _FAILURE_KIND_OTHER


def _retryable_for_kind(kind: str) -> bool:
    normalized = str(kind or "").strip().lower()
    if normalized == _FAILURE_KIND_QUOTA:
        return False
    return normalized in {_FAILURE_KIND_RATE_LIMIT, _FAILURE_KIND_TIMEOUT, _FAILURE_KIND_TRANSPORT}


def build_llm_error_meta(
    *,
    message: str,
    status_code: Optional[int] = None,
    provider_error_code: str = "",
    provider_error_type: str = "",
    kind: Optional[str] = None,
    retryable: Optional[bool] = None,
    raw_error: str = "",
) -> LLMErrorMeta:
    clean_message = str(message or "").strip() or "LLM request failed."
    clean_status = _coerce_optional_int(status_code)
    clean_code = str(provider_error_code or "").strip()
    clean_type = str(provider_error_type or "").strip()
    resolved_kind = str(kind or "").strip().lower()
    if resolved_kind not in _KNOWN_FAILURE_KINDS:
        resolved_kind = classify_llm_failure_kind(
            status_code=clean_status,
            provider_error_code=clean_code,
            provider_error_type=clean_type,
            message=clean_message,
        )
    resolved_retryable = _retryable_for_kind(resolved_kind) if retryable is None else bool(retryable)
    return {
        "kind": resolved_kind,
        "message": clean_message,
        "status_code": clean_status,
        "provider_error_code": clean_code,
        "provider_error_type": clean_type,
        "retryable": resolved_retryable,
        "raw_error": str(raw_error or clean_message),
    }


def normalize_llm_error_meta(raw_error: Any) -> Optional[LLMErrorMeta]:
    if raw_error is None:
        return None

    if isinstance(raw_error, dict):
        message = str(raw_error.get("message") or raw_error.get("error") or "").strip() or str(raw_error)
        status_code = _coerce_optional_int(raw_error.get("status_code"))
        if status_code is None:
            status_code = _extract_status_code_from_text(message)
        return build_llm_error_meta(
            message=message,
            status_code=status_code,
            provider_error_code=str(raw_error.get("provider_error_code") or raw_error.get("code") or ""),
            provider_error_type=str(raw_error.get("provider_error_type") or raw_error.get("type") or ""),
            kind=str(raw_error.get("kind") or ""),
            retryable=raw_error.get("retryable") if isinstance(raw_error.get("retryable"), bool) else None,
            raw_error=str(raw_error.get("raw_error") or message),
        )

    message = str(raw_error).strip()
    status_code = _extract_status_code_from_text(message)
    return build_llm_error_meta(
        message=message,
        status_code=status_code,
        raw_error=message,
    )


def is_rate_limit_or_quota_kind(kind: Any) -> bool:
    normalized = str(kind or "").strip().lower()
    return normalized in {_FAILURE_KIND_RATE_LIMIT, _FAILURE_KIND_QUOTA}

