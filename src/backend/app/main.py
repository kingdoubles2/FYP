import json
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llm_eval.failure_assistant import (
    build_assistant_prompt_bundle,
    build_case_evidence,
    build_pipeline_context,
    generate_failure_explanation,
    generate_suggested_test,
    load_backend_llm_settings,
    parse_spec_document,
    safe_json_dumps,
    safe_json_loads,
    sha256_text,
)
from llm_eval.ollama_client import OllamaClient
from spec_parser.parser import parse_openapi
from test_generator.generator import generate_test_cases

from .auth import create_access_token, hash_password, verify_password
from .db import SessionLocal
from .deps import get_current_user
from .init_db import init_db
from .models_db import LLMRunInsight, Spec, SpecArtifact, TestRun, User


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


def _validate_credentials(req: RegisterRequest | LoginRequest) -> None:
    email = req.email.strip().lower()
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Please provide a valid email address.")
    if len(req.password.encode("utf-8")) > 72:
        raise HTTPException(status_code=400, detail="Password must be 72 bytes or fewer.")


def _json_load_or_default(raw: Optional[str], default: Any) -> Any:
    parsed = safe_json_loads(raw)
    return parsed if parsed is not None else default


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
    settings = load_backend_llm_settings()
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
    try:
        _ = _get_owned_spec(db, current_user.id, spec_id)
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
    finally:
        db.close()

    return {
        "run_id": run_row.id,
        "spec_id": spec_id,
        "api_title": suite_data["api_title"],
        "api_version": suite_data["api_version"],
        "base_url": resolved_base_url,
        "summary": summary,
        "results": results,
    }


@app.get("/api/specs")
def list_specs(current_user: User = Depends(get_current_user), limit: Optional[int] = 50) -> list[dict[str, Any]]:
    safe_limit = max(1, min(limit or 50, 200))
    db = SessionLocal()
    try:
        rows = (
            db.query(Spec)
            .filter(Spec.user_id == current_user.id)
            .order_by(Spec.id.desc())
            .limit(safe_limit)
            .all()
        )

        return [
            {
                "id": row.id,
                "filename": row.filename,
                "title": row.title,
                "version": row.version,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
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
        client = OllamaClient(
            base_url=str(settings["base_url"]),
            timeout_seconds=int(settings["timeout_seconds"]),
        )
        payload = generate_failure_explanation(
            client=client,
            model=str(settings["model"]),
            evidence=bundle["evidence"],
            prompt_bundle=bundle["prompt_bundle"],
            word_min=int(settings["word_min"]),
            word_max=int(settings["word_max"]),
            retry_invalid_output=int(settings["retry_invalid_output"]),
            ollama_options=settings.get("ollama_options") or {},
        )

        return {
            "run_id": run_id,
            "test_id": test_id,
            "mode": "explanation",
            "payload": payload,
        }
    finally:
        db.close()


@app.post("/api/tests/{run_id}/cases/{test_id}/llm/suggest-test")
def suggest_test_for_failure(run_id: int, test_id: str, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
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
        client = OllamaClient(
            base_url=str(settings["base_url"]),
            timeout_seconds=int(settings["timeout_seconds"]),
        )
        existing_ids = [str(case.get("test_id") or "") for case in (bundle["suite_data"].get("test_cases") or [])]
        payload = generate_suggested_test(
            client=client,
            model=str(settings["model"]),
            evidence=bundle["evidence"],
            prompt_bundle=bundle["prompt_bundle"],
            original_test_case=bundle["test_case"],
            case_result=case_result,
            existing_test_ids=existing_ids,
            retry_invalid_output=int(settings["retry_invalid_output"]),
            ollama_options=settings.get("ollama_options") or {},
        )
        return {
            "run_id": run_id,
            "test_id": test_id,
            "mode": "suggest_test",
            "payload": payload,
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
        db.commit()
        return {"deleted": deleted_count}
    finally:
        db.close()
