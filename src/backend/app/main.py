import json
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from spec_parser.parser import parse_openapi
from test_generator.generator import generate_test_cases

from .auth import create_access_token, hash_password, verify_password
from .db import SessionLocal
from .deps import get_current_user
from .init_db import init_db
from .models_db import Spec, User


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


def _validate_credentials(req: RegisterRequest | LoginRequest) -> None:
    email = req.email.strip().lower()
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Please provide a valid email address.")
    if len(req.password.encode("utf-8")) > 72:
        raise HTTPException(status_code=400, detail="Password must be 72 bytes or fewer.")


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
