from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .auth import decode_access_token
from .db import SessionLocal
from .models_db import User


bearer_scheme = HTTPBearer(auto_error=True)


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> User:
    try:
        payload = decode_access_token(credentials.credentials)
        sub = payload.get("sub")
        if sub is None:
            raise ValueError("Token missing subject.")
        user_id = int(sub)
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token. {exc}") from exc

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found.")
        return user
    finally:
        db.close()

