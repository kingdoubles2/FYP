from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, UniqueConstraint, func

from .db import Base


class Spec(Base):
    __tablename__ = "specs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    title = Column(String, nullable=False)
    version = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("email", name="uq_users_email"),)

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

