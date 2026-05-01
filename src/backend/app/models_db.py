from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func

from .db import Base


class Spec(Base):
    __tablename__ = "specs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    title = Column(String, nullable=False)
    version = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class SpecArtifact(Base):
    __tablename__ = "spec_artifacts"
    __table_args__ = (UniqueConstraint("spec_id", name="uq_spec_artifacts_spec_id"),)

    id = Column(Integer, primary_key=True, index=True)
    spec_id = Column(Integer, ForeignKey("specs.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    spec_hash = Column(String, nullable=True)
    spec_raw_text = Column(Text, nullable=False)
    parsed_ir_json = Column(Text, nullable=False)
    generated_suite_json = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class TestRun(Base):
    __tablename__ = "test_runs"

    id = Column(Integer, primary_key=True, index=True)
    spec_id = Column(Integer, ForeignKey("specs.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    api_title = Column(String, nullable=False)
    api_version = Column(String, nullable=False)
    base_url = Column(String, nullable=False)
    suite_snapshot_json = Column(Text, nullable=False)
    results_json = Column(Text, nullable=False)
    summary_json = Column(Text, nullable=False)
    auth_meta_json = Column(Text, nullable=False)
    timeout_seconds = Column(Integer, nullable=False, default=10)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class LLMRunInsight(Base):
    __tablename__ = "llm_run_insights"
    __table_args__ = (UniqueConstraint("run_id", "test_id", "mode", name="uq_llm_run_insight_case_mode"),)

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("test_runs.id"), nullable=False, index=True)
    spec_id = Column(Integer, ForeignKey("specs.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    test_id = Column(String, nullable=False, index=True)
    mode = Column(String, nullable=False, index=True)
    model = Column(String, nullable=True)
    payload_json = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class UserLLMSettings(Base):
    __tablename__ = "user_llm_settings"
    __table_args__ = (UniqueConstraint("user_id", name="uq_user_llm_settings_user_id"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    active_model_id = Column(String, nullable=True)
    custom_instruction = Column(Text, nullable=False, default="")
    saved_models_json = Column(Text, nullable=False, default="[]")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("email", name="uq_users_email"),)

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

