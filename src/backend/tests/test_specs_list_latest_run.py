import json
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app import main
from backend.app.db import Base
from backend.app.models_db import Spec, TestRun, User


class ListSpecsLatestRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            future=True,
            connect_args={"check_same_thread": False},
        )
        self.TestSessionLocal = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
            future=True,
        )
        Base.metadata.create_all(bind=self.engine)

    def tearDown(self) -> None:
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    def _create_user(self, email: str) -> int:
        db = self.TestSessionLocal()
        try:
            row = User(email=email, password_hash="x")
            db.add(row)
            db.commit()
            db.refresh(row)
            return int(row.id)
        finally:
            db.close()

    def _create_spec(self, user_id: int, name: str) -> int:
        db = self.TestSessionLocal()
        try:
            row = Spec(
                user_id=user_id,
                filename=f"{name}.json",
                title=f"{name} API",
                version="1.0.0",
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return int(row.id)
        finally:
            db.close()

    def _create_run(self, *, spec_id: int, user_id: int, summary: Any) -> int:
        db = self.TestSessionLocal()
        try:
            if isinstance(summary, str):
                summary_raw = summary
            else:
                summary_raw = json.dumps(summary)
            row = TestRun(
                spec_id=spec_id,
                user_id=user_id,
                api_title="Example API",
                api_version="1.0.0",
                base_url="https://api.example.com",
                suite_snapshot_json=json.dumps({"test_cases": []}),
                results_json=json.dumps([]),
                summary_json=summary_raw,
                auth_meta_json=json.dumps({"mode": "none"}),
                timeout_seconds=10,
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return int(row.id)
        finally:
            db.close()

    def test_list_specs_includes_latest_run_summary(self) -> None:
        user_id = self._create_user("user1@example.com")
        spec_id = self._create_spec(user_id, "pets")
        self._create_run(
            spec_id=spec_id,
            user_id=user_id,
            summary={"total": 12, "passed": 8, "failed": 3, "skipped": 1},
        )
        latest_run_id = self._create_run(
            spec_id=spec_id,
            user_id=user_id,
            summary={"total": 12, "passed": 9, "failed": 2, "skipped": 1},
        )

        with patch.object(main, "SessionLocal", self.TestSessionLocal):
            payload = main.list_specs(current_user=SimpleNamespace(id=user_id))

        row = next(item for item in payload if int(item["id"]) == spec_id)
        self.assertIsInstance(row.get("latest_run"), dict)
        self.assertEqual(int(row["latest_run"]["id"]), latest_run_id)
        self.assertEqual(
            row["latest_run"]["summary"],
            {"total": 12, "passed": 9, "failed": 2, "skipped": 1},
        )

    def test_list_specs_returns_null_latest_run_when_no_runs_exist(self) -> None:
        user_id = self._create_user("user2@example.com")
        spec_id = self._create_spec(user_id, "billing")

        with patch.object(main, "SessionLocal", self.TestSessionLocal):
            payload = main.list_specs(current_user=SimpleNamespace(id=user_id))

        row = next(item for item in payload if int(item["id"]) == spec_id)
        self.assertIsNone(row.get("latest_run"))

    def test_list_specs_uses_latest_run_per_spec_only(self) -> None:
        user_id = self._create_user("user3@example.com")
        spec_id = self._create_spec(user_id, "orders")
        self._create_run(
            spec_id=spec_id,
            user_id=user_id,
            summary={"total": 5, "passed": 5, "failed": 0, "skipped": 0},
        )
        latest_run_id = self._create_run(
            spec_id=spec_id,
            user_id=user_id,
            summary={"total": 5, "passed": 4, "failed": 1, "skipped": 0},
        )

        with patch.object(main, "SessionLocal", self.TestSessionLocal):
            payload = main.list_specs(current_user=SimpleNamespace(id=user_id))

        row = next(item for item in payload if int(item["id"]) == spec_id)
        self.assertEqual(int(row["latest_run"]["id"]), latest_run_id)
        self.assertEqual(
            row["latest_run"]["summary"],
            {"total": 5, "passed": 4, "failed": 1, "skipped": 0},
        )

    def test_list_specs_enforces_user_isolation_for_latest_run_selection(self) -> None:
        owner_id = self._create_user("owner@example.com")
        other_user_id = self._create_user("other@example.com")
        spec_id = self._create_spec(owner_id, "inventory")
        owner_run_id = self._create_run(
            spec_id=spec_id,
            user_id=owner_id,
            summary={"total": 7, "passed": 6, "failed": 1, "skipped": 0},
        )
        _ = self._create_run(
            spec_id=spec_id,
            user_id=other_user_id,
            summary={"total": 7, "passed": 0, "failed": 7, "skipped": 0},
        )

        with patch.object(main, "SessionLocal", self.TestSessionLocal):
            payload = main.list_specs(current_user=SimpleNamespace(id=owner_id))

        row = next(item for item in payload if int(item["id"]) == spec_id)
        self.assertEqual(int(row["latest_run"]["id"]), owner_run_id)
        self.assertEqual(
            row["latest_run"]["summary"],
            {"total": 7, "passed": 6, "failed": 1, "skipped": 0},
        )

    def test_list_specs_malformed_summary_defaults_to_zero_counts(self) -> None:
        user_id = self._create_user("user4@example.com")
        spec_id = self._create_spec(user_id, "weather")
        self._create_run(
            spec_id=spec_id,
            user_id=user_id,
            summary="not-json",
        )

        with patch.object(main, "SessionLocal", self.TestSessionLocal):
            payload = main.list_specs(current_user=SimpleNamespace(id=user_id))

        row = next(item for item in payload if int(item["id"]) == spec_id)
        self.assertEqual(
            row["latest_run"]["summary"],
            {"total": 0, "passed": 0, "failed": 0, "skipped": 0},
        )


if __name__ == "__main__":
    unittest.main()
