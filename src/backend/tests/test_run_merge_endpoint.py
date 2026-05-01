import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app import main
from backend.app.db import Base
from backend.app.models_db import LLMRunInsight, Spec, TestRun, User


class RunMergeEndpointTests(unittest.TestCase):
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

    def _create_spec(self, *, user_id: int, filename: str) -> int:
        db = self.TestSessionLocal()
        try:
            row = Spec(
                user_id=user_id,
                filename=filename,
                title=f"{filename} API",
                version="1.0.0",
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return int(row.id)
        finally:
            db.close()

    def _create_run(self, *, user_id: int, spec_id: int) -> int:
        db = self.TestSessionLocal()
        try:
            row = TestRun(
                user_id=user_id,
                spec_id=spec_id,
                api_title="Sample API",
                api_version="1.0.0",
                base_url="https://api.example.com",
                suite_snapshot_json=json.dumps({"test_cases": []}),
                results_json=json.dumps([]),
                summary_json=json.dumps({"total": 0, "passed": 0, "failed": 0, "skipped": 0}),
                auth_meta_json=json.dumps({"mode": "none", "provided": False}),
                timeout_seconds=10,
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return int(row.id)
        finally:
            db.close()

    def _create_insight(self, *, user_id: int, spec_id: int, run_id: int, test_id: str, mode: str) -> int:
        db = self.TestSessionLocal()
        try:
            row = LLMRunInsight(
                user_id=user_id,
                spec_id=spec_id,
                run_id=run_id,
                test_id=test_id,
                mode=mode,
                model="gpt-4.1-mini",
                payload_json=json.dumps({"mode": mode}),
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return int(row.id)
        finally:
            db.close()

    def _count_runs(self, *, user_id: int) -> int:
        db = self.TestSessionLocal()
        try:
            return int(db.query(TestRun).filter(TestRun.user_id == user_id).count())
        finally:
            db.close()

    def _get_run(self, run_id: int) -> TestRun:
        db = self.TestSessionLocal()
        try:
            row = db.query(TestRun).filter(TestRun.id == run_id).first()
            self.assertIsNotNone(row)
            return row
        finally:
            db.close()

    def _build_case(self, *, test_id: str, expected_status: int) -> dict:
        return {
            "test_id": test_id,
            "title": f"Case {test_id}",
            "category": "happy_path",
            "method": "GET",
            "path": f"/{test_id.lower()}",
            "steps": [
                {
                    "step_number": 1,
                    "action": "Execute request",
                    "input_data": {
                        "path_params": {},
                        "query_params": {},
                        "headers": {},
                    },
                }
            ],
            "expected_result": {
                "status_code": expected_status,
            },
        }

    def _build_result(self, *, test_id: str, outcome: str) -> dict:
        return {
            "test_id": test_id,
            "title": f"Case {test_id}",
            "method": "GET",
            "path": f"/{test_id.lower()}",
            "expected_status": 200,
            "expected_status_any_of": None,
            "actual_status": 200 if outcome == "PASS" else 500,
            "outcome": outcome,
            "response_snippet": "",
            "response_body": "",
            "response_headers": {},
            "request_path_params": {},
            "request_query_params": {},
            "request_headers": {},
            "request_body": None,
            "error_message": "" if outcome == "PASS" else "Failure",
            "duration_ms": 42,
            "final_url": "https://api.example.com",
        }

    def test_individual_rerun_merges_into_existing_run_updates_summary_and_clears_case_insights(self) -> None:
        user_id = self._create_user("merge-owner@example.com")
        spec_id = self._create_spec(user_id=user_id, filename="orders.yaml")
        tc_1 = self._build_case(test_id="TC-001", expected_status=200)
        tc_2 = self._build_case(test_id="TC-002", expected_status=200)
        tc_2_updated = self._build_case(test_id="TC-002", expected_status=201)

        first_results = [
            self._build_result(test_id="TC-001", outcome="PASS"),
            self._build_result(test_id="TC-002", outcome="PASS"),
        ]
        rerun_results = [
            self._build_result(test_id="TC-002", outcome="FAIL"),
        ]

        with (
            patch.object(main, "SessionLocal", self.TestSessionLocal),
            patch("test_runner.run_test.resolve_base_url", return_value="https://api.example.com"),
            patch(
                "test_runner.run_test.run_suite",
                side_effect=[
                    (first_results, {"total": 2, "passed": 2, "failed": 0, "skipped": 0}),
                    (rerun_results, {"total": 1, "passed": 0, "failed": 1, "skipped": 0}),
                ],
            ),
        ):
            first = main.run_generated_tests(
                req=main.RunTestsRequest(
                    spec_id=spec_id,
                    api_title="Orders API",
                    api_version="1.0.0",
                    base_url="https://api.example.com",
                    test_cases=[tc_1, tc_2],
                ),
                current_user=SimpleNamespace(id=user_id),
            )
            run_id = int(first["run_id"])
            self._create_insight(user_id=user_id, spec_id=spec_id, run_id=run_id, test_id="TC-001", mode="analysis")
            self._create_insight(user_id=user_id, spec_id=spec_id, run_id=run_id, test_id="TC-002", mode="analysis")
            self._create_insight(user_id=user_id, spec_id=spec_id, run_id=run_id, test_id="TC-002", mode="suggest_test")

            rerun = main.run_generated_tests(
                req=main.RunTestsRequest(
                    spec_id=spec_id,
                    api_title="Orders API",
                    api_version="1.0.0",
                    base_url="https://api.example.com",
                    test_cases=[tc_2_updated],
                    merge_into_run_id=run_id,
                ),
                current_user=SimpleNamespace(id=user_id),
            )

        self.assertEqual(int(rerun["run_id"]), run_id)
        self.assertEqual(self._count_runs(user_id=user_id), 1)
        self.assertEqual(
            rerun["summary"],
            {"total": 2, "passed": 1, "failed": 1, "skipped": 0},
        )
        merged_by_id = {str(row.get("test_id")): row for row in rerun["results"]}
        self.assertEqual(str(merged_by_id["TC-001"]["outcome"]), "PASS")
        self.assertEqual(str(merged_by_id["TC-002"]["outcome"]), "FAIL")

        stored = self._get_run(run_id)
        stored_suite = json.loads(stored.suite_snapshot_json)
        stored_cases = stored_suite.get("test_cases") if isinstance(stored_suite.get("test_cases"), list) else []
        stored_case_by_id = {str(case.get("test_id")): case for case in stored_cases if isinstance(case, dict)}
        self.assertEqual(
            int(stored_case_by_id["TC-002"]["expected_result"]["status_code"]),
            201,
        )

        db = self.TestSessionLocal()
        try:
            tc1_insights = (
                db.query(LLMRunInsight)
                .filter(
                    LLMRunInsight.user_id == user_id,
                    LLMRunInsight.run_id == run_id,
                    LLMRunInsight.test_id == "TC-001",
                )
                .count()
            )
            tc2_insights = (
                db.query(LLMRunInsight)
                .filter(
                    LLMRunInsight.user_id == user_id,
                    LLMRunInsight.run_id == run_id,
                    LLMRunInsight.test_id == "TC-002",
                )
                .count()
            )
        finally:
            db.close()

        self.assertEqual(int(tc1_insights), 1)
        self.assertEqual(int(tc2_insights), 0)

    def test_merge_run_id_spec_mismatch_is_rejected(self) -> None:
        user_id = self._create_user("mismatch@example.com")
        spec_a = self._create_spec(user_id=user_id, filename="a.yaml")
        spec_b = self._create_spec(user_id=user_id, filename="b.yaml")
        run_id = self._create_run(user_id=user_id, spec_id=spec_a)

        with (
            patch.object(main, "SessionLocal", self.TestSessionLocal),
            patch("test_runner.run_test.resolve_base_url", return_value="https://api.example.com"),
            patch("test_runner.run_test.run_suite", return_value=([], {"total": 0, "passed": 0, "failed": 0, "skipped": 0})),
        ):
            with self.assertRaises(HTTPException) as ctx:
                main.run_generated_tests(
                    req=main.RunTestsRequest(
                        spec_id=spec_b,
                        api_title="Mismatch API",
                        api_version="1.0.0",
                        base_url="https://api.example.com",
                        test_cases=[self._build_case(test_id="TC-001", expected_status=200)],
                        merge_into_run_id=run_id,
                    ),
                    current_user=SimpleNamespace(id=user_id),
                )

        self.assertEqual(ctx.exception.status_code, 400)

    def test_merge_run_id_requires_run_ownership(self) -> None:
        owner_id = self._create_user("owner@example.com")
        other_id = self._create_user("other@example.com")
        owner_spec = self._create_spec(user_id=owner_id, filename="owner.yaml")
        other_spec = self._create_spec(user_id=other_id, filename="other.yaml")
        other_run_id = self._create_run(user_id=other_id, spec_id=other_spec)

        with (
            patch.object(main, "SessionLocal", self.TestSessionLocal),
            patch("test_runner.run_test.resolve_base_url", return_value="https://api.example.com"),
            patch("test_runner.run_test.run_suite", return_value=([], {"total": 0, "passed": 0, "failed": 0, "skipped": 0})),
        ):
            with self.assertRaises(HTTPException) as ctx:
                main.run_generated_tests(
                    req=main.RunTestsRequest(
                        spec_id=owner_spec,
                        api_title="Owner API",
                        api_version="1.0.0",
                        base_url="https://api.example.com",
                        test_cases=[self._build_case(test_id="TC-001", expected_status=200)],
                        merge_into_run_id=other_run_id,
                    ),
                    current_user=SimpleNamespace(id=owner_id),
                )

        self.assertEqual(ctx.exception.status_code, 404)

    def test_full_runs_without_merge_still_create_new_rows(self) -> None:
        user_id = self._create_user("full-run@example.com")
        spec_id = self._create_spec(user_id=user_id, filename="full.yaml")
        case = self._build_case(test_id="TC-001", expected_status=200)
        result = [self._build_result(test_id="TC-001", outcome="PASS")]

        with (
            patch.object(main, "SessionLocal", self.TestSessionLocal),
            patch("test_runner.run_test.resolve_base_url", return_value="https://api.example.com"),
            patch("test_runner.run_test.run_suite", return_value=(result, {"total": 1, "passed": 1, "failed": 0, "skipped": 0})),
        ):
            first = main.run_generated_tests(
                req=main.RunTestsRequest(
                    spec_id=spec_id,
                    api_title="Full API",
                    api_version="1.0.0",
                    base_url="https://api.example.com",
                    test_cases=[case],
                ),
                current_user=SimpleNamespace(id=user_id),
            )
            second = main.run_generated_tests(
                req=main.RunTestsRequest(
                    spec_id=spec_id,
                    api_title="Full API",
                    api_version="1.0.0",
                    base_url="https://api.example.com",
                    test_cases=[case],
                ),
                current_user=SimpleNamespace(id=user_id),
            )

        self.assertNotEqual(int(first["run_id"]), int(second["run_id"]))
        self.assertEqual(self._count_runs(user_id=user_id), 2)


if __name__ == "__main__":
    unittest.main()
