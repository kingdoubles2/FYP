import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app import main
from backend.app.db import Base
from backend.app.models_db import LLMRunInsight, Spec, SpecArtifact, TestRun, User


class LogisticsEndpointTests(unittest.TestCase):
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

    def _create_spec(self, *, user_id: int, filename: str, title: str, version: str = "1.0.0") -> int:
        db = self.TestSessionLocal()
        try:
            row = Spec(
                user_id=user_id,
                filename=filename,
                title=title,
                version=version,
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return int(row.id)
        finally:
            db.close()

    def _create_artifact(self, *, user_id: int, spec_id: int) -> int:
        db = self.TestSessionLocal()
        try:
            row = SpecArtifact(
                user_id=user_id,
                spec_id=spec_id,
                spec_hash=f"hash-{spec_id}",
                spec_raw_text="openapi: 3.0.0",
                parsed_ir_json=json.dumps({"endpoints": []}),
                generated_suite_json=json.dumps({"test_cases": []}),
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return int(row.id)
        finally:
            db.close()

    def _create_run(
        self,
        *,
        user_id: int,
        spec_id: int,
        summary: dict | None = None,
        results: list | None = None,
        auth_meta: dict | None = None,
        base_url: str = "https://api.example.com",
    ) -> int:
        db = self.TestSessionLocal()
        try:
            row = TestRun(
                user_id=user_id,
                spec_id=spec_id,
                api_title="Sample API",
                api_version="1.0.0",
                base_url=base_url,
                suite_snapshot_json=json.dumps(
                    {
                        "test_cases": [
                            {
                                "test_id": "TC-001",
                                "title": "Sample test",
                                "method": "GET",
                                "path": "/health",
                            }
                        ]
                    }
                ),
                results_json=json.dumps(
                    results
                    if isinstance(results, list)
                    else [
                        {
                            "test_id": "TC-001",
                            "outcome": "PASS",
                            "duration_ms": 120,
                            "actual_status": 200,
                        }
                    ]
                ),
                summary_json=json.dumps(
                    summary
                    if isinstance(summary, dict)
                    else {"total": 1, "passed": 1, "failed": 0, "skipped": 0}
                ),
                auth_meta_json=json.dumps(
                    auth_meta
                    if isinstance(auth_meta, dict)
                    else {"mode": "none", "provided": False}
                ),
                timeout_seconds=10,
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return int(row.id)
        finally:
            db.close()

    def _create_insight(
        self,
        *,
        user_id: int,
        spec_id: int,
        run_id: int,
        test_id: str,
        mode: str,
        model: str,
        payload: dict,
    ) -> int:
        db = self.TestSessionLocal()
        try:
            row = LLMRunInsight(
                user_id=user_id,
                spec_id=spec_id,
                run_id=run_id,
                test_id=test_id,
                mode=mode,
                model=model,
                payload_json=json.dumps(payload),
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return int(row.id)
        finally:
            db.close()

    def _fetch_run_row(self, run_id: int) -> TestRun:
        db = self.TestSessionLocal()
        try:
            row = db.query(TestRun).filter(TestRun.id == run_id).first()
            self.assertIsNotNone(row)
            return row
        finally:
            db.close()

    def test_list_logistics_runs_owner_scoped_paginated_and_filtered(self) -> None:
        owner_id = self._create_user("owner@example.com")
        other_id = self._create_user("other@example.com")
        spec_pet = self._create_spec(user_id=owner_id, filename="petstore.yaml", title="Petstore API", version="1.0")
        spec_weather = self._create_spec(user_id=owner_id, filename="weather.yaml", title="Weather API", version="2.0")
        other_spec = self._create_spec(user_id=other_id, filename="other.yaml", title="Other API", version="1.0")

        run_1 = self._create_run(
            user_id=owner_id,
            spec_id=spec_pet,
            summary={"total": 3, "passed": 3, "failed": 0, "skipped": 0},
        )
        run_2 = self._create_run(
            user_id=owner_id,
            spec_id=spec_weather,
            summary={"total": 4, "passed": 2, "failed": 2, "skipped": 0},
            results=[
                {"test_id": "TC-001", "outcome": "PASS", "duration_ms": 100},
                {"test_id": "TC-002", "outcome": "FAIL", "duration_ms": 220},
            ],
            auth_meta={"mode": "bearer", "provided": True, "token": "super-secret-token"},
        )
        _ = self._create_run(
            user_id=other_id,
            spec_id=other_spec,
            summary={"total": 2, "passed": 0, "failed": 2, "skipped": 0},
        )
        _ = self._create_insight(
            user_id=owner_id,
            spec_id=spec_weather,
            run_id=run_2,
            test_id="TC-001",
            mode="explanation",
            model="gpt-4.1-mini",
            payload={"mode": "explanation", "model": "gpt-4.1-mini", "explanation": "Sample"},
        )

        with patch.object(main, "SessionLocal", self.TestSessionLocal):
            page_one = main.list_logistics_runs(
                current_user=SimpleNamespace(id=owner_id),
                limit=1,
                state="all",
            )

            self.assertEqual(len(page_one["items"]), 1)
            self.assertTrue(bool(page_one["has_more"]))
            first_row = page_one["items"][0]
            self.assertEqual(int(first_row["run"]["id"]), run_2)
            self.assertEqual(first_row["run"]["summary"]["failed"], 2)
            self.assertEqual(int(first_row["run"]["duration_ms"]), 320)
            self.assertEqual(first_row["auth"], {"provided": True, "mode": "bearer"})
            self.assertTrue(first_row["ai"]["has_any"])
            self.assertEqual(first_row["ai"]["model_list"], ["gpt-4.1-mini"])

            page_two = main.list_logistics_runs(
                current_user=SimpleNamespace(id=owner_id),
                limit=5,
                before_run_id=page_one["next_before_run_id"],
                state="all",
            )
            self.assertEqual([int(item["run"]["id"]) for item in page_two["items"]], [run_1])

            failed_only = main.list_logistics_runs(
                current_user=SimpleNamespace(id=owner_id),
                limit=10,
                state="failed",
            )
            self.assertEqual([int(item["run"]["id"]) for item in failed_only["items"]], [run_2])

            passed_only = main.list_logistics_runs(
                current_user=SimpleNamespace(id=owner_id),
                limit=10,
                state="passed",
            )
            self.assertEqual([int(item["run"]["id"]) for item in passed_only["items"]], [run_1])

            by_query = main.list_logistics_runs(
                current_user=SimpleNamespace(id=owner_id),
                limit=10,
                spec_query="weather",
                state="all",
            )
            self.assertEqual([int(item["run"]["id"]) for item in by_query["items"]], [run_2])

        self.assertNotEqual(run_1, run_2)

    def test_logistics_detail_owner_scope_and_auth_secrecy(self) -> None:
        owner_id = self._create_user("owner2@example.com")
        other_id = self._create_user("other2@example.com")
        owner_spec = self._create_spec(user_id=owner_id, filename="owner.yaml", title="Owner API")
        other_spec = self._create_spec(user_id=other_id, filename="other.yaml", title="Other API")
        owner_run = self._create_run(
            user_id=owner_id,
            spec_id=owner_spec,
            summary={"total": 2, "passed": 1, "failed": 1, "skipped": 0},
            auth_meta={"mode": "api_key", "provided": True, "header": "X-API-Key", "api_key": "dont-leak-me"},
        )
        other_run = self._create_run(user_id=other_id, spec_id=other_spec)
        _ = self._create_insight(
            user_id=owner_id,
            spec_id=owner_spec,
            run_id=owner_run,
            test_id="TC-001",
            mode="suggest_test",
            model="qwen3-coder:latest",
            payload={"mode": "suggest_test", "model": "qwen3-coder:latest", "can_apply": True},
        )

        with patch.object(main, "SessionLocal", self.TestSessionLocal):
            detail = main.get_logistics_run_detail(run_id=owner_run, current_user=SimpleNamespace(id=owner_id))
            self.assertEqual(int(detail["run"]["id"]), owner_run)
            self.assertEqual(detail["auth"]["mode"], "api_key")
            self.assertTrue(bool(detail["auth"]["provided"]))
            self.assertEqual(detail["auth"]["header"], "X-API-Key")
            self.assertTrue(isinstance(detail["results"], list) and len(detail["results"]) > 0)
            serialized = json.dumps(detail)
            self.assertNotIn("dont-leak-me", serialized)

            listing = main.list_logistics_runs(current_user=SimpleNamespace(id=owner_id), limit=10, state="all")
            listing_serialized = json.dumps(listing)
            self.assertNotIn("dont-leak-me", listing_serialized)

            with self.assertRaises(HTTPException) as ctx:
                main.get_logistics_run_detail(run_id=other_run, current_user=SimpleNamespace(id=owner_id))
            self.assertEqual(ctx.exception.status_code, 404)

    def test_latest_run_endpoint_includes_persisted_llm_outputs_for_latest_owner_run(self) -> None:
        owner_id = self._create_user("latest-owner@example.com")
        other_id = self._create_user("latest-other@example.com")
        owner_spec = self._create_spec(user_id=owner_id, filename="latest-owner.yaml", title="Latest Owner API")
        other_spec = self._create_spec(user_id=other_id, filename="latest-other.yaml", title="Latest Other API")

        older_run = self._create_run(
            user_id=owner_id,
            spec_id=owner_spec,
            summary={"total": 1, "passed": 0, "failed": 1, "skipped": 0},
        )
        latest_run = self._create_run(
            user_id=owner_id,
            spec_id=owner_spec,
            summary={"total": 2, "passed": 0, "failed": 2, "skipped": 0},
        )
        other_run = self._create_run(
            user_id=other_id,
            spec_id=other_spec,
            summary={"total": 1, "passed": 0, "failed": 1, "skipped": 0},
        )
        _ = self._create_insight(
            user_id=owner_id,
            spec_id=owner_spec,
            run_id=older_run,
            test_id="TC-OLD",
            mode="explanation",
            model="gpt-4.1-mini",
            payload={"mode": "explanation", "explanation": "Older run insight"},
        )
        _ = self._create_insight(
            user_id=owner_id,
            spec_id=owner_spec,
            run_id=latest_run,
            test_id="TC-LATEST-1",
            mode="analysis",
            model="gpt-4.1-mini",
            payload={"mode": "analysis", "explanation": {"explanation": "Latest 1"}, "suggestion": {}},
        )
        _ = self._create_insight(
            user_id=owner_id,
            spec_id=owner_spec,
            run_id=latest_run,
            test_id="TC-LATEST-2",
            mode="suggest_test",
            model="qwen3-coder:latest",
            payload={"mode": "suggest_test", "reason": "Latest 2"},
        )
        _ = self._create_insight(
            user_id=other_id,
            spec_id=other_spec,
            run_id=other_run,
            test_id="TC-OTHER",
            mode="explanation",
            model="gpt-4.1-mini",
            payload={"mode": "explanation", "explanation": "Other user insight"},
        )

        with patch.object(main, "SessionLocal", self.TestSessionLocal):
            payload = main.get_latest_run_for_spec(spec_id=owner_spec, current_user=SimpleNamespace(id=owner_id))

        self.assertEqual(int(payload["spec_id"]), owner_spec)
        self.assertEqual(int(payload["latest_run"]["id"]), latest_run)
        outputs = payload.get("llm_outputs")
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 2)
        test_ids = {str(item.get("test_id") or "") for item in outputs if isinstance(item, dict)}
        self.assertEqual(test_ids, {"TC-LATEST-1", "TC-LATEST-2"})
        self.assertNotIn("TC-OLD", test_ids)
        self.assertNotIn("TC-OTHER", test_ids)
        for item in outputs:
            self.assertIn("id", item)
            self.assertIn("test_id", item)
            self.assertIn("mode", item)
            self.assertIn("model", item)
            self.assertIn("payload", item)
            self.assertIn("created_at", item)
            self.assertIn("updated_at", item)

    def test_delete_single_log_removes_run_and_linked_insights(self) -> None:
        user_id = self._create_user("delete-single@example.com")
        spec_id = self._create_spec(user_id=user_id, filename="delete.yaml", title="Delete API")
        run_id = self._create_run(user_id=user_id, spec_id=spec_id)
        _ = self._create_insight(
            user_id=user_id,
            spec_id=spec_id,
            run_id=run_id,
            test_id="TC-001",
            mode="explanation",
            model="gpt-4.1-mini",
            payload={"mode": "explanation", "model": "gpt-4.1-mini"},
        )

        with patch.object(main, "SessionLocal", self.TestSessionLocal):
            deleted = main.delete_logistics_run(run_id=run_id, current_user=SimpleNamespace(id=user_id))
            self.assertEqual(deleted["deleted"], 1)
            self.assertEqual(int(deleted["run_id"]), run_id)
            self.assertEqual(int(deleted["deleted_llm_insights"]), 1)

        db = self.TestSessionLocal()
        try:
            run_count = db.query(TestRun).filter(TestRun.user_id == user_id).count()
            insight_count = db.query(LLMRunInsight).filter(LLMRunInsight.user_id == user_id).count()
            self.assertEqual(run_count, 0)
            self.assertEqual(insight_count, 0)
        finally:
            db.close()

    def test_clear_logs_removes_runs_and_insights_but_keeps_specs_and_artifacts(self) -> None:
        user_id = self._create_user("clear-logs@example.com")
        spec_id = self._create_spec(user_id=user_id, filename="clear.yaml", title="Clear API")
        artifact_id = self._create_artifact(user_id=user_id, spec_id=spec_id)
        run_id = self._create_run(user_id=user_id, spec_id=spec_id)
        _ = self._create_insight(
            user_id=user_id,
            spec_id=spec_id,
            run_id=run_id,
            test_id="TC-001",
            mode="analysis",
            model="qwen3-coder:latest",
            payload={"mode": "analysis", "explanation": {}, "suggestion": {}},
        )

        with patch.object(main, "SessionLocal", self.TestSessionLocal):
            payload = main.clear_logistics_runs(current_user=SimpleNamespace(id=user_id))
            self.assertEqual(int(payload["deleted_runs"]), 1)
            self.assertEqual(int(payload["deleted_llm_insights"]), 1)

        db = self.TestSessionLocal()
        try:
            self.assertEqual(db.query(Spec).filter(Spec.user_id == user_id).count(), 1)
            self.assertEqual(db.query(SpecArtifact).filter(SpecArtifact.user_id == user_id).count(), 1)
            self.assertEqual(db.query(TestRun).filter(TestRun.user_id == user_id).count(), 0)
            self.assertEqual(db.query(LLMRunInsight).filter(LLMRunInsight.user_id == user_id).count(), 0)
            self.assertIsNotNone(db.query(SpecArtifact).filter(SpecArtifact.id == artifact_id).first())
        finally:
            db.close()

    def test_llm_endpoints_persist_insight_records_with_upsert(self) -> None:
        user_id = self._create_user("llm-log@example.com")
        spec_id = self._create_spec(user_id=user_id, filename="llm.yaml", title="LLM API")
        run_id = self._create_run(
            user_id=user_id,
            spec_id=spec_id,
            summary={"total": 1, "passed": 0, "failed": 1, "skipped": 0},
            results=[{"test_id": "TC-FAIL", "outcome": "FAIL", "duration_ms": 155}],
        )
        run_row = self._fetch_run_row(run_id)

        bundle = {
            "run_row": run_row,
            "case_result": {"outcome": "FAIL", "actual_status": 500},
            "settings": {"model": "qwen3-coder:latest"},
            "suite_data": {"test_cases": [{"test_id": "TC-FAIL"}]},
            "test_case": {
                "test_id": "TC-FAIL",
                "title": "Failing case",
                "method": "GET",
                "path": "/fail",
                "steps": [{"step_number": 1, "action": "Execute request", "input_data": {}}],
                "expected_result": {"status_code": 200},
            },
            "evidence": {"case_id": "spec::TC-FAIL"},
            "prompt_bundle": {"case_evidence": {}, "pipeline_context": {}},
        }

        first_explanation = {
            "runtime": {"provider": "openai", "model": "gpt-4.1-mini", "client": object(), "options": {}},
            "payload": {"mode": "explanation", "model": "gpt-4.1-mini", "explanation": "First"},
        }
        second_explanation = {
            "runtime": {"provider": "openai", "model": "gpt-4.1-mini", "client": object(), "options": {}},
            "payload": {"mode": "explanation", "model": "gpt-4.1-mini", "explanation": "Updated"},
        }
        analysis_explanation = {
            "runtime": {"provider": "openai", "model": "gpt-4.1-mini", "client": object(), "options": {}},
            "payload": {"mode": "explanation", "model": "gpt-4.1-mini", "explanation": "Analysis"},
        }
        suggestion_payload = {
            "mode": "suggest_test",
            "model": "qwen3-coder:latest",
            "signal": "status_mismatch",
            "skipped": True,
            "eligible_for_generation": False,
            "can_apply": False,
        }

        with (
            patch.object(main, "SessionLocal", self.TestSessionLocal),
            patch.object(main, "_load_run_case_bundle", return_value=bundle),
            patch.object(main, "_resolve_effective_llm_settings", return_value={"active_model_id": "builtin:ollama:qwen3-coder:latest"}),
            patch.object(
                main,
                "_build_runtime_from_active_model",
                return_value={"provider": "openai", "client": object(), "model": "gpt-4.1-mini", "options": {}},
            ),
            patch.object(
                main,
                "_generate_failure_explanation_or_raise",
                side_effect=[first_explanation, second_explanation, analysis_explanation],
            ),
            patch.object(
                main,
                "generate_suggested_test",
                return_value=suggestion_payload,
            ),
        ):
            main.explain_failed_case(run_id=run_id, test_id="TC-FAIL", current_user=SimpleNamespace(id=user_id))
            main.explain_failed_case(run_id=run_id, test_id="TC-FAIL", current_user=SimpleNamespace(id=user_id))
            main.analyze_failure_with_suggestion(run_id=run_id, test_id="TC-FAIL", current_user=SimpleNamespace(id=user_id))
            main.suggest_test_for_failure(
                run_id=run_id,
                test_id="TC-FAIL",
                req=main.SuggestTestRequest(explanation={"signal": "status_mismatch"}),
                current_user=SimpleNamespace(id=user_id),
            )

        db = self.TestSessionLocal()
        try:
            rows = (
                db.query(LLMRunInsight)
                .filter(LLMRunInsight.user_id == user_id, LLMRunInsight.run_id == run_id)
                .order_by(LLMRunInsight.mode.asc())
                .all()
            )
            by_mode = {str(row.mode): row for row in rows}
            self.assertEqual(set(by_mode.keys()), {"analysis", "explanation", "suggest_test"})
            self.assertEqual(len(rows), 3)
            explanation_payload = json.loads(by_mode["explanation"].payload_json)
            self.assertEqual(explanation_payload.get("explanation"), "Updated")
            self.assertEqual(str(by_mode["explanation"].model), "gpt-4.1-mini")

            analysis_payload = json.loads(by_mode["analysis"].payload_json)
            self.assertIn("explanation", analysis_payload)
            self.assertIn("suggestion", analysis_payload)

            suggest_payload = json.loads(by_mode["suggest_test"].payload_json)
            self.assertEqual(bool(suggest_payload.get("skipped")), bool(suggestion_payload.get("skipped")))
        finally:
            db.close()

if __name__ == "__main__":
    unittest.main()
