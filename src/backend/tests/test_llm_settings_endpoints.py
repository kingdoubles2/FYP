from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app import main
from backend.app.db import Base
from backend.app.models_db import User


def _backend_settings() -> dict:
    return {
        "model": "qwen3-coder:latest",
        "base_url": "http://localhost:11434",
        "timeout_seconds": 120,
        "word_target": 55,
        "word_min": 40,
        "word_max": 100,
        "retry_invalid_output": 1,
        "max_items_per_section": 12,
        "snippet_chars": 300,
        "max_similar_failures": 5,
        "max_full_context_items": 80,
        "ollama_options": {},
    }


def _backend_catalog() -> dict:
    return {
        "default_model": "qwen3-coder:latest",
        "models": ["qwen3-coder:latest"],
        "ollama_base_url": "http://localhost:11434",
    }


class LlmSettingsEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True, connect_args={"check_same_thread": False})
        self.TestSessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)
        Base.metadata.create_all(bind=self.engine)

        db = self.TestSessionLocal()
        try:
            user = User(email="tester@example.com", password_hash="x")
            db.add(user)
            db.commit()
            db.refresh(user)
            self.user_id = int(user.id)
        finally:
            db.close()

    def tearDown(self) -> None:
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    def test_get_settings_defaults_to_builtin_qwen(self) -> None:
        with (
            patch.object(main, "SessionLocal", self.TestSessionLocal),
            patch.object(main, "load_backend_llm_settings", return_value=_backend_settings()),
            patch.object(main, "load_backend_model_catalog", return_value=_backend_catalog()),
        ):
            payload = main.get_llm_settings(current_user=SimpleNamespace(id=self.user_id))

        self.assertEqual(payload["active_model_id"], "builtin:ollama:qwen3-coder:latest")
        self.assertEqual(payload["default_model_id"], "builtin:ollama:qwen3-coder:latest")
        self.assertEqual(len(payload["models"]), 1)
        self.assertEqual(payload["models"][0]["provider"], "ollama")
        self.assertEqual(payload["models"][0]["model"], "qwen3-coder:latest")

    def test_add_select_remove_external_model_masks_api_key(self) -> None:
        with (
            patch.object(main, "SessionLocal", self.TestSessionLocal),
            patch.object(main, "load_backend_llm_settings", return_value=_backend_settings()),
            patch.object(main, "load_backend_model_catalog", return_value=_backend_catalog()),
        ):
            added_payload = main.add_llm_model(
                req=main.AddLLMModelRequest(
                    provider="openai",
                    model="gpt-4.1-mini",
                    label="OpenAI GPT 4.1 Mini",
                    api_key="sk-test-1234567890",
                ),
                current_user=SimpleNamespace(id=self.user_id),
            )

            added_entry = next((row for row in added_payload["models"] if row.get("source") == "user"), None)
            self.assertIsNotNone(added_entry)
            self.assertTrue(bool(added_entry["has_api_key"]))
            self.assertNotIn("sk-test-1234567890", str(added_entry["api_key_masked"]))

            selected_payload = main.update_llm_settings(
                req=main.UpdateLLMSettingsRequest(active_model_id=str(added_entry["id"])),
                current_user=SimpleNamespace(id=self.user_id),
            )
            self.assertEqual(str(selected_payload["active_model_id"]), str(added_entry["id"]))

            deleted_payload = main.delete_llm_model(
                model_id=str(added_entry["id"]),
                current_user=SimpleNamespace(id=self.user_id),
            )

        remaining_ids = {str(row.get("id")) for row in deleted_payload["models"]}
        self.assertNotIn(str(added_entry["id"]), remaining_ids)
        self.assertEqual(deleted_payload["active_model_id"], "builtin:ollama:qwen3-coder:latest")

    def test_discover_openai_models_returns_normalized_catalog(self) -> None:
        client_mock = SimpleNamespace(
            list_models=lambda: ([{"id": "gpt-4.1", "label": "gpt-4.1"}], None),
        )
        with (
            patch.object(main, "load_backend_llm_settings", return_value=_backend_settings()),
            patch.object(main, "OpenAIClient", return_value=client_mock) as openai_ctor,
        ):
            payload = main.discover_provider_models(
                provider="openai",
                req=main.DiscoverProviderModelsRequest(api_key="sk-test"),
                current_user=SimpleNamespace(id=self.user_id),
            )

        self.assertEqual(payload["provider"], "openai")
        self.assertEqual(payload["models"], [{"id": "gpt-4.1", "label": "gpt-4.1"}])
        self.assertTrue(bool(openai_ctor.called))

    def test_discover_provider_models_requires_api_key(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            main.discover_provider_models(
                provider="openai",
                req=main.DiscoverProviderModelsRequest(api_key=""),
                current_user=SimpleNamespace(id=self.user_id),
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("API key", str(ctx.exception.detail))

    def test_discover_provider_models_rejects_unsupported_provider(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            main.discover_provider_models(
                provider="ollama",
                req=main.DiscoverProviderModelsRequest(api_key="test"),
                current_user=SimpleNamespace(id=self.user_id),
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Unsupported provider", str(ctx.exception.detail))

    def test_discover_provider_models_surfaces_upstream_error(self) -> None:
        client_mock = SimpleNamespace(list_models=lambda: (None, "upstream timeout"))
        with patch.object(main, "AnthropicClient", return_value=client_mock):
            with self.assertRaises(HTTPException) as ctx:
                main.discover_provider_models(
                    provider="anthropic",
                    req=main.DiscoverProviderModelsRequest(api_key="claude-test"),
                    current_user=SimpleNamespace(id=self.user_id),
                )
        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIn("Unable to load models", str(ctx.exception.detail))


if __name__ == "__main__":
    unittest.main()
