"""API tests with a mocked Translator."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main


class StubTranslator:
    """Returns a deterministic fake translation."""

    device = "cpu"
    adapter_path = "models/ru-uz-lora"
    config = SimpleNamespace(model=SimpleNamespace(base_model="facebook/nllb-200-distilled-600M"))

    def translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        if "boom" in texts:
            raise RuntimeError("generation failed")
        return [f"[{tgt_lang}] {t}" for t in texts]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(main, "build_translator", StubTranslator)
    with TestClient(main.app) as c:
        yield c


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "model_loaded": True,
        "device": "cpu",
        "base_model": "facebook/nllb-200-distilled-600M",
        "adapter": "models/ru-uz-lora",
    }


def test_translate(client: TestClient) -> None:
    body = {"text": "Привет, как дела?", "src_lang": "rus_Cyrl", "tgt_lang": "uzn_Latn"}
    resp = client.post("/translate", json=body)
    assert resp.status_code == 200
    assert resp.json() == {
        "translation": "[uzn_Latn] Привет, как дела?",
        "src_lang": "rus_Cyrl",
        "tgt_lang": "uzn_Latn",
    }


def test_translate_batch(client: TestClient) -> None:
    resp = client.post(
        "/translate/batch",
        json={"texts": ["Salom", "Rahmat"], "src_lang": "uzn_Latn", "tgt_lang": "rus_Cyrl"},
    )
    assert resp.status_code == 200
    assert resp.json()["translations"] == ["[rus_Cyrl] Salom", "[rus_Cyrl] Rahmat"]


@pytest.mark.parametrize(
    "body",
    [
        {"text": "", "src_lang": "rus_Cyrl", "tgt_lang": "uzn_Latn"},
        {"text": "   ", "src_lang": "rus_Cyrl", "tgt_lang": "uzn_Latn"},
        {"text": "x", "src_lang": "eng_Latn", "tgt_lang": "uzn_Latn"},
        {"text": "x", "src_lang": "rus_Cyrl", "tgt_lang": "rus_Cyrl"},
        {"text": "x" * (main.MAX_INPUT_CHARS + 1), "src_lang": "rus_Cyrl", "tgt_lang": "uzn_Latn"},
    ],
)
def test_translate_validation(client: TestClient, body: dict[str, Any]) -> None:
    assert client.post("/translate", json=body).status_code == 422


def test_batch_size_limit(client: TestClient) -> None:
    resp = client.post("/translate/batch", json={"texts": ["a"] * (main.MAX_BATCH_SIZE + 1)})
    assert resp.status_code == 422


def test_internal_error_is_500(client: TestClient) -> None:
    resp = client.post("/translate", json={"text": "boom"})
    assert resp.status_code == 500
    assert resp.json()["detail"] == "Internal translation error"


def test_model_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> None:
        raise OSError("weights missing")

    monkeypatch.setattr(main, "build_translator", broken)
    with TestClient(main.app) as c:
        assert c.get("/health").json()["status"] == "error"
        assert c.post("/translate", json={"text": "Привет"}).status_code == 503
