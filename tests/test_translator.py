"""Smoke tests for Translator with mocked HF loading (no 600M download)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ru_uz_mt.config import AppConfig, GenerateConfig
from ru_uz_mt.inference import translator as translator_mod
from ru_uz_mt.inference.translator import Translator, generate_translations

LANG_IDS = {"rus_Cyrl": 1001, "uzn_Latn": 1002}


class FakeEncoding(dict):
    def to(self, device: str) -> FakeEncoding:
        self.device = device
        return self


class FakeTokenizer:
    """Mimics the parts of the NLLB tokenizer used by the translator."""

    def __init__(self) -> None:
        self.src_lang: str | None = None
        self.calls: list[dict[str, Any]] = []

    def __call__(self, texts: list[str], **kwargs: Any) -> FakeEncoding:
        self.calls.append({"texts": list(texts), "src_lang": self.src_lang, **kwargs})
        return FakeEncoding(input_ids=list(texts))

    def convert_tokens_to_ids(self, token: str) -> int:
        return LANG_IDS[token]

    def batch_decode(self, generated: list[str], skip_special_tokens: bool = True) -> list[str]:
        return generated


class FakeModel:
    """Echo model: 'translates' by tagging the input with the forced target token id."""

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.device: str | None = None

    def generate(self, input_ids: list[str], **kwargs: Any) -> list[str]:
        self.generate_calls.append(kwargs)
        return [f"<{kwargs['forced_bos_token_id']}>{t}" for t in input_ids]

    def to(self, device: str) -> FakeModel:
        self.device = device
        return self

    def eval(self) -> FakeModel:
        return self


@pytest.fixture
def cfg(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "model": {"device": "cpu", "adapter_path": str(tmp_path / "adapter")},
            "generate": {"num_beams": 2, "max_length": 32, "batch_size": 2},
        }
    )


@pytest.fixture
def fake_hf(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    import transformers

    created: dict[str, Any] = {"tokenizer": FakeTokenizer(), "model": FakeModel()}
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: created["tokenizer"])
    )
    monkeypatch.setattr(
        transformers.AutoModelForSeq2SeqLM, "from_pretrained", staticmethod(lambda *a, **k: created["model"])
    )
    return created


def test_generate_translations_sets_nllb_tags_and_batches() -> None:
    tok, model = FakeTokenizer(), FakeModel()
    gen = GenerateConfig(num_beams=3, max_length=40, batch_size=2)
    out = generate_translations(model, tok, ["a", "b", "c"], "rus_Cyrl", "uzn_Latn", gen, "cpu")
    assert out == ["<1002>a", "<1002>b", "<1002>c"]
    assert [c["texts"] for c in tok.calls] == [["a", "b"], ["c"]]
    assert all(c["src_lang"] == "rus_Cyrl" for c in tok.calls)
    assert model.generate_calls[0] == {"forced_bos_token_id": 1002, "num_beams": 3, "max_length": 40}


def test_translator_without_adapter_uses_base(cfg: AppConfig, fake_hf: dict[str, Any]) -> None:
    tr = Translator(cfg)
    assert tr.adapter_path is None
    assert tr.device == "cpu" and fake_hf["model"].device == "cpu"
    assert tr.translate("Salom", "uzn_Latn", "rus_Cyrl") == "<1001>Salom"
    assert tr.translate_batch(["Привет", "Пока"], "rus_Cyrl", "uzn_Latn") == ["<1002>Привет", "<1002>Пока"]
    assert tr.translate_batch([], "rus_Cyrl", "uzn_Latn") == []


def test_translator_applies_lora_adapter(
    cfg: AppConfig, fake_hf: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import peft

    adapter_dir = Path(cfg.adapter_path)
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 8}), encoding="utf-8")

    class FakePeft:
        loaded_from: str | None = None

        @classmethod
        def from_pretrained(cls, model: FakeModel, path: str) -> FakePeft:
            cls.loaded_from = path
            inst = cls()
            inst.base = model
            return inst

        def merge_and_unload(self) -> FakeModel:
            return self.base

    monkeypatch.setattr(peft, "PeftModel", FakePeft)
    tr = Translator(cfg)
    assert FakePeft.loaded_from == str(adapter_dir)
    assert tr.adapter_path == str(adapter_dir)
    assert tr.model is fake_hf["model"]

    baseline = Translator(cfg, use_adapter=False)
    assert baseline.adapter_path is None


@pytest.mark.parametrize(("src", "tgt"), [("eng_Latn", "uzn_Latn"), ("rus_Cyrl", "rus_Cyrl")])
def test_translator_rejects_bad_langs(cfg: AppConfig, fake_hf: dict[str, Any], src: str, tgt: str) -> None:
    tr = Translator(cfg)
    with pytest.raises(ValueError):
        tr.translate("x", src, tgt)


def test_module_exposes_public_api() -> None:
    assert hasattr(translator_mod, "Translator") and hasattr(translator_mod, "generate_translations")
