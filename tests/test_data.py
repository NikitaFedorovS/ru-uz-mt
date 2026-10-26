"""Tests for cleaning, normalization and splitting (no network, no model)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from ru_uz_mt.config import DataConfig, load_config
from ru_uz_mt.data.normalize import (
    OKINA,
    TUTUQ,
    clean_text,
    is_cyrillic,
    looks_like_russian,
    normalize_uzbek_apostrophes,
    uzbek_cyr_to_lat,
)
from ru_uz_mt.data.prepare import clean_corpus, clean_pair, prepare, split_pairs, to_records
from ru_uz_mt.utils import read_jsonl


@pytest.fixture
def cfg(tmp_path: Path) -> DataConfig:
    return DataConfig(raw_dir=str(tmp_path / "raw"), processed_dir=str(tmp_path / "processed"))


@pytest.mark.parametrize(
    ("cyr", "lat"),
    [
        ("Салом", "Salom"),
        ("ўзбек тили", f"o{OKINA}zbek tili"),
        ("Ёшлар", "Yoshlar"),
        ("ер", "yer"),
        ("келди", "keldi"),
        ("қишлоқ", "qishloq"),
        ("ғалаба", f"g{OKINA}alaba"),
        ("Тошкент шаҳри", "Toshkent shahri"),
        ("ЎЗБЕКИСТОН", f"O{OKINA}ZBEKISTON"),
        ("маъно", f"ma{TUTUQ}no"),
    ],
)
def test_uzbek_cyr_to_lat(cyr: str, lat: str) -> None:
    assert uzbek_cyr_to_lat(cyr) == lat


def test_apostrophe_normalization() -> None:
    assert normalize_uzbek_apostrophes("o'zbek g‘alaba ma'no") == f"o{OKINA}zbek g{OKINA}alaba ma{TUTUQ}no"


def test_script_detection() -> None:
    assert is_cyrillic("Привет, мир")
    assert not is_cyrillic("Salom dunyo")
    assert looks_like_russian("Мы были здесь")
    assert not looks_like_russian("Биз шу ерда эдик, қишлоқда")


def test_clean_text_collapses_whitespace() -> None:
    assert clean_text("  Привет \t​ мир \n") == "Привет мир"


def test_clean_pair_filters(cfg: DataConfig) -> None:
    stats: Counter[str] = Counter()
    assert clean_pair("Привет, мир!", "Salom, dunyo!", cfg, stats) == ("Привет, мир!", "Salom, dunyo!")
    assert clean_pair("", "Salom", cfg, stats) is None
    assert clean_pair("Да", "Ha, bu juda uzun gap emasmi yoki shunchaki ko'p so'zmi?", cfg, stats) is None
    assert clean_pair("Мы были здесь", "Мы были здесь", cfg, stats) is None  # Russian on UZ side
    assert clean_pair("Hello world", "Salom dunyo", cfg, stats) is None  # wrong script on RU side
    assert stats["empty"] == 1 and stats["ratio"] == 1 and stats["uz_cyrillic"] == 1
    assert stats["wrong_script"] == 1


def test_clean_pair_transliterates_or_drops_cyrillic(cfg: DataConfig) -> None:
    stats: Counter[str] = Counter()
    assert clean_pair("Ташкент", "Тошкент", cfg, stats) == ("Ташкент", "Toshkent")
    strict = cfg.model_copy(update={"normalize_uzbek_to_latin": False})
    assert clean_pair("Ташкент", "Тошкент", strict, stats) is None


def test_clean_pair_length_limits(cfg: DataConfig) -> None:
    short_cfg = cfg.model_copy(update={"max_len": 3})
    assert clean_pair("раз два три четыре", "bir ikki uch tort", short_cfg, Counter()) is None


def test_clean_corpus_dedup(cfg: DataConfig) -> None:
    pairs = [("Привет", "Salom"), ("привет", "salom"), ("Спасибо", "Rahmat"), ("", "x")]
    out, stats = clean_corpus(pairs, cfg)
    assert out == [("Привет", "Salom"), ("Спасибо", "Rahmat")]
    assert stats["duplicate"] == 1 and stats["kept"] == 2 and stats["total"] == 4


def test_langid_filter(cfg: DataConfig) -> None:
    def fake_langid(text: str) -> str:
        return "ru" if is_cyrillic(text) else ("tr" if "merhaba" in text.lower() else "uz")

    out, stats = clean_corpus([("Привет", "Salom"), ("Привет всем", "Merhaba herkese")], cfg, fake_langid)
    assert out == [("Привет", "Salom")]
    assert stats["langid"] == 1


def test_split_no_leakage_and_sizes() -> None:
    pairs = [(f"предложение {i}", f"gap {i}") for i in range(1000)]
    splits = split_pairs(pairs, (0.98, 0.01, 0.01), seed=1, max_eval_pairs=5)
    assert len(splits["val"]) == 5 and len(splits["test"]) == 5 and len(splits["train"]) == 990
    sets = [set(s) for s in splits.values()]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert split_pairs(pairs, (0.98, 0.01, 0.01), seed=1) == split_pairs(pairs, (0.98, 0.01, 0.01), seed=1)


def test_to_records_bidirectional() -> None:
    recs = to_records([("Привет", "Salom")])
    assert recs == [
        {
            "src": "Привет",
            "tgt": "Salom",
            "direction": "ru-uz",
            "src_lang": "rus_Cyrl",
            "tgt_lang": "uzn_Latn",
        },
        {
            "src": "Salom",
            "tgt": "Привет",
            "direction": "uz-ru",
            "src_lang": "uzn_Latn",
            "tgt_lang": "rus_Cyrl",
        },
    ]
    assert len(to_records([("Привет", "Salom")], bidirectional=False)) == 1


def test_prepare_end_to_end_opus(cfg: DataConfig) -> None:
    corpus_dir = Path(cfg.raw_dir) / "toy"
    corpus_dir.mkdir(parents=True)
    ru = [f"Это предложение номер {i}." for i in range(50)] + ["", "Это предложение номер 0."]
    uz = [f"Bu {i}-raqamli gap." for i in range(50)] + ["Bo'sh", "Bu 0-raqamli gap."]
    (corpus_dir / "toy.ru-uz.ru").write_text("\n".join(ru) + "\n", encoding="utf-8")
    (corpus_dir / "toy.ru-uz.uz").write_text("\n".join(uz) + "\n", encoding="utf-8")

    cfg = cfg.model_copy(update={"opus_corpora": ["toy"], "splits": (0.8, 0.1, 0.1)})
    counts = prepare(cfg)
    assert counts == {"test": 10, "val": 10, "train": 80}  # 50 unique pairs x 2 directions
    for split in ("train", "val", "test"):
        rows = list(read_jsonl(Path(cfg.processed_dir) / f"{split}.jsonl"))
        assert rows and all(r["src"] and r["tgt"] for r in rows)
        assert {r["direction"] for r in rows} == {"ru-uz", "uz-ru"}
    assert (Path(cfg.processed_dir) / "stats.json").exists()


def test_config_validation_and_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("train:\n  num_epochs: 3\n", encoding="utf-8")
    monkeypatch.setenv("RU_UZ_MT__TRAIN__NUM_EPOCHS", "1")
    monkeypatch.setenv("RU_UZ_MT__LORA__TARGET_MODULES", "[q_proj, k_proj, v_proj]")
    cfg = load_config(cfg_file)
    assert cfg.train.num_epochs == 1
    assert cfg.lora.target_modules == ["q_proj", "k_proj", "v_proj"]
    assert cfg.adapter_path == cfg.train.output_dir
    with pytest.raises(ValueError):
        DataConfig(splits=(0.5, 0.5, 0.5))


def test_repo_config_loads() -> None:
    cfg = load_config(Path(__file__).parents[1] / "configs" / "train_config.yaml")
    assert cfg.model.src_lang == "rus_Cyrl" and cfg.model.tgt_lang == "uzn_Latn"
