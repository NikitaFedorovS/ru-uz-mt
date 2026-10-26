"""Load, clean, normalize and split the RU–UZ parallel corpus.

Output: ``{processed_dir}/{train,val,test}.jsonl`` with records
``{"src", "tgt", "direction", "src_lang", "tgt_lang"}``.

CLI: ``python -m ru_uz_mt.data.prepare --config configs/train_config.yaml``
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import typer

from ru_uz_mt.config import DataConfig, load_config
from ru_uz_mt.data.normalize import (
    clean_text,
    is_cyrillic,
    looks_like_russian,
    normalize_uzbek_apostrophes,
    script_ratio,
    uzbek_cyr_to_lat,
)
from ru_uz_mt.utils import setup_logging, write_jsonl

logger = logging.getLogger(__name__)

RU_CODE = "rus_Cyrl"
UZ_CODE = "uzn_Latn"
DIRECTIONS: dict[str, tuple[str, str]] = {"ru-uz": (RU_CODE, UZ_CODE), "uz-ru": (UZ_CODE, RU_CODE)}

Pair = tuple[str, str]  # (russian, uzbek)


class LangIdentifier(Protocol):
    """Anything that returns an ISO-639-1 language code for a sentence."""

    def __call__(self, text: str) -> str: ...


@dataclass
class FastTextLangId:
    """Wrapper around a fastText ``lid.176`` model (optional dependency)."""

    model_path: str

    def __post_init__(self) -> None:
        try:
            import fasttext  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("Language filtering needs `pip install fasttext-wheel`") from e
        self._model = fasttext.load_model(self.model_path)

    def __call__(self, text: str) -> str:
        labels, _ = self._model.predict(text.replace("\n", " "), k=1)
        return labels[0].removeprefix("__label__")


# --------------------------------------------------------------------------- loading


def iter_opus_pairs(raw_dir: str | Path, corpora: Iterable[str]) -> Iterator[Pair]:
    """Yield (ru, uz) pairs from Moses-format OPUS files (``<corpus>/*.ru`` + ``*.uz``)."""
    raw_dir = Path(raw_dir)
    for corpus in corpora:
        corpus_dir = raw_dir / corpus
        ru_files = sorted(corpus_dir.glob("*.ru"))
        if not ru_files:
            logger.warning("No *.ru files in %s — run scripts/download_data.sh first", corpus_dir)
            continue
        for ru_file in ru_files:
            uz_file = ru_file.with_suffix(".uz")
            if not uz_file.exists():
                logger.warning("Missing pair file for %s", ru_file)
                continue
            n = 0
            with ru_file.open(encoding="utf-8") as fr, uz_file.open(encoding="utf-8") as fu:
                for ru, uz in zip(fr, fu, strict=True):
                    n += 1
                    yield ru.rstrip("\n"), uz.rstrip("\n")
            logger.info("Loaded %d pairs from %s", n, corpus)


def iter_hf_pairs(dataset_name: str, dataset_config: str | None) -> Iterator[Pair]:
    """Yield (ru, uz) pairs from a HF dataset with a ``translation: {ru, uz}`` column."""
    from datasets import load_dataset

    ds_dict = load_dataset(dataset_name, dataset_config)
    for split_name, split in ds_dict.items():
        logger.info("Loaded %d rows from %s[%s]", len(split), dataset_name, split_name)
        for row in split:
            tr = row["translation"]
            yield tr["ru"], tr["uz"]


def load_pairs(cfg: DataConfig) -> Iterator[Pair]:
    """Dispatch to the configured corpus source."""
    if cfg.source == "opus":
        return iter_opus_pairs(cfg.raw_dir, cfg.opus_corpora)
    assert cfg.dataset_name is not None
    return iter_hf_pairs(cfg.dataset_name, cfg.dataset_config)


# --------------------------------------------------------------------------- cleaning


def normalize_uzbek(text: str, cfg: DataConfig) -> str | None:
    """Bring Uzbek text to Latin script. Returns ``None`` if the sentence should be dropped."""
    if is_cyrillic(text):
        if looks_like_russian(text) or not cfg.normalize_uzbek_to_latin:
            return None
        text = uzbek_cyr_to_lat(text)
    if cfg.normalize_apostrophes:
        text = normalize_uzbek_apostrophes(text)
    return text


def clean_pair(
    ru: str,
    uz: str,
    cfg: DataConfig,
    stats: Counter[str],
    langid: LangIdentifier | None = None,
) -> Pair | None:
    """Apply all per-pair filters. Increments ``stats[<reason>]`` for every drop."""
    ru, uz = clean_text(ru), clean_text(uz)
    if not ru or not uz:
        stats["empty"] += 1
        return None
    uz_norm = normalize_uzbek(uz, cfg)
    if uz_norm is None:
        stats["uz_cyrillic"] += 1
        return None
    uz = uz_norm
    if script_ratio(ru, "cyrillic") < 0.5 or script_ratio(uz, "latin") < 0.5:
        stats["wrong_script"] += 1
        return None
    ru_words, uz_words = len(ru.split()), len(uz.split())
    if not (cfg.min_len <= ru_words <= cfg.max_len and cfg.min_len <= uz_words <= cfg.max_len):
        stats["length"] += 1
        return None
    lo, hi = sorted((len(ru), len(uz)))
    if hi / max(lo, 1) > cfg.length_ratio:
        stats["ratio"] += 1
        return None
    if ru.lower() == uz.lower():
        stats["copy"] += 1
        return None
    if langid is not None and (langid(ru) != "ru" or langid(uz) != "uz"):
        stats["langid"] += 1
        return None
    return ru, uz


def clean_corpus(
    pairs: Iterable[Pair],
    cfg: DataConfig,
    langid: LangIdentifier | None = None,
) -> tuple[list[Pair], Counter[str]]:
    """Clean and deduplicate a stream of pairs.

    Returns:
        Unique clean pairs (original order) and a counter with drop reasons.
    """
    stats: Counter[str] = Counter()
    seen: set[Pair] = set()
    out: list[Pair] = []
    for ru, uz in pairs:
        stats["total"] += 1
        cleaned = clean_pair(ru, uz, cfg, stats, langid)
        if cleaned is None:
            continue
        key = (cleaned[0].lower(), cleaned[1].lower())
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)
        out.append(cleaned)
    stats["kept"] = len(out)
    return out, stats


# --------------------------------------------------------------------------- splitting


def split_pairs(
    pairs: list[Pair],
    ratios: tuple[float, float, float],
    seed: int,
    max_eval_pairs: int | None = None,
) -> dict[str, list[Pair]]:
    """Shuffle and split pairs into train/val/test.

    Splitting happens on *pairs* (before direction expansion) so both directions of a
    sentence always land in the same split — no train/test leakage.
    """
    shuffled = pairs.copy()
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)

    def _size(ratio: float) -> int:
        size = max(1, round(n * ratio)) if ratio > 0 and n >= 3 else 0
        return min(size, max_eval_pairs) if max_eval_pairs else size

    n_val, n_test = _size(ratios[1]), _size(ratios[2])
    return {
        "test": shuffled[:n_test],
        "val": shuffled[n_test : n_test + n_val],
        "train": shuffled[n_test + n_val :],
    }


def to_records(pairs: Iterable[Pair], bidirectional: bool = True) -> list[dict[str, Any]]:
    """Expand (ru, uz) pairs into directional records with NLLB language tags."""
    records: list[dict[str, Any]] = []
    for ru, uz in pairs:
        records.append(_record(ru, uz, "ru-uz"))
        if bidirectional:
            records.append(_record(uz, ru, "uz-ru"))
    return records


def _record(src: str, tgt: str, direction: str) -> dict[str, Any]:
    src_lang, tgt_lang = DIRECTIONS[direction]
    return {"src": src, "tgt": tgt, "direction": direction, "src_lang": src_lang, "tgt_lang": tgt_lang}


# --------------------------------------------------------------------------- pipeline


def prepare(cfg: DataConfig, max_pairs: int | None = None) -> dict[str, int]:
    """Run the full pipeline and write JSONL splits. Returns record counts per split."""
    langid = FastTextLangId(cfg.langid_model_path) if cfg.langid_model_path else None
    raw = load_pairs(cfg)
    pairs, stats = clean_corpus(raw, cfg, langid)
    if not pairs:
        raise RuntimeError("No pairs left after cleaning — check data.raw_dir / data.source")
    logger.info("Cleaning stats: %s", dict(stats))
    if max_pairs is not None and len(pairs) > max_pairs:
        random.Random(cfg.seed).shuffle(pairs)
        pairs = pairs[:max_pairs]
        logger.info("Subsampled to %d pairs (--max-pairs)", max_pairs)

    splits = split_pairs(pairs, cfg.splits, cfg.seed, cfg.max_test_pairs)
    out_dir = Path(cfg.processed_dir)
    counts: dict[str, int] = {}
    for name, split in splits.items():
        recs = to_records(split, cfg.bidirectional)
        random.Random(cfg.seed).shuffle(recs)
        counts[name] = write_jsonl(out_dir / f"{name}.jsonl", recs)
        logger.info("Wrote %s: %d records (%d pairs)", name, counts[name], len(split))
    (out_dir / "stats.json").write_text(
        json.dumps({"cleaning": dict(stats), "records": counts}, indent=2), encoding="utf-8"
    )
    return counts


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    config: str = typer.Option("configs/train_config.yaml", help="Path to YAML config"),
    max_pairs: int | None = typer.Option(None, help="Keep at most N clean pairs (quick runs)"),
) -> None:
    """Prepare data/processed/{train,val,test}.jsonl."""
    setup_logging()
    prepare(load_config(config).data, max_pairs=max_pairs)


if __name__ == "__main__":
    app()
