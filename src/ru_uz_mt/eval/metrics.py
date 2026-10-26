"""Corpus-level MT metrics (sacrebleu)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def corpus_scores(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    """Compute corpus BLEU and chrF (chrF is the key metric for agglutinative Uzbek)."""
    import sacrebleu

    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have equal length")
    if not hypotheses:
        return {"bleu": 0.0, "chrf": 0.0}
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return {"bleu": round(bleu.score, 2), "chrf": round(chrf.score, 2)}


def group_by_direction(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group JSONL records by their ``direction`` field, preserving order."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[rec["direction"]].append(rec)
    return dict(groups)
