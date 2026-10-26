"""Evaluate baseline NLLB vs. LoRA fine-tuned model on the test split (BLEU, chrF, opt. COMET).

CLI: ``python -m ru_uz_mt.eval.evaluate --config configs/train_config.yaml``
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any

import typer

from ru_uz_mt.config import AppConfig, load_config
from ru_uz_mt.eval.metrics import corpus_scores, group_by_direction
from ru_uz_mt.utils import read_jsonl, setup_logging, write_jsonl

logger = logging.getLogger(__name__)


def comet_score(sources: list[str], hyps: list[str], refs: list[str], model_name: str) -> float:
    """System-level COMET score (needs the optional ``unbabel-comet`` package)."""
    try:
        from comet import download_model, load_from_checkpoint  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("COMET needs `pip install unbabel-comet`") from e
    model = load_from_checkpoint(download_model(model_name))
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hyps, refs, strict=True)]
    return round(float(model.predict(data, batch_size=16, progress_bar=False).system_score) * 100, 2)


def evaluate_system(
    translator: Any,
    records: list[dict[str, Any]],
    cfg: AppConfig,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Translate every record and score each direction.

    Returns:
        ``{direction: {"bleu", "chrf", ["comet"]}}`` and per-sentence outputs.
    """
    scores: dict[str, dict[str, float]] = {}
    outputs: list[dict[str, Any]] = []
    for direction, recs in group_by_direction(records).items():
        sources = [r["src"] for r in recs]
        refs = [r["tgt"] for r in recs]
        logger.info("Translating %d sentences (%s)", len(recs), direction)
        hyps = translator.translate_batch(sources, recs[0]["src_lang"], recs[0]["tgt_lang"])
        scores[direction] = corpus_scores(hyps, refs)
        if cfg.eval.compute_comet:
            scores[direction]["comet"] = comet_score(sources, hyps, refs, cfg.eval.comet_model)
        outputs.extend(
            {"direction": direction, "src": s, "ref": r, "hyp": h}
            for s, r, h in zip(sources, refs, hyps, strict=True)
        )
    return scores, outputs


def format_table(results: dict[str, dict[str, dict[str, float]]]) -> str:
    """Render ``{system: {direction: metrics}}`` as a Markdown table with deltas."""
    systems = list(results)
    directions = sorted({d for sys_scores in results.values() for d in sys_scores})
    metrics = sorted({m for s in results.values() for d in s.values() for m in d})
    header = ["direction", "metric", *systems]
    if {"baseline", "finetuned"} <= set(systems):
        header.append("Δ")
    rows = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for d in directions:
        for m in metrics:
            vals = [results[s].get(d, {}).get(m) for s in systems]
            cells = [d, m, *("-" if v is None else f"{v:.2f}" for v in vals)]
            if "Δ" in header:
                b, f = results["baseline"].get(d, {}).get(m), results["finetuned"].get(d, {}).get(m)
                cells.append("-" if b is None or f is None else f"{f - b:+.2f}")
            rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def run_evaluation(
    cfg: AppConfig,
    max_eval_samples: int | None = None,
    skip_baseline: bool = False,
) -> dict[str, Any]:
    """Score baseline and fine-tuned systems; write ``results/metrics.json``."""
    from ru_uz_mt.inference.translator import Translator

    test_path = Path(cfg.data.processed_dir) / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} not found — run `make data` first")
    records = list(read_jsonl(test_path))
    if max_eval_samples is not None:
        records = records[:max_eval_samples]
    logger.info("Test records: %d", len(records))

    results_dir = Path(cfg.eval.results_dir)
    systems: list[tuple[str, bool]] = [] if skip_baseline else [("baseline", False)]
    has_adapter = (Path(cfg.adapter_path) / "adapter_config.json").exists()
    if has_adapter:
        systems.append(("finetuned", True))
    else:
        logger.warning("No adapter at %s — only the baseline will be evaluated", cfg.adapter_path)

    results: dict[str, dict[str, dict[str, float]]] = {}
    for name, use_adapter in systems:
        translator = Translator(cfg, use_adapter=use_adapter)
        results[name], outputs = evaluate_system(translator, records, cfg)
        write_jsonl(results_dir / f"predictions_{name}.jsonl", outputs)
        del translator
        gc.collect()

    table = format_table(results)
    print("\n" + table + "\n")
    payload = {
        "base_model": cfg.model.base_model,
        "adapter": cfg.adapter_path if has_adapter else None,
        "num_test_records": len(records),
        "generate": cfg.generate.model_dump(),
        "results": results,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (results_dir / "metrics.md").write_text(table + "\n", encoding="utf-8")
    logger.info("Saved metrics to %s", results_dir / "metrics.json")
    return payload


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    config: str = typer.Option("configs/train_config.yaml", help="Path to YAML config"),
    max_eval_samples: int | None = typer.Option(None, help="Use first N test records"),
    skip_baseline: bool = typer.Option(False, help="Only evaluate the fine-tuned model"),
) -> None:
    """Compute BLEU/chrF for baseline vs fine-tuned."""
    setup_logging()
    run_evaluation(load_config(config), max_eval_samples, skip_baseline)


if __name__ == "__main__":
    app()
