"""LoRA / QLoRA fine-tuning of NLLB-200 on bidirectional RU↔UZ data.

CLI: ``python -m ru_uz_mt.train.train_lora --config configs/train_config.yaml``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import typer

from ru_uz_mt.config import AppConfig, load_config
from ru_uz_mt.utils import read_jsonl, resolve_device, setup_logging

logger = logging.getLogger(__name__)


def load_model(cfg: AppConfig, device: str) -> Any:
    """Load the base seq2seq model (4-bit NF4 when QLoRA is on) and attach LoRA adapters."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForSeq2SeqLM

    kwargs: dict[str, Any] = {}
    if cfg.lora.use_qlora:
        if device != "cuda":
            raise RuntimeError("QLoRA (bitsandbytes 4-bit) requires a CUDA GPU")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}

    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.base_model, **kwargs)
    if cfg.lora.use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
    )
    model = get_peft_model(model, lora_config)  # freezes all base weights
    model.print_trainable_parameters()
    return model


def build_tokenize_fn(tokenizer: Any, max_seq_len: int) -> Any:
    """Return a batched ``datasets.map`` function for mixed-direction batches.

    NLLB needs ``src_lang``/``tgt_lang`` set on the tokenizer, so each batch is
    tokenized per language pair and re-assembled in the original order. Labels then
    start with the target language token, which is what the decoder learns to emit.
    """

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        n = len(batch["src"])
        out: dict[str, list[Any]] = {
            "input_ids": [None] * n,
            "attention_mask": [None] * n,
            "labels": [None] * n,
        }
        pairs = sorted(set(zip(batch["src_lang"], batch["tgt_lang"], strict=True)))
        for src_lang, tgt_lang in pairs:
            idx = [
                i for i in range(n) if batch["src_lang"][i] == src_lang and batch["tgt_lang"][i] == tgt_lang
            ]
            tokenizer.src_lang, tokenizer.tgt_lang = src_lang, tgt_lang
            enc = tokenizer(
                [batch["src"][i] for i in idx],
                text_target=[batch["tgt"][i] for i in idx],
                max_length=max_seq_len,
                truncation=True,
            )
            for j, i in enumerate(idx):
                for key in out:
                    out[key][i] = enc[key][j]
        return out

    return tokenize


def load_split(path: Path, max_samples: int | None) -> Any:
    """Load a processed JSONL split as a ``datasets.Dataset``."""
    from datasets import Dataset

    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `make data` first")
    records = list(read_jsonl(path))
    if max_samples is not None:
        records = records[:max_samples]
    return Dataset.from_list(records)


def make_trainer_class() -> type:
    """Build a ``Seq2SeqTrainer`` subclass that also reports BLEU/chrF on each evaluation."""
    from transformers import Seq2SeqTrainer

    from ru_uz_mt.eval.metrics import corpus_scores, group_by_direction
    from ru_uz_mt.inference.translator import generate_translations

    class MTTrainer(Seq2SeqTrainer):
        """Seq2SeqTrainer + generation-based BLEU/chrF on a small val subset."""

        def __init__(self, *args: Any, bleu_records: list[dict[str, Any]], app_cfg: AppConfig, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.bleu_records = bleu_records
            self.app_cfg = app_cfg

        def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
            metrics = super().evaluate(*args, **kwargs)
            if not self.bleu_records:
                return metrics
            model = self.accelerator.unwrap_model(self.model)
            was_training = model.training
            model.eval()
            extra: dict[str, float] = {}
            for direction, recs in group_by_direction(self.bleu_records).items():
                hyps = generate_translations(
                    model,
                    self.processing_class,
                    [r["src"] for r in recs],
                    recs[0]["src_lang"],
                    recs[0]["tgt_lang"],
                    self.app_cfg.generate,
                    str(self.args.device),
                    self.app_cfg.train.max_seq_len,
                )
                for name, value in corpus_scores(hyps, [r["tgt"] for r in recs]).items():
                    extra[f"eval_{name}_{direction}"] = value
            if was_training:
                model.train()
            self.log(extra)
            metrics.update(extra)
            return metrics

    return MTTrainer


def train(
    cfg: AppConfig,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Fine-tune and save only the LoRA adapter to ``cfg.train.output_dir``."""
    import torch
    from transformers import (
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    tc = cfg.train
    set_seed(tc.seed)
    device = resolve_device(cfg.model.device)
    logger.info("Device: %s | QLoRA: %s", device, cfg.lora.use_qlora)

    data_dir = Path(cfg.data.processed_dir)
    train_ds = load_split(data_dir / "train.jsonl", max_train_samples)
    val_ds = load_split(data_dir / "val.jsonl", max_eval_samples)
    bleu_n = min(tc.eval_bleu_samples, len(val_ds))
    bleu_records = val_ds.select(range(bleu_n)).to_list() if bleu_n else []
    logger.info("Train: %d | val: %d | val-for-BLEU: %d", len(train_ds), len(val_ds), bleu_n)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model)
    tokenize = build_tokenize_fn(tokenizer, tc.max_seq_len)
    cols = train_ds.column_names
    train_tok = train_ds.map(tokenize, batched=True, remove_columns=cols, desc="Tokenize train")
    val_tok = val_ds.map(tokenize, batched=True, remove_columns=cols, desc="Tokenize val")

    model = load_model(cfg, device)
    use_cuda = device == "cuda"
    bf16 = use_cuda and torch.cuda.is_bf16_supported()

    args = Seq2SeqTrainingArguments(
        output_dir=tc.output_dir,
        learning_rate=tc.learning_rate,
        num_train_epochs=tc.num_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=tc.batch_size,
        per_device_eval_batch_size=tc.batch_size,
        gradient_accumulation_steps=tc.grad_accumulation,
        warmup_steps=tc.warmup_steps,
        eval_strategy="steps",
        eval_steps=tc.eval_steps,
        save_strategy="steps",
        save_steps=tc.save_steps,
        save_total_limit=2,
        logging_steps=tc.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        label_names=["labels"],
        bf16=bf16,
        fp16=use_cuda and not bf16 and not cfg.lora.use_qlora,
        report_to=[] if tc.report_to == "none" else [tc.report_to],
        seed=tc.seed,
        dataloader_pin_memory=use_cuda,
    )
    callbacks = [EarlyStoppingCallback(tc.early_stopping_patience)] if tc.early_stopping_patience else []
    trainer = make_trainer_class()(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=tc.pad_to_multiple_of
        ),
        processing_class=tokenizer,
        callbacks=callbacks,
        bleu_records=bleu_records,
        app_cfg=cfg,
    )

    result = trainer.train()
    final_metrics = trainer.evaluate()

    out_dir = Path(tc.output_dir)
    trainer.model.save_pretrained(out_dir)  # PeftModel -> adapter weights only
    tokenizer.save_pretrained(out_dir)
    summary = {"train": result.metrics, "eval": final_metrics}
    (out_dir / "train_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved LoRA adapter to %s | final eval: %s", out_dir, final_metrics)
    return summary


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    config: str = typer.Option("configs/train_config.yaml", help="Path to YAML config"),
    max_train_samples: int | None = typer.Option(None, help="Use first N train records"),
    max_eval_samples: int | None = typer.Option(None, help="Use first N val records"),
    max_steps: int | None = typer.Option(None, help="Stop after N optimizer steps"),
) -> None:
    """Run LoRA fine-tuning."""
    setup_logging()
    train(load_config(config), max_train_samples, max_eval_samples, max_steps)


if __name__ == "__main__":
    app()
