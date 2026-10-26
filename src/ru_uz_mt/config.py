"""Project configuration: Pydantic models + YAML/env loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

ENV_PREFIX = "RU_UZ_MT__"
DEFAULT_CONFIG_PATH = "configs/train_config.yaml"

SUPPORTED_LANGS: tuple[str, ...] = ("rus_Cyrl", "uzn_Latn")


class ModelConfig(BaseModel):
    """Base model and language codes."""

    base_model: str = "facebook/nllb-200-distilled-600M"
    src_lang: str = "rus_Cyrl"
    tgt_lang: str = "uzn_Latn"
    adapter_path: str | None = None
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"


class DataConfig(BaseModel):
    """Corpus source, cleaning filters and split settings."""

    source: Literal["opus", "hf"] = "opus"
    raw_dir: str = "data/raw"
    opus_corpora: list[str] = Field(default_factory=lambda: ["wikimedia", "TED2020", "QED"])
    dataset_name: str | None = None
    dataset_config: str | None = None
    max_len: int = Field(128, gt=0)
    min_len: int = Field(1, ge=1)
    length_ratio: float = Field(3.0, ge=1.0)
    normalize_uzbek_to_latin: bool = True
    normalize_apostrophes: bool = True
    langid_model_path: str | None = None
    splits: tuple[float, float, float] = (0.98, 0.01, 0.01)
    max_test_pairs: int | None = 2000
    processed_dir: str = "data/processed"
    bidirectional: bool = True
    seed: int = 42

    @field_validator("splits")
    @classmethod
    def _splits_sum_to_one(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        if abs(sum(v) - 1.0) > 1e-6 or any(x < 0 for x in v):
            raise ValueError(f"splits must be non-negative and sum to 1.0, got {v}")
        return v

    @model_validator(mode="after")
    def _check_source(self) -> DataConfig:
        if self.source == "hf" and not self.dataset_name:
            raise ValueError("data.dataset_name is required when data.source == 'hf'")
        if self.min_len > self.max_len:
            raise ValueError("data.min_len must be <= data.max_len")
        return self


class LoraSettings(BaseModel):
    """LoRA / QLoRA hyperparameters."""

    r: int = Field(16, gt=0)
    alpha: int = Field(32, gt=0)
    dropout: float = Field(0.05, ge=0.0, lt=1.0)
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    use_qlora: bool = False


class TrainConfig(BaseModel):
    """Trainer hyperparameters."""

    output_dir: str = "models/ru-uz-lora"
    learning_rate: float = Field(2e-4, gt=0)
    num_epochs: float = Field(3, gt=0)
    batch_size: int = Field(16, gt=0)
    grad_accumulation: int = Field(1, gt=0)
    warmup_steps: int = Field(0, ge=0)
    eval_steps: int = Field(1000, gt=0)
    save_steps: int = Field(1000, gt=0)
    logging_steps: int = Field(50, gt=0)
    max_seq_len: int = Field(128, gt=0)
    # Pad batches to a multiple of N: 8 suits CUDA tensor cores; = max_seq_len gives fixed shapes,
    # which keeps the MPS allocator from fragmenting and swapping.
    pad_to_multiple_of: int | None = Field(8, gt=0)
    early_stopping_patience: int = Field(3, ge=0)
    eval_bleu_samples: int = Field(200, ge=0)
    report_to: Literal["none", "mlflow", "wandb"] = "none"
    seed: int = 42


class GenerateConfig(BaseModel):
    """Decoding parameters."""

    num_beams: int = Field(4, gt=0)
    max_length: int = Field(128, gt=0)
    batch_size: int = Field(16, gt=0)


class EvalConfig(BaseModel):
    """Evaluation settings."""

    results_dir: str = "results"
    compute_comet: bool = False
    comet_model: str = "Unbabel/wmt22-comet-da"


class AppConfig(BaseModel):
    """Root configuration object."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    lora: LoraSettings = Field(default_factory=LoraSettings)
    train: TrainConfig = Field(default_factory=TrainConfig)
    generate: GenerateConfig = Field(default_factory=GenerateConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    @property
    def adapter_path(self) -> str:
        """Adapter directory used for inference (falls back to the training output dir)."""
        return self.model.adapter_path or self.train.output_dir


def _apply_env_overrides(raw: dict[str, Any], environ: dict[str, str]) -> dict[str, Any]:
    """Overlay ``RU_UZ_MT__SECTION__FIELD=value`` env vars onto the raw config dict.

    Values are parsed as YAML so that numbers, booleans and lists work
    (e.g. ``RU_UZ_MT__LORA__TARGET_MODULES='[q_proj, v_proj]'``).
    """
    for key, value in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        parts = key[len(ENV_PREFIX) :].lower().split("__")
        if len(parts) != 2:
            continue
        section, field = parts
        raw.setdefault(section, {})
        if not isinstance(raw[section], dict):
            raw[section] = {}
        raw[section][field] = yaml.safe_load(value)
    return raw


def load_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
    """Read a YAML config, apply env overrides and validate it.

    Args:
        path: YAML file path. Defaults to ``$RU_UZ_MT_CONFIG`` or ``configs/train_config.yaml``.

    Returns:
        Validated :class:`AppConfig`.
    """
    load_dotenv()
    path = path or os.environ.get("RU_UZ_MT_CONFIG", DEFAULT_CONFIG_PATH)
    config_path = Path(path)
    raw: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    elif path != DEFAULT_CONFIG_PATH:
        raise FileNotFoundError(f"Config not found: {config_path}")
    raw = _apply_env_overrides(raw, dict(os.environ))
    return AppConfig.model_validate(raw)
