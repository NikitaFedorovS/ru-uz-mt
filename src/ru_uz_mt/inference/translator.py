"""Inference: NLLB base model + optional LoRA adapter."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ru_uz_mt.config import SUPPORTED_LANGS, AppConfig, GenerateConfig

logger = logging.getLogger(__name__)


def generate_translations(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    src_lang: str,
    tgt_lang: str,
    gen_cfg: GenerateConfig,
    device: str,
    max_input_len: int = 256,
) -> list[str]:
    """Translate ``texts`` (single direction) in mini-batches.

    Sets ``tokenizer.src_lang`` before encoding and forces the target language
    token as the first generated token (``forced_bos_token_id``), as NLLB requires.
    """
    import torch

    tokenizer.src_lang = src_lang
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
    outputs: list[str] = []
    for start in range(0, len(texts), gen_cfg.batch_size):
        batch = texts[start : start + gen_cfg.batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_input_len
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **enc,
                forced_bos_token_id=forced_bos,
                num_beams=gen_cfg.num_beams,
                max_length=gen_cfg.max_length,
            )
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return outputs


class Translator:
    """Loads NLLB (+ LoRA adapter if present) once and translates RU↔UZ text."""

    def __init__(
        self,
        config: AppConfig,
        adapter_path: str | None = None,
        use_adapter: bool = True,
    ) -> None:
        """Load tokenizer and model.

        Args:
            config: Project config (base model, device, generation params).
            adapter_path: LoRA adapter dir; defaults to ``config.adapter_path``.
            use_adapter: If False, serve the plain base model (baseline).
        """
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        from ru_uz_mt.utils import resolve_device

        self.config = config
        self.gen_cfg = config.generate
        self.device = resolve_device(config.model.device)
        self.adapter_path: str | None = None

        base = config.model.base_model
        logger.info("Loading base model %s on %s", base, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model = AutoModelForSeq2SeqLM.from_pretrained(base, dtype=dtype)

        path = adapter_path or config.adapter_path
        if use_adapter and (Path(path) / "adapter_config.json").exists():
            from peft import PeftModel

            logger.info("Applying LoRA adapter from %s", path)
            model = PeftModel.from_pretrained(model, path)
            model = model.merge_and_unload()  # no LoRA overhead at inference time
            self.adapter_path = path
        elif use_adapter:
            logger.warning("No LoRA adapter at %s — serving the base model", path)

        self.model = model.to(self.device).eval()

    @staticmethod
    def validate_langs(src_lang: str, tgt_lang: str) -> None:
        """Raise ``ValueError`` for unsupported or identical language codes."""
        for lang in (src_lang, tgt_lang):
            if lang not in SUPPORTED_LANGS:
                raise ValueError(f"Unsupported language {lang!r}; expected one of {SUPPORTED_LANGS}")
        if src_lang == tgt_lang:
            raise ValueError("src_lang and tgt_lang must differ")

    def translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        """Translate a list of texts from ``src_lang`` to ``tgt_lang``."""
        self.validate_langs(src_lang, tgt_lang)
        if not texts:
            return []
        return generate_translations(
            self.model, self.tokenizer, texts, src_lang, tgt_lang, self.gen_cfg, self.device
        )

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate a single text."""
        return self.translate_batch([text], src_lang, tgt_lang)[0]
