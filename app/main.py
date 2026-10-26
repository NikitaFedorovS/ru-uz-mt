"""FastAPI service: RU↔UZ translation over HTTP.

Run: ``uvicorn app.main:app --host 0.0.0.0 --port 8000``
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ru_uz_mt.config import load_config
from ru_uz_mt.utils import setup_logging

logger = logging.getLogger("ru_uz_mt.api")

MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "2000"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "32"))
REQUEST_TIMEOUT_S = float(os.environ.get("REQUEST_TIMEOUT_S", "60"))

LangCode = Literal["rus_Cyrl", "uzn_Latn"]


class TranslateRequest(BaseModel):
    """Single-text translation request."""

    text: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)
    src_lang: LangCode = "rus_Cyrl"
    tgt_lang: LangCode = "uzn_Latn"


class TranslateResponse(BaseModel):
    """Single-text translation response."""

    translation: str
    src_lang: LangCode
    tgt_lang: LangCode


class BatchTranslateRequest(BaseModel):
    """Batch translation request (one direction)."""

    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)
    src_lang: LangCode = "rus_Cyrl"
    tgt_lang: LangCode = "uzn_Latn"


class BatchTranslateResponse(BaseModel):
    """Batch translation response."""

    translations: list[str]
    src_lang: LangCode
    tgt_lang: LangCode


class HealthResponse(BaseModel):
    """Service status."""

    status: Literal["ok", "loading", "error"]
    model_loaded: bool
    device: str | None = None
    base_model: str | None = None
    adapter: str | None = None


def build_translator() -> Any:
    """Create the Translator from config/env (patched in tests)."""
    from ru_uz_mt.inference.translator import Translator

    return Translator(load_config())


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup; keep it on ``app.state``."""
    setup_logging()
    app.state.translator = None
    app.state.load_error = None
    # One model instance is not safe for concurrent generate() calls.
    app.state.lock = asyncio.Lock()
    try:
        app.state.translator = await asyncio.to_thread(build_translator)
        logger.info("Model ready")
    except Exception as e:  # keep serving /health so the failure is visible
        logger.exception("Failed to load model")
        app.state.load_error = str(e)
    yield


app = FastAPI(title="RU↔UZ Translator", version="0.1.0", lifespan=lifespan)


def _get_translator(request: Request) -> Any:
    translator = request.app.state.translator
    if translator is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return translator


async def _run(request: Request, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
    """Translate in a worker thread with a lock and a timeout; map errors to HTTP codes."""
    translator = _get_translator(request)
    if src_lang == tgt_lang:
        raise HTTPException(status_code=422, detail="src_lang and tgt_lang must differ")
    if any(len(t) > MAX_INPUT_CHARS for t in texts):
        raise HTTPException(status_code=422, detail=f"Each text must be <= {MAX_INPUT_CHARS} chars")
    if any(not t.strip() for t in texts):
        raise HTTPException(status_code=422, detail="Texts must not be blank")
    try:
        async with request.app.state.lock:
            return await asyncio.wait_for(
                asyncio.to_thread(translator.translate_batch, texts, src_lang, tgt_lang),
                timeout=REQUEST_TIMEOUT_S,
            )
    except asyncio.TimeoutError as e:  # not the builtin TimeoutError on Python 3.10
        raise HTTPException(status_code=504, detail="Translation timed out") from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("Translation failed")
        raise HTTPException(status_code=500, detail="Internal translation error") from e


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness + model status."""
    translator = request.app.state.translator
    if translator is None:
        return HealthResponse(
            status="error" if request.app.state.load_error else "loading", model_loaded=False
        )
    return HealthResponse(
        status="ok",
        model_loaded=True,
        device=translator.device,
        base_model=translator.config.model.base_model,
        adapter=translator.adapter_path,
    )


@app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest, request: Request) -> TranslateResponse:
    """Translate one text."""
    [translation] = await _run(request, [req.text], req.src_lang, req.tgt_lang)
    return TranslateResponse(translation=translation, src_lang=req.src_lang, tgt_lang=req.tgt_lang)


@app.post("/translate/batch", response_model=BatchTranslateResponse)
async def translate_batch(req: BatchTranslateRequest, request: Request) -> BatchTranslateResponse:
    """Translate up to ``MAX_BATCH_SIZE`` texts in one direction."""
    translations = await _run(request, req.texts, req.src_lang, req.tgt_lang)
    return BatchTranslateResponse(translations=translations, src_lang=req.src_lang, tgt_lang=req.tgt_lang)
