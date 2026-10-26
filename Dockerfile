# Two variants via build-arg:
#   docker build --build-arg VARIANT=gpu -t ru-uz-mt:gpu .   (CUDA 12.1 runtime)
#   docker build --build-arg VARIANT=cpu -t ru-uz-mt:cpu .   (python:3.10-slim)
ARG VARIANT=gpu

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base-gpu
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3-pip curl \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu121

FROM python:3.10-slim AS base-cpu
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

FROM base-${VARIANT} AS final
ARG TORCH_INDEX
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface
WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install torch --index-url ${TORCH_INDEX} \
    && python -m pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY configs ./configs
RUN python -m pip install --no-deps .

RUN useradd -m appuser && mkdir -p /cache/huggingface /app/models && chown -R appuser /cache /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
    CMD curl -fs http://localhost:8000/health | grep -q '"model_loaded":true' || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
