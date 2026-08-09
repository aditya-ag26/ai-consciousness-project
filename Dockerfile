# Backend image: FastAPI + FAISS retrieval + embedding model.
#
# Built in two stages so the compilers and pip caches used to install PyTorch
# never reach the published image. The embedding model is baked in at build
# time, which keeps cold starts fast and lets the container run without
# reaching out to Hugging Face.

# ---------- build stage ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Download the embedding model into the image rather than on first request, so
# a cold start does no network I/O.
ENV FASTEMBED_CACHE_DIR=/opt/fastembed
RUN python -c "\
from fastembed import TextEmbedding; \
TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2', \
cache_dir='/opt/fastembed')"

# ---------- runtime stage ----------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    FASTEMBED_CACHE_DIR=/opt/fastembed \
    # The model is already present, so never attempt a download at runtime.
    HF_HUB_OFFLINE=1 \
    EMBEDDING_BACKEND=onnx

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/fastembed /opt/fastembed

WORKDIR /app

# Only what the service actually reads: application code, configuration, and
# the prebuilt vector store.
COPY src/ ./src/
COPY config/ ./config/
COPY data/06_models/faiss_index/ ./data/06_models/faiss_index/

# Run as an unprivileged user so a compromise inside the container cannot
# modify the application or its dependencies.
# The model cache is written to at startup, so it must belong to the runtime
# user as well; otherwise the loader logs permission errors on every boot.
RUN useradd --create-home --uid 10001 appuser \
 && chown -R appuser:appuser /app /opt/fastembed
USER appuser

# Cloud Run assigns the port at runtime; 8080 is its default for local runs.
ENV PORT=8080
EXPOSE 8080

# Shell form is required so ${PORT} is expanded at runtime; `exec` keeps uvicorn
# as PID 1 so it receives the platform's shutdown signals directly.
CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
