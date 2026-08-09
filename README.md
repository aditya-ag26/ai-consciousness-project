<div align="center">

# AI Consciousness Research Assistant

Retrieval-augmented question answering over consciousness research, with cited
sources and a calibrated refusal policy for out-of-scope questions.

[Live demo](https://consciousness-web-630962135302.asia-south1.run.app) ·
[API health](https://consciousness-api-630962135302.asia-south1.run.app/health) ·
[Deployment guide](deploy/README.md)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Cloud Run](https://img.shields.io/badge/Cloud_Run-4285F4?logo=googlecloud&logoColor=white)
![Tests](https://img.shields.io/badge/tests-58_passing-brightgreen)

</div>

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Evaluation](#evaluation)
- [Performance](#performance)
- [Security](#security)
- [Deployment](#deployment)
- [Development](#development)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Authors](#authors)

## Overview

The assistant answers questions using a corpus of 153 arXiv papers and 5 expert
transcripts covering consciousness, sentience, and philosophy of mind. Every
answer cites the documents it was drawn from.

Questions outside that corpus are declined rather than answered from the
language model's own knowledge. The decision is made by scoring the question
against the corpus before generation, using a threshold selected from a labelled
evaluation set rather than chosen by hand.

> **Note on the live demo.** The deployed instance uses a free-tier hosted model
> limited to 20 generations per day. When that quota is exhausted the API returns
> `503` with an explanatory message; retrieval and the refusal policy continue to
> work. Running locally with Ollama has no such limit.

## Features

**Calibrated refusal policy.** Each question is embedded and matched against the
corpus. If the nearest document exceeds a distance threshold, the question is
declined and the language model is never invoked. The threshold is tuned by
sweeping it against 56 labelled cases and selecting the value that maximises F1.

**Conversational context.** Follow-up questions such as "why is it called that?"
carry no standalone meaning and would fail retrieval. They are rewritten into
self-contained questions against the conversation history before retrieval runs.
Off-topic questions asked mid-conversation are still declined, because rewriting
resolves references without importing an unrelated topic.

**Interchangeable model backends.** Generation runs against a local Ollama model
or the hosted Gemini API. Embeddings run on PyTorch or ONNX Runtime, both
producing identical vectors, which allows the production container to ship
without PyTorch.

**Source attribution.** Responses include the titles and arXiv categories of the
documents used, and a `refused` flag so clients can render declined questions
distinctly.

**Production deployment.** Two containerised services on Cloud Run behind
HTTPS, deployed by CI on every merge to `main`, with secrets held in Secret
Manager and a least-privilege runtime identity.

## Architecture

```
                    ┌──────────────────────────────────────────────┐
  Browser ─────────►│  consciousness-web   (nginx + React bundle)  │
                    └───────────────────────┬──────────────────────┘
                                            │  HTTPS, CORS-whitelisted
                    ┌───────────────────────▼──────────────────────┐
                    │  consciousness-api   (FastAPI)               │
                    │                                              │
                    │   1. embed the question      (ONNX MiniLM)   │
                    │   2. search the corpus       (FAISS)         │
                    │   3. relevance guardrail  ──► decline ────────┼──► no model call
                    │   4. build prompt + history                  │
                    │   5. generate                ────────────────┼──► Gemini API
                    └──────────────────────────────────────────────┘
```

A question that fails step 3 never reaches step 5, so refusals consume no model
quota. This property is also what makes load testing possible on a free tier.

### Follow-up resolution

```
"why is it called that?"
        │
        ├─ retrieve as written ──► nearest document 1.43 ──► above threshold
        │
        ├─ rewrite against the conversation
        │     → "Why is the hard problem of consciousness called hard?"
        │
        └─ retrieve again ──► nearest document 0.65 ──► answer, with sources
```

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 20+
- A [Google AI Studio API key](https://aistudio.google.com/apikey), or
  [Ollama](https://ollama.com) for a fully offline setup

### Installation

```bash
git clone https://github.com/aditya-ag26/ai-consciousness-project.git
cd ai-consciousness-project

python -m venv venv
./venv/Scripts/activate          # Windows
# source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
cp .env.example .env
```

Set a generation backend in `.env`:

```bash
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-key-here
```

Or run entirely offline with no API key:

```bash
LLM_PROVIDER=ollama
# ollama serve && ollama pull qwen2.5:0.5b-instruct
```

### Running

```bash
uvicorn src.api.main:app --reload        # terminal 1 — API on :8000
cd frontend && npm install && npm run dev # terminal 2 — UI on :5173
```

The vector store is committed to the repository, so no indexing step is
required. To rebuild it from source documents:

```bash
python -m src.data.build_vector_store
```

### Docker

```bash
docker compose up --build
```

Serves the frontend on `:5173` and the API on `:8000`. Add
`--profile local-llm` to start Ollama alongside them.

## Configuration

`config/config.yaml` defines all tunable behaviour: models, retrieval depth,
the relevance threshold, prompt templates, chunk sizes, and history window.

Settings that differ between environments are overridable through environment
variables. See `.env.example` for the annotated list.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `ollama` | `gemini` or `ollama` |
| `GOOGLE_API_KEY` | — | Required when `LLM_PROVIDER=gemini` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama host |
| `EMBEDDING_BACKEND` | `onnx` | `onnx` or `torch` |
| `ALLOWED_ORIGINS` | `http://localhost:5173` | CORS whitelist, comma-separated |
| `RATE_LIMIT_REQUESTS` | `30` | Requests per window, per IP |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit window |
| `ENABLE_DOCS` | `true` | Serve `/docs`; disabled in production |

### Model backends

| | `ollama` | `gemini` |
| --- | --- | --- |
| API key | Not required | `GOOGLE_API_KEY` |
| Data handling | Stays on the machine | Sent to Google |
| Memory | ~1 GB | Negligible |
| Latency | 10–15 s on CPU | 3–5 s |

| | `onnx` | `torch` |
| --- | --- | --- |
| Runtime | ONNX Runtime (fastembed) | PyTorch (sentence-transformers) |
| Image size | 1.11 GB | 2.52 GB |
| Cold start | 19 s | 110 s |

Both embedding backends produce identical unit-normalised vectors, verified by
`tests/rag_pipeline/test_embeddings.py`, so a FAISS index built with one is
readable by the other.

## API Reference

| Method | Endpoint | Response | Purpose |
| --- | --- | --- | --- |
| `GET` | `/health` | `200` | Liveness and model readiness |
| `POST` | `/ask` | `200` | Stateless single question |
| `POST` | `/sessions` | `201` | Create a chat session |
| `GET` | `/sessions/{id}/messages` | `200` | Replay a conversation |
| `POST` | `/sessions/{id}/messages` | `200` | Ask within a session |
| `DELETE` | `/sessions/{id}` | `204` | End a session |

### Example

```bash
curl -X POST https://consciousness-api-630962135302.asia-south1.run.app/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the hard problem of consciousness?","length":"short"}'
```

```json
{
  "answer": "The hard problem of consciousness is the question of how subjective experience arises from brain matter...",
  "sources": ["David Chalmers", "Detecting Qualia in Natural and Artificial Agents (cs.AI)"],
  "refused": false
}
```

### Error responses

| Status | Meaning |
| --- | --- |
| `400` | Invalid `length` value |
| `404` | Unknown session |
| `422` | Request body failed schema validation |
| `429` | Rate limit exceeded; includes `Retry-After` |
| `503` | Generation backend unavailable or out of quota |

Declined questions return `200` with `refused: true`, not an error status — the
request succeeded, and the refusal is the answer.

## Evaluation

The pipeline is measured against 56 labelled cases in
`config/eval_dataset.yaml`, covering on-topic questions with expected sources,
off-topic questions, context-dependent follow-ups, and adversarial near-misses
that borrow the corpus vocabulary without being on topic.

```bash
python -m src.evaluation
python -m src.evaluation --json report.json
```

Retrieval and refusal scoring depend only on the embedding model, so results are
reproducible without consuming model quota.

### Results

| Metric | Result |
| --- | --- |
| Retrieval hit rate @4 | 100% (20/20) |
| Mean reciprocal rank | 0.938 |
| Refusal accuracy | 92.9% |
| Refusal precision / recall | 92.3% / 92.3% |
| Refusal F1 | 0.923 |

| Category | Result |
| --- | --- |
| On-topic questions answered | 20/20 |
| Off-topic questions declined | 18/18 |
| Adversarial near-misses declined | 8/10 |
| Off-topic mid-conversation declined | 2/2 |
| Context-dependent follow-ups resolved | 4/6 |

### Threshold selection

The harness sweeps the relevance threshold and reports F1 at each value:

| Threshold | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| 0.80 | 94.6% | 100.0% | 88.5% | 0.939 |
| 0.90 | 92.9% | 95.8% | 88.5% | 0.920 |
| **0.95** | **92.9%** | **92.3%** | **92.3%** | **0.923** |
| 1.15 | 85.7% | 78.1% | 96.2% | 0.862 |
| 1.40 | 78.6% | 68.4% | 100.0% | 0.813 |

An initial threshold of 1.15 scored well against straightforward questions.
Adding adversarial cases — for example "how do I raise awareness for my
startup?" — reduced its F1 to 0.842 and motivated the sweep.

Some overlap between classes is irreducible. The corpus contains neuroscience
and AI papers, so "what is emotional intelligence in the workplace" (distance
0.841) scores closer than the on-topic "what did Hameroff propose about
microtubules" (0.926). The prompt retains its own refusal instruction as a
second layer for these cases.

## Performance

500 requests at 50 concurrent connections against the deployed service:

```bash
python -m src.evaluation.loadtest --url <api-url> --concurrency 50 --requests 500
```

| Metric | Cold (autoscaling) | Warm (steady state) |
| --- | --- | --- |
| Throughput | 29.7 req/s | 36.3 req/s |
| p50 latency | 0.62 s | 1.02 s |
| p95 latency | 10.05 s | 2.82 s |
| p99 latency | 11.27 s | 3.01 s |
| Error rate | 0.00% | 0.00% |

The first run was scaling from one instance to three; requests arriving on a
booting instance waited for it. Tail latency in that run reflects autoscaling
behaviour rather than a throughput limit.

**Scope.** The harness sends out-of-scope questions, which exercise the full
request path — HTTP, middleware, embedding, FAISS, serialisation — without
invoking the language model. Generation adds approximately 3–5 s and was not
load tested, as the free tier permits 20 requests per day. These figures should
not be quoted as end-to-end chat latency.

## Security

The API is intentionally public, since browsers hold no cloud credentials.
Protection is applied at the application layer.

| Control | Implementation |
| --- | --- |
| CORS | Explicit origin whitelist, credentials disabled |
| Rate limiting | Sliding window per client IP, `/health` exempt |
| Input validation | Length and emptiness constraints on every field |
| Response headers | `nosniff`, `DENY` framing, no referrer, restricted permissions policy |
| Secret storage | Secret Manager, injected at runtime, never in images or logs |
| Runtime identity | Dedicated service account with a single permission |
| Container user | Unprivileged, non-root |
| CI authentication | Workload Identity Federation; no long-lived keys |

The refusal policy also functions as a cost and abuse control: out-of-scope
prompts never reach the language model, so the deployment cannot be used as a
general-purpose LLM proxy.

## Deployment

Two Cloud Run services, both scaling to zero when idle. Pushing to `main` runs
CI; on success, the deploy workflow builds both images, tags them with the
commit SHA, pushes to Artifact Registry, deploys, updates the API's CORS
whitelist, and smoke-tests the result.

Each deployed revision is traceable to the commit that produced it, and rollback
is a single traffic-shift command.

```bash
export PROJECT_ID=your-project-id
./deploy/cloudrun.sh              # both services
./deploy/cloudrun.sh api          # API only
```

First-time project setup, teardown, and cost notes are documented in
[`deploy/README.md`](deploy/README.md).

## Development

```bash
pytest                      # 58 tests
pytest -m "not slow"        # skip tests that load both embedding backends
ruff check src/ tests/      # lint
```

Test coverage includes the refusal decision, history windowing, session store
semantics including expiry, rate limiting, provider selection and error
translation, and the equivalence of the two embedding backends.

CI runs lint, backend tests, and a frontend type-check on every push. Only
`main` deploys, and only after CI passes on the same commit.

## Project Structure

```
config/              config.yaml and the labelled evaluation set
data/                transcripts, filtered metadata, prebuilt FAISS index
src/
  api/               FastAPI app, session store, rate limiting, headers
  data/              paper filtering and vector store construction
  rag_pipeline/      QueryBot, LLM and embedding backends
  evaluation/        evaluation harness, metrics, load test
frontend/            React interface, served by nginx in its own image
tests/               unit tests mirroring the src/ layout
deploy/              Cloud Run scripts and deployment documentation
.github/workflows/   CI and deployment pipelines
```

## Limitations

- **Session storage is in-process.** Conversations are lost when a revision is
  replaced and are not shared across instances. A shared store would be required
  to scale horizontally.
- **The evaluation set is small.** 56 hand-written cases means absolute
  percentages carry more uncertainty than their precision implies; the threshold
  sweep and relative comparisons are the more reliable signal.
- **Answer quality is not scored.** Assessing it properly requires human
  judgement or a model-based judge.
- **The corpus is a fixed snapshot** of 153 papers and 5 transcripts.
- **Follow-up resolution depends on model capability.** The two unresolved
  follow-ups are cases where a 0.5B local model failed to rewrite the question;
  the hosted model resolves them reliably.

## Authors

- **Aditya Agarwal** — [@aditya-ag26](https://github.com/aditya-ag26)
- **Siddhant Sharma** — [@sidsharmaa](https://github.com/sidsharmaa)

## Acknowledgments

- The researchers whose work forms the knowledge base
- [arXiv](https://arxiv.org/help/api/) for open access to paper metadata
- LangChain, sentence-transformers, FAISS, and fastembed
