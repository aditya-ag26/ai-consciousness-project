<div align="center">

# AI Consciousness Research Assistant

**A retrieval-augmented chat assistant over consciousness research — grounded in cited sources, and built to refuse anything it cannot support.**

[**Live demo**](https://consciousness-web-630962135302.asia-south1.run.app) · [API health](https://consciousness-api-630962135302.asia-south1.run.app/health)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Cloud Run](https://img.shields.io/badge/Google_Cloud_Run-4285F4?logo=googlecloud&logoColor=white)
![Tests](https://img.shields.io/badge/tests-58_passing-brightgreen)

</div>

---

## What this is

Ask a question about consciousness and get an answer assembled from 153 arXiv
papers and 5 expert transcripts, with every source cited. Ask about anything
else and it will tell you it cannot help, rather than inventing something.

That second behaviour is the point of the project. A retrieval system that
answers everything is easy; one that reliably knows the edge of its own
knowledge takes measurement.

**Live:** https://consciousness-web-630962135302.asia-south1.run.app

> The deployed demo runs on a free-tier hosted model capped at 20 generations
> per day. If it reports that the model quota is exhausted, retrieval and the
> guardrail still work — off-topic questions are still correctly declined — and
> generation returns when the quota resets. Running locally has no such limit.

---

## Why it is interesting

**It knows what it does not know.** Every question is scored against the corpus
before generation. If the nearest document is too far away, the question is
declined and the language model is never called. The cutoff was not guessed: a
labelled evaluation set sweeps it and picks the value that maximises F1.

**It stays coherent across a conversation.** "Why is it called that?" carries no
meaning on its own, so retrieval alone would reject it. Follow-ups are rewritten
into standalone questions against the conversation first — which resolves them
without letting a genuinely off-topic question sneak in on the back of context.

**Every model is swappable.** Generation runs on a local Ollama model or a hosted
Gemini model; embeddings run on PyTorch or ONNX Runtime. Both embedding backends
produce identical vectors, which is what allowed the container to ship without
PyTorch and cut cold start from 110 s to 19 s.

**It is measured, not asserted.** 58 tests, a 56-case labelled evaluation set, a
threshold sweep, and a load test against the live deployment.

---

## How it works

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

A question that fails step 3 never reaches step 5. That is what makes the
guardrail cheap as well as safe: refusals cost no model quota at all.

### Answering a follow-up

```
"why is it called that?"
        │
        ├─ retrieve as written ──► nearest document 1.43 ──► too far
        │
        ├─ rewrite against the conversation
        │     → "Why is the hard problem of consciousness called hard?"
        │
        └─ retrieve again ──► nearest document 0.65 ──► answer, with sources
```

An off-topic question asked mid-conversation goes through the same path and is
still declined, because rewriting resolves references without importing a topic
into a question that already has its own.

---

## Results

### Retrieval and guardrail

Measured over 56 hand-labelled cases (`config/eval_dataset.yaml`) with
`python -m src.evaluation`:

| Metric | Result |
| --- | --- |
| Retrieval hit rate @4 | **100%** (20/20) |
| Mean reciprocal rank | **0.938** |
| Guardrail accuracy | **92.9%** |
| Guardrail precision / recall | **92.3% / 92.3%** |
| Guardrail F1 | **0.923** |

By category: 20/20 on-topic answered · 18/18 off-topic declined · 8/10
adversarial near-misses declined · 2/2 off-topic mid-conversation declined ·
4/6 context-dependent follow-ups resolved.

**The threshold is tuned, not chosen.** The harness sweeps it and reports F1 at
each value:

| Threshold | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| 0.80 | 94.6% | 100.0% | 88.5% | 0.939 |
| 0.90 | 92.9% | 95.8% | 88.5% | 0.920 |
| **0.95** | **92.9%** | **92.3%** | **92.3%** | **0.923** |
| 1.15 | 85.7% | 78.1% | 96.2% | 0.862 |
| 1.40 | 78.6% | 68.4% | 100.0% | 0.813 |

The first value tried was 1.15, which looked perfect against easy questions.
Adding adversarial near-misses — off-topic questions that borrow the corpus's
vocabulary, like *"how do I raise awareness for my startup?"* — dropped its F1
to 0.842 and exposed the real behaviour.

**Some overlap is irreducible.** The corpus contains neuroscience and AI papers,
so *"what is emotional intelligence in the workplace"* (distance 0.841) lands
nearer than the genuinely on-topic *"what did Hameroff propose about
microtubules"* (0.926). No single cutoff separates those, which is why the
prompt keeps its own refusal instruction as a second line of defence.

### Performance

500 requests at 50 concurrency against the live Cloud Run service
(`python -m src.evaluation.loadtest`):

| | Cold (autoscaling) | Warm (steady state) |
| --- | --- | --- |
| Throughput | 29.7 req/s | **36.3 req/s** |
| p50 latency | 0.62 s | **1.02 s** |
| p95 latency | 10.05 s | **2.82 s** |
| p99 latency | 11.27 s | **3.01 s** |
| Error rate | **0.00%** | **0.00%** |

The gap between the runs is the interesting part: the first was scaling one
instance to three, and requests landing on a booting instance waited for it.
Tail latency here is an autoscaling property, not a throughput ceiling.

**These figures cover the retrieval path.** Load testing uses out-of-scope
questions, which exercise the entire request path — HTTP, middleware,
embedding, FAISS, serialisation — without calling the language model. Generation
adds roughly 3–5 s and was not load tested, because the free tier allows 20
requests per day. Reporting these numbers as end-to-end chat latency would
overstate them.

### Cold start

| | PyTorch backend | ONNX backend |
| --- | --- | --- |
| Image size | 2.52 GB | **1.11 GB** |
| Cold start | 110 s | **19 s** |

Both run the same MiniLM weights and produce identical vectors — pinned by a
test — so the FAISS index and the tuned threshold carry over unchanged.

---

## Built with

| Layer | Choice |
| --- | --- |
| API | FastAPI, Pydantic, Uvicorn |
| Retrieval | FAISS, sentence-transformers MiniLM, LangChain |
| Embedding runtime | ONNX Runtime (fastembed) in production, PyTorch for development |
| Generation | Google Gemini (hosted) or Ollama (local) |
| Frontend | React 19, TypeScript, Vite, nginx |
| Infrastructure | Docker, Google Cloud Run, Artifact Registry, Secret Manager |
| CI/CD | GitHub Actions with Workload Identity Federation |
| Quality | pytest (58 tests), ruff |

---

## Running it locally

**Prerequisites:** Python 3.11+, Node.js 20+, and either an
[AI Studio API key](https://aistudio.google.com/apikey) or
[Ollama](https://ollama.com) for a fully offline setup.

```bash
git clone https://github.com/aditya-ag26/ai-consciousness-project.git
cd ai-consciousness-project

python -m venv venv && ./venv/Scripts/activate    # Windows
# python3 -m venv venv && source venv/bin/activate  # macOS / Linux

pip install -r requirements.txt
cp .env.example .env
```

Choose a generation backend in `.env`:

```bash
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-key-here

# …or run entirely offline, with no API key:
# LLM_PROVIDER=ollama
# (then: ollama serve && ollama pull qwen2.5:0.5b-instruct)
```

Start the API and the frontend:

```bash
uvicorn src.api.main:app --reload      # terminal 1
cd frontend && npm install && npm run dev   # terminal 2
```

Open http://localhost:5173. The knowledge base is committed to the repository,
so there is nothing to build first.

### With Docker

```bash
docker compose up --build
```

Frontend on `:5173`, API on `:8000`. Add `--profile local-llm` to bring up
Ollama alongside them.

---

## Configuration

`config/config.yaml` holds everything tunable: models, retrieval depth, the
relevance threshold, prompt templates, chunking, and history window. No
behaviour is hardcoded.

Environment variables override the settings that differ between a laptop and a
container — see `.env.example` for the full list.

| Variable | Purpose |
| --- | --- |
| `LLM_PROVIDER` | `gemini` or `ollama` |
| `GOOGLE_API_KEY` | required for `gemini` |
| `EMBEDDING_BACKEND` | `onnx` or `torch` |
| `ALLOWED_ORIGINS` | CORS whitelist |
| `RATE_LIMIT_REQUESTS` | per-IP request cap |
| `ENABLE_DOCS` | serve `/docs`; off in production |

---

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | liveness, plus whether the model finished loading |
| `POST` | `/ask` | one-off question, no conversation history |
| `POST` | `/sessions` | start a chat session |
| `GET` | `/sessions/{id}/messages` | replay a conversation |
| `POST` | `/sessions/{id}/messages` | ask within a session, using its history |
| `DELETE` | `/sessions/{id}` | end a session |

```bash
curl -X POST https://consciousness-api-630962135302.asia-south1.run.app/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the hard problem of consciousness?","length":"short"}'
```

Responses carry `refused: true` when the guardrail declines, so a client can
render that differently from an answer. A generation outage returns **503** with
a specific reason rather than a hang — the service is healthy, its upstream is
not.

---

## Security

The API is deliberately public — a browser has no cloud credentials — so the
protections are at the application layer:

- **CORS whitelist**, never a wildcard, with credentials disabled.
- **Per-IP rate limiting.** Each answer costs real compute or metered quota, so
  an unthrottled endpoint is a practical denial-of-service risk.
- **Input caps** on every field, so a large body cannot reach the model.
- **Security headers** on both services (`nosniff`, `DENY` framing, no referrer,
  restrictive permissions policy).
- **Secrets in Secret Manager**, injected at runtime. The API key never appears
  in an image, a config listing, or deployment logs.
- **A dedicated service account** whose only permission is reading that one
  secret — not the default account, which carries project-wide Editor.
- **Non-root container**, running as an unprivileged user.
- **Keyless CI.** GitHub authenticates through Workload Identity Federation and
  receives a short-lived token, so no service-account key exists to leak.

The guardrail is also a security control: it prevents the deployment being used
as a free general-purpose LLM proxy, because off-topic prompts never reach the
model.

---

## Testing and evaluation

```bash
pytest                                    # 58 tests
ruff check src/ tests/                    # lint
python -m src.evaluation                  # labelled-set evaluation
python -m src.evaluation.loadtest --url … # load test a deployment
```

Tests cover the guardrail decision, history windowing, session store semantics
including expiry, rate limiting, provider selection and failure translation, and
the equivalence of the two embedding backends — the invariant that lets the
container ship without PyTorch.

CI runs lint, tests, and a frontend type-check on every push. Only `main`
deploys, and only after CI passes on that commit.

---

## Deployment

Two Cloud Run services, both scaling to zero when idle. Pushing to `main`
triggers CI; if it passes, the deploy workflow builds both images, tags them
with the commit SHA, pushes to Artifact Registry, deploys, updates the API's
CORS whitelist with the frontend's URL, and smoke-tests the result.

Every deployed revision is traceable to the exact commit that produced it, and
rollback is a single traffic-shift command.

Manual deployment, and the one-time project setup, are documented in
[`deploy/README.md`](deploy/README.md).

---

## Project structure

```
config/            config.yaml (all tunable behaviour) and the evaluation set
data/              transcripts, filtered metadata, prebuilt FAISS index
src/
  api/             FastAPI app, session store, rate limiting, headers
  data/            pipelines that filter papers and build the vector store
  rag_pipeline/    QueryBot, pluggable LLM and embedding backends
  evaluation/      labelled-set harness, metrics, load test
frontend/          React chat interface, served by nginx in its own image
tests/             unit tests mirroring the src/ layout
deploy/            Cloud Run scripts and deployment documentation
.github/workflows/ CI and deployment pipelines
```

---

## Limitations

Stated plainly, because they shape how the results should be read:

- **Sessions live in one container's memory.** They are lost when a revision is
  replaced and are not shared across instances. Fine for short conversations;
  a shared store is the fix if this ever needs to survive scaling.
- **The evaluation set is small** (56 hand-written cases), so absolute
  percentages carry more uncertainty than their precision suggests. The
  threshold sweep and the relative comparisons are the more durable signal.
- **Answer quality is not scored.** Doing that properly needs human judgement or
  a model-based judge.
- **The corpus is a snapshot** of 153 papers and 5 transcripts, not the field.
- **Follow-up rewriting depends on the model.** The two missed follow-ups are
  cases where a 0.5B local model failed to rewrite the question; the hosted
  model resolves them reliably.

---

## Authors

**Aditya Agarwal** — [@aditya-ag26](https://github.com/aditya-ag26)
**Siddhant Sharma** — [@sidsharmaa](https://github.com/sidsharmaa)

## Acknowledgments

- The researchers whose work forms the knowledge base
- [arXiv](https://arxiv.org/help/api/) for open access to paper metadata
- LangChain, sentence-transformers, FAISS, and fastembed
