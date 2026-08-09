# Deploying to Google Cloud Run

Two Cloud Run services: the API (FastAPI, FAISS, ONNX embeddings) and the
frontend (nginx serving the built React bundle). Generation happens over the
Gemini API, so no model weights are hosted.

```
Browser ──► consciousness-web ──► consciousness-api ──► Gemini API
            (nginx + bundle)      (FastAPI + FAISS)
```

Both services scale to zero, so an idle deployment costs nothing.

## Why the order matters

Vite inlines environment variables when the bundle is **built**, so the frontend
image has to know the API's URL before it exists as an image. The API in turn
rejects cross-origin requests from origins it has not been told about. That
gives a fixed sequence:

1. Deploy the API, which mints its URL.
2. Build and deploy the frontend against that URL.
3. Update the API's `ALLOWED_ORIGINS` with the frontend's URL.

`cloudrun.sh` does exactly this.

## One-time setup

Set your project id and preferred region:

```bash
export PROJECT_ID="your-project-id"
export REGION="asia-south1"          # pick a region near your users
gcloud config set project "$PROJECT_ID"
```

Enable the services this deployment uses. GCP APIs are off by default, so this
is an explicit opt-in rather than a formality:

```bash
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com
```

Create the private Docker registry that Cloud Run pulls images from:

```bash
gcloud artifacts repositories create consciousness \
  --repository-format=docker \
  --location="$REGION" \
  --description="Images for the consciousness assistant"
```

Let the local Docker client authenticate to it:

```bash
gcloud auth configure-docker "${REGION}-docker.pkg.dev"
```

Store the Gemini key in Secret Manager rather than passing it as a plain
environment variable, so it is never printed in deployment output or visible in
the service's configuration:

```bash
printf '%s' "$GOOGLE_API_KEY" | gcloud secrets create gemini-api-key --data-file=-
```

Grant the service account Cloud Run runs as permission to read it:

```bash
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
gcloud secrets add-iam-policy-binding gemini-api-key \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

## Deploying

```bash
./deploy/cloudrun.sh            # both services
./deploy/cloudrun.sh api        # just the API
./deploy/cloudrun.sh frontend   # just the frontend
```

To ship a code change, run the same command again; Cloud Run shifts traffic to
the new revision once it passes its health check, and keeps the old revision for
rollback.

## Verifying

```bash
API_URL="$(gcloud run services describe consciousness-api --region "$REGION" --format 'value(status.url)')"
curl "$API_URL/health"
```

Then open the frontend URL, ask an on-topic question, ask a follow-up, and ask
something off-topic to confirm the guardrail still declines it.

## Cost

Everything here targets the always-free tier: 2M requests, 180k vCPU-seconds and
360k GiB-seconds per month. With `--min-instances 0` there is no idle cost, at
the price of a cold start (about 19 s) on the first request after a quiet
period. Setting `--min-instances 1` removes the cold start but keeps an instance
running continuously, which does consume the free allowance and then bills.

Set a budget alert under **Billing → Budgets & alerts** before deploying. GCP
has no hard spending cap, so the alert is the safety net.

## Rolling back

```bash
gcloud run revisions list --service consciousness-api --region "$REGION"
gcloud run services update-traffic consciousness-api \
  --region "$REGION" --to-revisions REVISION_NAME=100
```

## Tearing down

```bash
gcloud run services delete consciousness-api --region "$REGION"
gcloud run services delete consciousness-web --region "$REGION"
gcloud artifacts repositories delete consciousness --location "$REGION"
```

## Known limitations

Sessions live in the memory of a single container, so they are lost when a
revision is replaced or an idle instance is reclaimed, and they are not shared
if Cloud Run runs more than one instance. That is acceptable for a demo whose
conversations are short; a shared store (Redis, Firestore) is the fix if this
ever needs to survive scaling.
