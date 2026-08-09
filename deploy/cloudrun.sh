#!/usr/bin/env bash
#
# Deploys both services to Cloud Run.
#
# Written to be read before it is run: every step is a plain gcloud command, so
# it doubles as a record of what the deployment actually consists of.
#
#   ./deploy/cloudrun.sh              # deploy both services
#   ./deploy/cloudrun.sh api          # deploy only the API
#   ./deploy/cloudrun.sh frontend     # deploy only the frontend
#
# Requires: gcloud (authenticated), docker, and GOOGLE_API_KEY in the environment.

set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID to your Google Cloud project id}"
REGION="${REGION:-asia-south1}"
PROJECT_NUMBER="${PROJECT_NUMBER:-$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')}"
REPO="${REPO:-consciousness}"
API_SERVICE="${API_SERVICE:-consciousness-api}"
WEB_SERVICE="${WEB_SERVICE:-consciousness-web}"

REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}"
TARGET="${1:-all}"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$1"; }

# Cloud Run answers on two hostnames per service: the current
# SERVICE-PROJECTNUMBER.REGION.run.app form and a legacy
# SERVICE-HASH-REGIONCODE.a.run.app one. `status.url` returns only one, so the
# canonical form is constructed and the legacy form read, and both are treated
# as valid origins. Whitelisting only one silently breaks the app for anyone who
# happens to use the other.
service_url() {
  echo "https://${1}-${PROJECT_NUMBER}.${REGION}.run.app"
}

service_legacy_url() {
  gcloud run services describe "$1" --region "$REGION" --format 'value(status.url)'
}

deploy_api() {
  log "Building the API image for linux/amd64 (Cloud Run's architecture)"
  docker build --platform linux/amd64 -t "${REGISTRY}/api:latest" .

  log "Pushing the API image to Artifact Registry"
  docker push "${REGISTRY}/api:latest"

  log "Deploying the API to Cloud Run"
  # --allow-unauthenticated: the frontend calls this from the browser, where no
  #   Google credentials exist. Abuse is handled by the app's own rate limiting.
  # --min-instances 0: scale to zero so an idle service costs nothing.
  # --set-secrets: the key is read from Secret Manager at start, never baked in.
  gcloud run deploy "$API_SERVICE" \
    --image "${REGISTRY}/api:latest" \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 3 \
    --timeout 120 \
    --set-secrets "GOOGLE_API_KEY=gemini-api-key:latest" \
    --set-env-vars "LLM_PROVIDER=gemini,EMBEDDING_BACKEND=onnx,ENABLE_DOCS=false"

  log "API deployed at $(service_url "$API_SERVICE")"
}

deploy_frontend() {
  local api_url
  api_url="$(service_url "$API_SERVICE")"
  log "Building the frontend against ${api_url}"
  # Vite inlines environment variables at build time, so the API URL has to be
  # supplied here rather than at deploy time.
  docker build --platform linux/amd64 \
    --build-arg "VITE_API_BASE_URL=${api_url}" \
    -t "${REGISTRY}/web:latest" ./frontend

  log "Pushing the frontend image to Artifact Registry"
  docker push "${REGISTRY}/web:latest"

  log "Deploying the frontend to Cloud Run"
  gcloud run deploy "$WEB_SERVICE" \
    --image "${REGISTRY}/web:latest" \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --memory 256Mi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 3

  local web_url origins
  web_url="$(service_url "$WEB_SERVICE")"
  origins="${web_url},$(service_legacy_url "$WEB_SERVICE")"

  log "Allowing the browser to call the API from ${origins}"
  # The API rejects cross-origin requests from anywhere it is not told about, so
  # this has to happen after the frontend has a URL. ^;^ changes the delimiter
  # gcloud splits on, so the comma inside the value is not read as a separator.
  gcloud run services update "$API_SERVICE" \
    --region "$REGION" \
    --update-env-vars "^;^ALLOWED_ORIGINS=${origins}"

  log "Frontend deployed at ${web_url}"
}

case "$TARGET" in
  api) deploy_api ;;
  frontend) deploy_frontend ;;
  all) deploy_api && deploy_frontend ;;
  *) echo "Usage: $0 [api|frontend|all]" >&2; exit 1 ;;
esac

log "Done"
