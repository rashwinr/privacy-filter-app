#!/usr/bin/env bash
# One-command Cloud Run deploy from source.
# Usage:
#   PROJECT_ID=my-gcp-project ./scripts/deploy.sh
#   PROJECT_ID=my-gcp-project REGION=asia-south1 SERVICE=privacy-filter \
#     GCS_BUCKET=my-bucket ./scripts/deploy.sh
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID env var (your GCP project)}"
REGION="${REGION:-asia-south1}"
SERVICE="${SERVICE:-privacy-filter}"
GCS_BUCKET="${GCS_BUCKET:-}"

echo "▸ Project: $PROJECT_ID  Region: $REGION  Service: $SERVICE"

ENV_VARS="MODEL_NAME=openai/privacy-filter,MODEL_DEVICE=cpu"
if [ -n "$GCS_BUCKET" ]; then
  ENV_VARS+=",STORAGE_BACKEND=gcs,GCS_BUCKET=$GCS_BUCKET"
  echo "▸ Storage: GCS bucket gs://$GCS_BUCKET"
else
  echo "▸ Storage: local filesystem (/tmp/data, ephemeral on Cloud Run)"
fi

gcloud run deploy "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --source . \
  --platform managed \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --concurrency 4 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 5 \
  --set-env-vars "$ENV_VARS"
