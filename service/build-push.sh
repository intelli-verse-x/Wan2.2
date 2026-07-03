#!/usr/bin/env bash
# Build and push the Wan2.2 local video service image to ECR.
#
# Mirrors content-factory's framepack/avatar-animator ECR pattern. The image
# wraps service/app.py (FastAPI) + service/worker.py (GPU worker) and implements
# the exact HTTP contract content-factory's Wan22VideoGenerator expects
# (POST /generate, GET /status/{id}, POST /upload, GET /health).
#
# Usage:
#   ./service/build-push.sh                 # tag = git short sha (+ :latest)
#   ./service/build-push.sh v1.0.0          # explicit tag
#   AWS_ACCOUNT_ID=970547373533 AWS_REGION=us-east-1 ./service/build-push.sh
#
# Prerequisites:
#   - AWS CLI configured with ECR push permissions (IAM s3-user likely lacks
#     ecr:PutImage — run as an operator/role with ECR-push rights).
#   - Docker installed and running.
#
# NOTE: the Dockerfile does `COPY . /app` and installs the repo-root
# requirements.txt, so the build context MUST be the Wan2.2 repo root (this
# script sets that automatically).

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-970547373533}"
AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REGISTRY="${ECR_REGISTRY:-${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com}"
REPO_NAME="${REPO_NAME:-content-factory-wan22}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default tag = git short sha when available, else a timestamp.
if [[ "${1:-}" != "" ]]; then
  TAG="$1"
elif git -C "${REPO_ROOT}" rev-parse --short HEAD >/dev/null 2>&1; then
  TAG="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
else
  TAG="$(date +%Y%m%d-%H%M%S)"
fi

echo "=== Wan2.2 service ECR Build & Push ==="
echo "  Registry: ${ECR_REGISTRY}"
echo "  Repo:     ${REPO_NAME}"
echo "  Tag:      ${TAG}"
echo "  Context:  ${REPO_ROOT}"
echo ""

# ── Step 1: Ensure ECR repo exists ──────────────────────────────────────────
echo "--- Ensuring ECR repository exists..."
aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1 || \
  aws ecr create-repository \
    --repository-name "${REPO_NAME}" \
    --region "${AWS_REGION}" \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256

# ── Step 2: Login to ECR ─────────────────────────────────────────────────────
echo "--- Logging in to ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# ── Step 3: Build (context = repo root, dockerfile = service/Dockerfile) ─────
echo "--- Building Docker image (linux/amd64)..."
docker build \
  --platform linux/amd64 \
  -t "${REPO_NAME}:${TAG}" \
  -f "${REPO_ROOT}/service/Dockerfile" \
  "${REPO_ROOT}"

# ── Step 4: Tag ───────────────────────────────────────────────────────────────
echo "--- Tagging..."
docker tag "${REPO_NAME}:${TAG}" "${ECR_REGISTRY}/${REPO_NAME}:${TAG}"
docker tag "${REPO_NAME}:${TAG}" "${ECR_REGISTRY}/${REPO_NAME}:latest"

# ── Step 5: Push ──────────────────────────────────────────────────────────────
echo "--- Pushing to ECR..."
docker push "${ECR_REGISTRY}/${REPO_NAME}:${TAG}"
docker push "${ECR_REGISTRY}/${REPO_NAME}:latest"

echo ""
echo "=== Done! ==="
echo "  Image: ${ECR_REGISTRY}/${REPO_NAME}:${TAG}"
echo ""
echo "  Point the wan22 Deployment at this tag:"
echo "    kubectl -n aicart set image deploy/wan22 \\"
echo "      wan22=${ECR_REGISTRY}/${REPO_NAME}:${TAG}"
echo ""
echo "  (or edit intelli-verse-kube-infra/content-factory/wan22/deployment.yaml"
echo "   and re-apply with kubectl apply -f)."
