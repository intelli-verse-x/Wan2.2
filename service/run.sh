#!/usr/bin/env bash
# Convenience launcher for the Wan2.2 local video service.
#
#   WAN_CKPT_DIR=/models/Wan2.2-TI2V-5B ./service/run.sh
#
# Defaults to the single-box embedded-worker mode (no Redis). To scale out,
# set WAN_REDIS_URL and run `python -m service.worker` on each GPU box.
set -euo pipefail

export WAN_TASK="${WAN_TASK:-ti2v-5B}"
export WAN_HOST="${WAN_HOST:-0.0.0.0}"
export WAN_PORT="${WAN_PORT:-8000}"

if [[ -z "${WAN_CKPT_DIR:-}" ]]; then
  echo "ERROR: set WAN_CKPT_DIR to the downloaded checkpoint directory." >&2
  exit 1
fi

cd "$(dirname "$0")/.."
exec uvicorn service.app:app --host "$WAN_HOST" --port "$WAN_PORT"
