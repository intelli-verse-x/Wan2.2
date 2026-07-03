# Wan2.2 Local Video Service

A small queued inference server that exposes the Wan2.2 generation pipelines
over HTTP. It implements the exact contract content-factory's
`Wan22VideoGenerator` client expects, so standing this up turns Wan2.2 into a
zero-marginal-cost, self-hosted video backend.

```
POST /generate          -> {"task_id": "..."}
GET  /status/{task_id}  -> {"status": "queued|running|completed|failed", "video_url": ..., "duration": ...}
POST /upload            -> {"url": "..."}     # multipart field "file"
GET  /health            -> {"status": "ok", "worker": {"state": "ready|loading|degraded|fatal", ...}}
GET  /files/...         -> generated mp4s + uploaded references (static)
```

## Why this exists

The Wan2.2 repo ships model code and a CLI (`generate.py`) but **no HTTP/queue
layer**. content-factory already has the client (`tools/generators/video/wan22.py`)
pointing at `http://wan22.aicart.svc.cluster.local:8000`; this service is the
server that was never stood up. A single GPU processes one job at a time and the
queue absorbs bursts.

## Architecture

```
            HTTP                       queue                     GPU
client ──────────────▶  app.py  ───────────────▶  worker.py ─────────────▶ runner.py
(wan22.py)            (FastAPI)   (Redis list or   (1 job at a time)   (WanTI2V/T2V/I2V)
                                  in-memory Queue)
```

- **Embedded mode (default, single box):** no Redis. The API starts the GPU
  worker in a background thread; queue lives in-process.
- **Scaled mode:** set `WAN_REDIS_URL`. Run the API and one
  `python -m service.worker` per GPU box; they share the Redis queue.

## Quick start

```bash
pip install -r requirements.txt           # repo ML stack (torch, etc.)
pip install -r service/requirements.txt   # fastapi, uvicorn, redis

# Download a checkpoint, e.g. Wan2.2-TI2V-5B (single 24GB GPU friendly):
#   huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir /models/Wan2.2-TI2V-5B

WAN_CKPT_DIR=/models/Wan2.2-TI2V-5B ./service/run.sh
```

### No GPU? Dry-run mode

To exercise the full HTTP + queue path on a machine without a GPU or model
weights (local dev, CI), set `WAN_DRY_RUN=1`. The worker skips model loading and
synthesizes a valid H.264 placeholder clip with `ffmpeg` (respecting the
requested size and frame count), so `/generate` → `/status` → `/files` all work
end to end. `WAN_CKPT_DIR` is not required in this mode.

```bash
WAN_DRY_RUN=1 uvicorn service.app:app --host 0.0.0.0 --port 8000
```

Smoke test:

```bash
curl -s localhost:8000/health
TASK=$(curl -s -X POST localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"a knight rides through a misty forest","width":1280,"height":704}' | jq -r .task_id)
watch -n2 "curl -s localhost:8000/status/$TASK"
```

## Configuration (env vars)

| Var | Default | Notes |
|---|---|---|
| `WAN_TASK` | `ti2v-5B` | `ti2v-5B` (t2v+i2v, 24fps, 24GB), `t2v-A14B`, `i2v-A14B` |
| `WAN_CKPT_DIR` | — | **Required.** Path to the downloaded checkpoint dir |
| `WAN_DEVICE_ID` | `0` | GPU index |
| `WAN_OUTPUT_DIR` | `/tmp/wan22_outputs` | Videos + uploads live here; served at `/files` |
| `WAN_PUBLIC_BASE_URL` | `http://localhost:$PORT` | Base URL clients use to fetch results |
| `WAN_REDIS_URL` | — | Set to run API/worker as separate processes |
| `WAN_EMBEDDED_WORKER` | auto | `true` when no Redis; force with `1`/`0` |
| `WAN_DRY_RUN` | `false` | Skip the model; synthesize placeholder mp4s (no GPU/ckpt needed) |
| `WAN_OFFLOAD_MODEL` | `true` | CPU-offload to save VRAM |
| `WAN_T5_CPU` | `false` | Keep T5 on CPU (low-VRAM boxes) |
| `WAN_CONVERT_DTYPE` | `true` | Convert model dtype (recommended single-GPU) |
| `WAN_DEFAULT_FRAMES` | `81` | Coerced to nearest `4n+1` |
| `WAN_DEFAULT_STEPS` / `WAN_DEFAULT_GUIDE_SCALE` / `WAN_DEFAULT_SHIFT` | model defaults | Per-request overridable |

Requested resolutions are snapped to the nearest size the chosen task was
trained on (e.g. TI2V-5B → `1280x704` / `704x1280`).

## Wiring into content-factory

Point the client at this service and make Wan the default backend:

```bash
export PREFER_SELFHOSTED=true
export SELFHOSTED_VIDEO_BASE_URL=http://<wan-host>:8000   # or WAN22_SERVER_URL
export DEFAULT_VIDEO_BACKEND=wan22                        # forces Wan to front of the chain
```

See `content-factory` `tools/generators/factory.py` (`DEFAULT_VIDEO_BACKEND`
promotion) and `docs/WAN_LOCAL_BACKEND.md`.

## Scope

This endpoint covers the text/image video tasks (`t2v`, `i2v`, `ti2v`).
Speech-to-video (`s2v-14B`) and `animate-14B` have richer inputs (audio, pose,
masks) and are handled by the dedicated character-performance pipeline (roadmap
P3), not this generic video endpoint.
