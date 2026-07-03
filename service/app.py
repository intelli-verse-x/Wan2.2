"""FastAPI front-end for the Wan2.2 queued service.

Implements the exact HTTP contract that content-factory's
``Wan22VideoGenerator`` client expects, so pointing
``WAN22_SERVER_URL`` / ``SELFHOSTED_VIDEO_BASE_URL`` at this server makes Wan2.2
a drop-in video backend.

Run::

    uvicorn service.app:app --host 0.0.0.0 --port 8000

In embedded-worker mode (no Redis) the GPU worker starts automatically in a
background thread.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import get_config
from .jobs import get_job_store
from .worker import start_embedded_worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("wan22.app")

app = FastAPI(title="Wan2.2 Local Video Service", version="1.0.0")

_cfg = get_config()
_store = get_job_store()
os.makedirs(_cfg.uploads_dir, exist_ok=True)
os.makedirs(_cfg.videos_dir, exist_ok=True)

# Serve generated videos and uploaded references back to the client.
app.mount("/files", StaticFiles(directory=_cfg.output_dir), name="files")

_worker_stop = None
_worker_health = None


class GenerateRequest(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 1280
    height: int = 720
    num_frames: int = 81
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: int = -1
    mode: str = "t2v"  # "t2v" | "i2v"
    reference_images: List[str] = []
    lora_path: Optional[str] = None


@app.on_event("startup")
def _startup() -> None:
    global _worker_stop, _worker_health
    logger.info(
        "Wan2.2 service up: task=%s embedded_worker=%s redis=%s dry_run=%s output=%s",
        _cfg.task, _cfg.embedded_worker, bool(_cfg.redis_url), _cfg.dry_run, _cfg.output_dir,
    )
    if _cfg.embedded_worker:
        _worker_stop, _worker_health = start_embedded_worker(_store, _cfg)


@app.on_event("shutdown")
def _shutdown() -> None:
    if _worker_stop is not None:
        _worker_stop.set()


@app.get("/health")
def health() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": "ok",
        "task": _cfg.task,
        "embedded_worker": _cfg.embedded_worker,
        "dry_run": _cfg.dry_run,
    }
    if _worker_health is not None:
        payload["worker"] = _worker_health.as_dict()
    return payload


@app.post("/generate")
def generate(req: GenerateRequest) -> Dict[str, str]:
    params = req.model_dump()
    task_id = _store.enqueue(params)
    logger.info("Enqueued job %s (mode=%s, %dx%d)", task_id, req.mode, req.width, req.height)
    return {"task_id": task_id}


@app.get("/status/{task_id}")
def status(task_id: str) -> Dict[str, Any]:
    job = _store.get(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown task_id")
    return {
        "task_id": task_id,
        "status": job["status"],
        "video_url": job.get("video_url"),
        "duration": job.get("duration"),
        "error": job.get("error"),
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> Dict[str, str]:
    safe_name = os.path.basename(file.filename or f"upload_{int(time.time()*1000)}")
    dest = os.path.join(_cfg.uploads_dir, f"{int(time.time()*1000)}_{safe_name}")
    contents = await file.read()
    with open(dest, "wb") as fh:
        fh.write(contents)
    rel = os.path.relpath(dest, _cfg.output_dir)
    url = f"{_cfg.public_base_url}/files/{rel}"
    return {"url": url}


@app.exception_handler(Exception)
async def _unhandled(_request, exc: Exception):  # noqa: ANN001
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"error": str(exc)})
