"""GPU worker loop for the Wan2.2 queued service.

Pulls one job at a time, runs it on the GPU, and writes the result back to the
job store. Run as a dedicated process when using Redis::

    python -m service.worker

or let the API start it in a background thread (embedded-worker mode) on a
single-box deployment.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .config import ServiceConfig, get_config
from .jobs import JobStore, get_job_store
from .runner import WanRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("wan22.worker")


@dataclass
class WorkerHealth:
    """Shared, mutable view of the worker's liveness for the /health endpoint.

    Field writes are single attribute assignments guarded by the GIL, which is
    sufficient for this coarse status sharing between the worker thread and the
    request handlers.
    """

    # starting -> loading -> ready | degraded | fatal
    state: str = "starting"
    model_loaded: bool = False
    last_error: Optional[str] = None
    jobs_completed: int = 0
    jobs_failed: int = 0
    started_at: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "alive": self.state in ("loading", "ready", "degraded"),
            "model_loaded": self.model_loaded,
            "last_error": self.last_error,
            "jobs_completed": self.jobs_completed,
            "jobs_failed": self.jobs_failed,
            "uptime_s": round(time.time() - self.started_at, 1),
        }


def _public_url(cfg: ServiceConfig, local_path: str) -> str:
    """Translate a local output path into a client-fetchable URL under /files."""
    import os

    rel = os.path.relpath(local_path, cfg.output_dir)
    return f"{cfg.public_base_url}/files/{rel}"


def _drain_as_failed(
    store: JobStore, stop_event: Optional[threading.Event], error: str,
    health: Optional[WorkerHealth],
) -> None:
    """Fail every queued job with a clear error instead of hanging forever.

    Used when the runner cannot even be constructed (bad task / missing
    checkpoint). Previously an exception here silently killed the worker thread
    and jobs sat in ``queued`` indefinitely while /health still said "ok".
    """
    logger.error("Worker in fatal state, failing incoming jobs: %s", error)
    while stop_event is None or not stop_event.is_set():
        job = store.dequeue(timeout=2.0)
        if not job:
            continue
        store.mark_failed(job["task_id"], error=error)
        if health is not None:
            health.jobs_failed += 1


def run_worker(
    store: Optional[JobStore] = None,
    cfg: Optional[ServiceConfig] = None,
    stop_event: Optional[threading.Event] = None,
    health: Optional[WorkerHealth] = None,
) -> None:
    cfg = cfg or get_config()
    store = store or get_job_store()
    health = health or WorkerHealth()

    logger.info("Worker starting: task=%s device=%d redis=%s dry_run=%s",
                cfg.task, cfg.device_id, bool(cfg.redis_url), cfg.dry_run)

    # Construct the runner. A failure here (unsupported task, missing checkpoint)
    # is fatal for generation, but the worker must stay alive to fail jobs fast
    # rather than dying and leaving them stuck in the queue.
    try:
        runner = WanRunner(cfg)
    except Exception as exc:  # noqa: BLE001
        health.state = "fatal"
        health.last_error = str(exc)
        logger.exception("Worker could not initialize runner")
        _drain_as_failed(store, stop_event, str(exc), health)
        return

    # Warm the model so the first request isn't penalized with load latency.
    health.state = "loading"
    try:
        runner.load()
        health.model_loaded = True
        health.state = "ready"
    except Exception as exc:  # noqa: BLE001
        health.state = "degraded"
        health.last_error = str(exc)
        logger.error("Model warm-load failed (will retry per-job): %s", exc)

    while stop_event is None or not stop_event.is_set():
        job = store.dequeue(timeout=2.0)
        if not job:
            continue

        task_id = job["task_id"]
        logger.info("Picked job %s", task_id)
        store.mark_running(task_id)
        t0 = time.time()
        try:
            local_path, duration = runner.generate(job["params"])
            url = _public_url(cfg, local_path)
            store.mark_completed(task_id, video_url=url, duration=duration)
            health.jobs_completed += 1
            health.model_loaded = True
            health.state = "ready"
            logger.info("Job %s completed in %.1fs -> %s", task_id, time.time() - t0, url)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Job %s failed", task_id)
            store.mark_failed(task_id, error=str(exc))
            health.jobs_failed += 1
            health.last_error = str(exc)


def start_embedded_worker(store: JobStore, cfg: ServiceConfig) -> "tuple[threading.Event, WorkerHealth]":
    """Start the worker in a daemon thread.

    Returns its ``(stop_event, health)`` pair so the API can both shut it down
    and report its liveness on /health.
    """
    stop_event = threading.Event()
    health = WorkerHealth()
    thread = threading.Thread(
        target=run_worker,
        kwargs={"store": store, "cfg": cfg, "stop_event": stop_event, "health": health},
        name="wan22-worker",
        daemon=True,
    )
    thread.start()
    return stop_event, health


if __name__ == "__main__":
    run_worker()
