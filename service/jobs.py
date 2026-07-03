"""Job store for the Wan2.2 queued service.

Two interchangeable backends implement the same interface:

* ``InMemoryJobStore`` — a ``queue.Queue`` + dict guarded by a lock. Only valid
  when the API and the worker live in the same process (embedded-worker mode).
* ``RedisJobStore`` — a Redis list (the queue) plus one hash per job. Lets the
  API and one-or-more GPU workers run as separate processes / pods.

``get_job_store()`` picks the backend from ``WAN_REDIS_URL``.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from typing import Any, Dict, Optional

from .config import get_config

# Job lifecycle states. "completed" / "failed" are terminal; the content-factory
# client also treats "error" as terminal, so we never emit that spelling here.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

_TERMINAL = {STATUS_COMPLETED, STATUS_FAILED}


def _new_job(params: Dict[str, Any]) -> Dict[str, Any]:
    now = time.time()
    return {
        "task_id": uuid.uuid4().hex,
        "status": STATUS_QUEUED,
        "params": params,
        "video_url": None,
        "duration": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
    }


class JobStore:
    """Abstract job store."""

    def enqueue(self, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def dequeue(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def update(self, task_id: str, **fields: Any) -> None:
        raise NotImplementedError

    # convenience helpers ------------------------------------------------
    def mark_running(self, task_id: str) -> None:
        self.update(task_id, status=STATUS_RUNNING)

    def mark_completed(self, task_id: str, video_url: str, duration: float) -> None:
        self.update(
            task_id,
            status=STATUS_COMPLETED,
            video_url=video_url,
            duration=duration,
            error=None,
        )

    def mark_failed(self, task_id: str, error: str) -> None:
        self.update(task_id, status=STATUS_FAILED, error=error)


class InMemoryJobStore(JobStore):
    def __init__(self) -> None:
        self._q: "queue.Queue[str]" = queue.Queue()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def enqueue(self, params: Dict[str, Any]) -> str:
        job = _new_job(params)
        with self._lock:
            self._jobs[job["task_id"]] = job
        self._q.put(job["task_id"])
        return job["task_id"]

    def dequeue(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        try:
            task_id = self._q.get(timeout=timeout)
        except queue.Empty:
            return None
        with self._lock:
            return dict(self._jobs.get(task_id) or {})

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(task_id)
            return dict(job) if job else None

    def update(self, task_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(task_id)
            if not job:
                return
            job.update(fields)
            job["updated_at"] = time.time()


class RedisJobStore(JobStore):
    _QUEUE_KEY = "wan:queue"

    def __init__(self, url: str) -> None:
        import redis  # imported lazily so in-memory mode needs no redis dep

        self._r = redis.Redis.from_url(url, decode_responses=True)

    @staticmethod
    def _job_key(task_id: str) -> str:
        return f"wan:job:{task_id}"

    def _write(self, job: Dict[str, Any]) -> None:
        self._r.set(self._job_key(job["task_id"]), json.dumps(job))

    def enqueue(self, params: Dict[str, Any]) -> str:
        job = _new_job(params)
        self._write(job)
        self._r.rpush(self._QUEUE_KEY, job["task_id"])
        return job["task_id"]

    def dequeue(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        popped = self._r.blpop(self._QUEUE_KEY, timeout=int(max(1, timeout)))
        if not popped:
            return None
        _key, task_id = popped
        return self.get(task_id)

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        raw = self._r.get(self._job_key(task_id))
        return json.loads(raw) if raw else None

    def update(self, task_id: str, **fields: Any) -> None:
        raw = self._r.get(self._job_key(task_id))
        if not raw:
            return
        job = json.loads(raw)
        job.update(fields)
        job["updated_at"] = time.time()
        self._write(job)


_STORE: Optional[JobStore] = None


def get_job_store() -> JobStore:
    global _STORE
    if _STORE is None:
        cfg = get_config()
        if cfg.redis_url:
            _STORE = RedisJobStore(cfg.redis_url)
        else:
            _STORE = InMemoryJobStore()
    return _STORE
