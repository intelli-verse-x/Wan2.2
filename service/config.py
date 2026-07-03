"""Environment-driven configuration for the Wan2.2 queued service.

Every knob is an env var so the same image runs as a bare process, a
docker container, or a k8s Deployment without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ServiceConfig:
    # --- model ---
    task: str               # one of WAN_CONFIGS keys, e.g. "ti2v-5B"
    ckpt_dir: str           # path to the downloaded checkpoint directory
    device_id: int

    # --- memory / sharding flags (passed straight to the pipeline ctor) ---
    t5_cpu: bool
    offload_model: bool
    convert_model_dtype: bool
    t5_fsdp: bool
    dit_fsdp: bool

    # --- generation defaults (overridable per request) ---
    default_frames: int
    default_steps: int | None
    default_guide_scale: float | None
    default_shift: float | None

    # --- server / storage ---
    host: str
    port: int
    output_dir: str
    public_base_url: str

    # --- queue ---
    redis_url: str | None
    embedded_worker: bool

    # --- testing / no-GPU ---
    dry_run: bool

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        port = int(os.getenv("WAN_PORT", "8000"))
        host = os.getenv("WAN_HOST", "0.0.0.0")
        public = os.getenv("WAN_PUBLIC_BASE_URL", f"http://localhost:{port}").rstrip("/")
        redis_url = os.getenv("WAN_REDIS_URL") or None
        # Default to an embedded worker thread only when there is no external
        # Redis queue (i.e. a single-box deployment). With Redis, run a
        # dedicated `python -m service.worker` process instead.
        embedded = _as_bool(os.getenv("WAN_EMBEDDED_WORKER"), default=redis_url is None)

        def _opt_int(name: str) -> int | None:
            raw = os.getenv(name)
            return int(raw) if raw not in (None, "") else None

        def _opt_float(name: str) -> float | None:
            raw = os.getenv(name)
            return float(raw) if raw not in (None, "") else None

        return cls(
            task=os.getenv("WAN_TASK", "ti2v-5B"),
            ckpt_dir=os.getenv("WAN_CKPT_DIR", ""),
            device_id=int(os.getenv("WAN_DEVICE_ID", "0")),
            t5_cpu=_as_bool(os.getenv("WAN_T5_CPU"), default=False),
            offload_model=_as_bool(os.getenv("WAN_OFFLOAD_MODEL"), default=True),
            convert_model_dtype=_as_bool(os.getenv("WAN_CONVERT_DTYPE"), default=True),
            t5_fsdp=_as_bool(os.getenv("WAN_T5_FSDP"), default=False),
            dit_fsdp=_as_bool(os.getenv("WAN_DIT_FSDP"), default=False),
            default_frames=int(os.getenv("WAN_DEFAULT_FRAMES", "81")),
            default_steps=_opt_int("WAN_DEFAULT_STEPS"),
            default_guide_scale=_opt_float("WAN_DEFAULT_GUIDE_SCALE"),
            default_shift=_opt_float("WAN_DEFAULT_SHIFT"),
            host=host,
            port=port,
            output_dir=os.getenv("WAN_OUTPUT_DIR", "/tmp/wan22_outputs"),
            public_base_url=public,
            redis_url=redis_url,
            embedded_worker=embedded,
            dry_run=_as_bool(os.getenv("WAN_DRY_RUN"), default=False),
        )

    @property
    def uploads_dir(self) -> str:
        return os.path.join(self.output_dir, "uploads")

    @property
    def videos_dir(self) -> str:
        return os.path.join(self.output_dir, "videos")


_CONFIG: ServiceConfig | None = None


def get_config() -> ServiceConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = ServiceConfig.from_env()
    return _CONFIG
