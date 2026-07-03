"""Wan2.2 model runner.

Lazily loads the configured Wan pipeline once per worker process and turns a
queued job's parameters into an mp4 on disk. Supports the three text/image
video tasks; speech-to-video and animate have richer inputs and are out of
scope for this generic video endpoint.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from .config import ServiceConfig

logger = logging.getLogger("wan22.runner")

_T2V_TASKS = {"t2v-A14B"}
_I2V_TASKS = {"i2v-A14B"}
_TI2V_TASKS = {"ti2v-5B"}
SUPPORTED_TASKS = _T2V_TASKS | _I2V_TASKS | _TI2V_TASKS

# fps each task samples at, used to derive clip duration. Mirrors the
# ``sample_fps`` in ``wan/configs`` but declared here so the dry-run (no-GPU)
# path can compute durations without importing the torch-backed ``wan`` package.
_TASK_FPS = {"ti2v-5B": 24, "t2v-A14B": 16, "i2v-A14B": 16}


def _coerce_frame_num(num_frames: int) -> int:
    """Wan requires frame counts of the form 4n+1."""
    if num_frames < 5:
        return 5
    return ((num_frames - 1) // 4) * 4 + 1


def _pick_supported_size(task: str, width: int, height: int) -> Tuple[Tuple[int, int], str]:
    """Snap a requested (w, h) to the nearest size the model was trained on.

    Matches orientation first, then closest area, so a 1280x720 request maps to
    a landscape supported size rather than a portrait one.
    """
    from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES

    options: List[str] = list(SUPPORTED_SIZES[task])
    want_landscape = width >= height
    want_area = max(width * height, 1)

    def _score(size_str: str) -> Tuple[int, float]:
        w, h = SIZE_CONFIGS[size_str]
        orientation_penalty = 0 if (w >= h) == want_landscape else 1
        area_delta = abs((w * h) - want_area) / want_area
        return (orientation_penalty, area_delta)

    best = min(options, key=_score)
    return SIZE_CONFIGS[best], best


class WanRunner:
    def __init__(self, config: ServiceConfig) -> None:
        self.cfg = config
        self._model = None
        self._wan_cfg = None
        if config.task not in SUPPORTED_TASKS:
            raise ValueError(
                f"WAN_TASK={config.task!r} is not supported by the video service. "
                f"Supported: {sorted(SUPPORTED_TASKS)}"
            )
        # A checkpoint is only required for real inference. Dry-run mode
        # (WAN_DRY_RUN=1) synthesizes placeholder clips so the full HTTP/queue
        # path can be exercised on machines without a GPU or model weights.
        if not config.dry_run and (not config.ckpt_dir or not os.path.isdir(config.ckpt_dir)):
            raise ValueError(
                f"WAN_CKPT_DIR={config.ckpt_dir!r} is not a valid checkpoint directory."
            )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return

        if self.cfg.dry_run:
            logger.info("Dry-run mode: skipping Wan model load for task=%s", self.cfg.task)
            self._model = "dry-run"
            return

        import wan
        from wan.configs import WAN_CONFIGS

        cfg = self.cfg
        wan_cfg = WAN_CONFIGS[cfg.task]
        self._wan_cfg = wan_cfg

        ctor_kwargs = dict(
            config=wan_cfg,
            checkpoint_dir=cfg.ckpt_dir,
            device_id=cfg.device_id,
            rank=0,
            t5_fsdp=cfg.t5_fsdp,
            dit_fsdp=cfg.dit_fsdp,
            use_sp=False,
            t5_cpu=cfg.t5_cpu,
            convert_model_dtype=cfg.convert_model_dtype,
        )

        logger.info("Loading Wan pipeline task=%s ckpt=%s", cfg.task, cfg.ckpt_dir)
        t0 = time.time()
        if cfg.task in _TI2V_TASKS:
            self._model = wan.WanTI2V(init_on_cpu=True, **ctor_kwargs)
        elif cfg.task in _T2V_TASKS:
            self._model = wan.WanT2V(init_on_cpu=True, **ctor_kwargs)
        elif cfg.task in _I2V_TASKS:
            self._model = wan.WanI2V(init_on_cpu=True, **ctor_kwargs)
        logger.info("Wan pipeline loaded in %.1fs", time.time() - t0)

    @property
    def fps(self) -> int:
        if self._wan_cfg is not None:
            return int(getattr(self._wan_cfg, "sample_fps", 16) or 16)
        return _TASK_FPS.get(self.cfg.task, 16)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(self, params: Dict[str, Any]) -> Tuple[str, float]:
        """Run one generation job. Returns (local_mp4_path, duration_seconds)."""
        self.load()
        cfg = self.cfg

        prompt: str = params.get("prompt", "") or ""
        neg_prompt: str = params.get("negative_prompt", "") or ""
        mode: str = params.get("mode", "t2v")
        ref_images: List[str] = params.get("reference_images") or []

        width = int(params.get("width", 1280))
        height = int(params.get("height", 720))
        frame_num = _coerce_frame_num(int(params.get("num_frames", cfg.default_frames)))
        seed = int(params.get("seed", -1))

        if cfg.dry_run:
            return self._generate_dry_run(prompt, width, height, frame_num, seed)

        from wan.configs import MAX_AREA_CONFIGS
        from wan.utils.utils import save_video

        wan_cfg = self._wan_cfg
        (size_w, size_h), size_str = _pick_supported_size(cfg.task, width, height)

        steps = params.get("num_inference_steps") or cfg.default_steps or wan_cfg.sample_steps
        guide = params.get("guidance_scale") or cfg.default_guide_scale or wan_cfg.sample_guide_scale
        shift = cfg.default_shift if cfg.default_shift is not None else wan_cfg.sample_shift

        img = self._load_first_image(ref_images) if mode == "i2v" or cfg.task in _I2V_TASKS else None
        if cfg.task in _I2V_TASKS and img is None:
            raise ValueError("i2v-A14B requires a reference image but none resolved")

        logger.info(
            "generate task=%s mode=%s size=%s frames=%d steps=%s guide=%s seed=%d img=%s",
            cfg.task, mode, size_str, frame_num, steps, guide, seed, bool(img),
        )

        # The three pipelines take different geometry arguments:
        #   * T2V  -> size only
        #   * TI2V -> size + max_area (+ optional img)
        #   * I2V  -> max_area + img  (NO size kwarg; passing it raises TypeError)
        gen_kwargs: Dict[str, Any] = dict(
            input_prompt=prompt,
            frame_num=frame_num,
            shift=shift,
            sample_solver="unipc",
            sampling_steps=int(steps),
            guide_scale=guide,
            n_prompt=neg_prompt,
            seed=seed,
            offload_model=cfg.offload_model,
        )

        if cfg.task in _TI2V_TASKS:
            gen_kwargs["size"] = (size_w, size_h)
            gen_kwargs["max_area"] = MAX_AREA_CONFIGS[size_str]
            gen_kwargs["img"] = img  # optional for TI2V (None => pure t2v)
        elif cfg.task in _I2V_TASKS:
            gen_kwargs["max_area"] = MAX_AREA_CONFIGS[size_str]
            gen_kwargs["img"] = img
        else:  # T2V
            gen_kwargs["size"] = (size_w, size_h)

        video = self._model.generate(**gen_kwargs)

        out_path = os.path.join(cfg.videos_dir, f"wan_{int(time.time()*1000)}_{seed}.mp4")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_video(
            tensor=video[None],
            save_file=out_path,
            fps=self.fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError("save_video produced no output file")

        duration = round(frame_num / float(self.fps), 3)
        return out_path, duration

    # ------------------------------------------------------------------
    # Dry-run (no-GPU) synthesis
    # ------------------------------------------------------------------
    def _generate_dry_run(
        self, prompt: str, width: int, height: int, frame_num: int, seed: int
    ) -> Tuple[str, float]:
        """Produce a real, playable placeholder mp4 without loading any model.

        Lets the full HTTP + queue path be smoke-tested on machines with no GPU
        or checkpoint. Uses ffmpeg's synthetic ``testsrc`` so the output is a
        valid H.264 file the content-factory client can download and validate.
        """
        import shutil
        import subprocess

        cfg = self.cfg
        fps = self.fps
        # ffmpeg's H.264 encoder needs even dimensions; clamp to a sane range.
        w = max(16, (int(width) // 2) * 2)
        h = max(16, (int(height) // 2) * 2)
        duration = round(frame_num / float(fps), 3)

        out_path = os.path.join(cfg.videos_dir, f"wan_dryrun_{int(time.time()*1000)}_{seed}.mp4")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                "WAN_DRY_RUN requires ffmpeg on PATH to synthesize placeholder clips"
            )

        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi",
            "-i", f"testsrc=size={w}x{h}:rate={fps}:duration={max(duration, 0.1)}",
            "-frames:v", str(frame_num),
            "-pix_fmt", "yuv420p",
            out_path,
        ]
        logger.info(
            "dry-run synth task=%s size=%dx%d frames=%d fps=%d dur=%.2fs prompt=%r",
            cfg.task, w, h, frame_num, fps, duration, (prompt[:60] + "…") if len(prompt) > 60 else prompt,
        )
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError(f"ffmpeg dry-run synthesis failed: {proc.stderr.strip()}")

        return out_path, duration

    # ------------------------------------------------------------------
    # Reference image resolution
    # ------------------------------------------------------------------
    def _load_first_image(self, refs: List[str]):
        if not refs:
            return None
        from PIL import Image

        local = self._resolve_to_local(refs[0])
        if not local or not os.path.isfile(local):
            return None
        return Image.open(local).convert("RGB")

    def _resolve_to_local(self, ref: str) -> Optional[str]:
        """Map a reference (server URL, /files path, or local path) to disk."""
        cfg = self.cfg
        if ref.startswith(cfg.public_base_url):
            ref = ref[len(cfg.public_base_url):]
        if ref.startswith("/files/"):
            return os.path.join(cfg.output_dir, ref[len("/files/"):])
        if ref.startswith("http://") or ref.startswith("https://"):
            return self._download_external(ref)
        if os.path.isfile(ref):
            return ref
        return None

    def _download_external(self, url: str) -> Optional[str]:
        try:
            import urllib.request

            dest = os.path.join(self.cfg.uploads_dir, f"ext_{int(time.time()*1000)}")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to download reference %s: %s", url, exc)
            return None
