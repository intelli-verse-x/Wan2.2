"""Self-contained test suite for the Wan2.2 local video service.

Runs on any machine (no GPU, no torch, no checkpoint) via ``WAN_DRY_RUN`` and a
set of injected fake ``wan`` modules. Exercises:

  * config / frame-coercion unit logic
  * the per-task ``generate`` kwargs contract (regression test for the i2v
    ``size`` vs ``max_area`` bug)
  * the dry-run runner producing a real, playable mp4
  * the full HTTP contract (/health, /generate, /status, /upload, /files)
  * fail-fast behaviour when the model cannot initialize

Usage::

    python3 -m service.test_service        # from repo root
    python3 service/test_service.py

Exits non-zero if any check fails.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import types

# --- tiny test harness ---------------------------------------------------
_PASS = 0
_FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  \033[32mPASS\033[0m {name}")
    else:
        _FAIL += 1
        print(f"  \033[31mFAIL\033[0m {name}" + (f"  -> {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def _ffprobe(path: str) -> dict:
    """Return {codec, width, height, duration} for an mp4, or {} if no ffprobe."""
    if not shutil.which("ffprobe"):
        return {}
    out = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration:stream=codec_name,width,height",
         "-of", "default=noprint_wrappers=1", path],
        capture_output=True, text=True,
    ).stdout
    d: dict = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k] = v
    return d


# --- fake wan modules so runner.generate can be tested without torch ------
def _install_fake_wan(save_calls: list) -> None:
    wan = types.ModuleType("wan")
    configs = types.ModuleType("wan.configs")
    configs.SIZE_CONFIGS = {
        "1280*704": (1280, 704), "704*1280": (704, 1280),
        "1280*720": (1280, 720), "720*1280": (720, 1280),
        "480*832": (480, 832), "832*480": (832, 480),
    }
    configs.MAX_AREA_CONFIGS = {k: w * h for k, (w, h) in configs.SIZE_CONFIGS.items()}
    configs.SUPPORTED_SIZES = {
        "t2v-A14B": ("720*1280", "1280*720", "480*832", "832*480"),
        "i2v-A14B": ("720*1280", "1280*720", "480*832", "832*480"),
        "ti2v-5B": ("704*1280", "1280*704"),
    }
    configs.WAN_CONFIGS = {}

    utils_pkg = types.ModuleType("wan.utils")
    utils_mod = types.ModuleType("wan.utils.utils")

    def _save_video(tensor=None, save_file=None, **kw):  # noqa: ANN001
        save_calls.append({"save_file": save_file, **kw})
        with open(save_file, "wb") as fh:
            fh.write(b"\x00" * 1024)  # non-empty placeholder

    utils_mod.save_video = _save_video
    utils_pkg.utils = utils_mod

    sys.modules["wan"] = wan
    sys.modules["wan.configs"] = configs
    sys.modules["wan.utils"] = utils_pkg
    sys.modules["wan.utils.utils"] = utils_mod


def _remove_fake_wan() -> None:
    for m in ["wan", "wan.configs", "wan.utils", "wan.utils.utils"]:
        sys.modules.pop(m, None)


class _CapturingModel:
    def __init__(self):
        self.captured = None

    def generate(self, **kwargs):
        self.captured = kwargs
        import numpy as np
        return np.zeros((3, 4, 8, 8), dtype=np.float32)  # C,T,H,W-ish


class _FakeWanCfg:
    sample_steps = 50
    sample_guide_scale = 5.0
    sample_shift = 5.0
    sample_fps = 24


# ------------------------------------------------------------------------
# Test 1: config parsing
# ------------------------------------------------------------------------
def test_config():
    section("config parsing")
    from service import config as cfgmod

    os.environ["WAN_DRY_RUN"] = "1"
    os.environ["WAN_TASK"] = "ti2v-5B"
    cfgmod._CONFIG = None
    c = cfgmod.get_config()
    check("WAN_DRY_RUN=1 -> dry_run True", c.dry_run is True)
    check("task read from env", c.task == "ti2v-5B", c.task)

    os.environ["WAN_DRY_RUN"] = "0"
    cfgmod._CONFIG = None
    check("WAN_DRY_RUN=0 -> dry_run False", cfgmod.get_config().dry_run is False)
    cfgmod._CONFIG = None


# ------------------------------------------------------------------------
# Test 2: frame coercion
# ------------------------------------------------------------------------
def test_frame_coercion():
    section("frame coercion (must be 4n+1)")
    from service.runner import _coerce_frame_num
    cases = {1: 5, 5: 5, 81: 81, 80: 77, 82: 81, 49: 49, 50: 49, 100: 97}
    for inp, expected in cases.items():
        got = _coerce_frame_num(inp)
        check(f"_coerce_frame_num({inp}) == {expected}", got == expected, f"got {got}")


# ------------------------------------------------------------------------
# Test 3: per-task generate kwargs (regression for i2v size/max_area bug)
# ------------------------------------------------------------------------
def test_generate_kwargs():
    section("per-task generate kwargs contract")
    save_calls: list = []
    _install_fake_wan(save_calls)
    try:
        from service.config import ServiceConfig
        from service.runner import WanRunner

        ckpt = tempfile.mkdtemp(prefix="fake_ckpt_")
        out = tempfile.mkdtemp(prefix="wan_out_")
        # a real reference image for the i2v path
        from PIL import Image
        ref = os.path.join(out, "ref.png")
        Image.new("RGB", (64, 64), (10, 20, 30)).save(ref)

        def _mk(task: str) -> ServiceConfig:
            return ServiceConfig(
                task=task, ckpt_dir=ckpt, device_id=0,
                t5_cpu=False, offload_model=True, convert_model_dtype=True,
                t5_fsdp=False, dit_fsdp=False,
                default_frames=81, default_steps=None, default_guide_scale=None,
                default_shift=None, host="0.0.0.0", port=8000,
                output_dir=out, public_base_url="http://localhost:8000",
                redis_url=None, embedded_worker=True, dry_run=False,
            )

        # ---- t2v: size only, no img, no max_area ----
        r = WanRunner(_mk("t2v-A14B"))
        r._model = _CapturingModel()
        r._wan_cfg = _FakeWanCfg()
        r.generate({"prompt": "x", "mode": "t2v", "width": 1280, "height": 720, "num_frames": 81})
        kw = r._model.captured
        check("t2v passes size", "size" in kw)
        check("t2v omits img", "img" not in kw)
        check("t2v omits max_area", "max_area" not in kw)

        # ---- i2v: max_area + img, NO size (the bug) ----
        r = WanRunner(_mk("i2v-A14B"))
        r._model = _CapturingModel()
        r._wan_cfg = _FakeWanCfg()
        r.generate({"prompt": "x", "mode": "i2v", "reference_images": [ref],
                    "width": 1280, "height": 720, "num_frames": 81})
        kw = r._model.captured
        check("i2v OMITS size (regression)", "size" not in kw,
              f"size leaked: {kw.get('size')}")
        check("i2v passes max_area", "max_area" in kw)
        check("i2v passes img (not None)", kw.get("img") is not None)

        # ---- ti2v: size + max_area + img ----
        r = WanRunner(_mk("ti2v-5B"))
        r._model = _CapturingModel()
        r._wan_cfg = _FakeWanCfg()
        r.generate({"prompt": "x", "mode": "t2v", "width": 1280, "height": 704, "num_frames": 81})
        kw = r._model.captured
        check("ti2v passes size", "size" in kw)
        check("ti2v passes max_area", "max_area" in kw)
        check("ti2v has img key (None ok for pure t2v)", "img" in kw)

        check("save_video invoked once per generate", len(save_calls) == 3, str(len(save_calls)))
    finally:
        _remove_fake_wan()


# ------------------------------------------------------------------------
# Test 4: dry-run runner produces a real mp4
# ------------------------------------------------------------------------
def test_dry_run_runner():
    section("dry-run runner synthesizes a valid mp4")
    from service.config import ServiceConfig
    from service.runner import WanRunner

    out = tempfile.mkdtemp(prefix="wan_dry_")
    cfg = ServiceConfig(
        task="ti2v-5B", ckpt_dir="", device_id=0,
        t5_cpu=False, offload_model=True, convert_model_dtype=True,
        t5_fsdp=False, dit_fsdp=False,
        default_frames=81, default_steps=None, default_guide_scale=None,
        default_shift=None, host="0.0.0.0", port=8000,
        output_dir=out, public_base_url="http://localhost:8000",
        redis_url=None, embedded_worker=True, dry_run=True,
    )
    r = WanRunner(cfg)  # must NOT raise despite empty ckpt_dir
    check("dry-run WanRunner builds without checkpoint", True)
    path, duration = r.generate(
        {"prompt": "test clip", "width": 1280, "height": 720, "num_frames": 49})
    check("output file exists & non-empty", os.path.isfile(path) and os.path.getsize(path) > 0)
    check("duration == 49/24 (2.042)", abs(duration - 49 / 24) < 0.01, str(duration))
    probe = _ffprobe(path)
    if probe:
        check("codec is h264", probe.get("codec_name") == "h264", probe.get("codec_name", "?"))
        check("resolution 1280x720", probe.get("width") == "1280" and probe.get("height") == "720",
              f"{probe.get('width')}x{probe.get('height')}")


# ------------------------------------------------------------------------
# helper: load a fresh app instance under given env
# ------------------------------------------------------------------------
def _fresh_app(env: dict):
    os.environ.update(env)
    import service.config as cfgmod
    import service.jobs as jobsmod
    cfgmod._CONFIG = None
    jobsmod._STORE = None
    import service.app as appmod
    appmod = importlib.reload(appmod)
    return appmod


# ------------------------------------------------------------------------
# Test 5: full HTTP contract in dry-run mode
# ------------------------------------------------------------------------
def test_http_dry_run():
    section("HTTP contract (dry-run) via TestClient")
    from fastapi.testclient import TestClient
    import time

    out = tempfile.mkdtemp(prefix="wan_http_")
    appmod = _fresh_app({
        "WAN_DRY_RUN": "1", "WAN_TASK": "ti2v-5B",
        "WAN_OUTPUT_DIR": out, "WAN_PUBLIC_BASE_URL": "http://testserver",
        "WAN_REDIS_URL": "",
    })

    with TestClient(appmod.app) as client:
        # health
        h = client.get("/health").json()
        check("/health status ok", h.get("status") == "ok")
        check("/health reports dry_run", h.get("dry_run") is True)
        # worker warms up in a thread
        for _ in range(50):
            w = client.get("/health").json().get("worker", {})
            if w.get("state") == "ready":
                break
            time.sleep(0.1)
        check("/health worker state ready", w.get("state") == "ready", str(w))

        # generate + poll to completion
        r = client.post("/generate", json={
            "prompt": "a knight in a misty forest", "width": 1280,
            "height": 720, "num_frames": 81})
        check("/generate returns task_id", "task_id" in r.json())
        tid = r.json()["task_id"]

        status = {}
        for _ in range(100):
            status = client.get(f"/status/{tid}").json()
            if status["status"] in ("completed", "failed"):
                break
            time.sleep(0.1)
        check("t2v job completed", status.get("status") == "completed", str(status))
        check("completed job has video_url", bool(status.get("video_url")))
        check("duration == 81/24 (3.375)", abs((status.get("duration") or 0) - 81 / 24) < 0.01,
              str(status.get("duration")))

        # download via /files
        vurl = status["video_url"]
        rel = vurl.split("/files/", 1)[1]
        dl = client.get(f"/files/{rel}")
        check("/files serves the video (200)", dl.status_code == 200, str(dl.status_code))
        check("/files body non-empty", len(dl.content) > 0)

        # upload + i2v
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (640, 640), (30, 90, 150)).save(buf, format="PNG")
        buf.seek(0)
        up = client.post("/upload", files={"file": ("ref.png", buf, "image/png")})
        check("/upload returns url", "url" in up.json(), str(up.json()))
        img_url = up.json()["url"]

        r2 = client.post("/generate", json={
            "prompt": "pan across", "mode": "i2v", "reference_images": [img_url],
            "width": 720, "height": 1280, "num_frames": 25})
        tid2 = r2.json()["task_id"]
        st2 = {}
        for _ in range(100):
            st2 = client.get(f"/status/{tid2}").json()
            if st2["status"] in ("completed", "failed"):
                break
            time.sleep(0.1)
        check("i2v (upload) job completed", st2.get("status") == "completed", str(st2))

        # unknown task id -> 404
        check("unknown task_id -> 404", client.get("/status/nope").status_code == 404)


# ------------------------------------------------------------------------
# Test 6: fail-fast when the model can't initialize
# ------------------------------------------------------------------------
def test_fail_fast():
    section("fail-fast on bad checkpoint (no silent hang)")
    from fastapi.testclient import TestClient
    import time

    out = tempfile.mkdtemp(prefix="wan_fail_")
    appmod = _fresh_app({
        "WAN_DRY_RUN": "0", "WAN_TASK": "ti2v-5B",
        "WAN_CKPT_DIR": "/tmp/definitely_not_a_ckpt_dir_xyz",
        "WAN_OUTPUT_DIR": out, "WAN_PUBLIC_BASE_URL": "http://testserver",
        "WAN_REDIS_URL": "",
    })

    with TestClient(appmod.app) as client:
        # worker should reach fatal state
        w = {}
        for _ in range(50):
            w = client.get("/health").json().get("worker", {})
            if w.get("state") == "fatal":
                break
            time.sleep(0.1)
        check("worker reaches 'fatal' state", w.get("state") == "fatal", str(w))
        check("worker not alive", w.get("alive") is False)
        check("worker reports last_error", bool(w.get("last_error")))

        # a submitted job must FAIL fast, not hang in 'queued'
        tid = client.post("/generate", json={"prompt": "x"}).json()["task_id"]
        status = {}
        for _ in range(50):
            status = client.get(f"/status/{tid}").json()
            if status["status"] in ("completed", "failed"):
                break
            time.sleep(0.1)
        check("job fails fast (not stuck queued)", status.get("status") == "failed", str(status))
        check("failed job carries error message", bool(status.get("error")))

    # reset env so it doesn't leak into other runs
    os.environ.pop("WAN_CKPT_DIR", None)


def main() -> int:
    print("Wan2.2 service test suite")
    test_config()
    test_frame_coercion()
    test_generate_kwargs()
    test_dry_run_runner()
    test_http_dry_run()
    test_fail_fast()
    print(f"\n{'='*40}\nRESULT: {_PASS} passed, {_FAIL} failed\n{'='*40}")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
