# Wan2.2 local video service.
# Builds on an env that already has the Wan2.2 ML stack installed. Mount the
# checkpoints at runtime (-v /host/models:/models) and set WAN_CKPT_DIR.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# System libs OpenCV needs at import time (libGL.so.1, libgthread).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

# Repo-root deps (torch is already in the base image) + service deps.
# flash_attn needs nvcc, which the -runtime base lacks; it is OPTIONAL —
# wan/modules/attention.py falls back to torch scaled_dot_product_attention.
# Install everything EXCEPT flash_attn first (so a flash_attn build failure can't
# abort the core deps like diffusers/transformers), then try flash_attn best-effort.
RUN grep -viE '^[[:space:]]*flash[_-]attn' requirements.txt > /tmp/reqs.core.txt && \
    pip install --no-cache-dir -r /tmp/reqs.core.txt
# Deps imported by wan/ but absent from the repo requirements.txt
# (einops/peft: model code; decord/librosa: s2v video/audio loaders).
RUN pip install --no-cache-dir einops peft decord librosa
RUN pip install --no-cache-dir flash_attn || \
    echo "flash_attn unavailable (no nvcc in runtime base) — using torch SDPA fallback"
RUN pip install --no-cache-dir -r service/requirements.txt

ENV WAN_TASK=ti2v-5B \
    WAN_HOST=0.0.0.0 \
    WAN_PORT=8000 \
    WAN_OUTPUT_DIR=/data/outputs

EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
