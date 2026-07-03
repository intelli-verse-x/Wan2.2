"""Wan2.2 queued local inference service.

Wraps the Wan2.2 generation pipelines (`WanT2V`, `WanI2V`, `WanTI2V`) behind a
small FastAPI + job-queue server so they can be driven over HTTP exactly like a
hosted provider. The HTTP contract matches what content-factory's
``Wan22VideoGenerator`` client already expects:

    POST /generate          -> {"task_id": "..."}
    GET  /status/{task_id}  -> {"status": "...", "video_url": ..., "duration": ...}
    POST /upload            -> {"url": "..."}
    GET  /health            -> 200

A single GPU processes one job at a time; the queue absorbs bursts.
"""
