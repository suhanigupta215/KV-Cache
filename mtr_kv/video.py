"""FFmpeg/PyAV adapter for real compressed-video motion-vector metadata."""
from __future__ import annotations

import json
import sys
import time
from typing import Callable, TextIO

from .config import Config
from .live import LiveController


def motion_vector_block(vector, width: int, height: int) -> dict | None:
    """Convert an AVMotionVector record to the live controller's pixel contract."""
    block_width = int(vector["w"])
    block_height = int(vector["h"])
    destination_x = int(vector["dst_x"])
    destination_y = int(vector["dst_y"])
    scale = int(vector["motion_scale"])
    if block_width <= 0 or block_height <= 0 or scale == 0:
        return None
    left = max(0, destination_x - block_width // 2)
    top = max(0, destination_y - block_height // 2)
    right = min(width, destination_x + (block_width + 1) // 2)
    bottom = min(height, destination_y + (block_height + 1) // 2)
    if right <= left or bottom <= top:
        return None
    return {
        "rect": [left, top, right - left, bottom - top],
        "mv": [float(vector["motion_x"]) / scale, float(vector["motion_y"]) / scale],
        "confidence": 1.0,
    }


def run_video(cfg: Config, source: str, sink: TextIO = sys.stdout, max_frames: int | None = None,
              semantic_model: bool = False, kv_inference: bool = False,
              compare_baseline: bool = False, device: str = "cpu",
              on_frame: Callable | None = None, should_stop: Callable | None = None,
              kv_reuse: bool = False, reuse_window: int = 8) -> None:
    """Decode a video and emit live controller results as JSONL.

    PyAV is optional because the synthetic simulator and JSONL controller
    do not require FFmpeg. Codec support depends on the installed FFmpeg.
    """
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if kv_reuse and not (semantic_model and kv_inference):
        raise ValueError("--kv-reuse requires --semantic-model and --kv-inference")
    if reuse_window <= 0:
        raise ValueError("reuse_window must be positive")
    try:
        import av
    except ImportError as error:
        raise RuntimeError("PyAV is required for video input: install it with .venv/bin/python -m pip install av") from error

    if (kv_inference or compare_baseline) and not semantic_model:
        raise ValueError("KV inference and baseline comparison require --semantic-model")
    if should_stop is not None and should_stop():
        return
    feedback = None
    if semantic_model:
        from .vision_feedback import ViTAttentionFeedback

        feedback = ViTAttentionFeedback(cfg.model.page_tokens, device=device,
                                        kv_reuse=kv_reuse, reuse_window=reuse_window)
        cfg.model.input_size = feedback.input_size
        cfg.model.patch_size = feedback.patch_size
        attention = feedback.model.blocks[-1].attn
        cfg.model.layers = 1
        cfg.model.kv_heads = attention.num_heads
        cfg.model.head_dim = attention.head_dim
        cfg.model.bytes_per_element = 2.0

    from .phone import open_video_source

    if should_stop is not None and should_stop():
        return
    with open_video_source(source, should_stop) as container:
        if not container.streams.video:
            raise ValueError("input contains no video stream")
        stream = container.streams.video[0]
        if not source.startswith("phone://"):
            stream.codec_context.options = {**stream.codec_context.options, "flags2": "+export_mvs"}
        if feedback is not None:
            cfg.model.crop = feedback.source_crop((stream.width, stream.height))
        controller = LiveController(cfg, (stream.width, stream.height))
        decoder = iter(container.decode(stream))
        generation = 0
        while True:
            if should_stop is not None and should_stop():
                break
            decode_start = time.perf_counter()
            try:
                frame = next(decoder)
            except StopIteration:
                break
            decode_seconds = time.perf_counter() - decode_start
            generation += 1
            frame_start = time.perf_counter()
            blocks = []
            motion_vectors = frame.side_data.get("MOTION_VECTORS")
            if motion_vectors is not None:
                blocks.extend(
                    block for vector in motion_vectors.to_ndarray()
                    if (block := motion_vector_block(vector, frame.width, frame.height)) is not None
                )
            frame_type = av.video.frame.PictureType(frame.pict_type).name
            result = controller.frame({
                "generation": generation,
                "frame_type": frame_type if frame_type in ("I", "P", "B") else "NONE",
                "blocks": blocks,
            })
            result["source_pts"] = frame.pts
            result["source_seconds"] = float(frame.time) if frame.time is not None else None
            result["source_codec"] = stream.codec_context.name
            result["source_kind"] = "android_usb_camera" if source.startswith("phone://") else "video"
            result["capture_dropped_frames"] = getattr(container, "dropped_frames", 0)
            result["source_size"] = [frame.width, frame.height]
            result["source_crop"] = list(cfg.model.crop)
            result["reconciliation_path"] = "tiled" if cfg.mtre.use_optimized_path else "oracle"
            result["decode_seconds"] = decode_seconds
            result["motion_vectors"] = len(blocks)
            result["inference_path"] = "stored_kv" if kv_inference else "standard" if semantic_model else "codec_only"
            result["kv_reuse_enabled"] = kv_reuse
            if feedback is not None and result["status"] == "READY":
                scores = feedback.page_scores(
                    frame.to_ndarray(format="rgb24"),
                    controller=controller if kv_inference else None,
                    generation=generation if kv_inference else None,
                    compare_baseline=compare_baseline,
                )
                result["semantic_feedback"] = controller.semantic({
                    "generation": generation, "page_scores": scores,
                })
                result["semantic_model"] = feedback.MODEL_NAME
                result["model_inference_seconds"] = feedback.last_inference_seconds
                if feedback.reuse_stats is not None:
                    result["kv_reuse"] = feedback.reuse_stats
                if kv_inference:
                    result["kv_consumed_pages"] = feedback.kv_consumed_pages
                    controller.tensor_store.sync_tiers(generation)
                else:
                    for page, (key, value) in feedback.page_kv().items():
                        controller.attach_kv(page, generation, key, value, device=device)
                if feedback.baseline_comparison is not None:
                    result["baseline_comparison"] = feedback.baseline_comparison
                result["tensor_storage"] = controller.tensor_store.snapshot()
            result["frame_processing_seconds"] = time.perf_counter() - frame_start
            result["end_to_end_frame_seconds"] = decode_seconds + result["frame_processing_seconds"]
            result["memory"] = controller.memory.snapshot()
            result["pages"] = [
                {"page_id": page, "tier": entry.tier.name if entry.tier is not None else None,
                 "state": controller.states[page].value,
                 "activity": controller.page_activity[page], "ready": entry.ready}
                for page, entry in controller.memory.pages.items()
            ]
            if on_frame is not None:
                on_frame(result, frame)
            sink.write(json.dumps(result) + "\n")
            sink.flush()
            if max_frames is not None and generation >= max_frames:
                break
