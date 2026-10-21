"""Summarize measured decoder/model/controller JSONL without simulator claims."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]


def summarize(records: list[dict]) -> dict:
    frames = [record for record in records if record.get("type") == "frame_result"]
    if not frames:
        raise ValueError("no frame_result records found")
    model_times = [record["model_inference_seconds"] * 1000 for record in frames
                   if "model_inference_seconds" in record]
    processing_times = [record["frame_processing_seconds"] * 1000 for record in frames
                        if "frame_processing_seconds" in record]
    end_to_end_times = [record["end_to_end_frame_seconds"] * 1000 for record in frames
                        if "end_to_end_frame_seconds" in record]
    comparisons = [record["baseline_comparison"] for record in frames
                   if "baseline_comparison" in record]
    payloads = [record["tensor_storage"]["payload_bytes"] for record in frames
                if "tensor_storage" in record]
    paths = sorted({record.get("inference_path", "unknown") for record in frames})
    reuse = [record["kv_reuse"] for record in frames if "kv_reuse" in record]
    return {
        "kind": "measured_video_evaluation",
        "frames": len(frames),
        "ready_frames": sum(record.get("status") == "READY" for record in frames),
        "metadata_fallback_rate": mean(bool(record.get("metadata_fallback")) for record in frames),
        "mean_motion_vectors_per_frame": mean(record.get("motion_vectors", 0) for record in frames),
        "inference_paths": paths,
        "source_codec": frames[0].get("source_codec"),
        "source_size": frames[0].get("source_size"),
        "semantic_model": frames[0].get("semantic_model"),
        "tensor_device": frames[-1].get("tensor_storage", {}).get("device"),
        "model_frames": len(model_times),
        "mean_model_inference_ms": mean(model_times) if model_times else None,
        "p95_model_inference_ms": _percentile(model_times, 0.95) if model_times else None,
        "mean_processing_ms_excluding_decode": mean(processing_times) if processing_times else None,
        "p95_processing_ms_excluding_decode": _percentile(processing_times, 0.95) if processing_times else None,
        "mean_end_to_end_frame_ms": mean(end_to_end_times) if end_to_end_times else None,
        "p95_end_to_end_frame_ms": _percentile(end_to_end_times, 0.95) if end_to_end_times else None,
        "peak_actual_tensor_payload_bytes": max(payloads) if payloads else None,
        "final_tensor_bytes_written": frames[-1].get("tensor_storage", {}).get("bytes_written"),
        "final_tensor_bytes_read": frames[-1].get("tensor_storage", {}).get("bytes_read"),
        "mean_kv_consumed_pages": mean(record.get("kv_consumed_pages", 0) for record in frames)
                                  if "stored_kv" in paths else None,
        "compared_frames": len(comparisons),
        "baseline_mean_inference_ms": mean(item["baseline_seconds"] * 1000 for item in comparisons)
                                      if comparisons else None,
        "top1_agreement_rate": mean(bool(item["same_top1"]) for item in comparisons)
                               if comparisons else None,
        "mean_abs_logit_error": mean(item["mean_abs_logit_error"] for item in comparisons)
                                if comparisons else None,
        "kv_reuse_frames": len(reuse),
        "kv_pages_reused": sum(item["pages_reused"] for item in reuse),
        "kv_pages_recomputed": sum(item["pages_recomputed"] for item in reuse),
        "kv_tokens_reused": sum(item["tokens_kv_reused"] for item in reuse),
        "peak_reuse_validation_input_bytes": max((item["validation_input_bytes"] for item in reuse), default=0),
        "limitations": [
            "ViT-Tiny is an image classifier, not a vision-language model.",
            "Only the final ViT attention block consumes tiered K/V; optional cross-frame reuse requires exact block-input equality, and earlier blocks still execute.",
            "End-to-end frame time includes decoder wait and Python overhead but excludes model loading; baseline and stored-K/V paths are not a controlled speedup study.",
            "Top-1 agreement is not downstream task quality and patent novelty is not established by these measurements.",
        ],
    }


def evaluate_file(input_path: str, output_path: str) -> Path:
    records = []
    for line_number, line in enumerate(Path(input_path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at line {line_number}: {error.msg}") from error
    result = summarize(records)
    result["input"] = input_path
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
