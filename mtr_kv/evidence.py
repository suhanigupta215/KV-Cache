"""Reproducible, paired real-video evidence capture for technical review."""
from __future__ import annotations

from hashlib import sha256
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
from statistics import median
import subprocess
import sys

from .config import load_config
from .evaluate import summarize
from .video import run_video


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision() -> dict:
    try:
        root = Path(__file__).resolve().parents[1]
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
        return {"head": head, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "dirty": None}


def _runtime() -> dict:
    import av
    import timm
    import torch

    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "cpu_count": os.cpu_count(), "pyav": av.__version__,
        "torch": torch.__version__, "timm": timm.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }


def _records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def render_report(manifest: dict) -> str:
    """Human-readable audit view generated only from the manifest."""
    lines = ["# MTR-KV measured-run review", "",
             f"Status: **{manifest.get('status', 'incomplete')}**", "",
             "This is exploratory ViT classifier evidence, not a patentability opinion, VLM result, or hardware speedup claim.",
             ""]
    if manifest.get("error"):
        lines.extend([f"Capture error: `{manifest['error']}`", ""])
    source = manifest.get("source", {})
    config = manifest.get("config", {})
    lines.extend(["## Provenance", "",
                  f"- Input SHA-256: `{source.get('sha256', 'unavailable')}`",
                  f"- Config SHA-256: `{config.get('sha256', 'unavailable')}`",
                  f"- Git HEAD: `{manifest.get('git', {}).get('head')}`; dirty: `{manifest.get('git', {}).get('dirty')}`",
                  f"- Device: `{manifest.get('device')}`; repeats: `{manifest.get('repeats')}`; warm-up frames excluded: `{manifest.get('warmup_frames')}`",
                  f"- Cross-frame reuse configuration: `{manifest.get('kv_reuse', {'enabled': False})}`",
                  "", "## Per-run measurements", "",
                  "| Repeat | Arm | Frames | Mean model ms | P95 model ms | Mean end-to-end ms | Peak tensor bytes | KV page reuses | Validation bytes | Fallback rate | Raw trace |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
    for run in manifest.get("runs", []):
        summary = run["summary"]
        lines.append("| {repeat} | {arm} | {frames} | {model} | {p95} | {total} | {bytes} | {reuses} | {validation} | {fallback} | `{trace}` |".format(
            repeat=run["repeat"], arm=run["arm"], frames=summary["frames"],
            model=summary.get("mean_model_inference_ms"), p95=summary.get("p95_model_inference_ms"),
            total=summary.get("mean_end_to_end_frame_ms"),
            bytes=summary.get("peak_actual_tensor_payload_bytes"),
            reuses=summary.get("kv_pages_reused", 0), validation=summary.get("peak_reuse_validation_input_bytes", 0),
            fallback=summary.get("metadata_fallback_rate"), trace=run["trace"]))
    quality_run = manifest.get("quality_run")
    if quality_run:
        quality = quality_run["summary"]
        lines.extend(["", "## Paired classifier check", "",
                      f"- Compared frames: {quality['compared_frames']}",
                      f"- Top-1 agreement: {quality['top1_agreement_rate']}",
                      f"- Mean absolute logit error: {quality['mean_abs_logit_error']}",
                      "- This does not measure downstream VLM task quality."])
    lines.extend(["", "## Interpretation boundary", "",
                  "Model-loading time is excluded. Latency includes framework/device effects and is not a controlled production speedup claim. "
                  "Inspect every raw JSONL file and its hash before quoting a number. Representative data and patent-agent review remain necessary.", ""])
    return "\n".join(lines)


def capture(source: str, output_dir: str, config_path: str = "configs/reference.json",
            repeats: int = 3, max_frames: int = 100, warmup_frames: int = 5,
            device: str = "cpu", kv_reuse: bool = False, reuse_window: int = 8) -> Path:
    """Capture alternating standard/tiered runs plus a paired quality run.

    Refuses to overwrite an existing directory. Only local regular files are
    accepted so that the input can be hashed and replayed exactly.
    """
    video = Path(source).resolve()
    config = Path(config_path).resolve()
    output = Path(output_dir).resolve()
    if not video.is_file() or not config.is_file():
        raise ValueError("evidence capture requires existing local video and config files")
    if repeats < 1 or max_frames < 1 or warmup_frames < 0 or warmup_frames >= max_frames:
        raise ValueError("repeats/max_frames must be positive and warmup_frames < max_frames")
    if reuse_window <= 0:
        raise ValueError("reuse_window must be positive")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "kind": "mtr_kv_exploratory_evidence_v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(video)}, "config": {"path": str(config)},
        "git": _revision(), "device": device,
        "repeats": repeats, "max_frames": max_frames, "warmup_frames": warmup_frames,
        "kv_reuse": {"enabled": kv_reuse, "window": reuse_window},
        "runs": [], "quality_run": None,
        "interpretation": "Exploratory classifier/CPU-or-device evidence, not VLM, GPU-controller, patentability, or production-throughput proof.",
    }
    try:
        saved_config = output / "config.json"
        shutil.copy2(config, saved_config)
        manifest["source"].update({"sha256": _hash_file(video), "bytes": video.stat().st_size})
        manifest["config"].update({"copy": saved_config.name, "sha256": _hash_file(saved_config)})
        manifest["runtime"] = _runtime()
        for repeat in range(repeats):
            order = ("standard", "tiered") if repeat % 2 == 0 else ("tiered", "standard")
            for arm in order:
                path = output / f"repeat-{repeat + 1:02d}-{arm}.jsonl"
                with path.open("w", encoding="utf-8") as sink:
                    run_video(load_config(str(saved_config)), str(video), sink, max_frames=max_frames,
                              semantic_model=True, kv_inference=arm == "tiered", device=device,
                              **({"kv_reuse": True, "reuse_window": reuse_window} if kv_reuse and arm == "tiered" else {}))
                records = _records(path)
                frames = [record for record in records if record.get("type") == "frame_result"]
                if len(frames) <= warmup_frames or any(frame.get("status") != "READY" for frame in frames):
                    raise RuntimeError(f"incomplete or non-ready frame stream in {path.name}")
                summary = summarize(frames[warmup_frames:])
                manifest["runs"].append({"repeat": repeat + 1, "arm": arm, "order": list(order),
                                         "trace": path.name, "trace_sha256": _hash_file(path),
                                         "summary": summary})
            pair = manifest["runs"][-2:]
            counts = {item["summary"]["frames"] for item in pair}
            if len(counts) != 1:
                raise RuntimeError(f"arm frame counts differ in repeat {repeat + 1}")
            timelines = []
            for item in pair:
                frames = [record for record in _records(output / item["trace"])
                          if record.get("type") == "frame_result"]
                timelines.append([(frame["generation"], frame.get("source_pts")) for frame in frames])
            if timelines[0] != timelines[1]:
                raise RuntimeError(f"arm frame timelines differ in repeat {repeat + 1}")

        quality_path = output / "quality-paired.jsonl"
        with quality_path.open("w", encoding="utf-8") as sink:
            run_video(load_config(str(saved_config)), str(video), sink, max_frames=max_frames,
                      semantic_model=True, kv_inference=True, compare_baseline=True, device=device,
                      **({"kv_reuse": True, "reuse_window": reuse_window} if kv_reuse else {}))
        quality_frames = [record for record in _records(quality_path) if record.get("type") == "frame_result"]
        if len(quality_frames) <= warmup_frames or any(frame.get("status") != "READY" for frame in quality_frames):
            raise RuntimeError("incomplete or non-ready quality stream")
        reference_frames = [record for record in _records(output / manifest["runs"][0]["trace"])
                            if record.get("type") == "frame_result"]
        if [(frame["generation"], frame.get("source_pts")) for frame in quality_frames] != [
            (frame["generation"], frame.get("source_pts")) for frame in reference_frames
        ]:
            raise RuntimeError("quality-run frame timeline differs from the comparison arms")
        quality = summarize(quality_frames[warmup_frames:])
        if quality["compared_frames"] != quality["frames"]:
            raise RuntimeError("paired quality comparison missing for one or more frames")
        if _hash_file(video) != manifest["source"]["sha256"]:
            raise RuntimeError("source video changed during evidence capture")
        manifest["quality_run"] = {"trace": quality_path.name, "trace_sha256": _hash_file(quality_path),
                                   "summary": quality}
        manifest["median_model_ms_by_arm"] = {
            arm: median(item["summary"]["mean_model_inference_ms"] for item in manifest["runs"]
                        if item["arm"] == arm)
            for arm in ("standard", "tiered")
        }
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "incomplete"
        manifest["error"] = str(error)
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8")
        (output / "report.md").write_text(render_report(manifest), encoding="utf-8")
    return output / "manifest.json"
