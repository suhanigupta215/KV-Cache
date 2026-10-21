"""Paired oracle/index measurements with identical preprocessing and storage."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
from statistics import median
import sys

from mtr_kv.config import load_config
from mtr_kv.evaluate import summarize
from mtr_kv.video import run_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/demo.mp4")
    parser.add_argument("--output", default="results/optimization-20261021/paired")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(args.input)
    manifest = {"kind": "paired_reconciliation_optimization", "input": str(source.resolve()),
                "input_sha256": sha256(source.read_bytes()).hexdigest(),
                "frames_per_run": 60, "warmup_frames": 6, "repeats": 3,
                "python": sys.version, "platform": platform.platform(), "runs": []}
    for repeat in range(3):
        order = (False, True) if repeat % 2 == 0 else (True, False)
        paired = {}
        for indexed in order:
            arm = "indexed" if indexed else "oracle"
            path = output / f"repeat-{repeat+1}-{arm}.jsonl"
            cfg = load_config("configs/reference.json")
            cfg.mtre.use_optimized_path = indexed
            with path.open("w") as sink:
                run_video(cfg, str(source), sink, max_frames=60, semantic_model=True,
                          kv_inference=True, kv_reuse=True, compare_baseline=True)
            frames = [json.loads(line) for line in path.read_text().splitlines()]
            decisions = [{key: frame[key] for key in ("generation", "status", "metadata_fallback",
                          "transition_predicted", "descriptors", "apply_statuses", "ready_tokens",
                          "pages", "source_crop", "memory", "semantic_feedback")} for frame in frames]
            digest = sha256(json.dumps(decisions, sort_keys=True).encode()).hexdigest()
            paired[arm] = digest
            manifest["runs"].append({"repeat": repeat+1, "arm": arm, "trace": path.name,
                                     "sha256": sha256(path.read_bytes()).hexdigest(),
                                     "decision_sha256": digest, "config": asdict(cfg),
                                     "summary": summarize(frames[6:])})
        if paired["oracle"] != paired["indexed"]:
            raise RuntimeError("Oracle and indexed controller decisions differ")
    manifest["decisions_identical"] = True
    manifest["median_end_to_end_ms"] = {
        arm: median(run["summary"]["mean_end_to_end_frame_ms"] for run in manifest["runs"] if run["arm"] == arm)
        for arm in ("oracle", "indexed")}
    times = manifest["median_end_to_end_ms"]
    manifest["observed_pipeline_ratio"] = times["oracle"] / times["indexed"]
    manifest["interpretation"] = "Generated sample, CPU software only, paired final-code paths; includes baseline comparison, excludes model loading and dashboard. Not a standard-model, VLM, GPU, or production speedup claim."
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"median_ms": times, "ratio": manifest["observed_pipeline_ratio"],
                      "decisions_identical": True}))


if __name__ == "__main__":
    main()
