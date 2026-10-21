"""Command-line entry point for deterministic MTR-KV experiments."""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path
from typing import Any

from .config import load_config
from .report import write_report
from .scenarios_lib import REFERENCE_SCENARIOS
from .simulator import Simulator


def _scenario(name: str):
    try:
        return REFERENCE_SCENARIOS[name]()
    except KeyError as error:
        choices = ", ".join(sorted(REFERENCE_SCENARIOS))
        raise ValueError(f"unknown scenario '{name}'; choose one of: {choices}") from error


def _run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    scenario = _scenario(args.scenario)
    modes = ("baseline", "codec_only", "full") if args.mode == "all" else (args.mode,)
    runs: dict[str, Any] = {}
    for mode in modes:
        simulator = Simulator(cfg, scenario, mode=mode)
        metrics = simulator.run_baseline() if mode == "baseline" else simulator.run_mtr()
        runs[mode] = {
            "metrics": metrics.to_dict(),
            "frames": [frame.to_dict() for frame in simulator.frame_records],
            "event_trace": simulator.events.export_trace(),
        }
    payload = {
        "config": cfg.to_dict(),
        "scenario": dataclasses.asdict(scenario),
        "runs": runs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["mode", "stream_id", "generation", "frame_index", "baseline_pages", "mtr_pages_t0", "mtr_pages_t1", "mtr_pages_t2", "mtr_pages_t3", "pages_avoided", "fallback", "semantic_rescues", "conflicts", "queue_peak", "stale_drops", "transition_predicted", "scene_cut_truth", "tier_switches"]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for mode, run in runs.items():
                for frame in run["frames"]:
                    writer.writerow({"mode": mode, **frame})
    print(output)
    return 0


def _report(args: argparse.Namespace) -> int:
    print(write_report(args.input, args.output))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="MTR-KV deterministic simulator")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a reference scenario")
    run.add_argument("--scenario", default="T01", choices=sorted(REFERENCE_SCENARIOS))
    run.add_argument("--mode", default="all", choices=("all", "baseline", "codec_only", "full", "semantic_only"))
    run.add_argument("--config", default="configs/reference.json")
    run.add_argument("--output", default="results/simulation.json")
    run.add_argument("--csv", default="results/frames.csv")
    run.set_defaults(handler=_run)
    report = commands.add_parser("report", help="render a Markdown report from simulation JSON")
    report.add_argument("--input", default="results/simulation.json")
    report.add_argument("--output", default="results/report.md")
    report.set_defaults(handler=_report)
    live = commands.add_parser("live", help="process live decoder metadata and semantic JSONL from stdin")
    live.add_argument("--config", default="configs/reference.json")
    live.add_argument("--width", type=int, required=True, help="source video width in pixels")
    live.add_argument("--height", type=int, required=True, help="source video height in pixels")
    live.set_defaults(handler=_live)
    video = commands.add_parser("video", help="decode a compressed video with PyAV and stream codec motion decisions")
    video.add_argument("input", help="video file or live URL accepted by FFmpeg")
    video.add_argument("--config", default="configs/reference.json")
    video.add_argument("--max-frames", type=int)
    video.add_argument("--semantic-model", action="store_true", help="run pretrained ViT attention feedback on decoded frames")
    video.add_argument("--kv-inference", action="store_true", help="consume controller-stored K/V during the ViT's last attention block")
    video.add_argument("--compare-baseline", action="store_true", help="measure output deviation from standard ViT inference")
    video.add_argument("--device", default="cpu", help="torch device for model and T0 tensors (default: cpu)")
    video.set_defaults(handler=_video)
    dashboard = commands.add_parser("dashboard", help="serve a local dashboard streaming real video decisions")
    dashboard.add_argument("input", help="video file or live URL accepted by FFmpeg")
    dashboard.add_argument("--config", default="configs/reference.json")
    dashboard.add_argument("--max-frames", type=int)
    dashboard.add_argument("--semantic-model", action="store_true")
    dashboard.add_argument("--kv-inference", action="store_true")
    dashboard.add_argument("--compare-baseline", action="store_true")
    dashboard.add_argument("--device", default="cpu")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8765)
    dashboard.set_defaults(handler=_dashboard)
    dashboard.add_argument("--unpaced", action="store_true", help="process as fast as possible instead of pacing to video timestamps")
    demo = commands.add_parser("demo", help="generate a compressed sample video and launch the live dashboard")
    demo.add_argument("--output", default="results/demo.mp4")
    demo.add_argument("--generate-only", action="store_true")
    demo.add_argument("--semantic-model", action="store_true")
    demo.add_argument("--kv-inference", action="store_true")
    demo.add_argument("--compare-baseline", action="store_true")
    demo.add_argument("--device", default="cpu")
    demo.add_argument("--port", type=int, default=8765)
    demo.set_defaults(handler=_demo)
    phone = commands.add_parser("phone", help="use an Android 12+ phone camera over USB (scrcpy + adb)")
    phone.add_argument("--list-devices", action="store_true", help="show connected Android devices and authorization status")
    phone.add_argument("--serial", help="select one connected device by adb serial")
    phone.add_argument("--camera-facing", default="back", choices=("back", "front", "external"))
    phone.add_argument("--size", type=int, default=640, help="maximum camera dimension (default: 640)")
    phone.add_argument("--fps", type=int, default=15, help="requested camera capture rate (default: 15)")
    phone.add_argument("--max-frames", type=int)
    phone.add_argument("--config", default="configs/reference.json")
    phone.add_argument("--semantic-model", action="store_true")
    phone.add_argument("--kv-inference", action="store_true")
    phone.add_argument("--compare-baseline", action="store_true")
    phone.add_argument("--device", default="cpu")
    phone.add_argument("--port", type=int, default=8767)
    phone.set_defaults(handler=_phone)
    evaluate = commands.add_parser("evaluate", help="summarize measured video JSONL")
    evaluate.add_argument("input", help="JSONL emitted by mtr_kv video")
    evaluate.add_argument("--output", default="results/live-evaluation.json")
    evaluate.set_defaults(handler=_evaluate)
    evidence = commands.add_parser("evidence", help="capture repeated paired measurements from a real local video")
    evidence.add_argument("input", help="existing local video file")
    evidence.add_argument("--output-dir", required=True, help="new directory for immutable run artifacts")
    evidence.add_argument("--config", default="configs/reference.json")
    evidence.add_argument("--repeats", type=int, default=3)
    evidence.add_argument("--max-frames", type=int, default=100)
    evidence.add_argument("--warmup-frames", type=int, default=5)
    evidence.add_argument("--device", default="cpu")
    evidence.set_defaults(handler=_evidence)
    for command in (video, dashboard, demo, phone, evidence):
        command.add_argument("--kv-reuse", action="store_true", help="reuse unchanged static-page K/V across frames (requires model-backed KV inference)")
        command.add_argument("--reuse-window", type=int, default=8, help="maximum age in processed generations before refreshing reused K/V (default: 8)")
    args = parser.parse_args()
    if getattr(args, "kv_reuse", False) and args.command != "evidence" and not (
        getattr(args, "semantic_model", False) and getattr(args, "kv_inference", False)
    ):
        parser.error("--kv-reuse requires --semantic-model and --kv-inference")
    if getattr(args, "reuse_window", 1) <= 0:
        parser.error("--reuse-window must be positive")
    return args.handler(args)


def _live(args: argparse.Namespace) -> int:
    from .live import run_stream

    try:
        run_stream(load_config(args.config), (args.width, args.height))
    except ValueError as error:
        raise SystemExit(str(error)) from error
    return 0


def _video(args: argparse.Namespace) -> int:
    from .video import run_video

    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")
    try:
        run_video(load_config(args.config), args.input, max_frames=args.max_frames,
                  semantic_model=args.semantic_model, kv_inference=args.kv_inference,
                  compare_baseline=args.compare_baseline, device=args.device,
                  kv_reuse=args.kv_reuse, reuse_window=args.reuse_window)
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


def _dashboard(args: argparse.Namespace) -> int:
    from .dashboard import create_server

    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")
    try:
        server = create_server(args.input, args.host, args.port, args.config, args.max_frames,
                               args.semantic_model, args.kv_inference, args.compare_baseline, args.device,
                               realtime=not args.unpaced, kv_reuse=args.kv_reuse, reuse_window=args.reuse_window)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"MTR-KV dashboard: http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _demo(args: argparse.Namespace) -> int:
    from .demo import create_demo_video

    if (args.kv_inference or args.compare_baseline) and not args.semantic_model:
        raise SystemExit("KV inference and baseline comparison require --semantic-model")
    try:
        path = Path(args.output)
        if not path.exists():
            create_demo_video(str(path))
    except (ImportError, OSError, ValueError) as error:
        raise SystemExit(f"Cannot generate demo video: {error}") from error
    print(f"Demo video: {path.resolve()}", flush=True)
    if args.generate_only:
        return 0
    args.input = str(path)
    args.host, args.config, args.max_frames, args.unpaced = "127.0.0.1", "configs/reference.json", None, False
    return _dashboard(args)


def _phone(args: argparse.Namespace) -> int:
    from .phone import PhoneOptions, connected_devices, require_tool
    import subprocess

    try:
        if args.list_devices:
            devices = connected_devices()
            print(json.dumps(devices, indent=2))
            if not devices:
                print("No Android device detected. Connect a data-capable USB cable, enable USB debugging, and accept the phone's authorization prompt.")
            return 0
        options = PhoneOptions(args.serial, args.camera_facing, args.size, args.fps)
        require_tool("adb")
        require_tool("scrcpy")
        args.input = options.uri()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error)) from error
    args.host, args.unpaced = "127.0.0.1", True
    return _dashboard(args)


def _evaluate(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_file

    try:
        print(evaluate_file(args.input, args.output))
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


def _evidence(args: argparse.Namespace) -> int:
    from .evidence import capture

    try:
        print(capture(args.input, args.output_dir, args.config, args.repeats,
                      args.max_frames, args.warmup_frames, args.device,
                      kv_reuse=args.kv_reuse, reuse_window=args.reuse_window))
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
