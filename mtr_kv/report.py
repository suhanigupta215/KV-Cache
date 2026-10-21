"""Generate a concise Markdown report from simulator JSON output."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def generate_report(payload: dict[str, Any]) -> str:
    """Build a report from emitted raw metrics without invented claims."""
    lines = [
        "# MTR-KV Simulation Report",
        "",
        f"Scenario: `{payload['scenario']['name']}`",
        f"Seed: `{payload['scenario']['seed']}`",
        f"Configuration: `{payload['config']['name']}` v{payload['config']['version']}",
        "",
        "| Mode | Pages avoided | Bytes saved | Semantic rescues | Fallback rate | Stale drops |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, run in payload["runs"].items():
        metrics = run["metrics"]
        lines.append(
            "| {mode} | {pages} | {bytes_saved} | {rescues} | {fallback_rate:.2%} | {stale} |".format(
                mode=mode,
                pages=_format_number(metrics["pages_avoided"]),
                bytes_saved=_format_number(metrics["bytes_saved"]),
                rescues=_format_number(metrics["semantic_rescues"]),
                fallback_rate=metrics["fallback_rate"],
                stale=_format_number(metrics["stale_drops"]),
            )
        )
    lines.extend(
        [
            "",
            "All values are simulator measurements from the accompanying JSON; they are not hardware performance claims.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(input_path: str | Path, output_path: str | Path) -> Path:
    """Read a simulation artifact and write its Markdown report."""
    with Path(input_path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(generate_report(payload), encoding="utf-8")
    return destination
