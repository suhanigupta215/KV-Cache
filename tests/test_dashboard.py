"""Static contract tests for the local MTR-KV presentation dashboard."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_shell_exposes_required_controls():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    for identifier in (
        "mode-select",
        "artifact-input",
        "load-status",
        "metric-grid",
        "residency-chart",
        "frame-table-body",
        "frame-explanation",
    ):
        assert identifier in html
    assert "renderDashboard" in script


def test_dashboard_script_defines_artifact_boundary():
    script = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    for function_name in (
        "normalizeArtifact",
        "availableModes",
        "renderDashboard",
        "loadArtifactFile",
    ):
        assert f"function {function_name}" in script


def test_dashboard_handles_loading_and_empty_frames():
    script = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    assert "Invalid result artifact" in script
    assert "No frame records are available" in script
    assert 'addEventListener("change"' in script


def test_dashboard_renders_loaded_metric_values_as_text():
    script = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    assert "metricGrid.replaceChildren" in script


def test_generated_result_artifact_has_dashboard_shape():
    payload = json.loads((ROOT / "results" / "simulation.json").read_text(encoding="utf-8"))
    assert payload["scenario"]["name"]
    assert {"baseline", "codec_only", "full"} <= set(payload["runs"])
    assert "metrics" in payload["runs"]["full"]
