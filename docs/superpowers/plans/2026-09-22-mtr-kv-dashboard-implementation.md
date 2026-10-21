# MTR-KV Presentation Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dependency-free local dashboard that explains MTR-KV and visualizes generated simulator results.

**Architecture:** The frontend is a self-contained static application under `frontend/`. It renders a built-in T03 example on load and accepts the existing `results/simulation.json` artifact through a browser file picker. Its JavaScript validates and normalizes the artifact before updating the pipeline narrative, metrics, chart, and frame timeline.

**Tech Stack:** Semantic HTML, CSS custom properties, vanilla JavaScript, Python `pytest` static-file contract tests.

**Spec:** `docs/superpowers/specs/2026-09-22-mtr-kv-dashboard-design.md`

## Global Constraints

- Do not add a framework, package manager, CDN, network request, or backend server.
- Use only the JSON contract emitted by `python -m mtr_kv run`.
- Preserve the simulator as the sole source of metrics and do not make hardware-performance claims.
- Support keyboard interaction, visible focus states, reduced-motion preferences, and narrow displays.
- Keep the dashboard runnable by opening `frontend/index.html` directly in a browser.

## Review Focus

- Invalid or malformed JSON must show an actionable error and retain the preceding valid dashboard state.
- A JSON artifact with only one available mode must select and render that mode safely.
- Missing optional metric fields must render a clear placeholder rather than `undefined` or `NaN`.
- A frame list with no rows must show an empty-state message without breaking the explanation panel.
- Narrow viewports and keyboard-only navigation must retain readable content and usable controls.

---

### Task 1: Define the static dashboard shell and regression contract

**Files:**
- Create: `frontend/index.html`
- Create: `frontend/styles.css`
- Create: `frontend/app.js`
- Create: `frontend/README.md`
- Create: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: no application data during this task.
- Produces: stable DOM IDs `mode-select`, `artifact-input`, `load-status`, `metric-grid`, `residency-chart`, `frame-table-body`, and `frame-explanation` for the rendering code.

- [ ] **Step 1: Write the failing static-contract test**

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -q`

Expected: FAIL because the frontend files do not exist.

- [ ] **Step 3: Create the accessible HTML shell and visual system**

Create the files with a header, hero, architecture pipeline, metrics section, chart section, timeline table, explanation panel, methodology notice, file input, and reset button. Use semantic `header`, `main`, `section`, `nav`, `table`, and `aside` elements. Add CSS tokens for charcoal, slate, cyan, violet, warning, and success states; render a responsive grid that collapses to one column below 760px.

- [ ] **Step 4: Add usage guidance**

Document that the user generates an artifact with:

```bash
.venv/bin/python -m mtr_kv run --scenario T03 --mode all
```

Then opens `frontend/index.html` and selects `results/simulation.json` through the **Load results** control.

- [ ] **Step 5: Run the static-contract test**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -q`

Expected: PASS.

### Task 2: Implement artifact normalization and mode-driven metric rendering

**Files:**
- Modify: `frontend/app.js`
- Modify: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: artifact object `{config, scenario, runs}` where every run has `metrics` and optionally `frames`.
- Produces: `normalizeArtifact(payload)`, `availableModes(data)`, `metricValue(value, suffix)`, and `renderDashboard(data, mode)`.

- [ ] **Step 1: Extend the failing contract test for data functions**

```python
def test_dashboard_script_defines_artifact_boundary():
    script = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    for function_name in (
        "normalizeArtifact",
        "availableModes",
        "renderDashboard",
        "loadArtifactFile",
    ):
        assert f"function {function_name}" in script
```

- [ ] **Step 2: Run the dashboard tests to verify the new assertion fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -q`

Expected: FAIL because the artifact-boundary functions are not present.

- [ ] **Step 3: Implement the normalizer and example data**

Embed a compact deterministic example containing `baseline`, `codec_only`, and `full` metrics plus a small frame trace. `normalizeArtifact` must throw a descriptive error if `scenario` is absent, `runs` is absent, or no run exposes an object-valued `metrics` field. It must leave frame arrays empty when absent and choose the first available mode when the requested mode is unavailable.

- [ ] **Step 4: Render the metric workspace**

Render pages avoided, bytes saved, semantic rescues, stale drops, fallback rate, and mode description. Use `—` for absent values. Populate the scenario name, seed, config label, and mode select from normalized data. The selected mode must be shown using `aria-current`-equivalent state on its descriptive text.

- [ ] **Step 5: Run the dashboard tests**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -q`

Expected: PASS.

### Task 3: Add residency visualization, timeline interaction, and resilient file loading

**Files:**
- Modify: `frontend/index.html`
- Modify: `frontend/styles.css`
- Modify: `frontend/app.js`
- Modify: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: normalized selected run `{metrics, frames}`.
- Produces: `renderResidencyChart(metrics)`, `renderFrameTimeline(frames)`, `selectFrame(frame)`, and `loadArtifactFile(event)`.

- [ ] **Step 1: Add a failing resilience contract test**

```python
def test_dashboard_handles_loading_and_empty_frames():
    script = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    assert "Invalid result artifact" in script
    assert "No frame records are available" in script
    assert "addEventListener(\"change" in script
```

- [ ] **Step 2: Run the dashboard tests to verify the new assertion fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -q`

Expected: FAIL because the resilience messages and file handler are not present.

- [ ] **Step 3: Implement chart and timeline rendering**

Render a CSS bar chart using baseline pages, hot pages, and resident pages. Scale bar widths by the largest available page count and include text values so color is not the only signal. Render table rows from frame records with buttons or keyboard-selectable rows. Show the selected frame’s generation, tier allocation, pages avoided, rescue status, fallback status, and plain-language explanation in the side panel. Display `No frame records are available` for an empty trace.

- [ ] **Step 4: Implement file loading and reset behavior**

Use `FileReader` to parse the chosen JSON locally. On success normalize and render it, update `load-status`, and retain it as the active artifact. On a read, parse, or schema failure, display `Invalid result artifact: <reason>` with `role="alert"`, retain existing visual data, and do not throw uncaught errors. Reset must restore the built-in example and clear the input.

- [ ] **Step 5: Run the dashboard tests**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -q`

Expected: PASS.

### Task 4: Verify the dashboard against actual simulator output

**Files:**
- Modify: `frontend/README.md`
- Modify: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `results/simulation.json` emitted by the existing CLI.
- Produces: documented manual verification procedure and a Python test that validates the sample artifact structure.

- [ ] **Step 1: Add an artifact-shape test**

```python
import json


def test_generated_result_artifact_has_dashboard_shape():
    payload = json.loads((ROOT / "results" / "simulation.json").read_text(encoding="utf-8"))
    assert payload["scenario"]["name"]
    assert {"baseline", "codec_only", "full"} <= set(payload["runs"])
    assert "metrics" in payload["runs"]["full"]
```

- [ ] **Step 2: Run the test to verify it passes with the generated artifact**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py::test_generated_result_artifact_has_dashboard_shape -q`

Expected: PASS.

- [ ] **Step 3: Add manual visual QA instructions**

Document these checks: open the file directly; verify the built-in full mode; load `results/simulation.json`; change all three modes; select a frame; load an invalid JSON file; use keyboard tab navigation; inspect at desktop and 760px widths.

- [ ] **Step 4: Run full verification**

Run: `.venv/bin/python -m pytest -q && .venv/bin/python -m compileall -q mtr_kv`

Expected: all Python regression tests pass and compilation exits successfully.
