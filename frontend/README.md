# MTR-KV Presentation Dashboard

For the complete synchronized video demonstration, run `.venv/bin/python -m mtr_kv demo` from the project root and open http://127.0.0.1:8765. Watch the decoded source beside the committed page tiers. Stop, restart, and save the measured trace using the live controls. See [the walkthrough](../docs/DEMO.md).

Add `--semantic-model --kv-inference --kv-reuse --compare-baseline` to observe validated cross-frame K/V reuse. The live table reports actual page reuses; see [the reuse contract](../docs/KV_REUSE.md) for invalidation and measurement limits.

Generate a simulator artifact first:

```bash
cd /Users/shaurya/MTR-KV
.venv/bin/python -m mtr_kv run --scenario T03 --mode all
```

Open `frontend/index.html` directly in a browser, select **Load results**, and choose `results/simulation.json`. The dashboard runs entirely in the browser and does not send the artifact anywhere.

For an automatically updating trace from a real video source, run from the project root:

```bash
.venv/bin/python -m pip install -r requirements-live.txt
.venv/bin/python -m mtr_kv dashboard path/to/input.mp4 --semantic-model --kv-inference --compare-baseline
```

Then open `http://127.0.0.1:8765`. This is a local read-only server; the live cards use measured decoder/controller/model records, while the existing strategy comparison remains a simulator view. The classifier top-1 check is not a VLM quality benchmark.
