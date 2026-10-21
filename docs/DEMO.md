# MTR-KV demo walkthrough

## Start and present

Run these commands from `/Users/shaurya/MTR-KV`:

```bash
.venv/bin/python -m pip install -r requirements-live.txt
.venv/bin/python -m mtr_kv demo
```

Open http://127.0.0.1:8765 and click **Restart run**. Keep the decoded frame and page tiles visible together.

1. **Still content (0–4.5 s):** the first intra frame has no inter-frame vectors and takes conservative fallback. Subsequent reliable low-motion frames accumulate temporal history before allowing colder placement.
2. **Motion (4.5–9 s):** the cyan object moves. Watch the actual vector count and the controller's page states; precise tier changes depend on page aggregation and motion thresholds.
3. **Still again (9–13.5 s):** residency responds to temporal persistence, rather than changing immediately on every small motion variation.
4. **Scene cut (13.5–18 s):** the background changes. Intra frames or missing vectors can produce conservative fallback, followed by normal temporal adaptation.

Page tiles show committed tiers after semantic correction, if enabled. Their tooltips include activity, temporal state, and readiness. They are logical pages in ID order; a page can contain tokens from multiple image rows. The preview is the same decoded generation as the measurements. It has no audio.

**Stop** disconnects this run; **Restart run** starts a fresh controller. **Save trace** downloads image-free JSONL that can be evaluated with the CLI. For long feeds it retains the latest 10,000 frames; use the CLI redirected to a file for a complete capture. The generated source video can also be played independently from `results/demo.mp4`. For a presentation recording, screen-record the dashboard while replaying it.

## Use your own input

```bash
.venv/bin/python -m mtr_kv dashboard /absolute/path/to/input.mp4
.venv/bin/python -m mtr_kv dashboard 'rtsp://camera/encoded-stream'
```

The source is chosen on the command line. The server reads locally; it does not upload the video. Use a codec/decoder that exports motion-vector side data. Unsupported metadata takes the explicitly shown conservative fallback path. Android 12+ USB camera input is available through `python -m mtr_kv phone`; see [phone setup](PHONE_CAMERA.md). Generic USB webcam capture and browser file upload are future adapters. Applications with their own decoder can continuously write frame and semantic records into `python -m mtr_kv live --width WIDTH --height HEIGHT` on stdin; stdout flushes a corresponding decision per record.

## Enable the measured model path

```bash
.venv/bin/python -m mtr_kv demo --semantic-model --kv-inference --compare-baseline --port 8766
```

Open http://127.0.0.1:8766. ViT-Tiny produces attention scores and K/V tensors; its final attention block consumes tiered K/V. T0 uses the selected torch device, T1 uses CPU int8, and T2 uses CPU compressed fp16. The table reports actual payload bytes, model time, and top-1 agreement. The warmup/model load may take time. Classification agreement can differ on some inputs because reduced precision is part of the tier representation.

## Capture measurements

```bash
.venv/bin/python -m mtr_kv video results/demo.mp4 > results/demo-trace.jsonl
.venv/bin/python -m mtr_kv evaluate results/demo-trace.jsonl --output results/demo-evaluation.json
.venv/bin/python -m mtr_kv video results/demo.mp4 --semantic-model --kv-inference --compare-baseline --max-frames 12 > results/demo-model-trace.jsonl
.venv/bin/python -m mtr_kv evaluate results/demo-model-trace.jsonl --output results/demo-model-evaluation.json
```

Processing rate reflects decoder plus controller/model time for that frame; it excludes preview encoding, timestamp pacing, browser rendering, and model loading. Dashboard completion time includes pacing and initialization. Neither number establishes a GPU speedup.

## Completion boundary

The working prototype includes deterministic reference experiments, generation-safe control and semantic correction, streaming JSONL input, real compressed-video ingestion, optional ViT-backed tensor storage/consumption, paired output comparisons, evaluation/evidence export, and the synchronized local demo.

Optional [cross-frame KV reuse](KV_REUSE.md) is implemented for the final ViT block, and the [Android USB camera adapter](PHONE_CAMERA.md) is implemented with physical-phone validation pending. Production GPU/RTL integration, a full vision-language model, generic raw webcam adapters, and downstream task-quality/performance studies remain future work. Those require specific hardware and a target model/workload. This demonstration is a working research prototype and cannot establish that a production real-time system is complete.
