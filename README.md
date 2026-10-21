# MTR-KV with Browser Video Selection

This updated copy of the MTR-KV project adds **Choose File** and **Analyze video**
to the local dashboard. It processes your selected video through the existing
decoder/controller/model pipeline and shows a completed or partial results summary.

Based on the project snapshot supplied from `Shaurya002800/MTR-KV`.
See [video picker instructions](docs/VIDEO_PICKER.md) for usage and limits.

# MTR-KV

Patent-oriented technical review materials are in [docs/patent-review/README.md](docs/patent-review/README.md). They distinguish implemented software from proposed hardware/VLM embodiments and are not a patentability opinion.

MTR-KV is a deterministic Python prototype for Motion-to-Residency Reconciliation: codec motion metadata is reconciled with visual-token geometry, converted into generation-tagged allocation descriptors, and committed by a tiered KV memory-controller model. Semantic feedback can later rescue important low-motion pages through the same safe control path.

The frozen architecture and prototype requirements are in `docs/MTR_KV_Frozen_Master_Architecture_and_Prototype_Specification_FINAL_v1_0.docx`. This repository implements the simulator, not a production GPU controller or RTL design.

## Run the reference experiment

```bash
.venv/bin/python -m mtr_kv run --scenario T03 --mode all
.venv/bin/python -m mtr_kv report
```

The first command writes deterministic raw output to `results/simulation.json` and per-frame records to `results/frames.csv`. The second command creates `results/report.md` from that JSON. Choose `T01`, `T02`, `T03`, `T04`, `T08`, or `multistream` to exercise the reference workloads. Pass `--config configs/reference.json` to use the frozen reference parameters, or point it to another saved configuration.

## Validate

The live pipeline now uses an oracle-checked spatial index, the model's actual
preprocessing crop, compact T2 tensor bytes, and selective tier synchronization.
See [optimization measurements and verification](docs/OPTIMIZATION.md) for
the paired experiment and its limits.

```bash
.venv/bin/python -m pytest -q
```

The test suite covers spatial reconciliation, temporal hysteresis, page allocation, generation safety, semantic rescue, scene transitions, missing-metadata fallback, and adversarial ordering.

## Live demonstration — start here

From the repository root, launch the prepared demonstration:

```bash
.venv/bin/python -m mtr_kv demo
```

Open **http://127.0.0.1:8765** and select **Restart run** to replay. The command creates `results/demo.mp4` if absent, then decodes that real H.264 file through the controller. The 18-second clip has four phases: still content, a moving object, still content again, and a scene cut. The four bars at the bottom indicate the current phase. Codec motion vectors are extracted from the encoded video, not manufactured for the dashboard.

The view shows the decoded frame alongside the same generation's committed T0/T1/T2 page mapping, fallback events, motion-vector counts, and measured processing rate. Each tile represents a logical KV page, not a rectangular image region. Video timestamps pace presentation; slower processing plays at the speed the machine can sustain. **Stop** closes the stream; **Restart run** decodes from the beginning. **Save trace** exports the most recent 10,000 measured records as JSONL, excluding preview images, for the `evaluate` command. Refreshing or opening another tab starts an independent run.

For your own video or a compressed live feed:

```bash
.venv/bin/python -m mtr_kv dashboard /absolute/path/to/video.mp4
.venv/bin/python -m mtr_kv dashboard 'rtsp://your-camera/stream'
```

These commands accept FFmpeg-supported files and network sources. You can also use **Analyze your own video** in the local dashboard: choose a file from your computer or gallery, then select **Analyze video**. The dashboard shows the decoded preview, frame decisions, and a results summary. See [video picker usage and limits](docs/VIDEO_PICKER.md). A raw webcam does not inherently supply codec motion vectors; use an encoded camera stream for this adapter. To connect your own decoder or application, use the JSONL `live` interface below. Generic webcam capture is not implemented.

### Android camera over USB

For Android 12+ phones, install `scrcpy` and `adb`, connect a data-capable USB cable, enable USB debugging, and approve the computer on the phone. Then run:

```bash
.venv/bin/python -m mtr_kv phone --list-devices
.venv/bin/python -m mtr_kv phone
```

Open **http://127.0.0.1:8767** and select **Connect camera**. The direct camera feed supplies real H.264 motion metadata to the same controller and preview. Use `--camera-facing front` for the front camera; add `--semantic-model --kv-inference --compare-baseline` for model-backed processing. Slow inference skips older captured frames to keep the preview current. See [phone setup and limitations](docs/PHONE_CAMERA.md).

To demonstrate actual model-produced K/V tensors being stored and read during attention, stop the first server with Ctrl-C and run:

```bash
.venv/bin/python -m mtr_kv demo --semantic-model --kv-inference --compare-baseline
```

The model-backed demo uses pretrained ViT-Tiny; first use may download weights. CPU inference can run slower than playback. The default codec-only demo requires no model weights and reports controller allocations rather than actual tensor storage. The model mode additionally reports real tensor payloads, inference time, and classifier agreement against standard attention. Add `--port 8766` to run a second server, or use `dashboard ... --unpaced` to process without presentation pacing.

See [the demo walkthrough](docs/DEMO.md) for a presentation script, measurement commands, and the boundary between the completed prototype and future production integration.

## Online control input

`mtr_kv live` consumes observations as newline-delimited JSON on standard input and flushes one result for each record. It retains temporal, page-table, and generation state across frames:

```bash
.venv/bin/python -m mtr_kv live --width 640 --height 480
```

Each `frame` record needs a monotonically increasing `generation`, a frame type (`I`, `P`, or `B`), and decoder blocks in source pixel coordinates. A block supplies `rect` (`[x, y, width, height]`), `mv` (`[dx, dy]`), optional `residual`, `confidence`, and `valid`. `blocks: []` triggers conservative T0 fallback. For example:

```json
{"type":"frame","generation":1,"frame_type":"P","blocks":[{"rect":[0,0,640,480],"mv":[0,0],"residual":0,"confidence":1}]}
{"type":"semantic","generation":1,"page_scores":{"0":0.95}}
```

The `frame_result` contains proposed MKADs, controller apply statuses, completion tokens, and the current memory snapshot. A `semantic` record can arrive after its frame and is rejected if its generation is stale. The producer must provide actual decoder motion metadata and model-derived page scores. This command implements the streaming control path; it does not decode video, execute a transformer, or allocate GPU KV tensors by itself. Those integrations require a selected decoder, model, and device runtime.

## Compressed video ingestion

Install the live runtime and feed a video file or FFmpeg-compatible live URL to the controller:

```bash
.venv/bin/python -m pip install -r requirements-live.txt
.venv/bin/python -m mtr_kv video path/to/input.mp4 > results/video-decisions.jsonl
.venv/bin/python -m mtr_kv video path/to/input.mp4 --semantic-model > results/video-with-semantic.jsonl
```

`video` enables FFmpeg's `export_mvs` flag and converts decoded motion-vector side data into source-space blocks. The first intra frame, unsupported codecs, and frames without exported vectors enter the conservative metadata fallback. Each output line includes the source presentation timestamp, extracted-vector count, MKADs, apply statuses, and ready tokens. Use `--max-frames N` for a bounded run. Residual energy is currently unavailable through this adapter and is not fabricated; codec-only comparisons should record that limitation.

`--semantic-model` runs a pretrained ViT-Tiny classifier on each decoded RGB frame and aggregates its class-token attention into visual-page scores before invoking SRCE. The first use downloads model weights. Its 16-pixel patch geometry is applied to the controller for that run. Class-token attention is a model signal, not ground-truth semantic importance, and this classifier is not a VLM. The output records its model ID and correction actions so it can be evaluated separately from codec-only decisions.

The default semantic run captures the ViT's last-block K/V tensors and stores them in the controller-selected tier **after** standard attention. `tensor_storage.payload_bytes`, `bytes_written`, `bytes_read`, and timing fields come from actual tensor operations. For an end-to-end consumption path, run:

```bash
.venv/bin/python -m mtr_kv video path/to/input.mp4 --semantic-model --kv-inference --compare-baseline --max-frames 100 > results/live.jsonl
.venv/bin/python -m mtr_kv evaluate results/live.jsonl --output results/live-evaluation.json
```

`--kv-inference` replaces only the ViT's final attention block with a path that writes patch K/V into controller-selected tiers and reads them back **inside the attention computation**. Prefix-token K/V stays in the model. The live controller's model geometry is set to this one ViT block rather than the 24-layer frozen reference geometry; its occupancy estimate still includes page padding and tier ratios, whereas `tensor_storage` reports actual payload bytes. `--compare-baseline` also runs the standard ViT on each frame and records top-1 agreement, mean absolute logit error, and separate inference time. These are observed classifier metrics, not VLM task quality or a speedup claim. Both paths must fit the controller's patch-page geometry. The processing-time measurement excludes video decoding. `--device` selects the model and T0 torch device; the default is CPU. T1/T2 are CPU-backed representations.

The JSONL records the decoder codec, source dimensions, decoding time, processing time, model time, controller decisions, tensor payload bytes, and optional classifier comparison. The evaluation JSON summarizes mean and p95 timings and explicitly lists limitations. For a local, automatically updating presentation view, run:

```bash
.venv/bin/python -m mtr_kv dashboard path/to/input.mp4 --semantic-model --kv-inference --compare-baseline --max-frames 100
```

Open `http://127.0.0.1:8765`. The **Live controller trace** streams measured frame records as decoding progresses; the built-in simulator example is hidden in live mode unless explicitly selected. Restarting the view reprocesses the source. The server binds to localhost by default and never uploads the video. Model weights may download on first use. A live FFmpeg-compatible URL can be supplied instead of a file, although the Python/CPU path does not promise source-frame-rate throughput.

## Model-facing KV tensor API

Install PyTorch in the chosen runtime, then create a `LiveController` and submit real K/V tensors once the corresponding `frame()` result is `READY`:

```python
controller.attach_kv(page_id, generation, key_tensor, value_tensor, device="cpu")
key_tensor, value_tensor = controller.read_kv(page_id, generation)
print(controller.tensor_store.snapshot())
```

The optional `TorchKVStore` uses fp16 on the chosen T0 device, symmetric int8 on CPU for T1, and lossless compressed fp16 on CPU for T2. It measures payload bytes moved and read/write time. `TorchKVStore.snapshot()` reports actual tensor payloads, while `TieredMemoryModel.snapshot()` reports the controller's configured capacity model; do not equate the two. The `--kv-inference` path consumes these representations in the final ViT attention block.

### Cross-frame KV reuse

Add `--kv-reuse` to a model-backed run to retain unchanged static-page K/V across consecutive frames. The controller must commit the new generation; the model then verifies exact page-input feature equality before skipping that page's K/V projection. Missing metadata, transitions, changed features, generation gaps, and window expiry force refresh. The dashboard and evaluation report actual reuse and validation-memory overhead.

```bash
.venv/bin/python -m mtr_kv demo --semantic-model --kv-inference --kv-reuse --compare-baseline --port 8768
```

See [the reuse contract and measurements](docs/KV_REUSE.md). This is bounded reuse in one ViT block; earlier blocks still execute. Full VLM inference, task-quality evaluation, production GPU-controller integration, and a controlled multi-device performance study remain future work. Patent novelty and claim scope require a separate prior-art and legal review; the software does not establish either.

## Invention disclosure and evidence

The [version 1.0 review baseline](output/docx/MTR_KV_Patent_Disclosure_v1_0.docx) adds three trace-linked operating-state captures, a feature-to-code-to-test map and one natural-scene video check. The [matching PDF](output/pdf/MTR_KV_Patent_Disclosure_v1_0.pdf) and [frozen evidence package](output/evidence/MTR_KV_Frozen_Evidence_v1_0.zip) preserve the results with SHA-256 verification. The H.264 natural-video run agrees with baseline top-1 on 170 of 171 frames and has no reuse hits; its original VP8 source decodes with metadata fallback throughout. Preserve this baseline and make paraphrases or later changes in a new version.

The earlier [revised patent disclosure](output/docx/MTR_KV_Patent_Disclosure_Revised.docx) follows the supplied IDF-B reference structure and includes frame-specific implementation witnesses. A [matching PDF](output/pdf/MTR_KV_Patent_Disclosure_Revised.pdf) and [revised evidence archive](output/evidence/MTR_KV_Revised_Evidence.zip) accompany it. The revised geometry figure uses the measured model crop recorded in the traces.

The completed [IDF-B PDF](output/pdf/MTR_KV_IDF_B_Completed.pdf) preserves the supplied form and appends a detailed technical annex. The [editable annex](output/docx/MTR_KV_IDF_B_Detailed.docx) includes diagrams, prototype screenshots, measured results, prior-work comparisons, and references. The [evidence archive](output/evidence/MTR_KV_Evidence.zip) preserves source snapshots, input video, raw traces, test reports, and a hash-verification script. These are technical records for inventor and institutional review, not a filed patent or a novelty opinion.
