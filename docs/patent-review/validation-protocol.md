# Reproducible technical validation protocol

**No representative dataset was available on 18 October 2026. No real-workload numbers are reported here.** The repo's synthetic results and short codec fixture must not be presented as production performance or VLM quality.

## Input selection and provenance

Before running, obtain videos with permission to use and report them. Pre-register at least three categories: mostly static with localized motion, high-motion/scene cuts, and semantically important low-motion content. Include different supported codecs/resolutions and document sampling criteria, source, license, original SHA-256, duration, frame rate, and codec. Hold out at least one set that was not used to tune thresholds. Do not infer “semantic importance” from model attention alone; obtain task labels or blinded human annotations.

## Existing executable capture

From the project root after `pip install -r requirements-live.txt`:

```bash
.venv/bin/python -m mtr_kv evidence /absolute/path/to/representative.mp4 \
  --output-dir results/evidence-study-001 --config configs/reference.json \
  --repeats 5 --max-frames 300 --warmup-frames 10 --device cpu
```

Use a **new** output directory for each study; existing directories are refused. `manifest.json` records source/config SHA-256, Git HEAD and dirty state, runtime/device versions, per-arm trace hashes, alternating run order, warm-up exclusion, per-arm summaries, and a separate paired classifier-output check. `report.md` is generated from that manifest for human review; it is not a substitute for the raw traces. The copied config and raw JSONL files must be retained together. An interrupted run has `status: incomplete` and is not evidence of performance. The file source is hashed again after capture to detect mutation. A local file is required; a live URL cannot be replayed exactly.

The two arms are: `standard` (standard ViT attention, K/V stored afterward) and `tiered` (final-block attention reads controller-tiered K/V). The quality run uses the same frame input for both paths and records top-1 agreement and mean absolute logit error. Warm-up frames are excluded from the summary but preserved in the raw trace. The capture command does **not** claim the arms are equivalent VLM systems or that a measured ratio implies a general speedup. Timing includes Python framework overhead; model loading is excluded. Standard and tiered have different work, and CPU quantization/compression can be slower despite reducing tensor payload.

## Audit and acceptance gates

1. Verify `status: complete`, all hashes, identical per-frame generation/PTS timeline in each arm, equal frame counts, all `READY`, and no uncaught errors. Do not cherry-pick failed or slow runs.
2. Report median across repeats **and all per-repeat values** for mean and p95 model latency, end-to-end latency, actual tensor payload peak, bytes read/written, fallback rate, and semantic-correction counts. Report denominators. Do not relabel the controller's configured capacity estimate as measured resident tensor bytes.
3. Compare paired output agreement to a pre-registered tolerance, then inspect disagreement frames. Top-1 agreement is only a classifier sanity check; a VLM requires separately labeled downstream tasks and quality metrics.
4. Run ablations with the same immutable input and code revision: codec-only, semantic-only/full, no temporal hysteresis, no scene prediction, several page sizes, and different T1/T2 representations. The current `mtr_kv evidence` command captures the two executable inference paths; ablation automation and a VLM-quality harness are **not yet implemented**.
5. For hardware or energy claims, repeat on an identified GPU/device runtime with synchronized timing, actual peak device memory, transfer bandwidth, power/energy, and multiple independent runs. The present machine reports neither CUDA nor MPS availability, so no hardware-performance conclusion can be produced here.

Only the first two arms and classifier agreement are automated now. A patent draft may describe an enabled architecture without benchmark numbers, but **numerical technical-effect assertions must be tied to a specific, auditable experiment**. This protocol supplies the path; a representative dataset and independent review are still necessary.
