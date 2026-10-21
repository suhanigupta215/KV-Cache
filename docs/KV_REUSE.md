# Cross-frame KV reuse

This optional extension reuses visual patch K/V in **one final ViT-Tiny
attention block**, across consecutive processed frames. It is a software
embodiment of temporal reuse attached to the documented codec-to-residency
sequence, not a full VLM cache or a hardware KMC.

## Run

```bash
.venv/bin/python -m mtr_kv demo --semantic-model --kv-inference --kv-reuse --compare-baseline --port 8768
```

Open http://127.0.0.1:8768. The live table includes **KV pages reused**;
the decision text distinguishes retained and refreshed pages. Still content
is a useful demonstration. Moving video may have no exact matches; a zero
hit rate is valid and is never replaced with simulated savings.

The same flags work with `video`, `dashboard`, and `phone`. Capture a trace:

```bash
.venv/bin/python -m mtr_kv video results/demo.mp4 --semantic-model --kv-inference --kv-reuse --compare-baseline --max-frames 60 > results/reuse-demo.jsonl
.venv/bin/python -m mtr_kv evaluate results/reuse-demo.jsonl --output results/reuse-evaluation.json
```

For alternating standard/tiered captures and a separate classifier comparison:

```bash
.venv/bin/python -m mtr_kv evidence results/demo.mp4 --output-dir results/reuse-evidence --kv-reuse --max-frames 60
```

The evidence directory must be new. Manifests record reuse configuration.
The default `--reuse-window 8` refreshes tensors when their original content
generation becomes older than eight processed generations.

## Consistency contract

1. Every new generation follows the normal MKAD/controller
   reserve/prepare/commit/ready path before retained tensors can be rebound.
2. Decoder metadata must be reliable, activity low, and the page STATIC.
   Intra frames, transitions, high activity, missing metadata, and skipped
   controller generations invalidate reuse.
3. The actual page input at the final attention block must be **exactly equal**
   to the preceding input. Low motion or unchanged pixels alone is insufficient:
   earlier global attention can change quiet-page features when other content
   changes.
4. The payload must belong to the immediately preceding generation and the
   same controller, within its origin-age window. Geometry, dtype, device,
   parameter replacement, and normal tracked parameter updates invalidate
   history. Mutating weights outside PyTorch version tracking (for example
   through `.data`) is unsupported; create a new adapter after changing models.
5. Query projection runs for every token. K/V projection runs only for prefix
   tokens and misses. Hits reuse stored payloads through the new ready mapping;
   tier changes migrate the representation. Old-generation reads remain illegal.
6. Attention reads all patch K/V through the store. Later semantic correction
   still goes through KMC. Migration preserves the original content generation,
   so promotion or repeated reuse cannot extend the age bound indefinitely.

Only the latest page-input features and payloads are retained. New controllers,
failed inference, and process restarts cannot reuse a previous source's history.
This is bounded temporal reuse, not a multi-frame attention context or an
approximate motion-warped feature cache.

## Measurements and limits

The `kv_reuse` trace reports pages reused/recomputed, K/V tokens projected or
skipped, source/origin generations, recomputation reasons, and retained
validation-input bytes. Evaluation totals these measurements. Validation
inputs are additional memory, reported separately from tensor-store payloads.

Earlier ViT blocks still execute. Equality checks, cloning, split projections,
compression, and transfers add overhead; no end-to-end speedup is established.
fp16/int8 representations introduce error, and splitting packed projections
can change floating-point rounding. The baseline comparison records observed
classifier agreement and logit differences.

Tests cover actual retention without rewriting, partial invalidation, changed
features despite quiet motion, stale reads, missing metadata, intra frames,
generation gaps, window expiry, parameter/controller changes, tier migration,
and the full offline ViT/encoded-video pipeline. Pretrained-model traces on the
generated sample are separate observations, not representative performance or
task-quality evidence. Neither tests nor reuse measurements establish novelty.
