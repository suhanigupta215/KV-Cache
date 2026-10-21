# MTR-KV technical specification — inventor/counsel working draft

**Implementation update, 21 October 2026:** optional bounded cross-frame reuse is now implemented for the final ViT block with exact feature validation and generation-safe tensor rebinding. See [the reuse contract](../KV_REUSE.md) and updated E6 evidence map. References below to absent cross-frame reuse describe the earlier embodiment; they do not negate this limited extension. Full VLM/hardware embodiments remain proposed. The frozen document is unchanged.

**Status (18 October 2026):** engineering disclosure for review, not a filed patent specification, final claims, novelty opinion, or assertion of measured VLM/GPU benefit. The [frozen master architecture](../MTR_KV_Frozen_Master_Architecture_and_Prototype_Specification_FINAL_v1_0.docx) remains the original design record. Statements marked **implemented** below describe this repository; statements marked **proposed** require inventor confirmation and additional enablement. Counsel must determine filing form, legal sufficiency, claim scope, and disclosure dates.

## Working title and abstract

**Working title:** Generation-safe codec-to-visual-K/V residency control with semantic correction.

**Technical abstract:** A video-processing path obtains available decoder motion and frame-type information, maps source-pixel regions to visual-token and logical-page regions, and derives early page-level activity and temporal state. A descriptor identifies a stream, generation, page range, target K/V residency tier, and operation. A controller validates the descriptor, reserves replacement storage before releasing an active mapping, prepares page mappings, publishes them, and emits a completion token. A later model-derived page signal can request a generation-matched tier correction through the same controller. Missing decoder metadata invokes a conservative route. The executable embodiment is a Python controller model coupled to the final self-attention block of a ViT image classifier; it does not establish a hardware implementation, cross-frame VLM cache, or performance advantage.

## Field, problem, and prior-art boundary

The field is memory control for video-transformer visual K/V data. Video decoder metadata is available at a different spatial granularity and at a different time than transformer attention. A direct motion-to-token mapping can miss important static objects, while an unversioned late correction can corrupt a newer page. The engineering problem is to produce a page-level placement decision early, preserve mapping integrity during updates, and permit later semantic correction without stale writes.

Motion vectors, residuals, transformer K/V caching, page tables, quantization, and memory tiers are not asserted novel individually. The proposed combination must be compared against the references and claim families in [prior-art-landscape.md](prior-art-landscape.md), especially CodecSight, CoPE-VideoLM, PagedAttention, and K/V cache controller publications. This draft deliberately contains no novelty or non-obviousness conclusion.

## Definitions and implemented data path

- A **generation** identifies a frame/control epoch. A **logical page** groups visual patch tokens; the reference configuration uses 32 tokens per page. A **tier** names a representation/placement preference (T0 hot, T1 warm, T2 cold, optional T3 reconstruction).
- A **codec observation** carries source-pixel block geometry, motion magnitude, confidence, and frame type. The PyAV adapter requests exported motion vectors; it does **not** extract real residual energy. Residual-related behavior in simulator scenarios is synthetic unless a future adapter supplies it.
- A **motion-keyed allocation descriptor (MKAD)** carries stream and generation identifiers, contiguous page range, operation, target tier, activity, confidence, and cause. In the current code this is a Python object, not a hardware packet.
- A **ready token** records stream, generation, page range, and a model fence counter after a successful controller commit. The counter models ordering; it is not a measured GPU/PCIe memory fence.

The implemented software embodiment processes a source video as follows:

1. `mtr_kv/video.py` decodes frames and reads exported motion-vector side data when present. Unsupported or absent metadata is reported rather than imputed as zero motion. The input frame is also converted to RGB for the model path.
2. `mtr_kv/motion.py` normalizes observations; `mtr_kv/mtre.py` overlaps source-space motion blocks with patch-token rectangles and aggregates token activity to logical pages. `mtr_kv/temporal.py` and `mtr_kv/predictor.py` retain temporal state and scene-transition signals.
3. `mtr_kv/mkad.py` maps high activity, transitions, and unknown state conservatively toward T0; intermediate activity toward T1; low activity toward T2. Adjacent compatible pages may be grouped in one descriptor. These are configurable policy choices, not proven optimal thresholds.
4. `mtr_kv/controller.py` checks descriptor validity and generation, reserves replacement blocks, constructs shadow mappings, updates the active page-table model, and emits ready tokens. A failed replacement reservation preserves the old mapping; a failed descriptor is not marked successfully seen and can be retried. Unsupported T3 is rejected before commit. This is a single-process behavioral transaction model, not an atomic hardware publication demonstrated under concurrent device access.
5. `mtr_kv/vision_feedback.py` runs a pretrained ViT-Tiny image classifier. In `--kv-inference`, its final attention block stores patch K/V through `mtr_kv/tensor_store.py` and reads the stored representation back into that attention computation. T0 is fp16 on the selected torch device, T1 is CPU int8, and T2 is CPU compressed fp16. The class-token K/V remains in the model. This is **one block of one image classifier**, not a VLM or cross-frame reusable K/V cache.
6. The ViT class-token attention is aggregated to page scores. `mtr_kv/semantic.py` can issue a low-codec/high-score rescue or controlled demotion; `mtr_kv/live.py` routes its correction through the same KMC. The proxy score arrives after a full classifier pass, so it does not demonstrate avoided model computation or labeled semantic correctness.

## Control sequence and failure cases

**Nominal allocation:** accept a valid, non-stale MKAD; reserve the full replacement range; prepare shadow mappings; publish page entries and committed generations in the Python model; emit `KV_READY`; permit subsequent tensor attachment/read. Duplicate descriptors do not allocate twice. A newer generation may replace an older mapping only after replacement storage is available.

**No capacity:** reservation raises an allocation error before the active mapping is released. The configured T2 fallback may be attempted; if it also fails, return `ABORT` without a new ready token. The current allocator uses one shared physical pool with headroom, not true physically separate tier banks or a guaranteed emergency bank.

**Late semantic update:** an update bearing an older generation is dropped. A matched update may change the tier through the controller path. Unknown/transitioning state and missing metadata favor conservative placement. These rules are tested at the software level; no device race or formal hardware proof is claimed.

## Best known executable method and reproduction

The repository's present reference uses `configs/reference.json` for its frozen baseline. The live ViT command adapts its controller geometry to the selected 16-pixel patch model instead of pretending the frozen 24-layer geometry is its actual tensor shape. Install `requirements-live.txt`, then run a local permitted video:

```bash
.venv/bin/python -m mtr_kv video /absolute/path/to/input.mp4 \
  --semantic-model --kv-inference --compare-baseline --max-frames 100 \
  > results/live.jsonl
.venv/bin/python -m mtr_kv evaluate results/live.jsonl \
  --output results/live-evaluation.json
```

For a provenance-preserving comparison, follow [validation-protocol.md](validation-protocol.md) and use `mtr_kv evidence` with an immutable licensed local video. Its `manifest.json`, `report.md`, copied config, and hashed raw traces are the primary technical record. `results/simulation.json` and the `results/patent-review/` T03/T04/T08 traces are **synthetic**, not representative video benchmarks. No representative dataset was supplied, so this document reports no numerical real-world effect.

## Proposed alternatives requiring separate support

The frozen architecture describes decoder-to-controller asynchronous signaling, GPU/device page tables, physically differentiated tier banks, prefetch, an optional T3 reconstruction tier, and visual-token K/V reuse across VideoLM layers or time windows. These are **proposed** in relation to this repository. Before describing any as an enabled or achieved embodiment, the inventors should supply implementation details, an operable timing/synchronization protocol, and tests or design records. In particular, a general-purpose GPU tensor transfer plus a Python page table is not equivalent to a hardware KMC.

## Drawing package for counsel

1. **Figure 1, system:** decoder signal extractor; pixel-to-token reconciler; temporal predictor; MKAD builder; controller/page table; tensor tier store; transformer; semantic feedback loop. Distinguish implemented software links from proposed hardware links visually.
2. **Figure 2, geometry:** source-pixel block and crop/resize transform; patch-token overlap; token-to-page aggregation. Include a missing-metadata branch.
3. **Figure 3, transaction:** generation validation, reserve, prepare, publish, ready; show allocation failure preserving old mapping and stale semantic correction dropping.
4. **Figure 4, measured software embodiment:** final ViT attention block patch K/V write/read, class-token bypass, T0/T1/T2 representations, and separate standard-classifier comparison arm. Do not depict VLM reuse as measured.

## Candidate claim *topics* for professional drafting

These are issue-spotting topics, **not claims**. Counsel should decide whether any combination is novel, non-obvious, enabled, and suitable for the filing jurisdictions:

- A method combining decoder-derived spatial evidence, source-to-token/page reconciliation, temporal state, and a generation-tagged page-tier descriptor **before** transformer attention.
- A controller sequence that preserves the previous page mapping when replacement reservation fails and emits a generation-associated completion token only after a successful model commit.
- A feedback loop in which later page-level model importance corrects codec-derived placement through the same generation-checked descriptor/controller path, including conservative handling of absent metadata.
- If separately enabled, a concrete decoder/device interface and cross-frame model cache protocol with defined page identity, fences, and consistency guarantees.

## Open facts and evidence required before filing assertions

Inventors must verify inventorship, conception records, public-disclosure history, ownership, and best known method; see [counsel-handoff.md](counsel-handoff.md). A representative dataset, labeled task-quality measure, pre-registered ablations, repeated raw runs, and independent review are absent. A patent agent should compare exact prior-art limitations and decide what can truthfully be claimed. Neither a passing software test suite nor this draft establishes patentability, freedom to operate, or a measured speedup.
