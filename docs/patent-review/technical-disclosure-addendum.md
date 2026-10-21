# Technical disclosure addendum for inventor and counsel review

This is an engineering description, **not proposed final claim language**. It supplements, and does not silently revise, the frozen master architecture. “Implemented” means observable in this repository; “proposed” means only specified in the frozen document.

**Implementation update, 21 October 2026:** optional bounded cross-frame reuse is now implemented in the final ViT block. It validates codec eligibility and exact block-input equality, rebinds tensors only after the new generation is ready, and skips patch K/V projection on hits. See [the reuse contract](../KV_REUSE.md), `tests/test_kv_reuse.py`, and updated E6 in the evidence map. Statements below about absent cross-frame reuse describe the earlier embodiment; full VLM/cross-window and hardware integration remain unimplemented. This update makes no novelty or speedup assertion.

## Problem and candidate technical contribution

Compressed video carries spatial and temporal evidence before transformer attention is complete. The proposed apparatus uses that early evidence to form page-level memory-placement intent for visual K/V, then permits generation-matched semantic feedback to correct low-motion/high-importance mistakes through the same controller protocol. The candidate distinction is **the particular cross-domain control sequence and safety/feedback relationships**, not motion vectors, token pruning, paging, quantization, or tiered storage individually. Whether even that combination is novel or inventive is unresolved; see the prior-art chart.

## Current executable embodiment

1. `mtr_kv.video.run_video` decodes a compressed input through PyAV/FFmpeg, requests exported motion vectors, converts source rectangles and vector scale, and labels metadata fallback. The adapter does **not** extract codec residual energy, connect to NVDEC hardware, or run at a guaranteed source frame rate.
2. `mtr_kv.motion.MotionNormalizer`, `mtr_kv.mtre.Mtre`, `mtr_kv.temporal.TemporalStateEngine`, and `mtr_kv.predictor.ScenePredictor` convert available source-space evidence into token/page activity and temporal state. Configuration controls thresholds and crop/resize geometry.
3. `mtr_kv.mkad.MkadBuilder` emits generation-tagged page-range/tier descriptors. `mtr_kv.controller.Kmc` and `mtr_kv.memory.TieredMemoryModel` model reserve, shadow prepare, commit, ready-token emission, stale rejection, and tier transitions. This is a **Python behavioral controller**, not physical GPU memory hardware, an RTL design, or a real asynchronous decoder-to-accelerator FIFO.
4. `mtr_kv.vision_feedback.ViTAttentionFeedback` runs a pretrained ViT-Tiny **image classifier**. In `--kv-inference` mode its final attention block writes patch K/V through `LiveController.attach_kv`, reads it through `read_kv`, and uses those retrieved tensors for attention. `mtr_kv.tensor_store.TorchKVStore` uses fp16 T0 on the selected torch device, CPU int8 T1, and CPU compressed fp16 T2. This is one attention block in one model; it is **not a VLM KV cache reused across frames**.
5. The class-token attention signal is aggregated into page scores. `mtr_kv.semantic.Srce` can issue generation-matched corrections, and the same KMC applies them. That attention score is a proxy, not a labeled semantic ground truth. The present model finishes a full inference pass before the score exists; correction influences stored residency, not a proven avoidance of that pass.
6. `mtr_kv.evidence.capture` can hash a user-supplied local video and configuration, alternate standard/tiered runs, save raw JSONL and a manifest, and run paired classifier-output comparison. It cannot manufacture a representative dataset or prove a device-level speedup.

## Proposed but unimplemented embodiments in the frozen disclosure

- A hardware decoder signal path to a hardware/GPU KMC with physical tier banks and an asynchronous FIFO.
- A selected VLM with visual-token K/V reuse over sliding windows and measured downstream task quality.
- Cross-layer/cross-time page identity, true on-device prefetch, optional T3 reconstruction, fixed-point RTL, and device-level power/bandwidth measurements.

These may be included in a disclosure only with precise enablement and inventor confirmation; they must **not** be described as demonstrated by the current code.

## Suggested drawings for counsel

- **Figure A — system:** decoder metadata → normalizer/MTRE/temporal state → descriptor → KMC/page table/tier representations → transformer → semantic correction returning to the descriptor path. Solid lines for implemented software connections; dashed lines for proposed hardware links.
- **Figure B — transaction:** generation-tagged descriptor → reserve → shadow prepare → atomic commit → ready token; show stale correction rejection and fallback.
- **Figure C — geometry:** source-pixel motion block overlap after crop/resize with patch token regions and a page aggregation example.
- **Figure D — measured embodiment:** actual last-block ViT K/V write/read sequence and the distinct standard comparison arm. Do not depict multi-layer VLM reuse as a measured result.

## Best-mode information to confirm with inventors

The executable reference uses `configs/reference.json`, the current PyAV/FFmpeg adapter, 32 tokens per page unless overridden, a pretrained ViT-Tiny attention proxy, and T0/T1/T2 tensor representations described above. Counsel should ask whether the inventors know a better embodiment, unpublished implementation, or hardware design not captured here; this document cannot certify the legally required best method on their behalf.
