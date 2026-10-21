# Optimized prototype validation 21 October 2026

The live controller now uses the tiled MTRE path by default. Configure
`mtre.use_optimized_path=false` to retain the exhaustive oracle for comparisons.
The original frozen architecture document is unchanged.

Implemented changes:

- Enable the previously unused live spatial index, preserve block accumulation
  order, handle empty token sets and duplicate observations, and aggregate
  logical pages once across all reconciled tokens.
- Derive the source crop from the actual selected model resize/center-crop
  transform rather than assuming a full-frame stretch.
- Encode T2 as a shape header and compressed contiguous fp16 bytes without
  per-tensor torch.save archives. This private in-memory format is not an
  external persistent file format.
- Apply only needed semantic tier migrations rather than rereading every
  page after attention for snapshot synchronization.

## Primary paired measurement

`results/optimization-20261021/paired-serial/manifest.json` preserves three
alternating oracle/indexed pairs using the same final code, input, crop,
representations, model, and reuse configuration. Each run decodes 60 frames;
the first six remain in the trace but are excluded from its summary.

| Repeat | Oracle mean frame ms | Indexed mean frame ms |
| --- | ---: | ---: |
| 1 | 55.570 | 26.467 |
| 2 | 55.944 | 26.484 |
| 3 | 55.347 | 27.909 |

The medians of the per-run means are 55.570 and 26.484 ms: an observed ratio
of 2.098, or 52.34% lower time, for this reconciliation-path comparison. All
paired controller-decision hashes match. Timings include the baseline
comparison performed on each frame, exclude model loading/rendering/pacing,
and describe the CPU/generated sample only. They are not a speedup over
standard ViT, a full VLM, or production GPU/RTL inference.

The full serial four-phase demonstration has 216 ready/compared frames,
862 page reuses, and top-1 agreement on all 216 frames. This is classifier
consistency on generated content, not semantic task accuracy. Retained
validation inputs add 150,528 bytes at peak, separate from tensor payloads.

## Verification

128 tests pass, including exact full token/page equality for randomized
indexed/oracle geometry, duplicate observations, preprocessing crop bounds,
fp16 compressed scalar/empty/multidimensional round trips, and tier-sync I/O.
Text and JUnit reports, traces, hashes, and runtime details are preserved with
the completed invention disclosure evidence package.

Reproduce a new paired run from the repository root:

```bash
PYTHONPATH=. .venv/bin/python experiments/benchmark_optimization.py --output results/new-paired-run
```

The output directory must be new. Preliminary pre-change and concurrent-load
measurements are retained in the results folder but are not the primary
controlled timing comparison.
