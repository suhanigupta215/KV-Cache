# MTR-KV Simulation Report

Scenario: `T03_semantic_rescue`
Seed: `20260922`
Configuration: `frozen-reference-v1` v1.0

| Mode | Pages avoided | Bytes saved | Semantic rescues | Fallback rate | Stale drops |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0 | 0 | 0 | 0.00% | 0 |
| codec_only | 80 | 377487360 | 0 | 0.00% | 0 |
| full | 52 | 245366784 | 20 | 0.00% | 0 |

All values are simulator measurements from the accompanying JSON; they are not hardware performance claims.
