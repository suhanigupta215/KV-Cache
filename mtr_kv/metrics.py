"""Hard metrics the simulator must produce (spec sections 55, 62, 89).

All results are generated from raw logs. A headline number without its
denominator is not sufficient: absolute pages/bytes accompany every
percentage. Metric definitions follow Appendix E exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .data_objects import Tier


@dataclass
class FrameRecord:
    """One record per frame (Appendix G schema)."""

    stream_id: int
    generation: int
    frame_index: int
    baseline_pages: int
    mtr_pages_t0: int
    mtr_pages_t1: int
    mtr_pages_t2: int
    mtr_pages_t3: int
    pages_avoided: int
    fallback: bool
    semantic_rescues: int
    conflicts: int
    queue_peak: int
    stale_drops: int
    transition_predicted: bool = False
    scene_cut_truth: bool = False
    tier_switches: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "generation": self.generation,
            "frame_index": self.frame_index,
            "baseline_pages": self.baseline_pages,
            "mtr_pages_t0": self.mtr_pages_t0,
            "mtr_pages_t1": self.mtr_pages_t1,
            "mtr_pages_t2": self.mtr_pages_t2,
            "mtr_pages_t3": self.mtr_pages_t3,
            "pages_avoided": self.pages_avoided,
            "fallback": self.fallback,
            "semantic_rescues": self.semantic_rescues,
            "conflicts": self.conflicts,
            "queue_peak": self.queue_peak,
            "stale_drops": self.stale_drops,
            "transition_predicted": self.transition_predicted,
            "scene_cut_truth": self.scene_cut_truth,
            "tier_switches": self.tier_switches,
        }


@dataclass
class AggregateMetrics:
    """Aggregate metrics with explicit denominators (spec section 55)."""

    total_frames: int = 0
    baseline_pages_total: int = 0
    hot_pages_total: int = 0
    resident_pages_total: int = 0
    baseline_bytes_total: int = 0
    physical_bytes_total: int = 0
    kv_write_bytes: int = 0
    promotion_bytes: int = 0
    demotion_bytes: int = 0
    reconstruction_bytes: int = 0
    total_descriptors: int = 0
    stale_drops: int = 0
    duplicates: int = 0
    fallbacks: int = 0
    promotions: int = 0
    demotions: int = 0
    conflicts: int = 0
    evaluated_pages: int = 0
    low_codec_candidates: int = 0
    rescues: int = 0
    scene_tp: int = 0
    scene_fp: int = 0
    scene_fn: int = 0
    scene_tn: int = 0
    state_switches: int = 0
    frame_state_fraction: dict[str, float] = field(default_factory=dict)
    tier_switches: int = 0
    queue_peak_overall: int = 0
    descriptor_count_hierarchical: int = 0
    descriptor_count_flat: int = 0
    latency_samples: list[float] = field(default_factory=list)

    @property
    def semantic_rescues(self) -> int:
        """Compatibility name used by the reference test specification."""
        return self.rescues

    # ------------------------------------------------------------------
    # Appendix E definitions
    # ------------------------------------------------------------------
    def pages_avoided(self) -> int:
        return self.baseline_pages_total - self.hot_pages_total

    def bytes_saved(self) -> int:
        return self.baseline_bytes_total - self.physical_bytes_total

    def total_traffic(self) -> int:
        return (
            self.kv_write_bytes
            + self.promotion_bytes
            + self.demotion_bytes
            + self.reconstruction_bytes
        )

    def fallback_rate(self) -> float:
        return self.fallbacks / max(self.total_frames, 1)

    def stale_drop_rate(self) -> float:
        return self.stale_drops / max(self.total_descriptors, 1)

    def conflict_rate(self) -> float:
        return self.conflicts / max(self.evaluated_pages, 1)

    def semantic_rescue_rate(self) -> float:
        return self.rescues / max(self.low_codec_candidates, 1)

    def mean_latency(self) -> float:
        return (
            sum(self.latency_samples) / len(self.latency_samples)
            if self.latency_samples
            else 0.0
        )

    def p95_latency(self) -> float:
        if not self.latency_samples:
            return 0.0
        s = sorted(self.latency_samples)
        idx = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
        return s[idx]

    def scene_confusion(self) -> dict[str, float]:
        total = max(self.scene_tp + self.scene_fp + self.scene_tn + self.scene_fn, 1)
        return {
            "accuracy": (self.scene_tp + self.scene_tn) / total,
            "false_positive_rate": self.scene_fp / max(self.scene_fp + self.scene_tn, 1),
            "false_negative_rate": self.scene_fn / max(self.scene_fn + self.scene_tp, 1),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_frames": self.total_frames,
            "baseline_pages_total": self.baseline_pages_total,
            "hot_pages_total": self.hot_pages_total,
            "resident_pages_total": self.resident_pages_total,
            "pages_avoided": self.pages_avoided(),
            "pages_avoided_pct": 100.0
            * self.pages_avoided()
            / max(self.baseline_pages_total, 1),
            "baseline_bytes_total": self.baseline_bytes_total,
            "physical_bytes_total": self.physical_bytes_total,
            "bytes_saved": self.bytes_saved(),
            "bytes_saved_pct": 100.0
            * self.bytes_saved()
            / max(self.baseline_bytes_total, 1),
            "total_traffic_bytes": self.total_traffic(),
            "kv_write_bytes": self.kv_write_bytes,
            "promotion_bytes": self.promotion_bytes,
            "demotion_bytes": self.demotion_bytes,
            "reconstruction_bytes": self.reconstruction_bytes,
            "total_descriptors": self.total_descriptors,
            "stale_drops": self.stale_drops,
            "stale_drop_rate": self.stale_drop_rate(),
            "duplicates": self.duplicates,
            "fallbacks": self.fallbacks,
            "fallback_rate": self.fallback_rate(),
            "promotions": self.promotions,
            "demotions": self.demotions,
            "conflicts": self.conflicts,
            "conflict_rate": self.conflict_rate(),
            "low_codec_candidates": self.low_codec_candidates,
            "semantic_rescues": self.rescues,
            "semantic_rescue_rate": self.semantic_rescue_rate(),
            "scene_confusion": self.scene_confusion(),
            "temporal_state_switches": self.state_switches,
            "frame_state_fraction": self.frame_state_fraction,
            "tier_switches": self.tier_switches,
            "queue_peak_overall": self.queue_peak_overall,
            "descriptor_count_hierarchical": self.descriptor_count_hierarchical,
            "descriptor_count_flat": self.descriptor_count_flat,
            "mean_latency": self.mean_latency(),
            "p95_latency": self.p95_latency(),
        }


def tier_sort_key(tier: Tier) -> int:
    return tier.value
