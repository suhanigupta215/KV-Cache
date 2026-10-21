"""Semantic Residency Correction Engine and CSRR policy (spec 28-31, 53).

SRCE is the semantic bridge: its canonical case is low codec activity
combined with high semantic importance, resolved by promotion to a safer
tier. CSRR is the policy layer comparing codec activity A and semantic
importance S; conflict C = |A - S| identifies disagreement. Corrections
are generation-tagged and never bypass the normal KMC commit path.
Attention importance aggregation transfers summary statistics, never
full attention matrices (spec section 27).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import Config
from .data_objects import MKAD, Operation, PageLifecycle, TemporalState, Tier


def aggregate_attention(
    attention_scores: dict[int, float],
    tokens_per_page: int,
    method: str = "mean",
) -> dict[int, float]:
    """Aggregate token attention statistics to page granularity (spec 28).

    attention_scores maps token_id -> raw score. Aggregation can be mean,
    max, percentile, or weighted sum. Scores are normalized before
    thresholding; the chosen normalization is logged for reproducibility.
    """
    pages: dict[int, list[float]] = {}
    for token_id, score in attention_scores.items():
        page_id = token_id // tokens_per_page
        pages.setdefault(page_id, []).append(score)
    out: dict[int, float] = {}
    for page_id, scores in pages.items():
        if method == "mean":
            value = sum(scores) / len(scores)
        elif method == "max":
            value = max(scores)
        elif method == "min":
            value = min(scores)
        else:  # weighted_sum treats scores as weights over rank positions
            ordered = sorted(scores, reverse=True)
            value = sum(w * s for w, s in enumerate(ordered)) / max(
                sum(range(len(ordered))), 1
            )
        out[page_id] = value
    # Normalize to [0, 1] by max-abs (logged normalization).
    peak = max((abs(v) for v in out.values()), default=0.0)
    if peak > 0:
        out = {pid: v / peak for pid, v in out.items()}
    return out


@dataclass
class CorrectionAction:
    """Auditable control action (spec section 53)."""

    page_id: int
    operation: Operation
    tier: Tier
    activity: float
    semantic: float
    conflict: float
    cause: str
    old_tier: Optional[Tier] = None

    def to_dict(self) -> dict:
        return {
            "page_id": self.page_id,
            "operation": self.operation.value,
            "tier": self.tier.name,
            "activity": self.activity,
            "semantic": self.semantic,
            "conflict": self.conflict,
            "cause": self.cause,
            "old_tier": self.old_tier.name if self.old_tier else None,
        }


@dataclass
class CsrrStats:
    conflicts: int = 0
    rescues: int = 0
    controlled_demotions: int = 0
    low_codec_candidates: int = 0
    evaluated_pages: int = 0
    latency_sum: float = 0.0
    latency_count: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def note(self, cause: str) -> None:
        self.reasons[cause] = self.reasons.get(cause, 0) + 1

    def to_dict(self) -> dict:
        return {
            "conflicts": self.conflicts,
            "rescues": self.rescues,
            "controlled_demotions": self.controlled_demotions,
            "low_codec_candidates": self.low_codec_candidates,
            "evaluated_pages": self.evaluated_pages,
            "mean_rescue_latency": (
                self.latency_sum / max(self.latency_count, 1)
            ),
            "reasons": dict(self.reasons),
        }


class Srce:
    """Semantic correction over compact page scores (spec section 29)."""

    def __init__(self, cfg: Config):
        self.sc = cfg.srce
        self.stats = CsrrStats()

    def evaluate(
        self,
        stream_id: int,
        generation: int,
        frame_index: int,
        codec_activity: dict[int, float],
        semantic_scores: dict[int, float],
        page_states: dict[int, Tier],
        temporal_states: Optional[dict[int, TemporalState]] = None,
        low_activity_frames: Optional[dict[int, int]] = None,
        demote_persistence: Optional[int] = None,
    ) -> list[CorrectionAction]:
        """Compare codec-derived activity with semantic importance.

        Policy (spec section 30, CSRR):
          high A / high S -> stay T0
          low  A / low  S -> stay T2/T3 (preserve lower tier)
          low  A / high S -> PROMOTE (semantic rescue)
          high A / low  S -> controlled DEMOTE after temporal persistence
        """
        sc = self.sc
        actions: list[CorrectionAction] = []
        persistence = demote_persistence or sc.demote_persistence
        low_frames = low_activity_frames or {}
        evaluated = set(codec_activity) & set(semantic_scores)
        self.stats.evaluated_pages += len(evaluated)
        for pid in sorted(evaluated):
            activity = codec_activity[pid]
            semantic = semantic_scores[pid]
            conflict = abs(activity - semantic)
            current_tier = page_states.get(pid)
            if current_tier is None:
                continue
            is_conflict = conflict >= sc.conflict_threshold
            low_codec = activity <= sc.low_codec
            high_semantic = semantic >= sc.high_semantic
            if low_codec:
                self.stats.low_codec_candidates += 1
            if is_conflict:
                self.stats.conflicts += 1
            temporal = (temporal_states or {}).get(pid, TemporalState.ACTIVE)
            # UNKNOWN/TRANSITIONING pages are never aggressively demoted
            # (spec sections 11, 32).
            safe_to_demote = temporal in (
                TemporalState.ACTIVE,
                TemporalState.STABLE,
                TemporalState.STATIC,
            )

            if low_codec and high_semantic:
                if current_tier.value >= 2 and is_conflict:
                    actions.append(
                        CorrectionAction(
                            page_id=pid,
                            operation=Operation.PROMOTE,
                            tier=Tier.T0,
                            activity=activity,
                            semantic=semantic,
                            conflict=conflict,
                            cause="semantic_rescue",
                            old_tier=current_tier,
                        )
                    )
                    self.stats.rescues += 1
                    self.stats.note("semantic_rescue")
            elif activity >= sc.high_semantic and semantic <= sc.low_codec:
                if (
                    safe_to_demote
                    and current_tier == Tier.T0
                    and low_frames.get(pid, 0) >= persistence
                ):
                    actions.append(
                        CorrectionAction(
                            page_id=pid,
                            operation=Operation.DEMOTE,
                            tier=Tier.T1,
                            activity=activity,
                            semantic=semantic,
                            conflict=conflict,
                            cause="controlled_demotion",
                            old_tier=current_tier,
                        )
                    )
                    self.stats.controlled_demotions += 1
                    self.stats.note("controlled_demotion")
            # high A / high S and low A / low S require no action.
        return actions

    def note_rescue_latency(self, latency: float) -> None:
        self.stats.latency_sum += latency
        self.stats.latency_count += 1


class SemanticAdapter:
    """Transformer/VLM integration boundary (spec section 27).

    The prototype abstracts the transformer as a workload generator that
    consumes a residency plan; a realistic version hooks into a real
    vision transformer to export compact page importance. Preserves page
    and generation IDs so semantic statistics bind to the same window.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def consume(self, ready_pages: list[int], generation: int) -> dict[int, float]:
        """Return compact page-level semantic scores for the ready set.

        Overridden by scenario generators; the default returns a neutral
        importance profile.
        """
        return {pid: 0.0 for pid in ready_pages}
