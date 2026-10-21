"""Predictive Scene-Transition Handling (spec sections 12, 40).

Divergence metric D_f = mean_i |A_i^f - A_i^(f-1)| combined with residual
jumps, frame-type changes, and high-change fraction produces a transition
score. Above threshold, the controller enters TRANSITIONING, expands the
active set, and optionally prefetches a T0 safety margin. Correctness
must not depend on this predictor: a disabled predictor leaves the
system safe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import Config, SceneConfig


@dataclass
class PredictionOutcome:
    transition_score: float
    transitioning: bool
    prefetch_pages: int = 0
    divergence: float = 0.0
    residual_jump: float = 0.0
    frame_type_change: bool = False
    high_change_fraction: float = 0.0

    def to_dict(self) -> dict:
        return {
            "transition_score": self.transition_score,
            "transitioning": self.transitioning,
            "prefetch_pages": self.prefetch_pages,
        }


@dataclass
class ScenePredictorStats:
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    history: list[PredictionOutcome] = field(default_factory=list)

    def confusion(self) -> dict[str, float]:
        tp, fp = self.true_positives, self.false_positives
        tn, fn = self.true_negatives, self.false_negatives
        total = max(tp + fp + tn + fn, 1)
        return {
            "accuracy": (tp + tn) / total,
            "false_positive_rate": fp / max(fp + tn, 1),
            "false_negative_rate": fn / max(fn + tp, 1),
            "precision": tp / max(tp + fp, 1),
        }


class ScenePredictor:
    """Generates a transition score from frame-to-frame divergence."""

    def __init__(self, cfg: Config):
        self.sc: SceneConfig = cfg.scene
        self.prev_activity: Optional[list[float]] = None
        self.prev_residual: Optional[float] = None
        self.prev_frame_type: Optional[str] = None
        self.stats = ScenePredictorStats()

    def reset(self) -> None:
        self.prev_activity = None
        self.prev_residual = None
        self.prev_frame_type = None

    def update(
        self,
        activity_per_page: list[float],
        mean_residual: float,
        frame_type: str,
        ground_truth_cut: bool = False,
    ) -> PredictionOutcome:
        """Update predictor with current frame statistics.

        ground_truth_cut is optional and used only for confusion-matrix
        metrics; the policy itself never depends on it.
        """
        if not self.sc.enabled:
            outcome = PredictionOutcome(transition_score=0.0, transitioning=False)
            if ground_truth_cut:
                self.stats.false_negatives += 1
            else:
                self.stats.true_negatives += 1
            self.stats.history.append(outcome)
            return outcome

        if self.prev_activity is None:
            divergence = 0.0
            high_change_fraction = 0.0
        else:
            n = min(len(self.prev_activity), len(activity_per_page))
            if n == 0:
                divergence = 0.0
                high_change_fraction = 0.0
            else:
                diffs = [
                    abs(activity_per_page[i] - self.prev_activity[i])
                    for i in range(n)
                ]
                divergence = sum(diffs) / n
                high_change_fraction = (
                    sum(1 for d in diffs if d > self.sc.threshold) / n
                )

        residual_jump = 0.0
        if self.prev_residual is not None:
            residual_jump = abs(mean_residual - self.prev_residual)
        frame_type_change = (
            self.prev_frame_type is not None and frame_type != self.prev_frame_type
        )

        score = (
            divergence
            + self.sc.residual_jump * (residual_jump > self.sc.residual_jump)
            + self.sc.frame_type_change_weight * frame_type_change
            + self.sc.high_change_fraction * (high_change_fraction >= self.sc.high_change_fraction)
        )
        # Scores are derived from floating-point differences.  Treat a
        # value mathematically equal to the configured threshold as a
        # transition despite representational roundoff.
        transitioning = score + 1e-12 >= self.sc.threshold

        if ground_truth_cut and transitioning:
            self.stats.true_positives += 1
        elif ground_truth_cut:
            self.stats.false_negatives += 1
        elif transitioning:
            self.stats.false_positives += 1
        else:
            self.stats.true_negatives += 1

        outcome = PredictionOutcome(
            transition_score=score,
            transitioning=transitioning,
            prefetch_pages=self.sc.prefetch_margin if transitioning else 0,
            divergence=divergence,
            residual_jump=residual_jump,
            frame_type_change=frame_type_change,
            high_change_fraction=high_change_fraction,
        )
        self.stats.history.append(outcome)

        self.prev_activity = list(activity_per_page)
        self.prev_residual = mean_residual
        self.prev_frame_type = frame_type
        return outcome
