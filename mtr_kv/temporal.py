"""Temporal Motion State Engine (spec sections 11, 49).

Persistence, hysteresis, and explicit temporal states: ACTIVE,
TRANSITIONING, STABLE, STATIC, UNKNOWN. UNKNOWN is entered on invalid
metadata or low confidence; TRANSITIONING on scene-divergence evidence.
State transitions override aggressive demotion rules. The same logic
maps later to a hardware FSM with counters and comparators.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import Config, TemporalConfig
from .data_objects import TemporalState


@dataclass
class TemporalRecord:
    """Per-region temporal state (spec section 49)."""

    state: TemporalState = TemporalState.ACTIVE
    low_counter: int = 0
    high_counter: int = 0
    last_activity: float = 1.0
    last_confidence: float = 1.0
    last_valid: bool = True
    last_transition_frame: int = -1
    state_switches: int = 0
    frame_history: list[tuple[int, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "low_counter": self.low_counter,
            "high_counter": self.high_counter,
            "state_switches": self.state_switches,
        }


class TemporalStateEngine:
    """Tracks temporal state per key (token region, page, or group)."""

    def __init__(self, cfg: Config):
        self.tc: TemporalConfig = cfg.temporal
        self.confidence_threshold = cfg.motion.confidence_threshold
        self.records: dict[str, TemporalRecord] = {}

    def record(self, key: str) -> TemporalRecord:
        if key not in self.records:
            self.records[key] = TemporalRecord()
        return self.records[key]

    def update(
        self,
        key: str,
        activity: float,
        confidence: float,
        metadata_valid: bool,
        frame_index: int,
        scene_transition: bool = False,
    ) -> TemporalState:
        """Advance one frame of temporal state for a region.

        Guards (spec section 45): every transition is explicit so the
        simulator can catch illegal state changes.
        """
        tc = self.tc
        rec = self.record(key)
        prev = rec.state
        rec.last_activity = activity
        rec.last_confidence = confidence
        rec.last_valid = metadata_valid

        if not metadata_valid or confidence < self.confidence_threshold:
            new_state = TemporalState.UNKNOWN
        elif scene_transition:
            new_state = TemporalState.TRANSITIONING
            rec.last_transition_frame = frame_index
        else:
            low = activity <= tc.low_threshold
            high = activity >= tc.high_threshold
            if low:
                rec.low_counter += 1
                rec.high_counter = 0
            elif high:
                # Hysteresis: only above the higher threshold exits the
                # low-activity path (spec section 11).
                rec.low_counter = 0
                rec.high_counter += 1
                new_state = TemporalState.ACTIVE
                self._transition(rec, new_state, frame_index)
                rec.state = new_state
                return new_state
            else:
                # Middle band: hysteresis holds current low-path progress,
                # state becomes STABLE while counters persist.
                new_state = (
                    rec.state
                    if rec.state in (TemporalState.STABLE, TemporalState.STATIC)
                    else TemporalState.STABLE
                )
                if rec.state not in (TemporalState.STABLE, TemporalState.STATIC):
                    self._transition(rec, new_state, frame_index)
                rec.state = new_state
                return new_state

            if rec.low_counter >= tc.persistence_frames:
                new_state = (
                    TemporalState.STATIC
                    if rec.low_counter >= tc.stable_frames
                    else TemporalState.STABLE
                )
            else:
                # A low-activity sample enters the low path immediately;
                # persistence controls only when it may become STATIC.
                new_state = TemporalState.STABLE

        self._transition(rec, new_state, frame_index)
        rec.state = new_state
        return new_state

    def _transition(
        self,
        rec: TemporalRecord,
        new_state: TemporalState,
        frame_index: int,
    ) -> None:
        if new_state != rec.state:
            rec.state_switches += 1
        rec.frame_history.append((frame_index, new_state.value))
        if len(rec.frame_history) > 512:
            rec.frame_history.pop(0)

    def reset_on_iframe(self, key: str, frame_index: int) -> None:
        """I-frames reset temporal state and widen residency (spec 33)."""
        rec = self.record(key)
        rec.low_counter = 0
        rec.high_counter = 0
        self._transition(rec, TemporalState.ACTIVE, frame_index)
        rec.state = TemporalState.ACTIVE

    def state_of(self, key: str) -> TemporalState:
        return self.record(key).state

    def persistence(self, key: str) -> float:
        """Normalized persistence in [0, 1] for the activity model."""
        rec = self.record(key)
        if rec.state in (TemporalState.STATIC, TemporalState.STABLE):
            return 1.0
        if rec.state == TemporalState.UNKNOWN:
            return 0.0
        return min(rec.low_counter / max(self.tc.persistence_frames, 1), 1.0)

    def frame_state_fraction(self) -> dict[str, float]:
        """Fraction of frames spent in each state (spec section 11)."""
        counts: dict[str, int] = {}
        total = 0
        for rec in self.records.values():
            for _, state in rec.frame_history:
                counts[state] = counts.get(state, 0) + 1
                total += 1
        if total == 0:
            return {s.value: 0.0 for s in TemporalState}
        return {k: v / total for k, v in sorted(counts.items())}

    def total_state_switches(self) -> int:
        return sum(r.state_switches for r in self.records.values())
