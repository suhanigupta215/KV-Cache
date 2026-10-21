"""MKAD builder (spec sections 17, 18).

Converts page-level reconciliation output into controller descriptors.
Allocation decision policy: HIGH/ACTIVE -> T0 HOT, MEDIUM/STABLE -> T1
WARM, LOW/STATIC -> T2 COLD, UNKNOWN and TRANSITIONING -> conservative
T0. Coalescing of contiguous same-decision ranges reduces control-plane
overhead; one-page-per-descriptor mode is available for observability.
"""
from __future__ import annotations

from typing import Optional, Sequence

from .config import Config
from .data_objects import MKAD, Operation, TemporalState, Tier
from .mtre import PageActivity


class MkadBuilder:
    """Emits versioned descriptors with explicit target tiers."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.temporal = cfg.temporal
        self.kmc = cfg.kmc
        self._next_id = 1

    def decide_tier(
        self,
        activity: float,
        state: TemporalState,
        confidence: float,
        confidence_threshold: float,
    ) -> Tier:
        """Provisional allocation decision (spec section 18).

        UNKNOWN and TRANSITIONING use conservative T0 behavior. Missing
        metadata is never interpreted as zero motion.
        """
        if state in (TemporalState.UNKNOWN, TemporalState.TRANSITIONING):
            return Tier.T0
        low = self.temporal.low_threshold
        high = self.temporal.high_threshold
        if activity >= high:
            return Tier.T0
        if activity >= low:
            return Tier.T1
        return Tier.T2

    def build(
        self,
        stream_id: int,
        generation: int,
        frame_index: int,
        pages: Sequence[PageActivity],
        states: dict[int, TemporalState],
        confidence_threshold: float = 0.25,
        coalesce: bool = True,
        prefetch_pages: int = 0,
        total_pages: int = 0,
    ) -> list[MKAD]:
        """Group pages into descriptors; optionally coalesce ranges."""
        descriptors: list[MKAD] = []
        ordered = sorted(pages, key=lambda p: p.page_id)
        i = 0
        while i < len(ordered):
            page = ordered[i]
            tier = self.decide_tier(
                page.activity,
                states.get(page.page_id, TemporalState.ACTIVE),
                page.confidence,
                confidence_threshold,
            )
            state = states.get(page.page_id, TemporalState.ACTIVE)
            j = i + 1
            if coalesce:
                while j < len(ordered):
                    nxt = ordered[j]
                    nxt_tier = self.decide_tier(
                        nxt.activity,
                        states.get(nxt.page_id, TemporalState.ACTIVE),
                        nxt.confidence,
                        confidence_threshold,
                    )
                    nxt_state = states.get(nxt.page_id, TemporalState.ACTIVE)
                    if (
                        nxt_tier == tier
                        and nxt.page_id == ordered[j - 1].page_id + 1
                        and nxt_state == state
                    ):
                        j += 1
                    else:
                        break
            count = j - i
            group = ordered[i:j]
            activity = sum(p.activity for p in group) / len(group)
            confidence = sum(p.confidence for p in group) / len(group)
            descriptors.append(
                MKAD(
                    stream_id=stream_id,
                    generation=generation,
                    page_start=page.page_id,
                    page_count=count,
                    activity=activity,
                    target_tier=tier,
                    operation=Operation.ALLOCATE,
                    confidence=confidence,
                    frame_index=frame_index,
                    cause="codec",
                    descriptor_id=self._next_id,
                )
            )
            self._next_id += 1
            i = j
        # Prefetch descriptors are lower priority than correctness work
        # (spec section 40); they are speculative, cancelable, and
        # clamped to the real logical page range.
        if prefetch_pages > 0:
            last = ordered[-1].page_id if ordered else 0
            avail = max(0, (total_pages if total_pages else last + 1) - (last + 1))
            count = min(prefetch_pages, avail)
            if count > 0:
                descriptors.append(
                    MKAD(
                        stream_id=stream_id,
                        generation=generation,
                        page_start=last + 1,
                        page_count=count,
                        activity=0.0,
                        target_tier=Tier.T0,
                        operation=Operation.PREFETCH,
                        confidence=1.0,
                        frame_index=frame_index,
                        prefetch=True,
                        priority=5,
                        cause="transition_safety",
                        descriptor_id=self._next_id,
                    )
                )
                self._next_id += 1
        return descriptors

    def correction_descriptor(
        self,
        stream_id: int,
        generation: int,
        frame_index: int,
        page_id: int,
        operation: Operation,
        tier: Tier,
        activity: float,
        confidence: float,
        cause: str,
    ) -> MKAD:
        """Generation-matched correction descriptor (spec section 53).

        Corrections never bypass the normal KMC commit path.
        """
        descriptor = MKAD(
            stream_id=stream_id,
            generation=generation,
            page_start=page_id,
            page_count=1,
            activity=activity,
            target_tier=tier,
            operation=operation,
            confidence=confidence,
            frame_index=frame_index,
            cause=cause,
            descriptor_id=self._next_id,
        )
        self._next_id += 1
        return descriptor
