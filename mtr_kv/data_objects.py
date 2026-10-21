"""Frozen data objects for MTR-KV (spec sections 05, 48).

Core objects: MotionBlock, TokenRegion, PageState, Tier, MKAD, Mapping,
Reservation, Event, KVReady, SemanticUpdate. All are serializable for
trace replay. Control fields are minimal; debug fields are added without
changing the control contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Tier(Enum):
    """Multi-tier KV residency model (spec section 19)."""

    T0 = 0  # HOT: preferred precision, fast memory
    T1 = 1  # WARM: compressed or reduced precision
    T2 = 2  # COLD: lower bandwidth or deferred
    T3 = 3  # RECONSTRUCT: no hot page; restorable on demand


# Explicit safe tier transitions (no partial or illegal moves; spec section 63).
VALID_TIER_TRANSITIONS = {
    Tier.T0: {Tier.T0, Tier.T1, Tier.T2, Tier.T3},
    Tier.T1: {Tier.T0, Tier.T1, Tier.T2, Tier.T3},
    Tier.T2: {Tier.T0, Tier.T1, Tier.T2, Tier.T3},
    Tier.T3: {Tier.T0, Tier.T1, Tier.T2, Tier.T3},
}


class TemporalState(Enum):
    """Temporal motion states (spec section 11)."""

    ACTIVE = "ACTIVE"
    TRANSITIONING = "TRANSITIONING"
    STABLE = "STABLE"
    STATIC = "STATIC"
    UNKNOWN = "UNKNOWN"


class PageLifecycle(Enum):
    """Page lifecycle FSM (spec section 45)."""

    FREE = "FREE"
    RESERVED = "RESERVED"
    MAPPED = "MAPPED"
    READY = "READY"
    IN_USE = "IN_USE"
    RETIRED = "RETIRED"
    STALE = "STALE"
    RECLAIM = "RECLAIM"


class Operation(Enum):
    """MKAD operation codes (spec section 17)."""

    ALLOCATE = "ALLOCATE"
    PROMOTE = "PROMOTE"
    DEMOTE = "DEMOTE"
    RETIRE = "RETIRED"
    PREFETCH = "PREFETCH"


class ApplyStatus(Enum):
    """KMC apply() results (spec section 51)."""

    APPLIED = "APPLIED"
    STALE_DROP = "STALE_DROP"
    DUPLICATE = "DUPLICATE"
    RETRY = "RETRY"
    FALLBACK = "FALLBACK"
    ABORT = "ABORT"


class FrameType(Enum):
    """Frame type is a first-class metadata field (spec section 33)."""

    I = "I"
    P = "P"
    B = "B"
    NONE = "NONE"


@dataclass
class MotionBlock:
    """Canonical decoder descriptor (spec sections 09, 10).

    rect is in normalized source coordinates: (x, y, w, h) in pixels of
    the original source frame. motion is a bounded magnitude, residual a
    bounded energy score, confidence stays independent from activity.
    """

    stream_id: int
    generation: int
    frame_index: int
    rect: tuple[float, float, float, float]
    motion: float
    residual: float
    confidence: float
    frame_type: FrameType
    metadata_valid: bool = True
    reference_frame_id: Optional[int] = None
    # Raw values retained for ablation/debug (spec section 10).
    raw_dx: float = 0.0
    raw_dy: float = 0.0
    raw_block_size: Optional[tuple[int, int]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "generation": self.generation,
            "frame_index": self.frame_index,
            "rect": list(self.rect),
            "motion": self.motion,
            "residual": self.residual,
            "confidence": self.confidence,
            "frame_type": self.frame_type.value,
            "metadata_valid": self.metadata_valid,
        }


@dataclass
class TokenRegion:
    """Source-space footprint of one visual token (spec section 05)."""

    token_id: int
    rect: tuple[float, float, float, float]
    page_id: int  # logical page: token_index // page_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "rect": list(self.rect),
            "page_id": self.page_id,
        }


@dataclass
class MKAD:
    """Motion-Keyed Allocation Descriptor (spec section 17).

    The presence of target_tier is deliberate: the codec-to-token pipeline
    explicitly controls residency. Versioned for future embodiments.
    """

    stream_id: int
    generation: int
    page_start: int
    page_count: int
    activity: float
    target_tier: Tier
    operation: Operation
    confidence: float
    frame_index: int = -1
    token_range: Optional[tuple[int, int]] = None
    valid: bool = True
    priority: int = 0
    prefetch: bool = False
    cause: str = "codec"  # codec | semantic_rescue | transition_safety |
    # confidence_fallback | tier_maintenance
    version: int = 1
    layer_ids: Optional[list[int]] = None
    fence_id: int = -1
    descriptor_id: int = -1  # unique id for duplicate detection

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "generation": self.generation,
            "page_start": self.page_start,
            "page_count": self.page_count,
            "activity": self.activity,
            "target_tier": self.target_tier.name,
            "operation": self.operation.value,
            "confidence": self.confidence,
            "frame_index": self.frame_index,
            "valid": self.valid,
            "prefetch": self.prefetch,
            "cause": self.cause,
            "descriptor_id": self.descriptor_id,
        }


@dataclass
class Mapping:
    """Page-table mapping entry (spec section 22).

    Generation is part of the mapping so stale events can be rejected
    locally. A page can move tiers without changing logical identity.
    """

    logical_page: int
    physical_block: int
    tier: Tier
    generation: int
    representation: str  # e.g. fp16, int8, entropy, none
    state: PageLifecycle

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_page": self.logical_page,
            "physical_block": self.physical_block,
            "tier": self.tier.name,
            "generation": self.generation,
            "representation": self.representation,
            "state": self.state.value,
        }


@dataclass
class Reservation:
    """Physical allocation stage before mapping publication (spec 24)."""

    logical_page: int
    physical_block: int
    tier: Tier
    generation: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_page": self.logical_page,
            "physical_block": self.physical_block,
            "tier": self.tier.name,
            "generation": self.generation,
        }


@dataclass
class KVReady:
    """Completion token: readiness only exists after commit (spec 52)."""

    stream_id: int
    generation: int
    page_start: int
    page_count: int
    fence_id: int
    status: str = "READY"

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "generation": self.generation,
            "page_start": self.page_start,
            "page_count": self.page_count,
            "fence_id": self.fence_id,
            "status": self.status,
        }


@dataclass
class SemanticUpdate:
    """Generation-matched semantic correction (spec sections 29, 53)."""

    stream_id: int
    generation: int
    logical_page: int
    activity: float
    semantic_score: float
    conflict: float
    operation: Operation
    target_tier: Tier
    cause: str
    frame_index: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "generation": self.generation,
            "logical_page": self.logical_page,
            "activity": self.activity,
            "semantic_score": self.semantic_score,
            "conflict": self.conflict,
            "operation": self.operation.value,
            "target_tier": self.target_tier.name,
            "cause": self.cause,
        }


@dataclass
class Event:
    """Deterministic event for the async engine (spec section 54)."""

    time: float
    priority: int  # lower runs first; correctness work beats prefetch
    seq: int  # tie-break for determinism
    type: str  # DECODE_READY, MKAD_READY, PAGE_RESERVED, PAGE_COMMITTED,
    # KV_READY, SEMANTIC_UPDATE, CORRECTION_COMMITTED
    stream_id: int
    generation: int
    payload: dict[str, Any] = field(default_factory=dict)

    def sort_key(self) -> tuple:
        return (self.time, self.priority, self.seq, self.type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "priority": self.priority,
            "seq": self.seq,
            "type": self.type,
            "stream_id": self.stream_id,
            "generation": self.generation,
            "payload": self.payload,
        }


def bytes_per_page(
    page_tokens: int,
    layers: int,
    kv_heads: int,
    head_dim: int,
    bytes_per_element: float,
) -> int:
    """KV page footprint (spec section 36): page_tokens * L * K * d * b * 2."""
    return int(page_tokens * layers * kv_heads * head_dim * bytes_per_element * 2)
