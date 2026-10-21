"""KV Memory Controller (spec sections 20, 22-26, 51, 52).

The frozen transaction is RESERVE -> PREPARE -> COMMIT -> READY. All four
stages are generation-validated. The controller validates MKADs, checks
generation, reserves physical blocks, prepares shadow mappings, commits
atomically, emits KV_READY, and performs tier transitions. It does not
infer semantic meaning from raw motion values: MTRE and MKAD have already
converted those signals into page-specific actions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import Config
from .data_objects import (
    ApplyStatus,
    KVReady,
    MKAD,
    Mapping,
    Operation,
    PageLifecycle,
    Tier,
)
from .memory import AllocationError, PageAllocator, TieredMemoryModel


@dataclass
class ControllerCounters:
    """Prototype metrics and plausible hardware counters (spec 20)."""

    descriptors_seen: int = 0
    reservations: int = 0
    commits: int = 0
    stale_drops: int = 0
    duplicates: int = 0
    promotions: int = 0
    demotions: int = 0
    fallbacks: int = 0
    retires: int = 0
    aborted_transactions: int = 0
    ready_tokens: int = 0
    prefetches: int = 0
    prefetch_cancels: int = 0
    coalesced: int = 0
    conservative_freezes: int = 0
    cause_counts: dict[str, int] = field(default_factory=dict)

    def bump_cause(self, cause: str) -> None:
        self.cause_counts[cause] = self.cause_counts.get(cause, 0) + 1

    def to_dict(self) -> dict:
        return {
            "descriptors_seen": self.descriptors_seen,
            "reservations": self.reservations,
            "commits": self.commits,
            "stale_drops": self.stale_drops,
            "duplicates": self.duplicates,
            "promotions": self.promotions,
            "demotions": self.demotions,
            "fallbacks": self.fallbacks,
            "retires": self.retires,
            "aborted_transactions": self.aborted_transactions,
            "ready_tokens": self.ready_tokens,
            "prefetches": self.prefetches,
            "prefetch_cancels": self.prefetch_cancels,
            "coalesced": self.coalesced,
            "conservative_freezes": self.conservative_freezes,
            "cause_counts": dict(self.cause_counts),
        }


class Kmc:
    """Stateful controller core; apply() processes one descriptor.

    Returns APPLIED, STALE_DROP, DUPLICATE, RETRY, FALLBACK, or ABORT
    (spec section 51). A failed commit leaves the active mapping
    unchanged — the key invariant.
    """

    def __init__(self, cfg: Config, memory: TieredMemoryModel):
        self.cfg = cfg
        self.kc = cfg.kmc
        self.mem = memory
        self.counters = ControllerCounters()
        self.fence_counter = 0
        self.descriptor_counter = 0
        self.seen_descriptors: set[int] = set()
        self.queue_depth = 0
        self.queue_peak = 0
        self.conservative_mode = False
        self.faulted = False  # controller fault injection hook
        # Per-(stream, page) last committed generation for stale checks.
        self.committed_generation: dict[tuple[int, int], int] = {}
        self.pending_tokens: list[KVReady] = []

    # ------------------------------------------------------------------
    # Public transactional interface
    # ------------------------------------------------------------------
    def apply(self, d: MKAD) -> ApplyStatus:
        """Process one descriptor through the generation-safe protocol."""
        self.counters.descriptors_seen += 1
        self.queue_depth += 1
        self.queue_peak = max(self.queue_peak, self.queue_depth)
        try:
            if not self._validate(d):
                return ApplyStatus.FALLBACK
            if self.cfg.effective_generation_checks():
                status = self._generation_check(d)
                if status is not None:
                    return status
            if d.operation == Operation.RETIRE:
                status = self._retire(d)
            elif d.operation == Operation.PROMOTE:
                status = self._promote(d)
            elif d.operation == Operation.DEMOTE:
                status = self._demote(d)
            else:
                status = self._allocate(d)
            if status == ApplyStatus.APPLIED and d.descriptor_id > 0:
                self.seen_descriptors.add(d.descriptor_id)
            return status
        except AllocationError:
            self.counters.aborted_transactions += 1
            if self.kc.allocation_failure_mode == "fallback_t2":
                status = self._fallback_lower_tier(d)
                if status == ApplyStatus.APPLIED and d.descriptor_id > 0:
                    self.seen_descriptors.add(d.descriptor_id)
                return status
            return ApplyStatus.ABORT
        finally:
            self.queue_depth = max(0, self.queue_depth - 1)

    def take_ready_tokens(self) -> list[KVReady]:
        tokens = self.pending_tokens
        self.pending_tokens = []
        return tokens

    # ------------------------------------------------------------------
    # Validation and generation checks
    # ------------------------------------------------------------------
    def _validate(self, d: MKAD) -> bool:
        if self.faulted and self.kc_enforce_state_freeze():
            # State freeze rule: preserve last committed mapping on
            # control-plane error (spec section 32).
            self.counters.conservative_freezes += 1
            return False
        if not d.valid:
            self.counters.fallbacks += 1
            self.counters.bump_cause("invalid_descriptor")
            return False
        if d.page_count <= 0:
            self.counters.fallbacks += 1
            self.counters.bump_cause("bad_page_range")
            return False
        if d.page_start < 0 or d.page_start + d.page_count > len(self.mem.pages):
            self.counters.fallbacks += 1
            self.counters.bump_cause("page_range_out_of_bounds")
            return False
        if d.target_tier == Tier.T3 and (self.kc.tier_levels < 4 or not self.kc.enable_t3):
            self.counters.fallbacks += 1
            self.counters.bump_cause("unsupported_tier")
            return False
        return True

    def kc_enforce_state_freeze(self) -> bool:
        return self.cfg.fallback.state_freeze_on_fault

    def _generation_check(self, d: MKAD) -> Optional[ApplyStatus]:
        """Stale descriptor rejection (spec sections 23, 26).

        Any descriptor older than the current page generation is a
        STALE_DROP. Duplicates are idempotent (DUPLICATE status, no
        double allocation).
        """
        if d.descriptor_id and d.descriptor_id in self.seen_descriptors:
            self.counters.duplicates += 1
            return ApplyStatus.DUPLICATE
        for pid in range(d.page_start, d.page_start + d.page_count):
            key = (d.stream_id, pid)
            current = self.committed_generation.get(key, -1)
            if d.generation < current:
                self.counters.stale_drops += 1
                return ApplyStatus.STALE_DROP
            if d.generation == current and d.operation in (
                Operation.ALLOCATE,
                Operation.PREFETCH,
            ):
                # Already owned by this generation: idempotent.
                self.counters.duplicates += 1
                return ApplyStatus.DUPLICATE
        return None

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------
    def _allocate(self, d: MKAD) -> ApplyStatus:
        pages = list(range(d.page_start, d.page_start + d.page_count))
        emergency_ok = d.operation == Operation.PROMOTE or d.prefetch is False
        blocks = self.mem.allocator.reserve(d.target_tier, len(pages), emergency_ok)
        self.counters.reservations += 1
        # PREPARE: fill shadow mappings for all expected entries.
        shadow: dict[int, Mapping] = {}
        for pid, block in zip(pages, blocks):
            shadow[pid] = Mapping(
                logical_page=pid,
                physical_block=block,
                tier=d.target_tier,
                generation=d.generation,
                representation=self.mem.representation[d.target_tier],
                state=PageLifecycle.RESERVED,
            )
        # Verify all expected entries before the ordering fence.
        if len(shadow) != len(pages):
            self.mem.allocator.release(blocks, d.target_tier)
            self.counters.aborted_transactions += 1
            return ApplyStatus.ABORT
        # COMMIT: publish atomically; active table never half-updated.
        self.fence_counter += 1
        for pid, mapping in shadow.items():
            entry = self.mem.pages[pid]
            self._release_old(entry, new_tier=d.target_tier)
            entry.mapping = mapping
            entry.tier = d.target_tier
            entry.generation = d.generation
            entry.lifecycle = PageLifecycle.MAPPED
            self.committed_generation[(d.stream_id, pid)] = d.generation
        self.counters.commits += 1
        self.counters.bump_cause(d.cause)
        if d.prefetch:
            self.counters.prefetches += 1
        # READY: completion token only after commit (spec section 52).
        self.mem.record_traffic("kv_write", len(pages))
        self.pending_tokens.append(
            KVReady(
                stream_id=d.stream_id,
                generation=d.generation,
                page_start=d.page_start,
                page_count=d.page_count,
                fence_id=self.fence_counter,
            )
        )
        self.counters.ready_tokens += 1
        for pid in pages:
            entry = self.mem.pages[pid]
            entry.lifecycle = PageLifecycle.READY
            entry.ready = True
            assert entry.mapping is not None
            entry.mapping.state = PageLifecycle.READY
        return ApplyStatus.APPLIED

    def _promote(self, d: MKAD) -> ApplyStatus:
        """Promotion may allocate from the emergency pool (spec 18, 51)."""
        status = self._retarget_tier(d, allow_emergency=True)
        if status == ApplyStatus.APPLIED:
            self.counters.promotions += 1
            self.counters.bump_cause(d.cause or "semantic_rescue")
        return status

    def _demote(self, d: MKAD) -> ApplyStatus:
        """Demotion frees the old representation only after commit."""
        status = self._retarget_tier(d, allow_emergency=False)
        if status == ApplyStatus.APPLIED:
            self.counters.demotions += 1
            self.counters.bump_cause(d.cause or "tier_maintenance")
        return status

    def _retarget_tier(self, d: MKAD, allow_emergency: bool) -> ApplyStatus:
        pages = list(range(d.page_start, d.page_start + d.page_count))
        # Every page must be resident and owned by this generation or newer
        # semantics applied to current ownership only.
        for pid in pages:
            entry = self.mem.pages[pid]
            if entry.lifecycle not in (
                PageLifecycle.MAPPED,
                PageLifecycle.READY,
                PageLifecycle.IN_USE,
            ):
                return ApplyStatus.RETRY
            key = (d.stream_id, pid)
            current = self.committed_generation.get(key, -1)
            if self.cfg.effective_generation_checks() and d.generation < current:
                self.counters.stale_drops += 1
                return ApplyStatus.STALE_DROP
        old_tier = self.mem.pages[pages[0]].tier
        assert old_tier is not None
        if old_tier == d.target_tier:
            self.counters.duplicates += 1
            return ApplyStatus.DUPLICATE
        if not self._tier_transition_valid(old_tier, d.target_tier):
            return ApplyStatus.ABORT
        new_blocks = self.mem.allocator.reserve(
            d.target_tier, len(pages), emergency_ok=allow_emergency
        )
        old_blocks = [self.mem.pages[pid].mapping.physical_block for pid in pages]  # type: ignore[union-attr]
        # Commit tier move atomically.
        self.fence_counter += 1
        for pid, block in zip(pages, new_blocks):
            entry = self.mem.pages[pid]
            old_mapping = entry.mapping
            assert old_mapping is not None
            moved = Mapping(
                logical_page=pid,
                physical_block=block,
                tier=d.target_tier,
                generation=entry.generation,
                representation=self.mem.representation[d.target_tier],
                state=old_mapping.state,
            )
            # Traffic cost of promotion/demotion (spec section 19).
            direction = "promotion" if d.target_tier.value < old_tier.value else "demotion"
            self.mem.record_traffic(direction, 1)
            # Release old physical block after commit of new state.
            self.mem.allocator.release([old_mapping.physical_block], old_tier)
            entry.mapping = moved
            entry.tier = d.target_tier
        self.counters.commits += 1
        return ApplyStatus.APPLIED

    def _retire(self, d: MKAD) -> ApplyStatus:
        pages = list(range(d.page_start, d.page_start + d.page_count))
        for pid in pages:
            entry = self.mem.pages[pid]
            key = (d.stream_id, pid)
            if self.cfg.effective_generation_checks():
                current = self.committed_generation.get(key, -1)
                if d.generation < current:
                    self.counters.stale_drops += 1
                    return ApplyStatus.STALE_DROP
            if entry.mapping is not None:
                old_tier = entry.tier
                assert old_tier is not None
                self.mem.allocator.release([entry.mapping.physical_block], old_tier)
            entry.mapping = None
            entry.tier = None
            entry.lifecycle = PageLifecycle.RETIRED
            entry.ready = False
            self.committed_generation[key] = max(
                self.committed_generation.get(key, -1), d.generation
            )
        self.counters.retires += 1
        self.counters.commits += 1
        return ApplyStatus.APPLIED

    def _release_old(self, entry, new_tier: Tier) -> None:
        """Release the previous physical mapping during reallocation."""
        if entry.mapping is not None and entry.tier is not None:
            self.mem.allocator.release([entry.mapping.physical_block], entry.tier)

    def _tier_transition_valid(self, old: Tier, new: Tier) -> bool:
        if not self.kc.enforce_generation_checks:
            return True
        from .data_objects import VALID_TIER_TRANSITIONS  # noqa: PLC0415

        return new in VALID_TIER_TRANSITIONS[old]

    def _fallback_lower_tier(self, d: MKAD) -> ApplyStatus:
        """Allocation failure fallback: conservative lower tier (spec 18)."""
        self.counters.fallbacks += 1
        self.counters.bump_cause("allocation_failure_fallback")
        fb = MKAD(
            stream_id=d.stream_id,
            generation=d.generation,
            page_start=d.page_start,
            page_count=d.page_count,
            activity=d.activity,
            target_tier=Tier.T2,
            operation=Operation.ALLOCATE,
            confidence=d.confidence,
            frame_index=d.frame_index,
            cause="confidence_fallback",
            descriptor_id=0,  # bypass duplicate check for fallback path
        )
        if d.target_tier == Tier.T2:
            return ApplyStatus.ABORT
        self.kc_fallback_count()
        try:
            return self._allocate(fb)
        except AllocationError:
            self.counters.aborted_transactions += 1
            return ApplyStatus.ABORT

    def kc_fallback_count(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Backpressure (spec section 39)
    # ------------------------------------------------------------------
    def note_queue_pressure(self, depth: int) -> str:
        """At critical depth, enter conservative mode: preserve last
        committed mapping rather than dropping random descriptors."""
        self.queue_depth = depth
        self.queue_peak = max(self.queue_peak, depth)
        if depth >= self.kc.critical_queue_depth:
            self.conservative_mode = True
            self.counters.conservative_freezes += 1
            return "conservative_freeze"
        self.conservative_mode = False
        return "normal"

    def coalesce(self, descriptors: list[MKAD]) -> list[MKAD]:
        """Coalesce adjacent descriptors with same generation, operation,
        and tier (spec sections 17, 39)."""
        if not self.kc.coalescing_enabled or len(descriptors) <= 1:
            return descriptors
        out: list[MKAD] = []
        for d in sorted(descriptors, key=lambda x: (x.generation, x.page_start)):
            if out:
                prev = out[-1]
                if (
                    prev.generation == d.generation
                    and prev.operation == d.operation
                    and prev.target_tier == d.target_tier
                    and prev.stream_id == d.stream_id
                    and prev.page_start + prev.page_count == d.page_start
                ):
                    prev.page_count += d.page_count
                    self.counters.coalesced += 1
                    continue
            out.append(d)
        return out

    def invariants(self) -> list[str]:
        """Required invariants (spec sections 52, 63): no duplicate
        physical ownership, no stale mutation, ready implies committed,
        non-negative free capacity."""
        problems: list[str] = []
        owner: dict[int, int] = {}
        for pid, entry in self.mem.pages.items():
            m = entry.mapping
            if m is None:
                continue
            if m.physical_block in owner:
                problems.append(
                    f"duplicate physical ownership block {m.physical_block}: "
                    f"pages {owner[m.physical_block]} and {pid}"
                )
            owner[m.physical_block] = pid
            if entry.ready and m.state not in (
                PageLifecycle.MAPPED,
                PageLifecycle.READY,
                PageLifecycle.IN_USE,
            ):
                problems.append(f"ready without commit on page {pid}")
            if entry.ready and entry.generation < 0:
                problems.append(f"stale mutation risk on page {pid}")
        if len(self.mem.allocator.free) < 0:
            problems.append("negative free capacity")
        return problems
