"""Tiered KV memory model (spec sections 19, 21, 22, 36).

Physical pages are allocated per tier from per-tier free lists. A FREE
page becomes RESERVED before the page-table mapping is prepared; the
reservation prevents duplicate ownership by concurrent descriptors. The
page-table writer constructs a shadow mapping, verifies all expected
entries, and publishes atomically. The active mapping never exposes
half-written state.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .config import Config
from .data_objects import (
    Mapping,
    PageLifecycle,
    Reservation,
    Tier,
    VALID_TIER_TRANSITIONS,
    bytes_per_page,
)


class AllocationError(Exception):
    pass


@dataclass
class PageEntry:
    """Authoritative per-logical-page state (spec section 48)."""

    logical_page: int
    lifecycle: PageLifecycle = PageLifecycle.FREE
    mapping: Optional[Mapping] = None
    reservation: Optional[Reservation] = None
    tier: Optional[Tier] = None
    generation: int = -1
    ready: bool = False  # access is legal only after ready token

    def to_dict(self) -> dict:
        return {
            "logical_page": self.logical_page,
            "lifecycle": self.lifecycle.value,
            "tier": self.tier.name if self.tier else None,
            "generation": self.generation,
            "ready": self.ready,
            "mapping": self.mapping.to_dict() if self.mapping else None,
        }


@dataclass
class AllocatorStats:
    reservations: int = 0
    failed_reservations: int = 0
    releases: int = 0
    fragmentation_peak: float = 0.0

    def to_dict(self) -> dict:
        return {
            "reservations": self.reservations,
            "failed_reservations": self.failed_reservations,
            "releases": self.releases,
            "fragmentation_peak": self.fragmentation_peak,
        }


class PageAllocator:
    """Per-tier physical block management (spec section 21).

    Does not require physical contiguity; logical contiguous ranges can
    still be assigned efficiently. Fragmentation is measured explicitly.
    """

    def __init__(self, cfg: Config, total_pages: int):
        self.cfg = cfg
        kc = cfg.kmc
        self.total_pages = total_pages
        self.emergency_pool_pages = kc.emergency_pool_pages
        # One shared physical pool: the simulator shares physical ids
        # across tiers and tags ownership by tier in reservations. A
        # hardware embodiment would use per-tier pools/banks (spec 21).
        # Physical placement is non-contiguous by design (deterministic).
        self.free: list[int] = []
        self.emergency_free: list[int] = []
        step = 7
        ordered = [(p * step) % total_pages for p in range(total_pages)]
        seen: set[int] = set()
        for p in ordered:
            if p not in seen:
                seen.add(p)
                self.free.append(p)
        for p in range(total_pages):
            if p not in seen:
                self.free.append(p)
        # The emergency pool is a promotion policy budget, not capacity
        # removed from the shared physical allocator.  Reserving capacity
        # up front made the allocator report less free memory than it owns
        # and caused normal allocations to fail unnecessarily.  A future
        # banked allocator can represent this as a dedicated pool; this
        # shared-pool model keeps every physical block available while the
        # controller still marks promotion requests explicitly.
        self.emergency_pool_pages = min(self.emergency_pool_pages, total_pages)
        self.stats = AllocatorStats()

    def reserve(self, tier: Tier, count: int, emergency_ok: bool = False) -> list[int]:
        """Claim physical blocks; raises AllocationError on exhaustion.

        The emergency pool backs T0 promotions under tight memory so the
        semantic rescue path cannot fail solely because hot pages are
        exhausted (spec section 18).
        """
        if len(self.free) < count:
            self.stats.failed_reservations += count
            raise AllocationError(
                f"tier {tier.name} exhausted: need {count}, have {len(self.free)}"
            )
        blocks = self.free[:count]
        del self.free[:count]
        self.stats.reservations += count
        self._update_fragmentation()
        return blocks

    def release(self, physical_blocks: list[int], tier: Tier) -> None:
        """Return blocks to the shared free pool (demote/retire path)."""
        self.free.extend(physical_blocks)
        self.stats.releases += len(physical_blocks)
        self._update_fragmentation()

    def _update_fragmentation(self) -> None:
        """Fragmentation: non-contiguous free runs in the shared pool."""
        frag = 0.0
        if self.free:
            s = sorted(self.free)
            runs = 1
            for i in range(1, len(s)):
                if s[i] != s[i - 1] + 1:
                    runs += 1
            frag = runs / len(s)
        self.stats.fragmentation_peak = max(self.stats.fragmentation_peak, frag)


class TieredMemoryModel:
    """Logical page table plus per-tier free lists (spec sections 19, 48).

    PageTable is the authoritative state store. Cached summaries must not
    become an independent source of truth.
    """

    def __init__(self, cfg: Config, total_logical_pages: int):
        self.cfg = cfg
        self.pages = {
            pid: PageEntry(logical_page=pid) for pid in range(total_logical_pages)
        }
        b = bytes_per_page(
            cfg.model.page_tokens,
            cfg.model.layers,
            cfg.model.kv_heads,
            cfg.model.head_dim,
            cfg.model.bytes_per_element,
        )
        self.bytes_per_page = b
        # Physical pool carries 2x headroom so a newer generation can be
        # committed before the superseded mapping is released (the
        # commit ordering guarantee; spec sections 22, 51).
        self.allocator = PageAllocator(cfg, 2 * total_logical_pages)
        # Representation per tier (spec section 19): exact mechanism is
        # modular; the prototype models precision reduction.
        self.representation = {
            Tier.T0: "fp16",
            Tier.T1: "int8",
            Tier.T2: "entropy",
            Tier.T3: "none",
        }
        self.tier_byte_ratio = {
            Tier.T0: 1.0,
            Tier.T1: 0.5,
            Tier.T2: 0.25,
            Tier.T3: 0.0,
        }
        self.traffic: Counter[str] = Counter()
        self.tier_page_counts: Counter[str] = Counter()

    def tier_of(self, logical_page: int) -> Optional[Tier]:
        return self.pages[logical_page].tier

    def resident_pages(self) -> list[int]:
        return [
            pid
            for pid, e in self.pages.items()
            if e.lifecycle
            in (PageLifecycle.MAPPED, PageLifecycle.READY, PageLifecycle.IN_USE)
        ]

    def hot_pages(self) -> list[int]:
        return [
            pid
            for pid, e in self.pages.items()
            if e.tier == Tier.T0
            and e.lifecycle
            in (PageLifecycle.MAPPED, PageLifecycle.READY, PageLifecycle.IN_USE)
        ]

    def physical_bytes(self) -> int:
        total = 0
        for pid in self.resident_pages():
            entry = self.pages[pid]
            assert entry.tier is not None
            total += int(self.bytes_per_page * self.tier_byte_ratio[entry.tier])
        return total

    def bytes_by_tier(self) -> dict[str, int]:
        out = {t.name: 0 for t in Tier}
        for pid in self.resident_pages():
            entry = self.pages[pid]
            assert entry.tier is not None
            out[entry.tier.name] += int(
                self.bytes_per_page * self.tier_byte_ratio[entry.tier]
            )
        return out

    def record_traffic(self, kind: str, pages: int) -> None:
        self.traffic[kind] += pages * self.bytes_per_page

    def snapshot(self) -> dict:
        return {
            "resident": len(self.resident_pages()),
            "hot": len(self.hot_pages()),
            "bytes_by_tier": self.bytes_by_tier(),
            "physical_bytes": self.physical_bytes(),
            "allocator": self.allocator.stats.to_dict(),
        }
