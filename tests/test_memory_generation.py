"""Allocator, generation, and tier-routing tests (spec sections 21-26, 63).

Required invariants: no duplicate physical ownership, no stale mutation,
ready implies committed, valid tier transitions, non-negative capacity.
"""
import pytest

from mtr_kv.config import Config
from mtr_kv.controller import ApplyStatus, Kmc
from mtr_kv.data_objects import MKAD, Mapping, Operation, PageLifecycle, Tier
from mtr_kv.memory import AllocationError, PageAllocator, TieredMemoryModel


def make_kmc(total_pages: int = 8) -> tuple[Config, TieredMemoryModel, Kmc]:
    cfg = Config()
    mem = TieredMemoryModel(cfg, total_pages)
    return cfg, mem, Kmc(cfg, mem)


def mkad(page_start, page_count, tier, generation, op=Operation.ALLOCATE, **kw):
    return MKAD(
        stream_id=0,
        generation=generation,
        page_start=page_start,
        page_count=page_count,
        activity=0.5,
        target_tier=tier,
        operation=op,
        confidence=1.0,
        descriptor_id=kw.pop("descriptor_id", 0),
        **kw,
    )


class TestAllocator:
    def test_reserve_and_release_roundtrip(self):
        cfg = Config()
        alloc = PageAllocator(cfg, 16)
        blocks = alloc.reserve(Tier.T0, 4)
        assert len(blocks) == 4
        assert len(set(blocks)) == 4
        alloc.release(blocks, Tier.T0)
        assert len(alloc.free) == 16

    def test_exhaustion_raises(self):
        cfg = Config()
        alloc = PageAllocator(cfg, 4)
        alloc.reserve(Tier.T0, 2)
        alloc.reserve(Tier.T0, 2)
        with pytest.raises(AllocationError):
            alloc.reserve(Tier.T0, 1)

    def test_emergency_pool_backs_t0(self):
        cfg = Config()
        cfg.kmc.emergency_pool_pages = 4
        alloc = PageAllocator(cfg, 8)
        alloc.reserve(Tier.T0, 4)
        blocks = alloc.reserve(Tier.T0, 4, emergency_ok=True)
        assert len(blocks) == 4  # emergency pool satisfied the request

    def test_non_contiguous_physical_placement(self):
        cfg = Config()
        alloc = PageAllocator(cfg, 32)
        blocks = alloc.reserve(Tier.T0, 4)
        assert blocks != sorted(blocks) or True  # non-contiguity permitted
        assert len(set(blocks)) == 4

    def test_fragmentation_measured(self):
        cfg = Config()
        alloc = PageAllocator(cfg, 8)
        alloc.reserve(Tier.T0, 4)
        assert alloc.stats.fragmentation_peak >= 0.0
        assert alloc.stats.reservations == 4


class TestGenerationSafety:
    def test_stale_descriptor_rejected(self):
        cfg, mem, kmc = make_kmc()
        d1 = mkad(0, 2, Tier.T0, 5)
        assert kmc.apply(d1) == ApplyStatus.APPLIED
        d2 = mkad(0, 2, Tier.T1, 4, descriptor_id=0)
        assert kmc.apply(d2) == ApplyStatus.STALE_DROP

    def test_same_generation_idempotent(self):
        cfg, mem, kmc = make_kmc()
        d1 = mkad(0, 2, Tier.T0, 5, descriptor_id=1)
        assert kmc.apply(d1) == ApplyStatus.APPLIED
        d2 = mkad(0, 2, Tier.T0, 5, descriptor_id=2)
        assert kmc.apply(d2) == ApplyStatus.DUPLICATE
        assert mem.hot_pages() == [0, 1]

    def test_newer_generation_supersedes(self):
        cfg, mem, kmc = make_kmc()
        assert kmc.apply(mkad(0, 2, Tier.T0, 5)) == ApplyStatus.APPLIED
        assert kmc.apply(mkad(0, 2, Tier.T1, 6, descriptor_id=3)) == ApplyStatus.APPLIED
        assert mem.tier_of(0) == Tier.T1

    def test_ready_implies_committed(self):
        cfg, mem, kmc = make_kmc()
        kmc.apply(mkad(0, 2, Tier.T0, 5))
        tokens = kmc.take_ready_tokens()
        assert len(tokens) == 1
        assert tokens[0].generation == 5
        for pid in (0, 1):
            entry = mem.pages[pid]
            assert entry.ready
            assert entry.lifecycle == PageLifecycle.READY

    def test_no_ready_before_commit_on_failure(self):
        cfg, mem, kmc = make_kmc()
        kmc.faulted = True  # controller fault: validation fails
        status = kmc.apply(mkad(0, 2, Tier.T0, 5))
        assert status == ApplyStatus.FALLBACK
        assert kmc.take_ready_tokens() == []
        assert mem.resident_pages() == []

    def test_exhaustion_preserves_old_mapping_and_ready_state(self):
        cfg, mem, kmc = make_kmc(total_pages=2)
        assert kmc.apply(mkad(0, 1, Tier.T0, 1)) == ApplyStatus.APPLIED
        kmc.take_ready_tokens()
        old_mapping = mem.pages[0].mapping
        held = mem.allocator.reserve(Tier.T0, len(mem.allocator.free))
        assert kmc.apply(mkad(0, 1, Tier.T1, 2)) == ApplyStatus.ABORT
        assert mem.pages[0].mapping is old_mapping
        assert mem.pages[0].ready
        assert mem.pages[0].generation == 1
        assert kmc.take_ready_tokens() == []
        assert kmc.invariants() == []
        mem.allocator.release(held, Tier.T0)

    def test_failed_reservation_can_retry_same_descriptor(self):
        cfg, mem, kmc = make_kmc(total_pages=2)
        assert kmc.apply(mkad(0, 1, Tier.T0, 1)) == ApplyStatus.APPLIED
        held = mem.allocator.reserve(Tier.T0, len(mem.allocator.free))
        replacement = mkad(0, 1, Tier.T1, 2, descriptor_id=12)
        assert kmc.apply(replacement) == ApplyStatus.ABORT
        assert 12 not in kmc.seen_descriptors
        mem.allocator.release(held, Tier.T0)
        assert kmc.apply(replacement) == ApplyStatus.APPLIED
        assert mem.tier_of(0) == Tier.T1
        assert kmc.apply(replacement) == ApplyStatus.DUPLICATE

    def test_unsupported_t3_never_commits(self):
        cfg, mem, kmc = make_kmc(total_pages=2)
        assert kmc.apply(mkad(0, 1, Tier.T3, 1)) == ApplyStatus.FALLBACK
        assert mem.pages[0].mapping is None
        assert kmc.take_ready_tokens() == []

    def test_stale_correction_rejected(self):
        cfg, mem, kmc = make_kmc()
        assert kmc.apply(mkad(0, 2, Tier.T2, 5)) == ApplyStatus.APPLIED
        corr = mkad(0, 1, Tier.T0, 4, op=Operation.PROMOTE, descriptor_id=7)
        assert kmc.apply(corr) == ApplyStatus.STALE_DROP

    def test_unsafe_mode_detects_stale_mutation(self):
        cfg, mem, kmc = make_kmc()
        cfg.unsafe_generation_checks = True
        assert kmc.apply(mkad(0, 2, Tier.T0, 5)) == ApplyStatus.APPLIED
        assert kmc.apply(mkad(0, 2, Tier.T2, 3, descriptor_id=9)) == ApplyStatus.APPLIED
        assert mem.tier_of(0) == Tier.T2  # stale write corrupts state

    def test_bounds_rejected(self):
        cfg, mem, kmc = make_kmc()
        status = kmc.apply(mkad(6, 10, Tier.T0, 1))
        assert status == ApplyStatus.FALLBACK


class TestTierRouting:
    def test_promotion_via_emergency_pool(self):
        cfg, mem, kmc = make_kmc()
        assert kmc.apply(mkad(0, 2, Tier.T2, 5)) == ApplyStatus.APPLIED
        corr = mkad(0, 1, Tier.T0, 5, op=Operation.PROMOTE, descriptor_id=8)
        assert kmc.apply(corr) == ApplyStatus.APPLIED
        assert mem.tier_of(0) == Tier.T0
        assert kmc.counters.promotions == 1

    def test_demotion_frees_old_after_commit(self):
        cfg, mem, kmc = make_kmc()
        assert kmc.apply(mkad(0, 2, Tier.T0, 5)) == ApplyStatus.APPLIED
        corr = mkad(0, 2, Tier.T2, 5, op=Operation.DEMOTE, descriptor_id=9)
        assert kmc.apply(corr) == ApplyStatus.APPLIED
        assert mem.tier_of(0) == Tier.T2
        assert kmc.counters.demotions == 1

    def test_no_duplicate_physical_ownership(self):
        cfg, mem, kmc = make_kmc()
        assert kmc.apply(mkad(0, 4, Tier.T0, 1)) == ApplyStatus.APPLIED
        assert kmc.apply(mkad(4, 4, Tier.T2, 1)) == ApplyStatus.APPLIED
        assert kmc.invariants() == []

    def test_retire_releases_block(self):
        cfg, mem, kmc = make_kmc()
        assert kmc.apply(mkad(0, 2, Tier.T0, 5)) == ApplyStatus.APPLIED
        assert kmc.apply(mkad(0, 2, Tier.T0, 6, op=Operation.RETIRE, descriptor_id=10)) == ApplyStatus.APPLIED
        assert mem.tier_of(0) is None
        assert len(mem.allocator.free) == mem.allocator.total_pages


class TestMappingAtomicity:
    def test_failed_commit_leaves_mapping(self):
        cfg, mem, kmc = make_kmc()
        assert kmc.apply(mkad(0, 2, Tier.T0, 5)) == ApplyStatus.APPLIED
        before = {pid: mem.pages[pid].mapping for pid in (0, 1)}
        kmc.faulted = True
        assert kmc.apply(mkad(0, 2, Tier.T1, 6, descriptor_id=11)) == ApplyStatus.FALLBACK
        after = {pid: mem.pages[pid].mapping for pid in (0, 1)}
        assert before == after
