"""Race tests: adversarial reordering, duplicates, late corrections
(spec sections 08, 24, 26, 63; Appendix F trace).

The desired result is that generation 11 never overwrites generation 12,
and duplicate descriptors do not double-allocate pages.
"""
from mtr_kv.config import Config
from mtr_kv.controller import ApplyStatus, Kmc
from mtr_kv.data_objects import (
    Event,
    MKAD,
    Operation,
    Tier,
)
from mtr_kv.events import EventEngine
from mtr_kv.memory import TieredMemoryModel
from mtr_kv.temporal import TemporalState, TemporalStateEngine


def mkad(page_start, page_count, tier, generation, op=Operation.ALLOCATE, desc=0):
    return MKAD(
        stream_id=0,
        generation=generation,
        page_start=page_start,
        page_count=page_count,
        activity=0.5,
        target_tier=tier,
        operation=op,
        confidence=1.0,
        descriptor_id=desc,
    )


def test_appendix_f_race_trace():
    cfg = Config()
    mem = TieredMemoryModel(cfg, 8)
    kmc = Kmc(cfg, mem)
    assert kmc.apply(mkad(4, 1, Tier.T0, 11, desc=1)) == ApplyStatus.APPLIED
    assert kmc.apply(mkad(4, 1, Tier.T0, 12, desc=2)) == ApplyStatus.APPLIED
    assert kmc.apply(mkad(4, 1, Tier.T0, 12, desc=3)) == ApplyStatus.DUPLICATE
    corr = mkad(4, 1, Tier.T0, 11, op=Operation.PROMOTE, desc=4)
    assert kmc.apply(corr) == ApplyStatus.STALE_DROP
    dup = mkad(4, 1, Tier.T0, 12, desc=5)
    assert kmc.apply(dup) == ApplyStatus.DUPLICATE
    assert kmc.committed_generation[(0, 4)] == 12
    assert kmc.invariants() == []
    tokens = kmc.take_ready_tokens()
    assert len(tokens) == 2
    assert {t.generation for t in tokens} == {11, 12}


def test_adversarial_reordering_never_corrupts():
    """Late g=11 commit after g=12 commit must not overwrite."""
    cfg = Config()
    mem = TieredMemoryModel(cfg, 8)
    kmc = Kmc(cfg, mem)
    g12 = mkad(4, 1, Tier.T0, 12, desc=1)
    assert kmc.apply(g12) == ApplyStatus.APPLIED
    late_commit = mkad(4, 1, Tier.T2, 11, desc=2)
    assert kmc.apply(late_commit) == ApplyStatus.STALE_DROP
    assert mem.tier_of(4) == Tier.T0
    assert kmc.committed_generation[(0, 4)] == 12


def test_event_engine_adversarial_mode_preserves_safety():
    """Variable transport delay reorders events; correctness is
    generation-based, not time-of-arrival based."""
    cfg = Config()
    cfg.events.scheduler_mode = "adversarial"
    cfg.events.mkad_delay = 0.1
    engine = EventEngine(cfg)
    seen = []
    engine.register("MKAD_READY", lambda e: seen.append(e.generation))
    engine.schedule(1.0, "MKAD_READY", 0, 11, {}, jitter=True)
    engine.schedule(1.0, "MKAD_READY", 0, 12, {}, jitter=True)
    engine.run()
    assert sorted(seen) == [11, 12]
    # Adversarial jitter must be able to reorder same-schedule events.
    assert len(engine.trace) == 2


def test_duplicate_descriptors_never_double_allocate():
    cfg = Config()
    mem = TieredMemoryModel(cfg, 8)
    kmc = Kmc(cfg, mem)
    for _ in range(5):
        kmc.apply(mkad(0, 4, Tier.T0, 7, desc=42))
    assert kmc.counters.reservations == 1
    assert kmc.counters.duplicates == 4
    assert kmc.invariants() == []


def test_allocation_failure_abort_safely():
    cfg = Config()
    cfg.kmc.allocation_failure_mode = "abort"
    mem = TieredMemoryModel(cfg, 4)
    kmc = Kmc(cfg, mem)
    assert kmc.apply(mkad(0, 4, Tier.T0, 1)) == ApplyStatus.APPLIED
    assert kmc.apply(mkad(0, 4, Tier.T2, 2, desc=3)) == ApplyStatus.APPLIED
    assert kmc.invariants() == []


def test_queue_overflow_conservative_freeze():
    cfg = Config()
    cfg.kmc.critical_queue_depth = 4
    mem = TieredMemoryModel(cfg, 8)
    kmc = Kmc(cfg, mem)
    assert kmc.note_queue_pressure(2) == "normal"
    assert kmc.note_queue_pressure(8) == "conservative_freeze"
    assert kmc.conservative_mode
    assert kmc.counters.conservative_freezes == 1


def test_reordering_does_not_break_invariants():
    cfg = Config()
    mem = TieredMemoryModel(cfg, 8)
    kmc = Kmc(cfg, mem)
    descriptors = [
        mkad(0, 2, Tier.T0, 3, desc=1),
        mkad(2, 2, Tier.T2, 5, desc=2),
        mkad(0, 2, Tier.T1, 4, desc=3),
        mkad(0, 2, Tier.T0, 2, desc=4),
        mkad(4, 2, Tier.T0, 6, desc=5),
        mkad(2, 2, Tier.T0, 4, desc=6),
        mkad(6, 1, Tier.T1, 1, desc=7),
    ]
    for d in descriptors:
        kmc.apply(d)
    assert kmc.invariants() == []
    assert kmc.committed_generation[(0, 0)] == 4
    assert kmc.committed_generation[(0, 2)] == 5


def test_temporal_state_machines_separated():
    """Temporal state is orthogonal to tier state (spec section 45)."""
    cfg = Config()
    engine = TemporalStateEngine(cfg)
    key = "page-0"
    for frame in range(6):
        engine.update(key, activity=0.05, confidence=1.0, metadata_valid=True, frame_index=frame)
    assert engine.state_of(key) == TemporalState.STATIC
    engine.update(key, activity=0.9, confidence=1.0, metadata_valid=True, frame_index=6)
    assert engine.state_of(key) == TemporalState.ACTIVE
    engine.update(key, activity=0.5, confidence=0.1, metadata_valid=True, frame_index=7)
    assert engine.state_of(key) == TemporalState.UNKNOWN


def test_duplicate_event_trace_deterministic():
    cfg = Config()
    cfg.events.seed = 7
    e1 = EventEngine(cfg)
    e1.schedule(1.0, "MKAD_READY", 0, 1, {}, jitter=True)
    e1.schedule(1.0, "MKAD_READY", 0, 2, {}, jitter=True)
    t1 = e1.export_trace()
    e2 = EventEngine(cfg)
    e2.schedule(1.0, "MKAD_READY", 0, 1, {}, jitter=True)
    e2.schedule(1.0, "MKAD_READY", 0, 2, {}, jitter=True)
    t2 = e2.export_trace()
    assert t1 == t2
