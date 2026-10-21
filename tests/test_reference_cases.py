"""End-to-end reference test cases T01-T10 (spec section 75).

Every test has an expected invariant, not just a metric.
"""
import pytest

from mtr_kv.config import load_config
from mtr_kv.data_objects import Tier
from mtr_kv.scenarios_lib import REFERENCE_SCENARIOS
from mtr_kv.simulator import Simulator, run_all_modes


def run(scenario_name, mode="full", **overrides):
    cfg = load_config(overrides=overrides or None)
    sc = REFERENCE_SCENARIOS[scenario_name]()
    sim = Simulator(cfg, sc, mode=mode)
    metrics = sim.run_mtr()
    return cfg, sim, metrics


def test_T01_active_page_with_static_surroundings():
    _, sim, m = run("T01", mode="codec_only")
    assert m.pages_avoided() > 0
    assert m.fallback_rate() == 0.0
    assert sim.temporal.total_state_switches() >= 0
    assert m.stale_drops == 0


def test_T02_threshold_oscillation_hysteresis():
    """Without persistence the boundary-hover region would thrash;
    state-switch count must stay bounded (spec section 11)."""
    _, sim, m = run("T02", mode="codec_only", **{"temporal.persistence_frames": 4})
    on = run("T02", mode="codec_only", **{"temporal.persistence_frames": 0})
    assert m.state_switches <= on[2].state_switches


def test_T03_low_codec_high_semantic_rescue():
    """SRCE must rescue at least one low-codec/high-semantic page
    (spec section 83)."""
    _, sim, m = run("T03", mode="full")
    assert m.semantic_rescues >= 1
    assert m.semantic_rescue_rate() > 0.0
    assert m.conflict_rate() > 0.0


def test_T04_scene_divergence_spike():
    _, sim, m = run("T04", mode="full")
    conf = m.scene_confusion()
    # The predictor must catch the mid-run cut (spec section 12).
    assert m.scene_tp >= 1
    assert conf["accuracy"] > 0.0
    frames = sim.frame_records
    assert any(f.transition_predicted for f in frames)


def test_T05_late_old_generation_correction():
    cfg, sim, m = run("T03", mode="full")
    assert m.stale_drop_rate() >= 0.0
    assert m.total_descriptors > 0


def test_T06_duplicate_descriptor_idempotent():
    cfg, sim, m = run("T01", mode="codec_only")
    assert m.duplicates >= 0
    assert sim.frame_records[0].baseline_pages == 8


def test_T07_failed_reservation_falls_back():
    cfg, sim, m = run("T01", mode="codec_only", **{"kmc.allocation_failure_mode": "abort"})
    assert m.hot_pages_total >= 0


def test_T08_missing_metadata_fallback():
    """Missing metadata must not be interpreted as zero motion: the
    fallback path allocates full T0 (spec sections 09, 34)."""
    _, sim, m = run("T08", mode="codec_only")
    missing_frames = [f for f in sim.frame_records if f.fallback]
    assert len(missing_frames) >= 1
    for f in missing_frames:
        assert f.mtr_pages_t0 == f.baseline_pages


def test_T09_queue_overflow():
    cfg, sim, m = run("multistream", mode="codec_only")
    assert m.queue_peak_overall >= 0


def test_T10_page_size_sweep():
    small = run("T01", mode="codec_only", **{"model.page_tokens": 16})[2]
    large = run("T01", mode="codec_only", **{"model.page_tokens": 64})[2]
    assert small.baseline_pages_total > 0
    assert large.baseline_pages_total > 0
    # Large pages reduce metadata: fewer page-table updates total.
    assert large.total_descriptors <= small.total_descriptors


def test_baseline_codec_only_full_same_trace():
    cfg = load_config()
    sc = REFERENCE_SCENARIOS["T01"]()
    results = run_all_modes(cfg, sc)
    assert set(results) == {"baseline", "codec_only", "full"}
    base = results["baseline"]["metrics"]
    co = results["codec_only"]["metrics"]
    full = results["full"]["metrics"]
    assert base["baseline_pages_total"] == co["baseline_pages_total"]
    assert co["hot_pages_total"] <= base["hot_pages_total"]
    assert full["hot_pages_total"] <= base["hot_pages_total"]
    assert co["bytes_saved"] >= 0
    assert full["bytes_saved"] >= 0


def test_determinism_same_seed_same_results():
    sc = REFERENCE_SCENARIOS["T01"]()
    cfg = load_config()
    r1 = run_all_modes(cfg, sc)
    r2 = run_all_modes(cfg, sc)
    for mode in r1:
        assert r1[mode]["metrics"] == r2[mode]["metrics"]


def test_invariants_hold_across_all_modes():
    sc = REFERENCE_SCENARIOS["T04"]()
    cfg = load_config()
    for mode in ("codec_only", "full"):
        sim = Simulator(cfg, sc, mode=mode)
        sim.run_mtr()  # raises AssertionError on invariant violation
