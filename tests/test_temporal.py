"""Temporal state engine and scene predictor tests (spec sections 11-12)."""
from mtr_kv.config import Config
from mtr_kv.data_objects import TemporalState
from mtr_kv.predictor import ScenePredictor
from mtr_kv.scenarios_lib import REFERENCE_SCENARIOS, render_frame
from mtr_kv.simulator import Simulator
from mtr_kv.temporal import TemporalStateEngine


def test_persistence_before_static():
    cfg = Config()
    engine = TemporalStateEngine(cfg)
    for frame in range(cfg.temporal.persistence_frames - 1):
        engine.update("k", 0.05, 1.0, True, frame)
    assert engine.state_of("k") != TemporalState.STATIC
    engine.update("k", 0.05, 1.0, True, cfg.temporal.persistence_frames - 1)
    assert engine.state_of("k") == TemporalState.STABLE


def test_static_after_stable_frames():
    cfg = Config()
    engine = TemporalStateEngine(cfg)
    for frame in range(cfg.temporal.stable_frames + 1):
        engine.update("k", 0.05, 1.0, True, frame)
    assert engine.state_of("k") == TemporalState.STATIC


def test_hysteresis_exit_requires_high_threshold():
    cfg = Config()
    engine = TemporalStateEngine(cfg)
    for frame in range(5):
        engine.update("k", 0.05, 1.0, True, frame)
    assert engine.state_of("k") in (TemporalState.STABLE, TemporalState.STATIC)
    # Middle band holds the low path (hysteresis).
    engine.update("k", 0.28, 1.0, True, 10)
    assert engine.state_of("k") in (TemporalState.STABLE, TemporalState.STATIC)
    # Only above the high threshold does it return to ACTIVE.
    engine.update("k", 0.5, 1.0, True, 11)
    assert engine.state_of("k") == TemporalState.ACTIVE


def test_unknown_on_low_confidence():
    cfg = Config()
    engine = TemporalStateEngine(cfg)
    engine.update("k", 0.5, 0.1, True, 0)
    assert engine.state_of("k") == TemporalState.UNKNOWN


def test_transitioning_overrides_static():
    cfg = Config()
    engine = TemporalStateEngine(cfg)
    for frame in range(6):
        engine.update("k", 0.05, 1.0, True, frame)
    assert engine.state_of("k") in (TemporalState.STABLE, TemporalState.STATIC)
    engine.update("k", 0.05, 1.0, True, 6, scene_transition=True)
    assert engine.state_of("k") == TemporalState.TRANSITIONING


def test_iframe_resets_temporal_state():
    cfg = Config()
    engine = TemporalStateEngine(cfg)
    for frame in range(6):
        engine.update("k", 0.05, 1.0, True, frame)
    engine.reset_on_iframe("k", 6)
    assert engine.state_of("k") == TemporalState.ACTIVE
    assert engine.record("k").low_counter == 0


def test_scene_predictor_detects_cut():
    cfg = Config()
    predictor = ScenePredictor(cfg)
    low = [0.02] * 8
    high = [0.6] * 8
    predictor.update(low, 0.05, "P", ground_truth_cut=False)
    outcome = predictor.update(high, 0.5, "P", ground_truth_cut=True)
    assert outcome.transitioning
    assert predictor.stats.true_positives == 1


def test_scene_predictor_disabled_is_safe():
    """A disabled predictor must leave the rest of the system safe
    (spec section 12)."""
    cfg = Config()
    cfg.scene.enabled = False
    predictor = ScenePredictor(cfg)
    low = [0.02] * 8
    high = [0.9] * 8
    predictor.update(low, 0.05, "P")
    outcome = predictor.update(high, 0.8, "P", ground_truth_cut=True)
    assert not outcome.transitioning
    assert predictor.stats.false_negatives == 1


def test_false_positive_tracked():
    cfg = Config()
    cfg.scene.threshold = 0.01  # over-sensitive predictor
    predictor = ScenePredictor(cfg)
    predictor.update([0.1] * 4, 0.05, "P", ground_truth_cut=False)
    outcome = predictor.update([0.11] * 4, 0.05, "P", ground_truth_cut=False)
    assert outcome.transitioning
    assert predictor.stats.false_positives == 1


def test_multistream_temporal_keys_independent():
    cfg = Config()
    engine = TemporalStateEngine(cfg)
    engine.update("s0-p0", 0.05, 1.0, True, 0)
    engine.update("s1-p0", 0.9, 1.0, True, 0)
    assert engine.state_of("s0-p0") != engine.state_of("s1-p0")
