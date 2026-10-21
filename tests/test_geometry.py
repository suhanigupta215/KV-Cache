"""Unit tests for coordinate transforms, overlap, and score aggregation
(spec section 63). These directly address the pixel-to-token mapping
feasibility gap under resize, crop, aspect-ratio padding, and multiple
patch sizes.
"""
import math

import pytest

from mtr_kv.config import Config
from mtr_kv.data_objects import FrameType, MotionBlock, TokenRegion
from mtr_kv.mtre import Mtre, intersection, token_rect
from mtr_kv.scenarios_lib import REFERENCE_SCENARIOS
from mtr_kv.simulator import Simulator, build_token_regions


def test_intersection_basic():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 5.0, 10.0, 10.0)
    assert intersection(a, b) == pytest.approx(25.0)
    assert intersection(a, (100.0, 100.0, 1.0, 1.0)) == 0.0
    assert intersection(a, (10.0, 0.0, 5.0, 5.0)) == 0.0  # touching edge


def test_intersection_contained():
    a = (0.0, 0.0, 100.0, 100.0)
    b = (10.0, 10.0, 10.0, 10.0)
    assert intersection(a, b) == pytest.approx(100.0)


def test_token_rect_identity_mapping():
    rect = token_rect(
        token_id=0,
        grid_cols=16,
        patch=(14.0, 14.0),
        origin=(0.0, 0.0),
        crop_offset=(0.0, 0.0),
        source_size=(224.0, 224.0),
        pre_scale=(1.0, 1.0),
    )
    assert rect == pytest.approx((0.0, 0.0, 14.0, 14.0))


def test_token_rect_inverse_stretch():
    rect = token_rect(
        token_id=0,
        grid_cols=16,
        patch=(14.0, 14.0),
        origin=(0.0, 0.0),
        crop_offset=(0.0, 0.0),
        source_size=(448.0, 224.0),
        pre_scale=(0.5, 1.0),
    )
    assert rect == pytest.approx((0.0, 0.0, 28.0, 14.0))


def test_token_rect_crop_offset():
    rect = token_rect(
        token_id=0,
        grid_cols=16,
        patch=(14.0, 14.0),
        origin=(0.0, 0.0),
        crop_offset=(16.0, 8.0),
        source_size=(224.0, 224.0),
        pre_scale=(1.0, 1.0),
    )
    assert rect == pytest.approx((16.0, 8.0, 14.0, 14.0))


def test_token_rect_invalid_mask_returns_none():
    rect = token_rect(
        token_id=5,
        grid_cols=16,
        patch=(14.0, 14.0),
        origin=(0.0, 0.0),
        crop_offset=(0.0, 0.0),
        source_size=(224.0, 224.0),
        pre_scale=(1.0, 1.0),
        valid_mask=[True, True, True, True, True, False],
    )
    assert rect is None


def test_token_rect_clips_to_source():
    rect = token_rect(
        token_id=255,
        grid_cols=16,
        patch=(14.0, 14.0),
        origin=(0.0, 0.0),
        crop_offset=(0.0, 0.0),
        source_size=(220.0, 220.0),
        pre_scale=(1.0, 1.0),
    )
    assert rect[0] + rect[2] <= 220.0 + 1e-6


def test_build_token_regions_full_coverage():
    cfg = Config()
    regions, num_tokens = build_token_regions(cfg, (1920, 1080))
    assert num_tokens == 256
    assert len(regions) == num_tokens
    total_area = sum(r.rect[2] * r.rect[3] for r in regions)
    assert total_area == pytest.approx(1.0, rel=1e-3)


def test_mtre_coverage_weighted_activity():
    cfg = Config()
    cfg.mtre.conservative_mixed_evidence = False
    mtre = Mtre(cfg)
    block = MotionBlock(
        stream_id=0,
        generation=0,
        frame_index=0,
        rect=(0.0, 0.0, 1.0, 1.0),
        motion=0.5,
        residual=0.4,
        confidence=1.0,
        frame_type=FrameType.P,
        metadata_valid=True,
    )
    tokens = [TokenRegion(token_id=0, rect=(0.0, 0.0, 1.0, 1.0), page_id=0)]
    result = mtre.reconcile([block], tokens)
    assert len(result.tokens) == 1
    expected = 0.5 * 0.5 + 0.25 * 0.4
    assert result.tokens[0].activity == pytest.approx(expected)


def test_mtre_mixed_evidence_conservative():
    cfg = Config()
    mtre = Mtre(cfg)
    active = MotionBlock(
        0, 0, 0, (0.0, 0.0, 0.5, 1.0), motion=1.0, residual=0.5, confidence=1.0,
        frame_type=FrameType.P, metadata_valid=True,
    )
    static = MotionBlock(
        0, 0, 0, (0.5, 0.0, 0.5, 1.0), motion=0.0, residual=0.0, confidence=1.0,
        frame_type=FrameType.P, metadata_valid=True,
    )
    tokens = [TokenRegion(token_id=0, rect=(0.0, 0.0, 1.0, 1.0), page_id=0)]
    result = mtre.reconcile([active, static], tokens)
    tok = result.tokens[0]
    assert tok.mixed_evidence
    # Conservative: high-change component dominates the plain average.
    assert tok.activity >= 0.45


def test_mtre_optimized_matches_oracle():
    cfg = Config()
    cfg.mtre.use_optimized_path = False
    sc = REFERENCE_SCENARIOS["T01"]()
    sim = Simulator(cfg, sc, mode="codec_only")
    regions, _ = build_token_regions(cfg, sc.source_size)
    import mtr_kv.scenarios_lib as sl

    plans = sc.frame_plans()
    blocks, _, _ = sl.render_frame(sc, plans[3], sim.normalizer)
    oracle = sim.mtre.reconcile(blocks, regions)
    optimized = sim.mtre.reconcile_optimized(blocks, regions)
    assert len(oracle.tokens) == len(optimized.tokens)
    for a, b in zip(oracle.tokens, optimized.tokens):
        assert a.token_id == b.token_id
        assert a.activity == pytest.approx(b.activity, abs=1e-9)
        assert a.mixed_evidence == b.mixed_evidence


def test_multi_patch_size_geometry():
    cfg = Config()
    cfg.model.patch_size = 16
    regions, num_tokens = build_token_regions(cfg, (640, 480))
    assert num_tokens == (224 // 16) ** 2
    assert len(regions) == num_tokens


def test_invalid_codec_block_excluded():
    cfg = Config()
    mtre = Mtre(cfg)
    block = MotionBlock(
        0, 0, 0, (0.0, 0.0, 1.0, 1.0), motion=1.0, residual=1.0, confidence=1.0,
        frame_type=FrameType.P, metadata_valid=False,
    )
    tokens = [TokenRegion(token_id=0, rect=(0.0, 0.0, 1.0, 1.0), page_id=0)]
    result = mtre.reconcile([block], tokens)
    assert result.tokens[0].activity == 0.0


@pytest.mark.parametrize("tiles", [1, 3, 8, 17])
def test_tiled_reconciliation_matches_complete_oracle_with_mixed_random_blocks(tiles):
    import random
    from dataclasses import asdict

    rng = random.Random(84)
    cfg = Config()
    cfg.mtre.spatial_index_tiles = tiles
    regions, _ = build_token_regions(cfg, (640, 360))
    blocks = [MotionBlock(0, 1, 1, (rng.uniform(-.1, .9), rng.uniform(-.1, .9),
                                    rng.uniform(.01, .4), rng.uniform(.01, .4)),
                          rng.random(), rng.random(), rng.random(), FrameType.P,
                          metadata_valid=i % 7 != 0) for i in range(120)]
    blocks.extend([blocks[4], blocks[4]])  # distinct observations can share identity
    mtre = Mtre(cfg)
    persistence = {i: rng.random() for i in range(8)}
    oracle = mtre.reconcile(blocks, regions, persistence)
    indexed = mtre.reconcile_optimized(blocks, regions, persistence)
    assert asdict(indexed) == asdict(oracle)
    assert mtre.reconcile_optimized(blocks, []).tokens == []


def test_model_center_crop_maps_token_bounds_to_actual_source_window():
    from torchvision.transforms import Compose, Resize, CenterCrop
    from mtr_kv.vision_feedback import ViTAttentionFeedback

    feedback = ViTAttentionFeedback.__new__(ViTAttentionFeedback)
    feedback.input_size = 224
    feedback.transform = Compose([Resize(256), CenterCrop(224)])
    cfg = Config()
    cfg.model.patch_size = 16
    cfg.model.crop = feedback.source_crop((640, 360))
    regions, _ = build_token_regions(cfg, (640, 360))
    # torchvision resizes 640x360 to 455x256, then crops at (116, 16).
    assert cfg.model.crop == pytest.approx((116 * 640 / 455, 22.5, 224 * 640 / 455, 315))
    assert min(r.rect[0] for r in regions) == pytest.approx(cfg.model.crop[0] / 640)
    assert max(r.rect[0] + r.rect[2] for r in regions) == pytest.approx((cfg.model.crop[0] + cfg.model.crop[2]) / 640)
