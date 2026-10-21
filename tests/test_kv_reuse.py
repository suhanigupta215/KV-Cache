"""Cross-frame reuse must preserve content identity and generation safety."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mtr_kv.config import Config
from mtr_kv.live import LiveController
from mtr_kv.vision_feedback import ViTAttentionFeedback


def make_controller():
    cfg = Config()
    cfg.model.input_size = 32
    cfg.model.patch_size = 16
    cfg.model.page_tokens = 2
    cfg.model.layers = 1
    cfg.model.kv_heads = 3
    cfg.model.head_dim = 4
    cfg.motion.score_mode = "motion_only"
    cfg.scene.enabled = False
    cfg.temporal.persistence_frames = cfg.temporal.stable_frames = 1
    return LiveController(cfg, (32, 32))


def advance(controller, generation, frame_type="P", blocks=None):
    if blocks is None:
        blocks = [{"rect": [0, 0, 32, 32], "mv": [0, 0], "confidence": 1}]
    result = controller.frame({"generation": generation, "frame_type": frame_type, "blocks": blocks})
    assert result["status"] == "READY"
    return result


@pytest.fixture
def adapter():
    from timm.layers import Attention

    torch.manual_seed(42)
    attention = Attention(12, num_heads=3, qkv_bias=True).eval()
    attention.fused_attn = False
    model = ViTAttentionFeedback.__new__(ViTAttentionFeedback)
    model.torch, model.device = torch, torch.device("cpu")
    model.model = SimpleNamespace(num_prefix_tokens=1)
    model.page_tokens, model.kv_reuse, model.reuse_window = 2, True, 8
    model._reuse_inputs, model._reuse_generation = {}, None
    model._reuse_owner, model._reuse_signature = None, None
    model.kv_consumed_pages = 0
    tokens = torch.randn(1, 5, 12)
    return model, attention, tokens


def infer(adapter, controller, generation, tokens=None):
    model, attention, original = adapter
    with torch.inference_mode():
        return model._stored_attention(attention, original if tokens is None else tokens,
                                       controller, generation, None, False)


def test_unchanged_static_pages_reuse_real_payload_without_rewriting(adapter):
    controller = make_controller()
    advance(controller, 1)
    first = infer(adapter, controller, 1)
    before = controller.tensor_store.snapshot()
    advance(controller, 2)
    second = infer(adapter, controller, 2)
    model, attention, tokens = adapter
    assert model.reuse_stats["pages_reused"] == 2
    assert model.reuse_stats["tokens_kv_projected"] == 1  # prefix only
    assert model.reuse_stats["tokens_kv_reused"] == 4
    assert controller.tensor_store.snapshot()["bytes_written"] == before["bytes_written"]
    # Splitting the packed projection changes GEMM shape for prefix K/V,
    # which can introduce normal floating-point rounding differences.
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)
    with torch.inference_mode():
        baseline = attention(tokens)
    assert torch.allclose(second, baseline, atol=.002, rtol=.002)
    with pytest.raises(ValueError, match="stale"):
        controller.read_kv(0, 1)
    assert controller.tensor_store.records[0].origin_generation == 1


def test_changed_model_input_is_recomputed_despite_zero_codec_motion(adapter):
    controller = make_controller()
    advance(controller, 1)
    infer(adapter, controller, 1)
    advance(controller, 2)
    changed = adapter[2].clone()
    changed[:, 1] += .5
    result = infer(adapter, controller, 2, changed)
    stats = adapter[0].reuse_stats
    assert stats["pages_reused"] == 1
    assert stats["recompute_reasons"]["model_input_changed"] == 1
    assert stats["tokens_kv_projected"] == 3
    with torch.inference_mode():
        baseline = adapter[1](changed)
    assert torch.allclose(result, baseline, atol=.002, rtol=.002)


@pytest.mark.parametrize("frame_type,blocks,generation", [
    ("P", [], 2), ("I", None, 2), ("P", None, 3),
    ("P", [{"rect": [0, 0, 32, 32], "mv": [32, 32], "confidence": 1}], 2),
])
def test_unreliable_active_iframe_or_skipped_generations_invalidate(adapter, frame_type, blocks, generation):
    controller = make_controller()
    advance(controller, 1)
    infer(adapter, controller, 1)
    advance(controller, generation, frame_type, blocks)
    infer(adapter, controller, generation)
    assert adapter[0].reuse_stats["pages_reused"] == 0


def test_window_refreshes_old_origins(adapter):
    adapter[0].reuse_window = 1
    controller = make_controller()
    for generation in range(1, 4):
        advance(controller, generation)
        infer(adapter, controller, generation)
        assert adapter[0].reuse_stats["pages_reused"] == (2 if generation == 2 else 0)
    assert adapter[0].reuse_stats["recompute_reasons"] == {"window_expired": 2}
    assert controller.tensor_store.records[0].origin_generation == 3


def test_weight_changes_and_new_controllers_invalidate_history(adapter):
    controller = make_controller()
    advance(controller, 1)
    infer(adapter, controller, 1)
    with torch.no_grad():
        adapter[1].qkv.weight.add_(.01)
    advance(controller, 2)
    infer(adapter, controller, 2)
    assert adapter[0].reuse_stats["pages_reused"] == 0
    other = make_controller()
    advance(other, 3)
    infer(adapter, other, 3)
    assert adapter[0].reuse_stats["pages_reused"] == 0


def test_reuse_retiering_preserves_origin_and_checks_source(adapter):
    from mtr_kv.data_objects import Tier

    controller = make_controller()
    advance(controller, 1)
    infer(adapter, controller, 1)
    advance(controller, 2)
    correction = controller.semantic({"generation": 2, "page_scores": {"0": 1}})
    assert correction["apply_statuses"] == ["APPLIED"]
    infer(adapter, controller, 2)
    assert adapter[0].reuse_stats["pages_reused"] == 2
    assert controller.tensor_store.records[0].tier == Tier.T0
    assert controller.tensor_store.records[0].origin_generation == 1
    with pytest.raises(KeyError, match="source"):
        controller.tensor_store.reuse(0, 2, 1)


def test_nonready_page_cannot_be_rebound(adapter):
    controller = make_controller()
    advance(controller, 1)
    infer(adapter, controller, 1)
    with pytest.raises(ValueError):
        controller.tensor_store.reuse(0, 2, 1)
