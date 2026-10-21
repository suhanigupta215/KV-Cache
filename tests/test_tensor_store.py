"""Tests physical KV payloads against controller commit and generation rules."""
import pytest

torch = pytest.importorskip("torch")

from mtr_kv.config import Config
from mtr_kv.controller import Kmc
from mtr_kv.data_objects import MKAD, Operation, Tier
from mtr_kv.memory import TieredMemoryModel
from mtr_kv.tensor_store import TorchKVStore
from mtr_kv.live import LiveController


def descriptor(generation, tier):
    return MKAD(0, generation, 0, 1, 0.0, tier, Operation.ALLOCATE, 1.0)


def test_tensor_store_requires_ready_mapping_and_generation():
    cfg = Config()
    kmc = Kmc(cfg, TieredMemoryModel(cfg, 2))
    store = TorchKVStore(kmc)
    key = torch.arange(16, dtype=torch.float16)
    with pytest.raises(ValueError):
        store.put(0, 1, key, key)
    kmc.apply(descriptor(1, Tier.T0))
    assert store.put(0, 1, key, key) == 64
    with pytest.raises(ValueError):
        store.get(0, 0)


@pytest.mark.parametrize("tier", [Tier.T0, Tier.T1, Tier.T2])
def test_tensor_store_roundtrip_real_payloads(tier):
    cfg = Config()
    kmc = Kmc(cfg, TieredMemoryModel(cfg, 2))
    kmc.apply(descriptor(1, tier))
    store = TorchKVStore(kmc)
    key = torch.linspace(-1, 1, 128)
    store.put(0, 1, key, key * 2)
    restored_key, restored_value = store.get(0, 1)
    assert torch.allclose(restored_key.float(), key, atol=0.02)
    assert torch.allclose(restored_value.float(), key * 2, atol=0.02)
    assert store.snapshot()["payload_bytes"] > 0


def test_live_controller_accepts_model_kv_only_after_ready():
    controller = LiveController(Config(), (640, 480))
    key = torch.ones(8)
    with pytest.raises(ValueError):
        controller.attach_kv(0, 1, key, key)
    controller.frame({"generation": 1, "blocks": []})
    controller.attach_kv(0, 1, key, key)
    assert torch.equal(controller.read_kv(0, 1)[0], key.half())
    controller.frame({"generation": 2, "blocks": []})
    with pytest.raises(ValueError):
        controller.read_kv(0, 1)


def test_tensor_store_counts_tier_migration_io():
    cfg = Config()
    kmc = Kmc(cfg, TieredMemoryModel(cfg, 2))
    assert kmc.apply(descriptor(1, Tier.T2)).value == "APPLIED"
    store = TorchKVStore(kmc)
    key = torch.linspace(-1, 1, 128)
    store.put(0, 1, key, key)
    before = store.snapshot()
    correction = MKAD(0, 1, 0, 1, 0.5, Tier.T0, Operation.PROMOTE, 1.0, descriptor_id=2)
    assert kmc.apply(correction).value == "APPLIED"
    recovered, _ = store.get(0, 1)
    after = store.snapshot()
    assert torch.allclose(recovered.float(), key, atol=0.001)
    assert after["bytes_written"] > before["bytes_written"]
    assert after["bytes_read"] > before["bytes_read"]
    assert store.records[0].tier == Tier.T0


@pytest.mark.parametrize("shape", [(), (0,), (3, 4), (3, 2, 7)])
def test_raw_compressed_fp16_preserves_shape_and_values(shape):
    kmc = Kmc(Config(), TieredMemoryModel(Config(), 2))
    kmc.apply(descriptor(1, Tier.T2))
    store = TorchKVStore(kmc)
    key = torch.randn(shape)
    store.put(0, 1, key, key)
    recovered, _ = store.get(0, 1)
    assert recovered.shape == key.shape
    assert torch.equal(recovered, key.half())


def test_tier_sync_does_not_count_spurious_reads():
    kmc = Kmc(Config(), TieredMemoryModel(Config(), 2))
    kmc.apply(descriptor(1, Tier.T2))
    store = TorchKVStore(kmc)
    store.put(0, 1, torch.ones(8), torch.ones(8))
    before = store.snapshot()
    store.sync_tiers(1)
    assert store.snapshot() == before
    kmc.apply(MKAD(0, 1, 0, 1, 1, Tier.T0, Operation.PROMOTE, 1))
    store.sync_tiers(1)
    assert store.records[0].tier == Tier.T0
    assert store.snapshot()["bytes_read"] > before["bytes_read"]
