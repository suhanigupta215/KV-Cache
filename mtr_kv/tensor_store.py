"""Physical KV tensor representations bound to committed logical mappings."""
from __future__ import annotations

from dataclasses import dataclass
import struct
import time
import zlib

from .data_objects import Tier


@dataclass
class TensorRecord:
    generation: int
    tier: Tier
    key: object
    value: object
    scale_key: object = None
    scale_value: object = None
    origin_generation: int | None = None


class TorchKVStore:
    """Store actual K/V tensors after the KMC makes a page ready.

    T0 holds fp16 tensors on the selected device. T1 uses symmetric int8
    quantization on CPU. T2 uses lossless zlib compression of fp16 tensors.
    Storage measurements count allocated payload bytes, not allocator overhead.
    """

    def __init__(self, kmc, device: str = "cpu"):
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("TorchKVStore requires PyTorch") from error
        self.torch = torch
        self.kmc = kmc
        self.device = torch.device(device)
        self.records: dict[int, TensorRecord] = {}
        self.bytes_written = 0
        self.bytes_read = 0
        self.write_seconds = 0.0
        self.read_seconds = 0.0
        self.pages_reused = 0

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            self.torch.mps.synchronize()

    def _mapping(self, page_id: int, generation: int):
        entry = self.kmc.mem.pages.get(page_id)
        if entry is None or not entry.ready or entry.mapping is None:
            raise ValueError("KV page is not committed and ready")
        if entry.generation != generation:
            raise ValueError("stale KV generation")
        return entry.mapping

    def _encode_tensor(self, tensor, tier: Tier):
        tensor = tensor.detach().contiguous().to(dtype=self.torch.float16)
        if tier == Tier.T0:
            return tensor.to(self.device), None
        tensor = tensor.cpu()
        if tier == Tier.T1:
            magnitude = tensor.abs().max().float()
            scale = max(magnitude.item() / 127.0, 1e-8)
            return (tensor.float() / scale).round().clamp(-127, 127).to(self.torch.int8), scale
        if tier == Tier.T2:
            # A private in-memory payload needs no pickle/ZIP archive. Keep
            # the shape header and compress only contiguous fp16 bytes.
            shape = tuple(tensor.shape)
            header = struct.pack("<I", len(shape)) + struct.pack("<" + "Q" * len(shape), *shape)
            return header + zlib.compress(tensor.numpy().tobytes()), None
        raise ValueError("T3 requires a reconstruction callback and cannot store KV")

    def _decode_tensor(self, encoded, tier: Tier, scale):
        if tier == Tier.T0:
            return encoded
        if tier == Tier.T1:
            return (encoded.float() * scale).to(self.device)
        dimensions = struct.unpack_from("<I", encoded)[0]
        shape = struct.unpack_from("<" + "Q" * dimensions, encoded, 4)
        raw = bytearray(zlib.decompress(encoded[4 + 8 * dimensions:]))
        if not raw:
            return self.torch.empty(shape, dtype=self.torch.float16, device=self.device)
        return self.torch.frombuffer(raw, dtype=self.torch.float16).reshape(shape).to(self.device)

    def sync_tiers(self, generation: int) -> None:
        """Apply semantic migrations without rereading unchanged pages."""
        for page_id, record in list(self.records.items()):
            if record.generation != generation:
                continue
            mapping = self._mapping(page_id, generation)
            if record.tier != mapping.tier:
                self._retier(page_id, record, mapping.tier)

    def _payload_bytes(self, encoded) -> int:
        if isinstance(encoded, bytes):
            return len(encoded)
        return encoded.numel() * encoded.element_size()

    def put(self, page_id: int, generation: int, key, value) -> int:
        mapping = self._mapping(page_id, generation)
        if key.shape != value.shape:
            raise ValueError("key and value tensors must have matching shapes")
        self._synchronize()
        start = time.perf_counter()
        encoded_key, scale_key = self._encode_tensor(key, mapping.tier)
        encoded_value, scale_value = self._encode_tensor(value, mapping.tier)
        record = TensorRecord(generation, mapping.tier, encoded_key, encoded_value, scale_key, scale_value, generation)
        self.records[page_id] = record
        self._synchronize()
        count = self._payload_bytes(encoded_key) + self._payload_bytes(encoded_value)
        self.bytes_written += count
        self.write_seconds += time.perf_counter() - start
        return count

    def reuse(self, page_id: int, generation: int, source_generation: int) -> None:
        """Bind validated unchanged content to a new committed generation.

        The model adapter must verify content identity first. This operation
        never makes an uncommitted page readable or accepts a stale source.
        """
        mapping = self._mapping(page_id, generation)
        record = self.records.get(page_id)
        if generation != source_generation + 1:
            raise ValueError("KV reuse requires consecutive generations")
        if record is None or record.generation != source_generation:
            raise KeyError("reuse source generation is unavailable")
        if record.tier != mapping.tier:
            self._retier(page_id, record, mapping.tier)
            record = self.records[page_id]
        record.generation = generation
        self.pages_reused += 1

    def get(self, page_id: int, generation: int):
        mapping = self._mapping(page_id, generation)
        record = self.records.get(page_id)
        if record is None or record.generation != generation:
            raise KeyError("KV tensors have not been materialized for this generation")
        if record.tier != mapping.tier:
            self._retier(page_id, record, mapping.tier)
            record = self.records[page_id]
        self._synchronize()
        start = time.perf_counter()
        key = self._decode_tensor(record.key, record.tier, record.scale_key)
        value = self._decode_tensor(record.value, record.tier, record.scale_value)
        self._synchronize()
        self.bytes_read += self._payload_bytes(record.key) + self._payload_bytes(record.value)
        self.read_seconds += time.perf_counter() - start
        return key, value

    def _retier(self, page_id: int, record: TensorRecord, target: Tier) -> None:
        self._synchronize()
        read_start = time.perf_counter()
        key = self._decode_tensor(record.key, record.tier, record.scale_key)
        value = self._decode_tensor(record.value, record.tier, record.scale_value)
        self._synchronize()
        self.bytes_read += self._payload_bytes(record.key) + self._payload_bytes(record.value)
        self.read_seconds += time.perf_counter() - read_start
        write_start = time.perf_counter()
        encoded_key, scale_key = self._encode_tensor(key, target)
        encoded_value, scale_value = self._encode_tensor(value, target)
        self.records[page_id] = TensorRecord(record.generation, target, encoded_key, encoded_value, scale_key, scale_value, record.origin_generation)
        self._synchronize()
        self.bytes_written += self._payload_bytes(encoded_key) + self._payload_bytes(encoded_value)
        self.write_seconds += time.perf_counter() - write_start

    def snapshot(self) -> dict:
        return {
            "payload_bytes": sum(self._payload_bytes(record.key) + self._payload_bytes(record.value) for record in self.records.values()),
            "bytes_written": self.bytes_written,
            "bytes_read": self.bytes_read,
            "write_seconds": self.write_seconds,
            "read_seconds": self.read_seconds,
            "device": str(self.device),
            "pages_reused": self.pages_reused,
        }
