"""Online controller for decoder metadata and model feedback streams.

Input is newline-delimited JSON. A producer sends frame records followed by
optional semantic records for the same generation. No synthetic workload is
generated here; the caller supplies observations as they arrive.
"""
from __future__ import annotations

import json
import math
import sys
from typing import TextIO

from .config import Config
from .controller import Kmc
from .data_objects import ApplyStatus, MKAD, Operation, TemporalState, Tier
from .memory import TieredMemoryModel
from .mkad import MkadBuilder
from .motion import MotionNormalizer
from .mtre import Mtre, PageActivity
from .predictor import ScenePredictor
from .semantic import Srce
from .simulator import build_token_regions
from .temporal import TemporalStateEngine


class LiveController:
    """Maintain one controller state across incoming frames and corrections."""

    def __init__(self, cfg: Config, source_size: tuple[int, int], stream_id: int = 0):
        if min(source_size) <= 0 or cfg.model.page_tokens <= 0:
            raise ValueError("source dimensions and page_tokens must be positive")
        self.cfg = cfg
        self.source_size = source_size
        self.stream_id = stream_id
        self.regions, tokens = build_token_regions(cfg, source_size)
        self.page_count = math.ceil(tokens / cfg.model.page_tokens)
        self.memory = TieredMemoryModel(cfg, self.page_count)
        self.kmc = Kmc(cfg, self.memory)
        self.normalizer = MotionNormalizer(cfg)
        self.mtre = Mtre(cfg)
        self.temporal = TemporalStateEngine(cfg)
        self.predictor = ScenePredictor(cfg)
        self.builder = MkadBuilder(cfg)
        self.srce = Srce(cfg)
        self.current_generation = -1
        self.page_activity: dict[int, float] = {}
        self.states: dict[int, TemporalState] = {}
        self.tensor_store = None
        self.reusable_pages: set[int] = set()

    def reuse_kv(self, page_id: int, generation: int, source_generation: int) -> None:
        """Carry model-validated static K/V through a new ready mapping."""
        if generation != self.current_generation or page_id not in self.reusable_pages:
            raise ValueError("page is not eligible for cross-frame KV reuse")
        if self.tensor_store is None:
            raise KeyError("no previous KV tensors are available")
        self.tensor_store.reuse(page_id, generation, source_generation)

    def attach_kv(self, page_id: int, generation: int, key, value, device: str = "cpu") -> int:
        """Materialize model-produced K/V tensors after the page is ready."""
        if self.tensor_store is None:
            from .tensor_store import TorchKVStore

            self.tensor_store = TorchKVStore(self.kmc, device=device)
        return self.tensor_store.put(page_id, generation, key, value)

    def read_kv(self, page_id: int, generation: int):
        """Resolve current-generation K/V through the committed page mapping."""
        if self.tensor_store is None:
            raise KeyError("no KV tensors have been materialized")
        return self.tensor_store.get(page_id, generation)

    def frame(self, message: dict) -> dict:
        generation = int(message["generation"])
        if generation <= self.current_generation:
            return {"type": "frame_result", "generation": generation, "status": "STALE_DROP"}
        blocks_raw = message.get("blocks")
        if blocks_raw is not None and not isinstance(blocks_raw, list):
            raise ValueError("blocks must be an array or null")
        frame_type = message.get("frame_type", "P")
        blocks = self.normalizer.normalize(
            self.stream_id, generation, generation, blocks_raw or [], frame_type=frame_type,
            source_size=self.source_size,
        )
        metadata_valid = bool(blocks) and all(block.metadata_valid for block in blocks)
        if metadata_valid:
            reconcile = self.mtre.reconcile_optimized if self.cfg.mtre.use_optimized_path else self.mtre.reconcile
            result = reconcile(blocks, self.regions)
            activity = {page: 0.0 for page in range(self.page_count)}
            confidence = {page: 0.0 for page in range(self.page_count)}
            for token in result.tokens:
                activity[token.page_id] = max(activity[token.page_id], token.activity)
                confidence[token.page_id] = max(confidence[token.page_id], token.confidence)
            metadata_valid = all(value > 0 for value in confidence.values())
        if not metadata_valid:
            activity = {page: 1.0 for page in range(self.page_count)}
            confidence = {page: 0.0 for page in range(self.page_count)}

        prediction = self.predictor.update(
            [activity[page] for page in range(self.page_count)],
            sum(block.residual for block in blocks) / len(blocks) if blocks else 0.0,
            frame_type,
        )
        self.states = {}
        for page in range(self.page_count):
            self.states[page] = self.temporal.update(
                f"s{self.stream_id}-p{page}", activity[page], confidence[page],
                metadata_valid, generation, scene_transition=prediction.transitioning,
            )
        if frame_type == "I":
            for page in range(self.page_count):
                self.temporal.reset_on_iframe(f"s{self.stream_id}-p{page}", generation)
                self.states[page] = self.temporal.state_of(f"s{self.stream_id}-p{page}")
        self.reusable_pages = {
            page for page in range(self.page_count)
            if generation == self.current_generation + 1 and metadata_valid
            and frame_type in ("P", "B") and not prediction.transitioning
            and self.states[page] == TemporalState.STATIC
            and activity[page] <= self.cfg.temporal.low_threshold
        }
        descriptors = self.builder.build(
            stream_id=self.stream_id, generation=generation, frame_index=generation,
            pages=[PageActivity(page, activity[page], 0.0, confidence[page], self.cfg.model.page_tokens) for page in range(self.page_count)],
            states=self.states, confidence_threshold=self.cfg.motion.confidence_threshold,
            coalesce=self.cfg.kmc.coalescing_enabled,
        )
        statuses = [self.kmc.apply(descriptor) for descriptor in descriptors]
        self.current_generation = generation
        self.page_activity = activity
        ready = self.kmc.take_ready_tokens()
        problems = self.kmc.invariants()
        if problems:
            raise RuntimeError(f"controller invariants failed: {problems}")
        return {
            "type": "frame_result", "generation": generation,
            "status": "READY" if len(ready) == len(descriptors) and all(status == ApplyStatus.APPLIED for status in statuses) else "NOT_READY",
            "metadata_fallback": not metadata_valid,
            "transition_predicted": prediction.transitioning,
            "reuse_eligible_pages": sorted(self.reusable_pages),
            "descriptors": [descriptor.to_dict() for descriptor in descriptors],
            "apply_statuses": [status.value for status in statuses],
            "ready_tokens": [token.to_dict() for token in ready],
            "memory": self.memory.snapshot(),
        }

    def semantic(self, message: dict) -> dict:
        generation = int(message["generation"])
        if generation != self.current_generation:
            return {"type": "semantic_result", "generation": generation, "status": "STALE_DROP"}
        scores_raw = message.get("page_scores")
        if not isinstance(scores_raw, dict):
            raise ValueError("page_scores must be an object keyed by logical page ID")
        scores = {int(page): float(score) for page, score in scores_raw.items()}
        if any(page < 0 or page >= self.page_count or not math.isfinite(score) or score < 0 or score > 1 for page, score in scores.items()):
            raise ValueError("page_scores must contain valid page IDs and scores in [0, 1]")
        actions = self.srce.evaluate(
            stream_id=self.stream_id, generation=generation, frame_index=generation,
            codec_activity=self.page_activity, semantic_scores=scores,
            page_states={page: self.memory.tier_of(page) for page in self.memory.resident_pages()},
            temporal_states=self.states,
        )
        statuses = []
        for action in actions:
            descriptor = self.builder.correction_descriptor(
                self.stream_id, generation, generation, action.page_id,
                action.operation, action.tier, action.activity, 1.0, action.cause,
            )
            statuses.append(self.kmc.apply(descriptor).value)
        problems = self.kmc.invariants()
        if problems:
            raise RuntimeError(f"controller invariants failed: {problems}")
        return {
            "type": "semantic_result", "generation": generation,
            "status": "APPLIED", "actions": [action.to_dict() for action in actions],
            "apply_statuses": statuses, "memory": self.memory.snapshot(),
        }


def run_stream(cfg: Config, source_size: tuple[int, int], source: TextIO = sys.stdin, sink: TextIO = sys.stdout) -> None:
    """Consume and emit one JSON object per line, flushing each result."""
    controller = LiveController(cfg, source_size)
    for line_number, line in enumerate(source, 1):
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("record must be a JSON object")
            if message.get("type") == "frame":
                result = controller.frame(message)
            elif message.get("type") == "semantic":
                result = controller.semantic(message)
            else:
                raise ValueError("type must be frame or semantic")
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            result = {"type": "error", "line": line_number, "message": str(error)}
        sink.write(json.dumps(result) + "\n")
        sink.flush()
