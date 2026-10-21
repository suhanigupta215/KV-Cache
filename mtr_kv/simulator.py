"""End-to-end MTR-KV simulator (spec sections 07, 46, 54, 56).

Replays the frozen execution flow: obtain codec metadata, normalize,
update temporal state and scene forecast, reconcile codec/token geometry,
generate MKADs, reserve/commit through the generation-safe KMC, emit
KV_READY, run transformer work, aggregate semantic importance, evaluate
SRCE conflicts, issue corrections, and reject stale updates.

Modes (spec section 56):
  baseline      - fully materializes all visual KV pages in T0
  codec_only    - codec metadata through MTRE/MKAD/KMC, no SRCE
  semantic_only - semantic-only control (waits for model evidence)
  full          - codec guidance plus SRCE and scene prediction
"""
from __future__ import annotations

import math
from typing import Optional

from .config import Config
from .controller import Kmc
from .data_objects import (
    MKAD,
    ApplyStatus,
    Operation,
    TemporalState,
    Tier,
    bytes_per_page,
)
from .events import EventEngine
from .metrics import AggregateMetrics, FrameRecord
from .mkad import MkadBuilder
from .motion import MotionNormalizer
from .mtre import (
    HierarchicalReconciler,
    Mtre,
    PageActivity,
    TokenRegion,
    token_rect,
)
from .memory import TieredMemoryModel
from .predictor import ScenePredictor
from .scenarios_lib import Scenario, render_frame, semantic_trace
from .semantic import Srce, aggregate_attention
from .temporal import TemporalStateEngine

TOKEN_GRID = 16  # 16x16 = 256 tokens for the 224/14 reference geometry


def build_token_regions(
    cfg: Config,
    source_size: tuple[int, int],
    grid: int = TOKEN_GRID,
) -> tuple[list[TokenRegion], int]:
    """Precise pixel-to-token mapping (spec section 15).

    Model input 224x224, patch 14 -> 16x16 grid. Inverse preprocessing
    maps each token patch back to source coordinates with crop/resize
    handling and a validity mask for padding.
    """
    model_input = cfg.model.input_size
    patch = cfg.model.patch_size
    grid = model_input // patch
    num_tokens = grid * grid
    src_w, src_h = source_size
    # The default spans the full source. A live model supplies the inverse
    # of its actual resize/center-crop transform, including fractional edges.
    crop_offset_x, crop_offset_y, crop_width, crop_height = cfg.model.crop
    if crop_width <= 0 or crop_height <= 0:
        crop_offset_x = crop_offset_y = 0.0
        crop_width, crop_height = src_w, src_h
    pre_scale_x = model_input / crop_width
    pre_scale_y = model_input / crop_height
    page_tokens = cfg.model.page_tokens
    regions: list[TokenRegion] = []
    for tid in range(num_tokens):
        col = tid % grid
        row = tid // grid
        rect = token_rect(
            token_id=tid,
            grid_cols=grid,
            patch=(patch, patch),
            origin=(0.0, 0.0),
            crop_offset=(crop_offset_x, crop_offset_y),
            source_size=(src_w, src_h),
            pre_scale=(pre_scale_x, pre_scale_y),
            valid_mask=None,
        )
        if rect is None:
            continue
        # Normalize token geometry into the same source-coordinate space
        # as codec blocks (spec section 15).
        nx = rect[0] / src_w
        ny = rect[1] / src_h
        nw = rect[2] / src_w
        nh = rect[3] / src_h
        regions.append(
            TokenRegion(token_id=tid, rect=(nx, ny, nw, nh), page_id=tid // page_tokens)
        )
    return regions, num_tokens


class Simulator:
    """Deterministic single-command simulator for all modes."""

    def __init__(self, cfg: Config, scenario: Scenario, mode: Optional[str] = None):
        self.cfg = cfg
        self.scenario = scenario
        self.mode = mode or cfg.mode
        self.normalizer = MotionNormalizer(cfg)
        self.temporal = TemporalStateEngine(cfg)
        self.predictor = ScenePredictor(cfg)
        self.mtre = Mtre(cfg)
        self.hier = HierarchicalReconciler(cfg, self.mtre)
        self.builder = MkadBuilder(cfg)
        self.srce = Srce(cfg)
        self.events = EventEngine(cfg)
        self.frame_records: list[FrameRecord] = []
        self.use_hierarchical = cfg.mtre.hierarchical

    # ------------------------------------------------------------------
    def _num_pages(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.cfg.model.page_tokens)

    def run_baseline(self) -> AggregateMetrics:
        """Baseline fully materializes required visual KV pages in T0."""
        cfg = self.cfg
        regions, num_tokens = build_token_regions(cfg, self.scenario.source_size)
        total_pages = self._num_pages(num_tokens)
        metrics = AggregateMetrics()
        page_bytes = bytes_per_page(
            cfg.model.page_tokens,
            cfg.model.layers,
            cfg.model.kv_heads,
            cfg.model.head_dim,
            cfg.model.bytes_per_element,
        )
        for plan in self.scenario.frame_plans():
            metrics.total_frames += 1
            metrics.baseline_pages_total += total_pages
            metrics.hot_pages_total += total_pages
            metrics.baseline_bytes_total += total_pages * page_bytes
            metrics.physical_bytes_total += total_pages * page_bytes
            metrics.kv_write_bytes += total_pages * page_bytes
        return metrics

    # ------------------------------------------------------------------
    def run_mtr(self) -> AggregateMetrics:
        """Codec-only, semantic-only, or full MTR-KV run."""
        cfg = self.cfg
        stream_id = 0
        regions, num_tokens = build_token_regions(cfg, self.scenario.source_size)
        total_pages = self._num_pages(num_tokens)
        page_tokens = cfg.model.page_tokens

        page_bytes = bytes_per_page(
            page_tokens,
            cfg.model.layers,
            cfg.model.kv_heads,
            cfg.model.head_dim,
            cfg.model.bytes_per_element,
        )
        memory = self._make_memory(total_pages)
        kmc = Kmc(cfg, memory)
        sem_trace = semantic_trace(self.scenario, page_tokens, num_tokens)
        semantic_enabled = self.mode in ("full", "semantic_only")
        codec_enabled = self.mode in ("codec_only", "full")

        metrics = AggregateMetrics()
        prev_page_activity: dict[int, float] = {}
        low_activity_frames: dict[int, int] = {}
        tier_state_prev: dict[int, Tier] = {}
        tier_switches = 0
        baseline_pages_per_frame = total_pages

        for plan in self.scenario.frame_plans():
            generation = plan.generation
            frame_index = plan.frame_index
            fallback_used = False
            transition_predicted = False

            # Step 1-2: obtain and normalize codec metadata.
            blocks, region_semantics, mean_residual = render_frame(
                self.scenario, plan, self.normalizer
            )
            if plan.metadata_missing or not blocks:
                fallback_used = True
                metrics.fallbacks += 1

            # Step 3: temporal state + scene-transition forecast.
            page_keys = {
                pid: f"s{stream_id}-p{pid}" for pid in range(total_pages)
            }
            per_page_activity: dict[int, float] = {pid: 0.0 for pid in range(total_pages)}
            per_page_conf: dict[int, float] = {pid: 0.0 for pid in range(total_pages)}

            if codec_enabled and not plan.metadata_missing:
                recon = self._reconcile(blocks, regions)
                for tok in recon.tokens:
                    per_page_activity[tok.page_id] = max(
                        per_page_activity[tok.page_id], tok.activity
                    )
                    per_page_conf[tok.page_id] = max(
                        per_page_conf[tok.page_id], tok.confidence
                    )
            elif plan.metadata_missing:
                # Missing metadata is not zero motion: route to UNKNOWN
                # and fall back conservatively (spec sections 09, 34).
                per_page_activity = {
                    pid: cfg.temporal.low_threshold for pid in range(total_pages)
                }
                per_page_conf = {pid: 0.0 for pid in range(total_pages)}

            # Scene predictor on page-activity divergence.
            outcome = self.predictor.update(
                [per_page_activity[pid] for pid in range(total_pages)],
                mean_residual / max(cfg.motion.residual_max, 1e-9),
                plan.frame_type,
                ground_truth_cut=plan.is_scene_cut,
            )
            transition_predicted = outcome.transitioning
            if outcome.transitioning:
                for pid in range(total_pages):
                    self.temporal.update(
                        page_keys[pid],
                        per_page_activity[pid],
                        per_page_conf[pid],
                        not plan.metadata_missing,
                        frame_index,
                        scene_transition=True,
                    )
            else:
                for pid in range(total_pages):
                    self.temporal.update(
                        page_keys[pid],
                        per_page_activity[pid],
                        per_page_conf[pid],
                        not plan.metadata_missing,
                        frame_index,
                    )
            if plan.frame_type == "I":
                for pid in range(total_pages):
                    self.temporal.reset_on_iframe(page_keys[pid], frame_index)

            states = {
                pid: self.temporal.state_of(page_keys[pid]) for pid in range(total_pages)
            }
            persistence = {
                pid: self.temporal.persistence(page_keys[pid]) for pid in range(total_pages)
            }

            # Steps 4-5: reconcile + MKAD generation, then async transport.
            descriptors: list[MKAD] = []
            if codec_enabled:
                hier_count = self._descriptor_count(blocks, regions)
                if frame_index == 0:
                    metrics.descriptor_count_hierarchical = hier_count
                    flat = total_pages
                    metrics.descriptor_count_flat = flat
                descriptors = self.builder.build(
                    stream_id=stream_id,
                    generation=generation,
                    frame_index=frame_index,
                    pages=[
                        PageActivity(
                            page_id=pid,
                            activity=per_page_activity[pid],
                            residual=0.0,
                            confidence=per_page_conf[pid],
                            token_count=page_tokens,
                        )
                        for pid in range(total_pages)
                    ],
                    states=states,
                    confidence_threshold=cfg.motion.confidence_threshold,
                    coalesce=cfg.kmc.coalescing_enabled,
                    prefetch_pages=outcome.prefetch_pages if cfg.scene.enabled else 0,
                    total_pages=total_pages,
                )
                for d in descriptors:
                    d.frame_index = frame_index
                # Async descriptor path: MKAD crosses the FIFO with a
                # transport delay (spec sections 08, 25). The event
                # engine models control latency; the controller applies
                # descriptors deterministically during the drain.
                for d in descriptors:
                    self.events.schedule(
                        self.events.now + cfg.events.mkad_delay,
                        "MKAD_READY",
                        stream_id,
                        generation,
                        {"descriptor": d.to_dict()},
                        priority=1 if not d.prefetch else 5,
                        jitter=cfg.events.scheduler_mode == "adversarial",
                    )
                    kmc.apply(d)

            # Steps 6-8: reserve, commit, KV_READY via the controller.
            if plan.metadata_missing:
                # Fallback policy: full T0 allocation (spec section 34).
                fb = MKAD(
                    stream_id=stream_id,
                    generation=generation,
                    page_start=0,
                    page_count=total_pages,
                    activity=1.0,
                    target_tier=Tier.T0,
                    operation=Operation.ALLOCATE,
                    confidence=0.0,
                    frame_index=frame_index,
                    cause="confidence_fallback",
                    descriptor_id=0,
                )
                kmc.apply(fb)
            else:
                self._drain_events(kmc)

            # Step 9: transformer work consumes the residency plan.
            ready = kmc.take_ready_tokens()
            resident = memory.resident_pages()

            # Step 10: aggregate semantic importance (compact stats only).
            semantic_scores: dict[int, float] = {}
            if semantic_enabled:
                token_scores = sem_trace.get(frame_index, {})
                page_scores = aggregate_attention(
                    token_scores, page_tokens, method=cfg.srce.aggregation
                )
                for pid in range(total_pages):
                    semantic_scores[pid] = page_scores.get(pid, 0.0)

            # Steps 11-13: SRCE conflict evaluation + corrections.
            rescues_this_frame = 0
            conflicts_this_frame = 0
            if semantic_enabled:
                actions = self.srce.evaluate(
                    stream_id=stream_id,
                    generation=generation,
                    frame_index=frame_index,
                    codec_activity={
                        pid: per_page_activity[pid] if codec_enabled else 1.0
                        for pid in range(total_pages)
                    },
                    semantic_scores=semantic_scores,
                    page_states={
                        pid: memory.tier_of(pid) or Tier.T0 for pid in resident
                    },
                    temporal_states={
                        pid: states.get(pid, TemporalState.ACTIVE)
                        for pid in range(total_pages)
                    },
                    low_activity_frames=low_activity_frames,
                )
                conflicts_this_frame = self.srce.stats.conflicts
                for action in actions:
                    corr = self.builder.correction_descriptor(
                        stream_id=stream_id,
                        generation=generation,
                        frame_index=frame_index,
                        page_id=action.page_id,
                        operation=action.operation,
                        tier=action.tier,
                        activity=action.activity,
                        confidence=1.0,
                        cause=action.cause,
                    )
                    self.events.schedule(
                        self.events.now + cfg.events.correction_delay,
                        "SEMANTIC_UPDATE",
                        stream_id,
                        generation,
                        {"action": action.to_dict()},
                        priority=2,
                        jitter=cfg.events.scheduler_mode == "adversarial",
                    )
                    status = kmc.apply(corr)
                    if status == ApplyStatus.APPLIED:
                        if action.operation == Operation.PROMOTE:
                            rescues_this_frame += 1
                            self.srce.note_rescue_latency(
                                cfg.events.semantic_delay + cfg.events.correction_delay
                            )
            self._drain_events(kmc)

            # Track low-activity persistence for controlled demotion.
            for pid in range(total_pages):
                if per_page_activity[pid] <= cfg.srce.low_codec:
                    low_activity_frames[pid] = low_activity_frames.get(pid, 0) + 1
                else:
                    low_activity_frames[pid] = 0

            # Step 14: carry tier state into the next frame.
            for pid in range(total_pages):
                current = memory.tier_of(pid)
                if pid in tier_state_prev and tier_state_prev[pid] != current:
                    tier_switches += 1
                tier_state_prev[pid] = current

            # Frame record (Appendix G schema).
            tiers = {t.name: 0 for t in Tier}
            for pid in memory.resident_pages():
                t = memory.tier_of(pid)
                if t is not None:
                    tiers[t.name] += 1
            record = FrameRecord(
                stream_id=stream_id,
                generation=generation,
                frame_index=frame_index,
                baseline_pages=baseline_pages_per_frame,
                mtr_pages_t0=len(memory.hot_pages()),
                mtr_pages_t1=tiers["T1"],
                mtr_pages_t2=tiers["T2"],
                mtr_pages_t3=tiers["T3"],
                pages_avoided=baseline_pages_per_frame - len(memory.hot_pages()),
                fallback=fallback_used,
                semantic_rescues=rescues_this_frame,
                conflicts=conflicts_this_frame,
                queue_peak=kmc.queue_peak,
                stale_drops=kmc.counters.stale_drops,
                transition_predicted=transition_predicted,
                scene_cut_truth=plan.is_scene_cut,
                tier_switches=tier_switches,
            )
            self.frame_records.append(record)

            # Aggregate metrics.
            metrics.total_frames += 1
            metrics.baseline_pages_total += baseline_pages_per_frame
            metrics.hot_pages_total += len(memory.hot_pages())
            metrics.resident_pages_total += len(memory.resident_pages())
            metrics.baseline_bytes_total += baseline_pages_per_frame * page_bytes
            metrics.physical_bytes_total += memory.physical_bytes()
            metrics.total_descriptors += len(descriptors) + (
                1 if plan.metadata_missing else 0
            )
            metrics.stale_drops = kmc.counters.stale_drops
            metrics.duplicates = kmc.counters.duplicates
            metrics.fallbacks = kmc.counters.fallbacks
            metrics.promotions = kmc.counters.promotions
            metrics.demotions = kmc.counters.demotions
            metrics.conflicts = self.srce.stats.conflicts
            metrics.evaluated_pages = self.srce.stats.evaluated_pages
            metrics.low_codec_candidates = self.srce.stats.low_codec_candidates
            metrics.rescues = self.srce.stats.rescues
            st = self.predictor.stats
            metrics.scene_tp, metrics.scene_fp = st.true_positives, st.false_positives
            metrics.scene_fn, metrics.scene_tn = st.false_negatives, st.true_negatives
            metrics.state_switches = self.temporal.total_state_switches()
            metrics.frame_state_fraction = self.temporal.frame_state_fraction()
            metrics.tier_switches = tier_switches
            metrics.queue_peak_overall = max(
                metrics.queue_peak_overall, kmc.queue_peak
            )
            for kind, value in memory.traffic.items():
                if kind == "kv_write":
                    metrics.kv_write_bytes = value
                elif kind == "promotion":
                    metrics.promotion_bytes = value
                elif kind == "demotion":
                    metrics.demotion_bytes = value
                elif kind == "reconstruction":
                    metrics.reconstruction_bytes = value
            metrics.latency_samples.append(
                cfg.events.decode_delay
                + cfg.events.mtre_delay
                + cfg.events.mkad_delay
                + cfg.events.commit_delay
            )

        # Safety gate: run must end with all invariants satisfied
        # (spec sections 54, 63).
        problems = kmc.invariants()
        if problems:
            raise AssertionError(f"KMC invariant violations: {problems}")
        return metrics

    # ------------------------------------------------------------------
    def _make_memory(self, total_pages: int):
        return TieredMemoryModel(self.cfg, total_pages)

    def _reconcile(self, blocks, regions):
        if self.use_hierarchical:
            return self.hier.run(blocks, regions)[0]
        return self.mtre.reconcile(blocks, regions)

    def _descriptor_count(self, blocks, regions) -> int:
        count = self.hier._count_descriptors(regions, blocks)
        return count

    def _drain_events(self, kmc: Kmc) -> None:
        # The prototype models transport with the event engine; the
        # controller applies deterministically at drain time.
        while self.events.pending() > 0:
            self.events.step()


def run_all_modes(
    cfg: Config, scenario: Scenario
) -> dict[str, dict]:
    """Baseline, codec-only, and full MTR-KV with the same input trace
    (spec sections 46, 55, 56)."""
    results: dict[str, dict] = {}
    for mode in ("baseline", "codec_only", "full"):
        sim = Simulator(cfg, scenario, mode=mode)
        if mode == "baseline":
            metrics = sim.run_baseline()
        else:
            metrics = sim.run_mtr()
        results[mode] = {
            "metrics": metrics.to_dict(),
            "frames": [fr.to_dict() for fr in sim.frame_records],
            "event_trace": sim.events.export_trace()
            if hasattr(sim, "events")
            else [],
            "scenario": scenario.name,
            "seed": scenario.seed,
            "config": cfg.to_dict(),
        }
    return results
