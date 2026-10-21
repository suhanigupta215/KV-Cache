"""Synthetic workload generator (spec section 57).

Models static backgrounds with moving objects, camera pans, compression
noise, small semantic events, scene cuts, missing metadata, multi-stream
bursts, and adversarial reorderings. Each region gets a controllable
temporal trajectory of motion, residual, confidence, and semantic
importance. Deterministic random seeds. Workloads deliberately trigger
each architecture component before real-video traces are introduced.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from .data_objects import MotionBlock


@dataclass
class RegionSpec:
    """Controllable temporal trajectory for one spatial region."""

    name: str
    rect: tuple[float, float, float, float]  # source pixels
    motion_profile: str = "static"
    # static | moving | pan | noise | semantic_event | cut | burst
    motion_amplitude: float = 0.0
    residual_level: float = 8.0
    confidence: float = 1.0
    valid: bool = True
    semantic_importance: float = 0.1
    semantic_spike_frame: Optional[int] = None
    block_size: tuple[int, int] = (16, 16)


@dataclass
class FramePlan:
    frame_index: int
    generation: int
    frame_type: str
    is_scene_cut: bool
    regions: list[RegionSpec] = field(default_factory=list)
    metadata_missing: bool = False


@dataclass
class Scenario:
    """Full scenario definition (spec section 57)."""

    name: str
    source_size: tuple[int, int]
    num_frames: int
    regions: list[RegionSpec]
    scene_cut_frames: list[int] = field(default_factory=list)
    missing_metadata_frames: list[int] = field(default_factory=list)
    iframe_frames: list[int] = field(default_factory=lambda: [0])
    seed: int = 20260922

    def frame_plans(self) -> list[FramePlan]:
        plans: list[FramePlan] = []
        for f in range(self.num_frames):
            frame_type = "I" if f in self.iframe_frames else "P"
            plans.append(
                FramePlan(
                    frame_index=f,
                    generation=f,
                    frame_type=frame_type,
                    is_scene_cut=f in self.scene_cut_frames,
                    regions=list(self.regions),
                    metadata_missing=f in self.missing_metadata_frames,
                )
            )
        return plans


def _rect_shares(r1, r2) -> bool:
    x1a, y1a, w1a, h1a = r1
    x1b, y1b, w1b, h1b = r2
    return not (
        x1a + w1a <= x1b or x1b + w1b <= x1a or y1a + h1a <= y1b or y1b + h1b <= y1a
    )


def render_frame(
    scenario: Scenario,
    plan: FramePlan,
    normalizer,
) -> tuple[list[MotionBlock], dict[int, float], float]:
    """Materialize canonical MotionBlocks for one frame.

    Returns blocks, per-region semantic importance (keyed by region
    index), and the mean residual. Deterministic given the scenario seed.
    """
    rng = random.Random(scenario.seed * 100003 + plan.frame_index)
    blocks: list[MotionBlock] = []
    semantics: dict[int, float] = {}
    residual_sum = 0.0
    residual_count = 0

    for region_index, region in enumerate(plan.regions):
        motion_amp = region.motion_amplitude
        residual = region.residual_level
        confidence = region.confidence
        valid = region.valid and not plan.metadata_missing
        semantic = region.semantic_importance

        if region.motion_profile == "moving":
            # Object motion: alternating direction each frame.
            phase = math.sin(plan.frame_index * 0.9 + region_index)
            dx = motion_amp * phase
            dy = motion_amp * 0.5 * math.cos(plan.frame_index * 0.7)
        elif region.motion_profile == "pan":
            dx = motion_amp
            dy = 0.0
        elif region.motion_profile == "noise":
            dx = rng.uniform(-motion_amp, motion_amp)
            dy = rng.uniform(-motion_amp, motion_amp)
            residual = region.residual_level * rng.uniform(0.5, 1.5)
        elif region.motion_profile == "semantic_event":
            if region.semantic_spike_frame is not None and (
                plan.frame_index >= region.semantic_spike_frame
            ):
                # Small visual event: low motion but real content change.
                dx, dy = 0.5, 0.5
                residual = max(residual, 90.0)
                semantic = region.semantic_importance
            else:
                dx = dy = 0.0
        elif region.motion_profile == "cut":
            # After a scene cut, motion evidence saturates.
            if any(f <= plan.frame_index for f in scenario.scene_cut_frames):
                dx = motion_amp
                dy = motion_amp * 0.5
            else:
                dx = dy = 0.0
        elif region.motion_profile == "burst":
            dx = motion_amp * rng.choice((-1.0, 1.0))
            dy = motion_amp * rng.uniform(-0.3, 0.3)
            confidence = region.confidence * rng.uniform(0.6, 1.0)
        else:  # static
            dx = dy = 0.0
            residual = region.residual_level * rng.uniform(0.8, 1.2)

        if plan.is_scene_cut and region.motion_profile != "cut":
            # Scene cut invalidates assumptions across the frame: residual
            # jumps everywhere (spec section 12).
            residual = max(residual, 120.0)

        block_size_x = max(
            int(region.rect[2] // max(1, (region.rect[2] / region.block_size[0]))), 4
        )
        block_size_y = max(
            int(region.rect[3] // max(1, (region.rect[3] / region.block_size[1]))), 4
        )
        step_x = max(block_size_x, 8)
        step_y = max(block_size_y, 8)
        rx, ry, rw, rh = region.rect
        y = ry
        while y < ry + rh - 1e-6:
            x = rx
            while x < rx + rw - 1e-6:
                bw = min(step_x, rx + rw - x)
                bh = min(step_y, ry + rh - y)
                blocks.append(
                    normalizer.normalize(
                        stream_id=0,
                        generation=plan.generation,
                        frame_index=plan.frame_index,
                        blocks=[
                            {
                                "rect": (x, y, bw, bh),
                                "mv": (dx, dy),
                                "residual": residual,
                                "confidence": confidence,
                                "valid": valid,
                                "block_size": (block_size_x, block_size_y),
                            }
                        ],
                        frame_type=plan.frame_type,
                        source_size=scenario.source_size,
                    )[0]
                )
                x += bw
            y += bh
        semantics[region_index] = semantic
        residual_sum += residual
        residual_count += 1

    return blocks, semantics, (residual_sum / max(residual_count, 1))


def semantic_trace(
    scenario: Scenario,
    tokens_per_page: int,
    num_tokens: int,
) -> dict[int, dict[int, float]]:
    """Per-frame token-level semantic scores for the adapter (spec 57).

    Returns {frame_index: {token_id: importance}}. Regions with a
    semantic spike light up their covered token area; other tokens stay
    low. The generator remains the source of truth for the proxy.
    """
    rng = random.Random(scenario.seed + 7919)
    trace: dict[int, dict[int, float]] = {}
    # Token grid geometry assumes the default 224/14 pipeline geometry
    # used by the reference config.
    for plan in scenario.frame_plans():
        scores: dict[int, float] = {}
        for region_index, region in enumerate(plan.regions):
            if region.motion_profile == "semantic_event":
                active = (
                    region.semantic_spike_frame is not None
                    and plan.frame_index >= region.semantic_spike_frame
                )
            else:
                active = region.semantic_importance > 0.5
            level = 0.9 if active else region.semantic_importance
            if level <= 0.0:
                continue
            # Light up tokens overlapping the region rect in normalized
            # source coordinates (approximate footprint).
            sx, sy = scenario.source_size
            x0 = region.rect[0] / sx
            y0 = region.rect[1] / sy
            x1 = (region.rect[0] + region.rect[2]) / sx
            y1 = (region.rect[1] + region.rect[3]) / sy
            grid = int(math.sqrt(num_tokens))
            cols = grid
            rows = grid
            for tid in range(num_tokens):
                col = tid % cols
                row = tid // cols
                u0, u1 = col / cols, (col + 1) / cols
                v0, v1 = row / rows, (row + 1) / rows
                overlap = max(0.0, min(u1, x1) - max(u0, x0)) * max(
                    0.0, min(v1, y1) - max(v0, y0)
                )
                token_area = 1.0 / (cols * rows)
                if overlap > 0.3 * token_area:
                    jitter = rng.uniform(0.85, 1.0)
                    scores[tid] = max(scores.get(tid, 0.0), level * jitter)
        trace[plan.frame_index] = scores
    return trace


# ----------------------------------------------------------------------
# Reference scenarios
# ----------------------------------------------------------------------
def scenario_static_background(seed: int = 20260922) -> Scenario:
    """T01 shape: one active page with static surroundings."""
    return Scenario(
        name="T01_static_background",
        source_size=(1920, 1080),
        num_frames=10,
        regions=[
            RegionSpec(
                name="background",
                rect=(0.0, 0.0, 1920.0, 1080.0),
                motion_profile="static",
                motion_amplitude=0.0,
                residual_level=6.0,
                semantic_importance=0.1,
            ),
            RegionSpec(
                name="moving_object",
                rect=(700.0, 400.0, 400.0, 300.0),
                motion_profile="moving",
                motion_amplitude=40.0,
                residual_level=40.0,
                semantic_importance=0.8,
            ),
        ],
        seed=seed,
    )


def scenario_threshold_oscillation(seed: int = 20260922) -> Scenario:
    """T02 shape: motion hovering around the tier boundary."""
    return Scenario(
        name="T02_threshold_oscillation",
        source_size=(1280, 720),
        num_frames=14,
        regions=[
            RegionSpec(
                name="hover_region",
                rect=(0.0, 0.0, 1280.0, 720.0),
                motion_profile="noise",
                motion_amplitude=1.6,
                residual_level=24.0,
                semantic_importance=0.4,
            )
        ],
        seed=seed,
    )


def scenario_semantic_rescue(seed: int = 20260922) -> Scenario:
    """T03 shape: low codec activity, high semantic importance."""
    return Scenario(
        name="T03_semantic_rescue",
        source_size=(1280, 720),
        num_frames=10,
        regions=[
            RegionSpec(
                name="quiet_but_important",
                rect=(300.0, 200.0, 500.0, 350.0),
                motion_profile="semantic_event",
                motion_amplitude=0.0,
                residual_level=10.0,
                semantic_importance=0.95,
                semantic_spike_frame=4,
            ),
            RegionSpec(
                name="filler_static",
                rect=(0.0, 0.0, 1280.0, 720.0),
                motion_profile="static",
                residual_level=5.0,
                semantic_importance=0.05,
            ),
        ],
        seed=seed,
    )


def scenario_scene_cut(seed: int = 20260922) -> Scenario:
    """T04 shape: scene divergence spike mid-run."""
    return Scenario(
        name="T04_scene_cut",
        source_size=(1920, 1080),
        num_frames=12,
        regions=[
            RegionSpec(
                name="scene",
                rect=(0.0, 0.0, 1920.0, 1080.0),
                motion_profile="cut",
                motion_amplitude=20.0,
                residual_level=30.0,
                semantic_importance=0.6,
            )
        ],
        scene_cut_frames=[6],
        seed=seed,
    )


def scenario_missing_metadata(seed: int = 20260922) -> Scenario:
    """T08 shape: decoder metadata disappears for some frames."""
    return Scenario(
        name="T08_missing_metadata",
        source_size=(1280, 720),
        num_frames=10,
        regions=[
            RegionSpec(
                name="mixed",
                rect=(0.0, 0.0, 1280.0, 720.0),
                motion_profile="moving",
                motion_amplitude=10.0,
                residual_level=35.0,
                semantic_importance=0.5,
            )
        ],
        missing_metadata_frames=[3, 4, 7],
        seed=seed,
    )


def scenario_multistream_burst(seed: int = 20260922) -> Scenario:
    """Multi-stream pressure: motion bursts and low-confidence frames."""
    return Scenario(
        name="multistream_burst",
        source_size=(1280, 720),
        num_frames=10,
        regions=[
            RegionSpec(
                name="burst_area",
                rect=(0.0, 0.0, 1280.0, 720.0),
                motion_profile="burst",
                motion_amplitude=18.0,
                residual_level=50.0,
                confidence=0.8,
                semantic_importance=0.5,
            )
        ],
        seed=seed,
    )


REFERENCE_SCENARIOS = {
    "T01": scenario_static_background,
    "T02": scenario_threshold_oscillation,
    "T03": scenario_semantic_rescue,
    "T04": scenario_scene_cut,
    "T08": scenario_missing_metadata,
    "multistream": scenario_multistream_burst,
}
