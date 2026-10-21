"""Configuration loading (spec sections 10, 42, 74).

All thresholds and parameters live in a versioned config object so runs
are reproducible and later translate into RTL configuration registers.
The simulator exposes equivalent configuration objects with the same
parameter names as the reference configuration.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ModelConfig:
    """Reference model geometry (spec section 74)."""

    input_size: int = 224
    patch_size: int = 14
    layers: int = 24
    kv_heads: int = 16
    head_dim: int = 128
    bytes_per_element: float = 2.0
    page_tokens: int = 32
    crop: tuple[float, float, float, float] = (0, 0, -1, -1)  # x, y, w, h in source px
    source_size: tuple[int, int] = (1920, 1080)


@dataclass
class MotionConfig:
    """Normalization parameters (spec sections 10, 16)."""

    activity_weights: tuple[float, float, float, float] = (0.5, 0.25, 0.10, 0.15)
    # (wm, wr, wp, wc) for motion, residual, persistence, confidence
    motion_max: float = 32.0  # |dx|+|dy| saturation bound in pixels
    residual_max: float = 255.0
    quantize_activity: bool = False
    quantization_levels: int = 16  # fixed-point emulation levels
    confidence_threshold: float = 0.25  # below -> UNKNOWN fallback
    score_mode: str = "full"  # motion_only | motion_residual | full


@dataclass
class TemporalConfig:
    """Persistence and hysteresis (spec sections 11, 74)."""

    persistence_frames: int = 4  # N low frames before STATIC
    low_threshold: float = 0.20  # enter STATIC/STABLE below this
    high_threshold: float = 0.35  # return to ACTIVE above this
    stable_frames: int = 6  # low frames before STABLE -> STATIC progression
    transition_hold_frames: int = 3


@dataclass
class SceneConfig:
    """Scene-transition predictor (spec section 12)."""

    enabled: bool = True
    threshold: float = 0.45  # transition score above this -> TRANSITIONING
    residual_jump: float = 0.30
    frame_type_change_weight: float = 0.20
    high_change_fraction: float = 0.60
    prefetch_margin: int = 8  # extra T0 pages prefetched on transition
    false_positive_budget: float = 0.15  # informational for sweeps


@dataclass
class MtreConfig:
    """Reconciliation parameters (spec sections 13-15)."""

    hierarchical: bool = True
    hierarchy_min_region: int = 2  # min tokens per leaf region
    hierarchy_heterogeneity: float = 0.15
    conservative_mixed_evidence: bool = True  # high-change dominates
    spatial_index_tiles: int = 8  # tiles per axis for optimized path
    use_optimized_path: bool = True  # oracle remains available for paired checks


@dataclass
class KmcConfig:
    """Controller parameters (spec sections 18-26)."""

    tier_levels: int = 3  # 2 = binary T0/T2, 3 = T0/T1/T2, 4 = +T3
    enable_t3: bool = False
    emergency_pool_pages: int = 8  # reserved T0 pages for promotions
    max_queue_depth: int = 256
    critical_queue_depth: int = 128
    coalescing_enabled: bool = True
    generation_checks_enabled: bool = True  # disable only in unsafe test mode
    enforce_generation_checks: bool = False  # test hook, set by unsafe mode
    reconstruction_latency: float = 3.0  # T3 restore cost in time units
    reconstruction_bytes_ratio: float = 0.25  # byte cost ratio vs page
    allocation_failure_mode: str = "fallback_t2"  # fallback_t2 | abort


@dataclass
class SrceConfig:
    """Semantic correction parameters (spec sections 28-31)."""

    enabled: bool = True
    low_codec: float = 0.20
    high_semantic: float = 0.70
    conflict_threshold: float = 0.40
    aggregation: str = "max"  # token->page aggregation: mean | max | min
    demote_persistence: int = 4  # high A / low S demotion needs persistence
    semantic_normalization: str = "max_abs"  # logged for reproducibility
    emergency_pool_only_promotions: bool = True


@dataclass
class FallbackConfig:
    """Safety/fallback policy (spec sections 32, 34)."""

    missing_metadata_mode: str = "full_t0"  # full_t0 | hold_previous
    low_confidence_mode: str = "unknown_state"
    queue_overflow_mode: str = "conservative_freeze"
    state_freeze_on_fault: bool = True


@dataclass
class EventConfig:
    """Async engine timing (spec sections 08, 25, 54)."""

    decode_delay: float = 1.0
    mtre_delay: float = 0.5
    mkad_delay: float = 0.2  # async FIFO transport delay
    commit_delay: float = 0.3
    semantic_delay: float = 1.5
    correction_delay: float = 0.2
    scheduler_mode: str = "normal"  # normal | adversarial
    adversarial_delay_jitter: float = 2.0
    seed: int = 20260922


@dataclass
class Config:
    """Versioned experiment configuration; stored with every result."""

    version: str = "1.0"
    name: str = "reference"
    model: ModelConfig = field(default_factory=ModelConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    mtre: MtreConfig = field(default_factory=MtreConfig)
    kmc: KmcConfig = field(default_factory=KmcConfig)
    srce: SrceConfig = field(default_factory=SrceConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    events: EventConfig = field(default_factory=EventConfig)
    num_streams: int = 1
    num_frames: int = 12
    mode: str = "full"  # baseline | codec_only | semantic_only | full
    unsafe_generation_checks: bool = False  # ablation-only (spec section 60)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def effective_generation_checks(self) -> bool:
        """Unsafe mode is an explicit ablation (spec section 60)."""
        if self.unsafe_generation_checks:
            return False
        return self.kmc.generation_checks_enabled

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)


def _apply_overrides(cfg: Config, overrides: Optional[dict[str, Any]]) -> Config:
    if not overrides:
        return cfg
    for key, value in overrides.items():
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        current = getattr(obj, parts[-1])
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(obj, parts[-1], value)
    return cfg


def load_config(path: Optional[str] = None, overrides: Optional[dict[str, Any]] = None) -> Config:
    """Load a config JSON file, then apply dotted-key overrides."""
    cfg = Config()
    if path:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for section_name, section in data.items():
            if section_name in ("version", "name", "mode", "num_streams", "num_frames",
                                "unsafe_generation_checks"):
                setattr(cfg, section_name, section)
                continue
            target = getattr(cfg, section_name, None)
            if target is None or not dataclasses.is_dataclass(target):
                continue
            for key, value in section.items():
                current = getattr(target, key)
                if isinstance(current, tuple) and isinstance(value, list):
                    value = tuple(value)
                setattr(target, key, value)
    return _apply_overrides(cfg, overrides)
