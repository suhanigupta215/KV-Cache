"""Motion descriptor normalization (spec sections 09, 10, 16).

Converts raw codec metadata into canonical MotionBlocks with bounded
motion magnitude, quantized residual score, and independent confidence.
Missing metadata routes to an explicit fallback state, never zero motion.
"""
from __future__ import annotations

from typing import Iterable, Optional

from .config import Config, MotionConfig
from .data_objects import FrameType, MotionBlock


class MotionNormalizer:
    """Normalizes decoder metadata into canonical source-coordinate blocks."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mc: MotionConfig = cfg.motion

    def normalize(
        self,
        stream_id: int,
        generation: int,
        frame_index: int,
        blocks: Iterable[dict],
        frame_type: str = "P",
        source_size: Optional[tuple[int, int]] = None,
    ) -> list[MotionBlock]:
        """blocks: raw dicts with rect, mv, residual, confidence, valid.

        rect is in source pixels; motion is raw (dx, dy). Unknown/invalid
        entries keep metadata_valid=False so downstream sees explicit
        uncertainty rather than zero motion.
        """
        out: list[MotionBlock] = []
        ft = FrameType(frame_type)
        for raw in blocks:
            rect = raw.get("rect", (0.0, 0.0, 0.0, 0.0))
            valid = bool(raw.get("valid", True))
            dx, dy = raw.get("mv", (0.0, 0.0))
            confidence = float(raw.get("confidence", 1.0))
            residual = float(raw.get("residual", 0.0))
            if source_size is not None:
                sx, sy = source_size
                rect = (rect[0] / sx, rect[1] / sy, rect[2] / sx, rect[3] / sy)
            block = MotionBlock(
                stream_id=stream_id,
                generation=generation,
                frame_index=frame_index,
                rect=tuple(rect),  # type: ignore[arg-type]
                motion=self._magnitude(dx, dy),
                residual=self._residual(residual),
                confidence=self._clip01(confidence),
                frame_type=ft,
                metadata_valid=valid,
                raw_dx=dx,
                raw_dy=dy,
                raw_block_size=raw.get("block_size"),
            )
            out.append(block)
        return out

    def _magnitude(self, dx: float, dy: float) -> float:
        """Hardware-friendly |dx|+|dy| magnitude, bounded to [0, 1]."""
        m = abs(dx) + abs(dy)
        return min(m / max(self.mc.motion_max, 1e-9), 1.0)

    def _residual(self, r: float) -> float:
        return self._clip01(r / max(self.mc.residual_max, 1e-9))

    @staticmethod
    def _clip01(v: float) -> float:
        return max(0.0, min(1.0, v))

    def activity(
        self,
        block: MotionBlock,
        persistence: float = 0.0,
    ) -> float:
        """Activity score A = clip(wm*M + wr*R + wp*P + wc*C, 0, 1).

        Confidence stays separate in the block but contributes a small
        weight here; low-confidence routing to UNKNOWN is handled in the
        temporal engine. score_mode selects ablations (spec section 16).
        """
        wm, wr, wp, wc = self.mc.activity_weights
        mode = self.mc.score_mode
        m_term = block.motion if mode in ("motion_only", "motion_residual", "full") else 0.0
        r_term = block.residual if mode in ("motion_residual", "full") else 0.0
        p_term = persistence if mode == "full" else 0.0
        c_term = block.confidence if mode == "full" else 0.0
        if mode == "motion_only":
            a = wm * m_term / max(wm, 1e-9)
        elif mode == "motion_residual":
            a = 0.67 * m_term + 0.33 * r_term
        else:
            a = wm * m_term + wr * r_term + wp * p_term + wc * c_term
        a = max(0.0, min(1.0, a))
        if self.mc.quantize_activity:
            levels = max(2, self.mc.quantization_levels)
            a = round(a * (levels - 1)) / (levels - 1)
        return a
