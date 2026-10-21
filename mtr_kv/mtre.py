"""Motion-to-Token Reconciliation Engine (spec sections 13-15).

Solves the codec-grid versus token-grid mismatch using normalized source
coordinates and coverage-weighted overlap. The naive correctness path is
retained as the oracle; a tiled optimized path is compared against it.
Mixed evidence is handled conservatively: when a token overlaps both
active and static evidence, the higher-change component can dominate to
avoid unsafe demotion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .config import Config, MtreConfig
from .data_objects import MotionBlock, TokenRegion


def intersection(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Area of rect intersection; rects are (x, y, w, h)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def token_rect(
    token_id: int,
    grid_cols: int,
    patch: tuple[float, float],
    origin: tuple[float, float],
    crop_offset: tuple[float, float],
    source_size: tuple[float, float],
    pre_scale: tuple[float, float],
    valid_mask: Optional[Sequence[bool]] = None,
) -> Optional[tuple[float, float, float, float]]:
    """Build the source-space rectangle of one token (spec section 15).

    Applies the inverse of visual preprocessing: the token lives in model
    space; it is mapped back through resize scale and crop offset into
    source-frame coordinates. Returns None for padding/invalid tokens,
    which are removed using the preprocessing validity mask.
    """
    col = token_id % grid_cols
    row = token_id // grid_cols
    mx = origin[0] + col * patch[0]
    my = origin[1] + row * patch[1]
    # Inverse transform: model px -> source px.
    sx = (mx / max(pre_scale[0], 1e-9)) + crop_offset[0]
    sy = (my / max(pre_scale[1], 1e-9)) + crop_offset[1]
    sw = patch[0] / max(pre_scale[0], 1e-9)
    sh = patch[1] / max(pre_scale[1], 1e-9)
    if valid_mask is not None and token_id < len(valid_mask) and not valid_mask[token_id]:
        return None
    # Clip against source bounds.
    width, height = source_size
    x1 = max(sx, 0.0)
    y1 = max(sy, 0.0)
    x2 = min(sx + sw, float(width))
    y2 = min(sy + sh, float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


@dataclass
class TokenActivity:
    token_id: int
    page_id: int
    activity: float
    residual: float
    confidence: float
    coverage_total: float
    mixed_evidence: bool = False

    def to_dict(self) -> dict:
        return {
            "token_id": self.token_id,
            "page_id": self.page_id,
            "activity": self.activity,
            "confidence": self.confidence,
            "coverage_total": self.coverage_total,
            "mixed_evidence": self.mixed_evidence,
        }


@dataclass
class PageActivity:
    page_id: int
    activity: float
    residual: float
    confidence: float
    token_count: int
    heterogeneous: bool = False

    def to_dict(self) -> dict:
        return {
            "page_id": self.page_id,
            "activity": self.activity,
            "confidence": self.confidence,
            "token_count": self.token_count,
            "heterogeneous": self.heterogeneous,
        }


@dataclass
class ReconciliationResult:
    tokens: list[TokenActivity] = field(default_factory=list)
    pages: list[PageActivity] = field(default_factory=list)

    def token_activity_by_page(self) -> dict[int, list[TokenActivity]]:
        out: dict[int, list[TokenActivity]] = {}
        for t in self.tokens:
            out.setdefault(t.page_id, []).append(t)
        return out


class Mtre:
    """Coverage-weighted reconciliation with conservative aggregation."""

    def __init__(self, cfg: Config):
        self.mc: MtreConfig = cfg.mtre
        self.motion_cfg = cfg.motion

    # ------------------------------------------------------------------
    # Correctness-first oracle (spec section 50)
    # ------------------------------------------------------------------
    def reconcile(
        self,
        blocks: Sequence[MotionBlock],
        tokens: Sequence[TokenRegion],
        persistence_by_page: Optional[dict[int, float]] = None,
    ) -> ReconciliationResult:
        """Check each token rect against intersecting codec blocks and
        accumulate coverage-weighted evidence."""
        result = ReconciliationResult()
        wm, wr, _wp, _wc = self.motion_cfg.activity_weights
        for tok in tokens:
            ta = tok.rect[2] * tok.rect[3]
            if ta <= 0:
                continue
            total = 0.0
            weight = 0.0
            high_total = 0.0
            high_weight = 0.0
            residual_acc = 0.0
            conf_acc = 0.0
            acts: list[float] = []
            mixed = False
            for b in blocks:
                if not b.metadata_valid:
                    continue
                ov = intersection(b.rect, tok.rect)
                if ov == 0.0:
                    continue
                cov = ov / ta
                act = self._block_activity(b)
                # act already carries the wm/wr weighting; coverage is
                # the aggregation weight across blocks.
                score = act
                acts.append(act)
                total += cov * score
                residual_acc += cov * b.residual
                conf_acc += cov * b.confidence
                weight += cov
                if act >= 0.10:
                    high_total += cov * score
                    high_weight += cov
            if weight <= 0:
                activity = 0.0
                residual = 0.0
                conf = 0.0
                mixed = False
            else:
                activity = total / weight
                residual = residual_acc / weight
                conf = conf_acc / weight
                # Conservative mixed evidence (spec section 13): when a
                # token overlaps both active and static evidence, the
                # higher-change component dominates to avoid unsafe
                # demotion.
                if self.mc.conservative_mixed_evidence and high_weight > 0:
                    has_low = any(a < 0.10 for a in acts)
                    has_high = any(a >= 0.10 for a in acts)
                    if has_low and has_high:
                        mixed = True
                        activity = max(activity, high_total / high_weight)
            result.tokens.append(
                TokenActivity(
                    token_id=tok.token_id,
                    page_id=tok.page_id,
                    activity=activity,
                    residual=residual,
                    confidence=conf,
                    coverage_total=weight,
                    mixed_evidence=mixed,
                )
            )
        self._aggregate_pages(result, persistence_by_page)
        return result

    def _block_activity(self, b: MotionBlock) -> float:
        wm, wr, _wp, _wc = self.motion_cfg.activity_weights
        a = wm * b.motion + wr * b.residual
        return max(0.0, min(1.0, a))

    def _aggregate_pages(
        self,
        result: ReconciliationResult,
        persistence_by_page: Optional[dict[int, float]],
    ) -> None:
        by_page = result.token_activity_by_page()
        for page_id, toks in sorted(by_page.items()):
            n = len(toks)
            if n == 0:
                continue
            acts = [t.activity for t in toks]
            mean = sum(acts) / n
            hetero = (max(acts) - min(acts)) > self.mc.hierarchy_heterogeneity
            if self.mc.conservative_mixed_evidence and any(t.mixed_evidence for t in toks):
                # Higher-change dominates within a mixed page.
                mean = max(mean, max(acts) * 0.5)
            persistence = (persistence_by_page or {}).get(page_id, 0.0)
            act = max(0.0, min(1.0, 0.8 * mean + 0.2 * persistence))
            result.pages.append(
                PageActivity(
                    page_id=page_id,
                    activity=act,
                    residual=sum(t.residual for t in toks) / n,
                    confidence=sum(t.confidence for t in toks) / n,
                    token_count=n,
                    heterogeneous=hetero,
                )
            )

    # ------------------------------------------------------------------
    # Tiled optimized path (spec sections 14, 50)
    # ------------------------------------------------------------------
    def reconcile_optimized(
        self,
        blocks: Sequence[MotionBlock],
        tokens: Sequence[TokenRegion],
        persistence_by_page: Optional[dict[int, float]] = None,
    ) -> ReconciliationResult:
        """Spatial-index version: each token checks only nearby blocks."""
        result = ReconciliationResult()
        if not tokens:
            return result
        if not blocks:
            return self.reconcile(blocks, tokens, persistence_by_page)
        tiles = max(1, self.mc.spatial_index_tiles)
        # Bounds of the token geometry define the index space.
        xs0 = min(t.rect[0] for t in tokens)
        ys0 = min(t.rect[1] for t in tokens)
        xs1 = max(t.rect[0] + t.rect[2] for t in tokens)
        ys1 = max(t.rect[1] + t.rect[3] for t in tokens)
        width = max(xs1 - xs0, 1e-9)
        height = max(ys1 - ys0, 1e-9)
        index: dict[tuple[int, int], list[tuple[int, MotionBlock]]] = {}
        for position, b in enumerate(blocks):
            if not b.metadata_valid:
                continue
            x, y, w, h = b.rect
            tx0 = int((x - xs0) / width * tiles)
            ty0 = int((y - ys0) / height * tiles)
            tx1 = int(math.floor((x + w - xs0) / width * tiles))
            ty1 = int(math.floor((y + h - ys0) / height * tiles))
            for tx in range(max(tx0, 0), min(tx1, tiles - 1) + 1):
                for ty in range(max(ty0, 0), min(ty1, tiles - 1) + 1):
                    index.setdefault((tx, ty), []).append((position, b))
        for tok in tokens:
            x, y, w, h = tok.rect
            tx0 = max(int((x - xs0) / width * tiles), 0)
            ty0 = max(int((y - ys0) / height * tiles), 0)
            tx1 = min(int(math.floor((x + w - xs0) / width * tiles)), tiles - 1)
            ty1 = min(int(math.floor((y + h - ys0) / height * tiles)), tiles - 1)
            candidates: list[tuple[int, MotionBlock]] = []
            seen: set[int] = set()
            for tx in range(tx0, tx1 + 1):
                for ty in range(ty0, ty1 + 1):
                    for position, b in index.get((tx, ty), ()):
                        if position not in seen:
                            seen.add(position)
                            candidates.append((position, b))
            # Match oracle summation order, including at overlapping tile edges.
            candidates.sort(key=lambda item: item[0])
            sub = self.reconcile([block for _, block in candidates], [tok], persistence_by_page)
            result.tokens.extend(sub.tokens)
        result.tokens.sort(key=lambda t: t.token_id)
        self._aggregate_pages(result, persistence_by_page)
        return result


class HierarchicalReconciler:
    """Coarse-first recursive reconciliation (spec section 14).

    Starts from coarse spatial regions and subdivides only where motion,
    residual, confidence, or spatial variance is too heterogeneous. The
    lowest level is bounded by token/page granularity. Reduces descriptor
    count versus per-token emission.
    """

    def __init__(self, cfg: Config, mtre: Mtre):
        self.cfg = cfg
        self.mtre = mtre
        self.mc = cfg.mtre

    def run(
        self,
        blocks: Sequence[MotionBlock],
        tokens: Sequence[TokenRegion],
        persistence_by_page: Optional[dict[int, float]] = None,
    ) -> tuple[ReconciliationResult, int]:
        """Returns the full reconciliation plus the descriptor count
        emitted at the coarse level (hierarchy enabled vs disabled metric,
        spec section 14)."""
        result = self.mtre.reconcile(blocks, tokens, persistence_by_page)
        descriptors = self._count_descriptors(tokens, blocks)
        return result, descriptors

    def _count_descriptors(
        self,
        tokens: Sequence[TokenRegion],
        blocks: Sequence[MotionBlock],
    ) -> int:
        """Recursive region emission count with threshold-controlled
        subdivision (echoes thresholded recursive MV analysis)."""
        if not tokens:
            return 0
        rects = [t.rect for t in tokens]
        x0 = min(r[0] for r in rects)
        y0 = min(r[1] for r in rects)
        x1 = max(r[0] + r[2] for r in rects)
        y1 = max(r[1] + r[3] for r in rects)
        region_tokens = list(tokens)
        return self._subdivide(
            x0, y0, x1 - x0, y1 - y0, region_tokens, blocks, depth=0
        )

    def _subdivide(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        tokens: list[TokenRegion],
        blocks: Sequence[MotionBlock],
        depth: int,
    ) -> int:
        if not tokens:
            return 0
        acts = []
        confs = []
        for b in blocks:
            if not b.metadata_valid:
                continue
            ov = intersection(b.rect, (x, y, w, h))
            if ov > 0:
                acts.append(self.mtre._block_activity(b))
                confs.append(b.confidence)
        homogeneous = True
        if len(acts) > 1:
            spread = max(acts) - min(acts)
            c_spread = max(confs) - min(confs) if confs else 0.0
            homogeneous = (
                spread <= self.mc.hierarchy_heterogeneity
                and c_spread <= self.mc.hierarchy_heterogeneity
            )
        min_side = math.sqrt(max(w * h, 1e-12))
        at_leaf = (
            len(tokens) <= self.mc.hierarchy_min_region
            or depth >= 8
            or min_side <= 1e-6
        )
        if homogeneous or at_leaf:
            return 1
        # Subdivide along the longer axis.
        count = 0
        if w >= h:
            xm = x + w / 2
            left = [t for t in tokens if t.rect[0] + t.rect[2] / 2 < xm]
            right = [t for t in tokens if t.rect[0] + t.rect[2] / 2 >= xm]
            count += self._subdivide(x, y, w / 2, h, left, blocks, depth + 1)
            count += self._subdivide(xm, y, w / 2, h, right, blocks, depth + 1)
        else:
            ym = y + h / 2
            top = [t for t in tokens if t.rect[1] + t.rect[3] / 2 < ym]
            bottom = [t for t in tokens if t.rect[1] + t.rect[3] / 2 >= ym]
            count += self._subdivide(x, y, w, h / 2, top, blocks, depth + 1)
            count += self._subdivide(x, ym, w, h / 2, bottom, blocks, depth + 1)
        return count
