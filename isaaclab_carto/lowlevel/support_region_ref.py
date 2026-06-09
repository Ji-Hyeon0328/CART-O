# isaaclab_carto/lowlevel/support_region_ref.py
#
# B8-p/B8-q: support-region based CoM/base reference generator.
#
# This is a flat-terrain geometric baseline:
#   stance foot positions -> support polygon -> support center -> base_ref
#
# It is intentionally simpler than Abdalla/Orsolino-style actuation-aware
# feasible regions. The goal is to close the simulation loop first.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class SupportRegionRefConfig:
    alpha: float = 0.35
    margin: float = 0.035
    max_shift_per_step: float = 0.004
    min_stance_legs: int = 3
    height_ref: float = 0.54
    roll_gain_from_y_error: float = 0.0
    pitch_gain_from_x_error: float = 0.0
    max_roll_ref: float = 0.10
    max_pitch_ref: float = 0.10


@dataclass
class SupportRegionRefOutput:
    base_ref: torch.Tensor          # [N, 6] = x,y,z,roll,pitch,yaw
    support_center_xy: torch.Tensor # [N, 2]
    current_xy: torch.Tensor        # [N, 2]
    target_xy: torch.Tensor         # [N, 2]
    margin_to_edge: torch.Tensor    # [N]
    swing_allowed: torch.Tensor     # [N] bool
    stance_count: torch.Tensor      # [N]
    polygon_xy: torch.Tensor        # [N, 4, 2]
    debug: Dict[str, torch.Tensor]


def _edge_distance_point_to_segment_2d(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ab = b - a
    ap = p - a
    denom = (ab * ab).sum(dim=-1).clamp_min(1.0e-9)
    t = ((ap * ab).sum(dim=-1) / denom).clamp(0.0, 1.0)
    proj = a + t.unsqueeze(-1) * ab
    return torch.linalg.norm(p - proj, dim=-1)


def _polygon_signed_area(poly: torch.Tensor) -> torch.Tensor:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * torch.sum(x * torch.roll(y, shifts=-1, dims=0) - y * torch.roll(x, shifts=-1, dims=0))


def _sort_polygon_ccw(points: torch.Tensor) -> torch.Tensor:
    c = points.mean(dim=0)
    ang = torch.atan2(points[:, 1] - c[1], points[:, 0] - c[0])
    poly = points[torch.argsort(ang)]
    if _polygon_signed_area(poly) < 0:
        poly = torch.flip(poly, dims=[0])
    return poly


def _point_inside_convex_polygon(p: torch.Tensor, poly: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    if poly.shape[0] < 3:
        return torch.tensor(False, device=p.device)
    a = poly
    b = torch.roll(poly, shifts=-1, dims=0)
    edge = b - a
    rel = p.unsqueeze(0) - a
    cross = edge[:, 0] * rel[:, 1] - edge[:, 1] * rel[:, 0]
    return torch.all(cross >= -eps)


def _margin_to_polygon_edges(p: torch.Tensor, poly: torch.Tensor) -> torch.Tensor:
    if poly.shape[0] < 3:
        return torch.tensor(-1.0, device=p.device, dtype=p.dtype)
    a = poly
    b = torch.roll(poly, shifts=-1, dims=0)
    d = _edge_distance_point_to_segment_2d(p.unsqueeze(0).expand_as(a), a, b)
    inside = _point_inside_convex_polygon(p, poly)
    margin = torch.min(d)
    return torch.where(inside, margin, -margin)


def _limit_shift(prev_xy: torch.Tensor, target_xy: torch.Tensor, max_step: float) -> torch.Tensor:
    delta = target_xy - prev_xy
    norm = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1.0e-9)
    scale = torch.clamp(max_step / norm, max=1.0)
    return prev_xy + scale * delta


def compute_support_region_ref(
    foot_pos_w: torch.Tensor,
    base_pos_w: torch.Tensor,
    base_rpy_w: torch.Tensor,
    stance_mask: torch.Tensor,
    prev_base_ref: torch.Tensor | None = None,
    cfg: SupportRegionRefConfig | None = None,
) -> SupportRegionRefOutput:
    if cfg is None:
        cfg = SupportRegionRefConfig()

    device = foot_pos_w.device
    dtype = foot_pos_w.dtype
    n = foot_pos_w.shape[0]

    stance_bool = stance_mask > 0.5
    stance_count = stance_bool.sum(dim=1)
    current_xy = base_pos_w[:, 0:2]

    if prev_base_ref is None:
        prev_xy = current_xy
        prev_yaw = base_rpy_w[:, 2]
    else:
        # Rate-limit against the previous reference, not the raw current base.
        prev_xy = prev_base_ref[:, 0:2]
        prev_yaw = prev_base_ref[:, 5]

    base_ref = torch.zeros((n, 6), device=device, dtype=dtype)
    support_center_xy = torch.zeros((n, 2), device=device, dtype=dtype)
    target_xy = torch.zeros((n, 2), device=device, dtype=dtype)
    margin_to_edge = torch.zeros((n,), device=device, dtype=dtype)
    swing_allowed = torch.zeros((n,), device=device, dtype=torch.bool)
    polygon_xy = torch.full((n, 4, 2), float("nan"), device=device, dtype=dtype)

    for env_id in range(n):
        stance_points = foot_pos_w[env_id, stance_bool[env_id], 0:2]

        if stance_points.shape[0] >= cfg.min_stance_legs:
            poly = _sort_polygon_ccw(stance_points)
            center = poly.mean(dim=0)
            margin_val = _margin_to_polygon_edges(current_xy[env_id], poly)
            allowed = bool((margin_val > cfg.margin).item())
            polygon_xy[env_id, 0:poly.shape[0], :] = poly
        else:
            center = current_xy[env_id]
            margin_val = torch.tensor(-1.0, device=device, dtype=dtype)
            allowed = False

        raw_target = (1.0 - cfg.alpha) * current_xy[env_id] + cfg.alpha * center
        limited_target = _limit_shift(prev_xy[env_id:env_id + 1], raw_target.unsqueeze(0), cfg.max_shift_per_step)[0]

        dx = limited_target[0] - current_xy[env_id, 0]
        dy = limited_target[1] - current_xy[env_id, 1]

        roll_ref = torch.clamp(cfg.roll_gain_from_y_error * dy, -cfg.max_roll_ref, cfg.max_roll_ref)
        pitch_ref = torch.clamp(cfg.pitch_gain_from_x_error * dx, -cfg.max_pitch_ref, cfg.max_pitch_ref)

        support_center_xy[env_id] = center
        target_xy[env_id] = limited_target
        margin_to_edge[env_id] = margin_val
        swing_allowed[env_id] = allowed

        base_ref[env_id, 0] = limited_target[0]
        base_ref[env_id, 1] = limited_target[1]
        base_ref[env_id, 2] = cfg.height_ref
        base_ref[env_id, 3] = roll_ref
        base_ref[env_id, 4] = pitch_ref
        base_ref[env_id, 5] = prev_yaw[env_id]

    return SupportRegionRefOutput(
        base_ref=base_ref,
        support_center_xy=support_center_xy,
        current_xy=current_xy,
        target_xy=target_xy,
        margin_to_edge=margin_to_edge,
        swing_allowed=swing_allowed,
        stance_count=stance_count,
        polygon_xy=polygon_xy,
        debug={"stance_mask": stance_mask.detach().clone(), "stance_count": stance_count.detach().clone()},
    )
