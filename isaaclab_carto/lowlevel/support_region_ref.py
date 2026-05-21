from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch

@dataclass
class SupportRegionRefConfig:
    alpha: float = 0.45
    margin: float = 0.03
    max_shift_per_step: float = 0.004
    min_stance_legs: int = 3
    height_ref: float = 0.54
    roll_gain_from_y_error: float = 0.0
    pitch_gain_from_x_error: float = 0.0
    max_roll_ref: float = 0.10
    max_pitch_ref: float = 0.10

@dataclass
class SupportRegionRefOutput:
    base_ref: torch.Tensor
    support_center_xy: torch.Tensor
    current_xy: torch.Tensor
    target_xy: torch.Tensor
    margin_to_edge: torch.Tensor
    swing_allowed: torch.Tensor
    stance_count: torch.Tensor
    polygon_xy: torch.Tensor
    debug: Dict[str, torch.Tensor]

def _sort_ccw(points: torch.Tensor) -> torch.Tensor:
    c = points.mean(dim=0)
    ang = torch.atan2(points[:, 1] - c[1], points[:, 0] - c[0])
    poly = points[torch.argsort(ang)]
    x, y = poly[:, 0], poly[:, 1]
    area = 0.5 * torch.sum(x * torch.roll(y, -1) - y * torch.roll(x, -1))
    return torch.flip(poly, dims=[0]) if area < 0 else poly

def _dist_to_segment(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ab = b - a
    ap = p - a
    t = ((ap * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-9)).clamp(0.0, 1.0)
    proj = a + t.unsqueeze(-1) * ab
    return torch.linalg.norm(p - proj, dim=-1)

def _inside_convex_ccw(p: torch.Tensor, poly: torch.Tensor) -> torch.Tensor:
    if poly.shape[0] < 3:
        return torch.tensor(False, device=p.device)
    a = poly
    b = torch.roll(poly, -1, dims=0)
    e = b - a
    r = p.unsqueeze(0) - a
    cross = e[:, 0] * r[:, 1] - e[:, 1] * r[:, 0]
    return torch.all(cross >= -1e-9)

def _signed_margin(p: torch.Tensor, poly: torch.Tensor) -> torch.Tensor:
    if poly.shape[0] < 3:
        return torch.tensor(-1.0, device=p.device, dtype=p.dtype)
    a = poly
    b = torch.roll(poly, -1, dims=0)
    d = _dist_to_segment(p.unsqueeze(0).expand_as(a), a, b).min()
    return torch.where(_inside_convex_ccw(p, poly), d, -d)

def _limit_step(prev: torch.Tensor, target: torch.Tensor, max_step: float) -> torch.Tensor:
    delta = target - prev
    n = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-9)
    return prev + torch.clamp(max_step / n, max=1.0) * delta

def compute_support_region_ref(
    foot_pos_w: torch.Tensor,        # [N,4,3], LF RF LH RH
    base_pos_w: torch.Tensor,        # [N,3]
    base_rpy_w: torch.Tensor,        # [N,3]
    stance_mask: torch.Tensor,       # [N,4], 1 stance 0 swing
    prev_base_ref: torch.Tensor | None = None,
    cfg: SupportRegionRefConfig | None = None,
) -> SupportRegionRefOutput:
    if cfg is None:
        cfg = SupportRegionRefConfig()
    device, dtype = foot_pos_w.device, foot_pos_w.dtype
    nenv = foot_pos_w.shape[0]
    stance_bool = stance_mask > 0.5
    stance_count = stance_bool.sum(dim=1)
    current_xy = base_pos_w[:, 0:2]
    prev_xy = current_xy if prev_base_ref is None else prev_base_ref[:, 0:2]
    prev_yaw = base_rpy_w[:, 2] if prev_base_ref is None else prev_base_ref[:, 5]

    base_ref = torch.zeros((nenv, 6), device=device, dtype=dtype)
    support_center_xy = torch.zeros((nenv, 2), device=device, dtype=dtype)
    target_xy = torch.zeros((nenv, 2), device=device, dtype=dtype)
    margin_to_edge = torch.zeros((nenv,), device=device, dtype=dtype)
    swing_allowed = torch.zeros((nenv,), device=device, dtype=torch.bool)
    polygon_xy = torch.full((nenv, 4, 2), float('nan'), device=device, dtype=dtype)

    for e in range(nenv):
        pts = foot_pos_w[e, stance_bool[e], 0:2]
        if pts.shape[0] >= cfg.min_stance_legs:
            poly = _sort_ccw(pts)
            center = poly.mean(0)
            margin = _signed_margin(current_xy[e], poly)
            polygon_xy[e, :poly.shape[0]] = poly
            allowed = bool((margin > cfg.margin).item())
        else:
            center = current_xy[e]
            margin = torch.tensor(-1.0, device=device, dtype=dtype)
            allowed = False
        raw = (1.0 - cfg.alpha) * current_xy[e] + cfg.alpha * center
        tgt = _limit_step(prev_xy[e:e+1], raw.unsqueeze(0), cfg.max_shift_per_step)[0]
        support_center_xy[e] = center
        target_xy[e] = tgt
        margin_to_edge[e] = margin
        swing_allowed[e] = allowed
        dx, dy = tgt[0] - current_xy[e, 0], tgt[1] - current_xy[e, 1]
        base_ref[e, 0] = tgt[0]
        base_ref[e, 1] = tgt[1]
        base_ref[e, 2] = cfg.height_ref
        base_ref[e, 3] = torch.clamp(cfg.roll_gain_from_y_error * dy, -cfg.max_roll_ref, cfg.max_roll_ref)
        base_ref[e, 4] = torch.clamp(cfg.pitch_gain_from_x_error * dx, -cfg.max_pitch_ref, cfg.max_pitch_ref)
        base_ref[e, 5] = prev_yaw[e]
    return SupportRegionRefOutput(base_ref, support_center_xy, current_xy, target_xy, margin_to_edge,
                                  swing_allowed, stance_count, polygon_xy,
                                  {"stance_mask": stance_mask.detach().clone()})
