# isaaclab_carto/lowlevel/force_mpc_objectives.py
#
# B5-e: beta objective cleanup.
#
# This file centralizes how beta_t = [beta_h, beta_v, beta_e]
# affects low-level forceMPC objectives.
#
# Intended semantics:
#
#   beta_h: height / base stability emphasis
#   beta_v: command velocity tracking emphasis
#   beta_e: energy / force effort / smoothness emphasis
#
# This lets the Objective Selector later connect to forceMPC cleanly.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class ForceMPCObjectiveConfig:
    # Height / vertical acceleration PD
    base_kp_h: float = 45.0
    base_kd_h: float = 10.0

    # Horizontal velocity tracking gain
    base_kp_vxy: float = 4.0

    # QP objective weights
    w_z_base: float = 5.0
    w_z_gain: float = 20.0

    w_xy_base: float = 2.0
    w_xy_gain: float = 15.0

    w_force_base: float = 1e-3
    w_force_gain: float = 6e-3

    # Force-rate/smoothing behavior
    rate_scale_gain: float = 0.45
    smoothing_gain: float = 0.10

    # Optional global multipliers
    height_gain_beta_scale: float = 1.5
    velocity_gain_beta_scale: float = 1.5


def normalize_beta(beta_t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    beta_t = torch.clamp(beta_t, min=eps)
    return beta_t / torch.clamp(beta_t.sum(dim=1, keepdim=True), min=eps)


def beta_to_objective_terms(
    beta_t: torch.Tensor,
    cfg: ForceMPCObjectiveConfig | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Map beta_t to forceMPC objective terms.

    Args:
        beta_t: [N,3] with [height/stability, velocity, energy]
        cfg

    Returns:
        Dictionary of tensors, each [N]
    """
    if cfg is None:
        cfg = ForceMPCObjectiveConfig()

    beta = normalize_beta(beta_t)
    beta_h = beta[:, 0]
    beta_v = beta[:, 1]
    beta_e = beta[:, 2]

    kp_h = cfg.base_kp_h * (0.5 + cfg.height_gain_beta_scale * beta_h)
    kd_h = cfg.base_kd_h * (0.5 + cfg.height_gain_beta_scale * beta_h)
    kp_vxy = cfg.base_kp_vxy * (0.5 + cfg.velocity_gain_beta_scale * beta_v)

    w_acc_z = cfg.w_z_base + cfg.w_z_gain * beta_h
    w_acc_xy = cfg.w_xy_base + cfg.w_xy_gain * beta_v
    w_force = cfg.w_force_base + cfg.w_force_gain * beta_e

    # Larger beta_e should discourage sudden force change.
    rate_scale = torch.clamp(1.0 - cfg.rate_scale_gain * beta_e, min=0.25, max=1.0)

    # Larger beta_e should smooth more.
    smoothing_extra = cfg.smoothing_gain * beta_e

    return {
        "beta": beta,
        "beta_h": beta_h,
        "beta_v": beta_v,
        "beta_e": beta_e,
        "kp_h": kp_h,
        "kd_h": kd_h,
        "kp_vxy": kp_vxy,
        "w_acc_z": w_acc_z,
        "w_acc_xy": w_acc_xy,
        "w_force": w_force,
        "rate_scale": rate_scale,
        "smoothing_extra": smoothing_extra,
    }


def make_beta_preset(num_envs: int, device, dtype, preset: str) -> torch.Tensor:
    if preset == "height":
        beta = torch.tensor([0.70, 0.20, 0.10], device=device, dtype=dtype)
    elif preset == "velocity":
        beta = torch.tensor([0.25, 0.65, 0.10], device=device, dtype=dtype)
    elif preset == "energy":
        beta = torch.tensor([0.25, 0.20, 0.55], device=device, dtype=dtype)
    elif preset == "balanced":
        beta = torch.tensor([0.45, 0.35, 0.20], device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown beta preset: {preset}")
    return beta.unsqueeze(0).repeat(num_envs, 1)


def summarize_objective_terms(terms: Dict[str, torch.Tensor], env_id: int = 0) -> Dict[str, object]:
    keys = [
        "beta",
        "kp_h",
        "kd_h",
        "kp_vxy",
        "w_acc_z",
        "w_acc_xy",
        "w_force",
        "rate_scale",
        "smoothing_extra",
    ]
    out: Dict[str, object] = {}
    for key in keys:
        value = terms[key]
        if value.ndim == 1:
            out[f"{key}_env{env_id}"] = float(value[env_id].detach().cpu())
        else:
            out[f"{key}_env{env_id}"] = value[env_id].detach().cpu().tolist()
    return out


def build_accel_target(
    x_hat: torch.Tensor,
    u_cmd: torch.Tensor,
    beta_t: torch.Tensor,
    h_ref: float,
    gravity: float,
    cfg: ForceMPCObjectiveConfig | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Build desired centroidal acceleration target:
        [ax, ay, az, alpha_x, alpha_y, alpha_z]

    Current model:
        ax/ay: velocity tracking
        az: gravity + height PD
        angular target: zero
    """
    terms = beta_to_objective_terms(beta_t, cfg=cfg)

    N = x_hat.shape[0]
    device = x_hat.device
    dtype = x_hat.dtype

    h = x_hat[:, 2]
    v_world_xy = x_hat[:, 6:8]
    vz = x_hat[:, 8]

    v_des_xy = u_cmd[:, 0:2]
    v_err_xy = v_des_xy - v_world_xy
    h_err = torch.full_like(h, h_ref) - h

    a_des = torch.zeros((N, 6), device=device, dtype=dtype)
    a_des[:, 0:2] = terms["kp_vxy"].unsqueeze(1) * v_err_xy
    a_des[:, 2] = gravity + terms["kp_h"] * h_err + terms["kd_h"] * (0.0 - vz)
    a_des[:, 3:6] = 0.0

    aux: Dict[str, torch.Tensor] = dict(terms)
    aux.update({
        "h": h,
        "vz": vz,
        "v_des_xy": v_des_xy,
        "v_world_xy": v_world_xy,
        "v_err_xy": v_err_xy,
        "h_err": h_err,
    })

    return a_des, aux


def build_weight_vector(
    beta_t: torch.Tensor,
    cfg: ForceMPCObjectiveConfig | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Build diagonal weights for 6D acceleration objective:
        [w_xy, w_xy, w_z, w_ang, w_ang, w_ang]
    """
    terms = beta_to_objective_terms(beta_t, cfg=cfg)
    N = beta_t.shape[0]
    device = beta_t.device
    dtype = beta_t.dtype

    W = torch.zeros((N, 6), device=device, dtype=dtype)
    W[:, 0] = terms["w_acc_xy"]
    W[:, 1] = terms["w_acc_xy"]
    W[:, 2] = terms["w_acc_z"]
    W[:, 3:6] = 0.05

    return W, terms
