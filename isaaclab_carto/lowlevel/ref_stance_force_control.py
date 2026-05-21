# isaaclab_carto/lowlevel/ref_stance_force_control.py
#
# B4-a: Ref.S-based stance force distribution.
#
# This connects:
#
#   thetaDecoder / thetaRefMapper
#       → Ref.S stance schedule
#       → stance-only vertical support force
#       → tau = J_foot^T f_stance
#       → effort residual action
#
# This is still not full forceMPC.
# It is an MPC-shaped interface sanity check before porting MATLAB forceMPC.

from __future__ import annotations

from typing import Dict, Tuple

import torch

from isaaclab_carto.lowlevel.support_force_control import (
    extract_foot_jacobians_action_order,
    compute_tau_jtf,
)


def extract_stance_mask_from_ref(
    ref: Dict[str, torch.Tensor],
    use_k: int = 0,
    min_stance_legs: int = 2,
) -> torch.Tensor:
    """
    Extract stance mask from Ref.S.

    Args:
        ref:
            theta_ref_mapper output.
            ref["S"] shape: [num_envs, 4, H], 1=stance, 0=swing.
        use_k:
            Horizon index to use. For current-time action, use k=0.
        min_stance_legs:
            Safety guard. If stance legs are fewer than this, force all stance.

    Returns:
        stance_mask: [num_envs, 4]
    """
    S = ref["S"]
    H = S.shape[-1]
    k = min(max(use_k, 0), H - 1)

    stance = (S[:, :, k] > 0.5).to(S.dtype)

    # Safety guard for debug.
    num_stance = stance.sum(dim=1)
    unsafe = num_stance < float(min_stance_legs)
    if torch.any(unsafe):
        stance[unsafe, :] = 1.0

    return stance


def compute_ref_stance_forces(
    robot,
    ref: Dict[str, torch.Tensor],
    h_ref: float = 0.67,
    mass: float = 32.5,
    gravity: float = 9.81,
    kp_h: float = 40.0,
    kd_h: float = 8.0,
    residual_ratio: float = 0.02,
    max_fz_per_foot: float = 6.0,
    force_sign: float = 1.0,
    use_k: int = 0,
    min_stance_legs: int = 2,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Compute stance-only vertical residual forces from Ref.S.

    fz_total = m*g + Kp_h*(h_ref - h) + Kd_h*(0 - vz)
    fz_residual = residual_ratio * fz_total
    distribute among stance legs from Ref.S

    Returns:
        f_feet: [num_envs, 4, 3]
        info
    """
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype
    num_envs = robot.data.joint_pos.shape[0]

    stance = extract_stance_mask_from_ref(
        ref=ref,
        use_k=use_k,
        min_stance_legs=min_stance_legs,
    ).to(device=device, dtype=dtype)

    h = robot.data.root_pos_w[:, 2]
    vz = robot.data.root_lin_vel_w[:, 2]

    h_ref_t = torch.full_like(h, h_ref)
    h_err = h_ref_t - h

    fz_total = mass * gravity + kp_h * h_err + kd_h * (0.0 - vz)
    fz_total_residual = residual_ratio * fz_total

    num_stance = torch.clamp(stance.sum(dim=1), min=1.0)
    fz_per_stance = fz_total_residual / num_stance
    fz_per_stance = torch.clamp(fz_per_stance, -max_fz_per_foot, max_fz_per_foot)

    f_feet = torch.zeros((num_envs, 4, 3), device=device, dtype=dtype)
    f_feet[:, :, 2] = force_sign * fz_per_stance.unsqueeze(1) * stance

    info = {
        "h_ref": h_ref,
        "h_mean": float(h.mean().detach().cpu()),
        "vz_mean": float(vz.mean().detach().cpu()),
        "h_err_mean": float(h_err.mean().detach().cpu()),
        "mass": mass,
        "gravity": gravity,
        "kp_h": kp_h,
        "kd_h": kd_h,
        "residual_ratio": residual_ratio,
        "fz_total_mean": float(fz_total.mean().detach().cpu()),
        "fz_total_residual_mean": float(fz_total_residual.mean().detach().cpu()),
        "fz_per_stance_mean": float(fz_per_stance.mean().detach().cpu()),
        "fz_per_foot_min": float(f_feet[:, :, 2].min().detach().cpu()),
        "fz_per_foot_max": float(f_feet[:, :, 2].max().detach().cpu()),
        "force_sign": force_sign,
        "use_k": use_k,
        "min_stance_legs": min_stance_legs,
        "stance_mask_env0": stance[0].detach().cpu().tolist(),
        "num_stance_env0": float(num_stance[0].detach().cpu()),
    }

    return f_feet, info


def make_ref_stance_support_torque(
    robot,
    ref: Dict[str, torch.Tensor],
    h_ref: float = 0.67,
    mass: float = 32.5,
    gravity: float = 9.81,
    kp_h: float = 40.0,
    kd_h: float = 8.0,
    residual_ratio: float = 0.02,
    max_fz_per_foot: float = 6.0,
    tau_scale: float = 1.0,
    max_tau: float = 3.0,
    linear_rows: str = "0_3",
    force_sign: float = 1.0,
    use_k: int = 0,
    min_stance_legs: int = 2,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Compute tau = J^T f using Ref.S stance schedule.

    Returns:
        tau: [num_envs, 12] in Isaac action joint order.
        info
    """
    Jv_feet, j_info = extract_foot_jacobians_action_order(
        robot=robot,
        linear_rows=linear_rows,
    )

    f_feet, f_info = compute_ref_stance_forces(
        robot=robot,
        ref=ref,
        h_ref=h_ref,
        mass=mass,
        gravity=gravity,
        kp_h=kp_h,
        kd_h=kd_h,
        residual_ratio=residual_ratio,
        max_fz_per_foot=max_fz_per_foot,
        force_sign=force_sign,
        use_k=use_k,
        min_stance_legs=min_stance_legs,
    )

    tau = compute_tau_jtf(Jv_feet, f_feet)
    tau = tau_scale * tau
    tau = torch.clamp(tau, -max_tau, max_tau)

    info: Dict[str, object] = {}
    info.update(j_info)
    info.update(f_info)
    info.update({
        "tau_scale": tau_scale,
        "max_tau": max_tau,
        "linear_rows": linear_rows,
        "tau_mean_abs": float(tau.abs().mean().detach().cpu()),
        "tau_max_abs": float(tau.abs().max().detach().cpu()),
        "f_feet_env0": f_feet[0].detach().cpu().tolist(),
    })

    return tau, info
