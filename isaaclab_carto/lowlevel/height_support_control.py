# isaaclab_carto/lowlevel/height_support_control.py
#
# B3-c height-feedback support torque.
#
# Controller idea:
#
#   fz_total = m*g + Kp_h * (h_ref - h) + Kd_h * (0 - vz)
#   fz_per_foot = fz_total / num_stance_feet
#   tau = J_foot^T f_foot
#
# Important:
#   This is still NOT full MPC/WBC.
#   It is a small residual support torque layer on top of implicit PD.
#
#   Use very small residual scaling first because implicit PD is already
#   doing most of the standing stabilization.

from __future__ import annotations

from typing import Dict, Tuple

import torch

from isaaclab_carto.lowlevel.support_force_control import (
    extract_foot_jacobians_action_order,
    compute_tau_jtf,
)


def estimate_robot_mass_from_name(robot, default_mass: float = 32.5) -> float:
    """
    Simple default mass for Spot-like robot.

    The exact USD mass may be different. This is only used for a residual
    support-force sanity test, not for final MPC/WBC.
    """
    return default_mass


def make_stance_mask_all(robot) -> torch.Tensor:
    """
    All-feet stance mask.

    Returns:
        stance_mask: [num_envs, 4]
    """
    num_envs = robot.data.joint_pos.shape[0]
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype
    return torch.ones((num_envs, 4), device=device, dtype=dtype)


def compute_height_feedback_forces(
    robot,
    h_ref: float = 0.67,
    mass: float = 32.5,
    gravity: float = 9.81,
    kp_h: float = 40.0,
    kd_h: float = 8.0,
    residual_ratio: float = 0.02,
    min_fz_per_foot: float = -5.0,
    max_fz_per_foot: float = 5.0,
    stance_mask: torch.Tensor | None = None,
    force_sign: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Compute per-foot vertical residual force from body height error.

    Args:
        robot:
            Isaac Lab articulation.
        h_ref:
            Desired base height in world z.
            From B0.5/B3 logs, stable height was roughly 0.673.
        mass:
            Approx robot mass. Used to form nominal mg.
        gravity:
            Gravity magnitude.
        kp_h, kd_h:
            Height feedback gains on total vertical force.
        residual_ratio:
            Only inject this ratio of the total computed support force.
            Start tiny: 0.01~0.03.
        min_fz_per_foot, max_fz_per_foot:
            Clamp residual force per foot.
        stance_mask:
            [num_envs, 4], 1=stance, 0=swing.
            For B3-c use all stance.
        force_sign:
            +1 or -1. Use +1 first.

    Returns:
        f_feet:
            [num_envs, 4, 3]
        info:
            diagnostics.
    """
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype
    num_envs = robot.data.joint_pos.shape[0]

    if stance_mask is None:
        stance_mask = make_stance_mask_all(robot)

    h = robot.data.root_pos_w[:, 2]
    vz = robot.data.root_lin_vel_w[:, 2]

    h_ref_t = torch.full_like(h, h_ref)
    h_err = h_ref_t - h

    # Total support-like force.
    # This includes mg, but we inject only a tiny residual_ratio.
    fz_total = mass * gravity + kp_h * h_err + kd_h * (0.0 - vz)
    fz_total_residual = residual_ratio * fz_total

    num_stance = torch.clamp(stance_mask.sum(dim=1), min=1.0)
    fz_per_stance = fz_total_residual / num_stance

    fz_per_stance = torch.clamp(
        fz_per_stance,
        min=min_fz_per_foot,
        max=max_fz_per_foot,
    )

    f_feet = torch.zeros((num_envs, 4, 3), device=device, dtype=dtype)
    f_feet[:, :, 2] = force_sign * fz_per_stance.unsqueeze(1) * stance_mask

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
    }

    return f_feet, info


def make_height_feedback_support_torque(
    robot,
    h_ref: float = 0.67,
    mass: float = 32.5,
    gravity: float = 9.81,
    kp_h: float = 40.0,
    kd_h: float = 8.0,
    residual_ratio: float = 0.02,
    max_fz_per_foot: float = 5.0,
    tau_scale: float = 1.0,
    max_tau: float = 2.0,
    linear_rows: str = "0_3",
    force_sign: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Make height-feedback support residual torque.

    Returns:
        tau: [num_envs, 12] in Isaac action joint order.
        info: diagnostics.
    """
    Jv_feet, j_info = extract_foot_jacobians_action_order(
        robot=robot,
        linear_rows=linear_rows,
    )

    f_feet, f_info = compute_height_feedback_forces(
        robot=robot,
        h_ref=h_ref,
        mass=mass,
        gravity=gravity,
        kp_h=kp_h,
        kd_h=kd_h,
        residual_ratio=residual_ratio,
        min_fz_per_foot=-max_fz_per_foot,
        max_fz_per_foot=max_fz_per_foot,
        stance_mask=None,
        force_sign=force_sign,
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
    })

    return tau, info
