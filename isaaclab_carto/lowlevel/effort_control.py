# isaaclab_carto/lowlevel/effort_control.py
#
# Effort-control utilities for B-step debugging.
#
# This is NOT MPC/WBC yet.

from __future__ import annotations

from typing import Dict

import torch


def make_zero_torque(robot) -> torch.Tensor:
    """Zero effort command."""
    return torch.zeros_like(robot.data.joint_pos)


def make_joint_pd_torque(
    robot,
    kp: float = 20.0,
    kd: float = 2.0,
    max_tau: float = 20.0,
    q_des: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Joint-space PD torque around default pose.

    Use this only as a torque-path sanity check.
    It is not a standing controller.
    """
    q = robot.data.joint_pos
    dq = robot.data.joint_vel

    if q_des is None:
        q_des = robot.data.default_joint_pos

    tau = kp * (q_des - q) - kd * dq
    return torch.clamp(tau, -max_tau, max_tau)


def get_spot_joint_name_to_index(robot) -> Dict[str, int]:
    joint_names = getattr(robot, "joint_names", None)
    if joint_names is None:
        raise RuntimeError("robot.joint_names not found.")
    return {name: i for i, name in enumerate(joint_names)}


def add_sine_probe_torque(
    tau: torch.Tensor,
    robot,
    step: int,
    dt: float,
    joint_name: str = "fl_hy",
    amp: float = 0.0,
    freq: float = 0.5,
) -> torch.Tensor:
    """
    Add tiny sine torque to one joint.
    Use amp <= 1.0 at first.
    """
    if amp == 0.0:
        return tau

    name_to_idx = get_spot_joint_name_to_index(robot)
    if joint_name not in name_to_idx:
        raise RuntimeError(f"Joint {joint_name} not found. Available: {list(name_to_idx.keys())}")

    idx = name_to_idx[joint_name]
    t = step * dt
    value = amp * torch.sin(
        torch.tensor(2.0 * 3.141592653589793 * freq * t, device=tau.device, dtype=tau.dtype)
    )
    tau[:, idx] += value
    return tau


def summarize_torque(tau: torch.Tensor) -> Dict[str, float]:
    return {
        "mean_abs": float(tau.abs().mean().detach().cpu()),
        "max_abs": float(tau.abs().max().detach().cpu()),
        "l2_mean": float(torch.linalg.norm(tau, dim=1).mean().detach().cpu()),
    }
