# isaaclab_carto/lowlevel/support_force_control.py
#
# B3-b support torque utilities.
#
# Goal:
#   Use Isaac Lab foot Jacobians to compute a small residual torque:
#
#       tau_support = J_foot^T f_support
#
# Important:
#   The output torque MUST be in Isaac Lab action joint order:
#
#       [fl_hx, fr_hx, hl_hx, hr_hx,
#        fl_hy, fr_hy, hl_hy, hr_hy,
#        fl_kn, fr_kn, hl_kn, hr_kn]
#
#   Earlier inspection code printed leg-grouped joint names for readability.
#   For actual action, this file uses the robot's native joint order directly.

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


SPOT_FOOT_NAMES = ["fl_foot", "fr_foot", "hl_foot", "hr_foot"]


def get_foot_indices(robot) -> Tuple[List[int], List[str]]:
    body_names = getattr(robot, "body_names", None)
    if body_names is None:
        raise RuntimeError("robot.body_names not found.")

    name_to_idx = {name: i for i, name in enumerate(body_names)}
    missing = [name for name in SPOT_FOOT_NAMES if name not in name_to_idx]
    if missing:
        raise RuntimeError(f"Missing foot names: {missing}. Available: {body_names}")

    return [name_to_idx[name] for name in SPOT_FOOT_NAMES], SPOT_FOOT_NAMES


def get_raw_jacobians(robot) -> torch.Tensor:
    view = getattr(robot, "root_physx_view", None)
    if view is None:
        view = getattr(robot, "_root_physx_view", None)
    if view is None:
        raise RuntimeError("root_physx_view not found.")

    if not hasattr(view, "get_jacobians"):
        raise RuntimeError("root_physx_view.get_jacobians not found.")

    return view.get_jacobians()


def extract_foot_jacobians_action_order(
    robot,
    linear_rows: str = "0_3",
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Extract foot Jacobians in native action joint order.

    Args:
        robot:
            Isaac Lab Articulation.
        linear_rows:
            "0_3" uses spatial rows 0:3.
            "3_6" uses spatial rows 3:6.
            Your inspect result showed rows 0:3 produce plausible tau_support,
            so use "0_3" first.

    Returns:
        Jv_feet:
            [num_envs, 4, 3, 12]
            foot linear Jacobian for feet [fl, fr, hl, hr],
            columns in robot.joint_names native action order.
        info:
            metadata.
    """
    J = get_raw_jacobians(robot)

    if J.ndim != 4:
        raise RuntimeError(f"Expected raw Jacobian [N, B, 6, C], got {tuple(J.shape)}")

    num_joints = robot.data.joint_pos.shape[1]
    C = J.shape[-1]

    if C == num_joints + 6:
        col_indices = list(range(6, 6 + num_joints))
        has_base_cols = True
    elif C == num_joints:
        col_indices = list(range(num_joints))
        has_base_cols = False
    else:
        # Best effort for floating-base articulation.
        if C >= num_joints + 6:
            col_indices = list(range(6, 6 + num_joints))
            has_base_cols = True
        else:
            raise RuntimeError(f"Cannot infer Jacobian columns. C={C}, num_joints={num_joints}")

    foot_indices, foot_names = get_foot_indices(robot)

    if linear_rows == "0_3":
        row_slice = slice(0, 3)
    elif linear_rows == "3_6":
        row_slice = slice(3, 6)
    else:
        raise ValueError(f"linear_rows must be '0_3' or '3_6', got {linear_rows}")

    Jv_feet = J[:, foot_indices, row_slice, :][:, :, :, col_indices]

    info = {
        "raw_shape": tuple(J.shape),
        "foot_indices": foot_indices,
        "foot_names": foot_names,
        "joint_names_action_order": list(getattr(robot, "joint_names", [])),
        "col_indices": col_indices,
        "has_base_cols": has_base_cols,
        "linear_rows": linear_rows,
        "Jv_norm": float(torch.linalg.norm(Jv_feet).detach().cpu()),
    }

    return Jv_feet, info


def make_vertical_support_forces(
    robot,
    fz_per_foot: float = 2.0,
    fx: float = 0.0,
    fy: float = 0.0,
    force_sign: float = 1.0,
) -> torch.Tensor:
    """
    Make per-foot support force.

    Args:
        fz_per_foot:
            Vertical force magnitude per foot.
            Start tiny, e.g. 1~3 N, because implicit PD is already stabilizing.
        fx, fy:
            Optional horizontal force per foot.
        force_sign:
            Use +1 first. If response is clearly opposite, test -1.

    Returns:
        f_feet: [num_envs, 4, 3]
    """
    num_envs = robot.data.joint_pos.shape[0]
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype

    f = torch.zeros((num_envs, 4, 3), device=device, dtype=dtype)
    f[:, :, 0] = force_sign * fx
    f[:, :, 1] = force_sign * fy
    f[:, :, 2] = force_sign * fz_per_foot
    return f


def compute_tau_jtf(Jv_feet: torch.Tensor, f_feet: torch.Tensor) -> torch.Tensor:
    """
    tau = sum_i J_i^T f_i

    Args:
        Jv_feet: [N, 4, 3, 12]
        f_feet: [N, 4, 3]

    Returns:
        tau: [N, 12] in action joint order.
    """
    return torch.einsum("nlik,nli->nk", Jv_feet, f_feet)


def make_support_torque(
    robot,
    fz_per_foot: float = 2.0,
    tau_scale: float = 1.0,
    max_tau: float = 3.0,
    linear_rows: str = "0_3",
    force_sign: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Compute small support residual torque.

    This is not full MPC/WBC.
    It is only B3-b sanity check for J^T f residual torque injection.

    Returns:
        tau_support: [N, 12] clipped.
        info: metadata.
    """
    Jv_feet, info = extract_foot_jacobians_action_order(robot, linear_rows=linear_rows)
    f_feet = make_vertical_support_forces(
        robot=robot,
        fz_per_foot=fz_per_foot,
        force_sign=force_sign,
    )

    tau = compute_tau_jtf(Jv_feet, f_feet)
    tau = tau_scale * tau
    tau = torch.clamp(tau, -max_tau, max_tau)

    info.update({
        "fz_per_foot": fz_per_foot,
        "tau_scale": tau_scale,
        "max_tau": max_tau,
        "force_sign": force_sign,
        "tau_mean_abs": float(tau.abs().mean().detach().cpu()),
        "tau_max_abs": float(tau.abs().max().detach().cpu()),
    })

    return tau, info
