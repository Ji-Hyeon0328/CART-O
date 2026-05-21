# isaaclab_carto/lowlevel/qps_wbc_bridge.py
#
# B8-c: QP-style WBC bridge.
#
# This is still not full floating-base rigid-body WBC.
# It builds a small joint-torque least-squares / QP-style bridge:
#
#   minimize_tau
#       w_f      ||tau - J_stance^T f_ref||^2
#     + w_swing  ||J_swing tau - y_swing||^2
#     + w_post   ||tau - tau_posture||^2
#     + w_reg    ||tau||^2
#     + w_rate   ||tau - tau_prev||^2
#
# subject approximately by projection:
#     |tau_i| <= max_total_tau
#
# Why this step:
#   Previous B8-b just added terms:
#       tau = J^T f_ref + J^T f_swing + tau_posture
#   B8-c instead lets the terms compete in a single weighted LS objective.
#
# Important:
#   - This is a bridge toward WBC-QP.
#   - It does not yet use floating-base M(q) qdd + h = S^T tau + Jc^T f.
#   - It is useful to debug objective balance, torque limits, and task-space coupling.

from __future__ import annotations

from typing import Dict, Tuple

import torch

from isaaclab_carto.lowlevel.support_force_control import (
    extract_foot_jacobians_action_order,
    compute_tau_jtf,
)
from isaaclab_carto.lowlevel.simple_wbc_ls import (
    extract_masks,
    extract_ref_foot_targets,
    get_foot_indices,
    default_spot_joint_pose,
)


class QPSWBCBridgeState:
    def __init__(self):
        self.prev_tau = None

    def reset(self):
        self.prev_tau = None


def build_posture_tau(
    robot,
    kp_posture: float = 0.8,
    kd_posture: float = 0.04,
    max_posture_tau: float = 0.25,
) -> torch.Tensor:
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype
    N = robot.data.joint_pos.shape[0]

    q = robot.data.joint_pos
    dq = robot.data.joint_vel
    q_nom = default_spot_joint_pose(device, dtype).unsqueeze(0).repeat(N, 1)

    tau = kp_posture * (q_nom - q) - kd_posture * dq
    return torch.clamp(tau, -max_posture_tau, max_posture_tau)


def build_swing_task_target(
    robot,
    ref: Dict[str, torch.Tensor],
    k: int = 0,
    kp_swing_xyz=(20.0, 20.0, 35.0),
    kd_swing_xyz=(1.5, 1.5, 2.5),
    max_task_cmd: float = 4.0,
    max_pos_err: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    """
    Build y_swing for the LS term:
        J_swing tau ≈ y_swing

    This y_swing is not a physical force; it is a task command vector shaped
    like foot-space effort. It lets the LS solve choose torque satisfying
    swing task while balancing force/posture/rate objectives.
    """
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype

    foot_indices = get_foot_indices(robot)
    xf = robot.data.body_pos_w[:, foot_indices, :]
    xfd = robot.data.body_lin_vel_w[:, foot_indices, :]

    xf_ref, xfd_ref = extract_ref_foot_targets(ref, k=k)
    _stance, swing = extract_masks(ref, k=k)

    kp = torch.tensor(kp_swing_xyz, device=device, dtype=dtype).view(1, 1, 3)
    kd = torch.tensor(kd_swing_xyz, device=device, dtype=dtype).view(1, 1, 3)

    pos_err = torch.clamp(xf_ref - xf, -max_pos_err, max_pos_err)
    vel_err = xfd_ref - xfd

    y = kp * pos_err + kd * vel_err
    y = torch.clamp(y, -max_task_cmd, max_task_cmd)
    y = y * swing.unsqueeze(-1)

    info = {
        "swing_mask_env0": swing[0].detach().cpu().tolist(),
        "xf_env0": xf[0].detach().cpu().tolist(),
        "xf_ref_env0": xf_ref[0].detach().cpu().tolist(),
        "swing_pos_err_env0": pos_err[0].detach().cpu().tolist(),
        "swing_vel_err_env0": vel_err[0].detach().cpu().tolist(),
        "y_swing_env0": y[0].detach().cpu().tolist(),
        "y_swing_mean_abs": float(y.abs().mean().detach().cpu()),
        "y_swing_max_abs": float(y.abs().max().detach().cpu()),
        "kp_swing_xyz": list(kp_swing_xyz),
        "kd_swing_xyz": list(kd_swing_xyz),
        "max_task_cmd": max_task_cmd,
        "max_pos_err": max_pos_err,
    }

    return y, swing, info


def solve_weighted_ls_per_env(
    A_blocks,
    b_blocks,
    reg_eps: float = 1e-6,
) -> torch.Tensor:
    """
    Solve min ||A tau - b||^2 for one env.
    A_blocks: list of [m,12]
    b_blocks: list of [m]
    """
    A = torch.cat(A_blocks, dim=0)
    b = torch.cat(b_blocks, dim=0)

    H = A.T @ A + reg_eps * torch.eye(A.shape[1], device=A.device, dtype=A.dtype)
    g = A.T @ b

    try:
        tau = torch.linalg.solve(H, g)
    except Exception:
        tau = torch.linalg.pinv(H) @ g
    return tau


def make_qps_wbc_bridge_torque(
    robot,
    ref: Dict[str, torch.Tensor],
    f_ref: torch.Tensor,
    state: QPSWBCBridgeState | None = None,
    k: int = 0,
    linear_rows: str = "0_3",
    kp_swing_xyz=(20.0, 20.0, 35.0),
    kd_swing_xyz=(1.5, 1.5, 2.5),
    max_task_cmd: float = 4.0,
    max_pos_err: float = 0.05,
    kp_posture: float = 0.8,
    kd_posture: float = 0.04,
    max_posture_tau: float = 0.20,
    w_force: float = 1.0,
    w_swing: float = 0.10,
    w_posture: float = 0.40,
    w_reg: float = 0.02,
    w_rate: float = 0.25,
    max_total_tau: float = 3.0,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype
    N = robot.data.joint_pos.shape[0]

    Jv_feet, j_info = extract_foot_jacobians_action_order(robot=robot, linear_rows=linear_rows)

    stance, swing = extract_masks(ref, k=k)
    tau_force = compute_tau_jtf(Jv_feet, f_ref)

    y_swing, swing_mask, swing_info = build_swing_task_target(
        robot=robot,
        ref=ref,
        k=k,
        kp_swing_xyz=kp_swing_xyz,
        kd_swing_xyz=kd_swing_xyz,
        max_task_cmd=max_task_cmd,
        max_pos_err=max_pos_err,
    )

    tau_post = build_posture_tau(
        robot=robot,
        kp_posture=kp_posture,
        kd_posture=kd_posture,
        max_posture_tau=max_posture_tau,
    )

    tau_prev = None
    if state is not None and state.prev_tau is not None:
        tau_prev = state.prev_tau.to(device=device, dtype=dtype)
    else:
        tau_prev = torch.zeros((N, 12), device=device, dtype=dtype)

    tau_out = torch.zeros((N, 12), device=device, dtype=dtype)

    I = torch.eye(12, device=device, dtype=dtype)

    for env_id in range(N):
        A_blocks = []
        b_blocks = []

        # 1) Force tracking in torque space: tau ≈ J^T f_ref
        A_blocks.append((w_force ** 0.5) * I)
        b_blocks.append((w_force ** 0.5) * tau_force[env_id])

        # 2) Swing task in foot-space: J_swing tau ≈ y_swing.
        # Use block rows only for swing legs.
        for leg in range(4):
            if swing[env_id, leg] > 0.5:
                J_leg = Jv_feet[env_id, leg, :, :]  # [3,12]
                y_leg = y_swing[env_id, leg, :]     # [3]
                A_blocks.append((w_swing ** 0.5) * J_leg)
                b_blocks.append((w_swing ** 0.5) * y_leg)

        # 3) Posture regularization: tau ≈ tau_post
        A_blocks.append((w_posture ** 0.5) * I)
        b_blocks.append((w_posture ** 0.5) * tau_post[env_id])

        # 4) Torque regularization: tau ≈ 0
        A_blocks.append((w_reg ** 0.5) * I)
        b_blocks.append(torch.zeros((12,), device=device, dtype=dtype))

        # 5) Torque rate: tau ≈ tau_prev
        A_blocks.append((w_rate ** 0.5) * I)
        b_blocks.append((w_rate ** 0.5) * tau_prev[env_id])

        tau = solve_weighted_ls_per_env(A_blocks, b_blocks)
        tau_out[env_id] = tau

    tau_out = torch.clamp(tau_out, -max_total_tau, max_total_tau)

    if state is not None:
        state.prev_tau = tau_out.detach().clone()

    info: Dict[str, object] = {
        "linear_rows": linear_rows,
        "stance_mask_env0": stance[0].detach().cpu().tolist(),
        "swing_mask_env0": swing[0].detach().cpu().tolist(),
        "w_force": w_force,
        "w_swing": w_swing,
        "w_posture": w_posture,
        "w_reg": w_reg,
        "w_rate": w_rate,
        "max_total_tau": max_total_tau,
        "tau_force_env0": tau_force[0].detach().cpu().tolist(),
        "tau_posture_env0": tau_post[0].detach().cpu().tolist(),
        "tau_prev_env0": tau_prev[0].detach().cpu().tolist(),
        "tau_qps_wbc_env0": tau_out[0].detach().cpu().tolist(),
        "tau_force_mean_abs": float(tau_force.abs().mean().detach().cpu()),
        "tau_force_max_abs": float(tau_force.abs().max().detach().cpu()),
        "tau_posture_mean_abs": float(tau_post.abs().mean().detach().cpu()),
        "tau_posture_max_abs": float(tau_post.abs().max().detach().cpu()),
        "tau_qps_wbc_mean_abs": float(tau_out.abs().mean().detach().cpu()),
        "tau_qps_wbc_max_abs": float(tau_out.abs().max().detach().cpu()),
    }
    info.update(j_info)
    info.update(swing_info)
    return tau_out, info
