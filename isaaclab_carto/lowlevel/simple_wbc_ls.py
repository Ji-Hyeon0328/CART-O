# isaaclab_carto/lowlevel/simple_wbc_ls.py
#
# B8-b: Simple WBC-LS style torque.
#
# This is not a full rigid-body-dynamics QP WBC yet.
# It is a bridge controller:
#
#   tau_total =
#       J_stance^T f_ref
#       + J_swing^T f_swing_task
#       + tau_posture
#
# where:
#   f_ref is produced by forceMPC v2
#   f_swing_task = Kp (xf_ref - xf) + Kd (xfd_ref - xfd)
#
# This is closer to WBC than the previous joint-space swing PD because
# swing tracking is now expressed in foot task space.
#
# Current limitations:
#   - No M(q) qdd + h = S^T tau + J^T f equality solve
#   - No contact acceleration constraint
#   - No true hierarchical QP
#   - Still residual torque on top of Isaac implicit PD

from __future__ import annotations

from typing import Dict, Tuple

import torch

from isaaclab_carto.lowlevel.support_force_control import (
    extract_foot_jacobians_action_order,
    compute_tau_jtf,
)


FOOT_NAMES = ["fl_foot", "fr_foot", "hl_foot", "hr_foot"]


def default_spot_joint_pose(device, dtype) -> torch.Tensor:
    q = torch.zeros((12,), device=device, dtype=dtype)
    q[4:8] = 0.65
    q[8:12] = -1.20
    return q


def get_foot_indices(robot):
    name_to_idx = {name: i for i, name in enumerate(robot.body_names)}
    return [name_to_idx[name] for name in FOOT_NAMES]


def extract_ref_foot_targets(ref: Dict[str, torch.Tensor], k: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ref.Xf_ref expected shape: [N, 3, 4, H]
    Ref.Xfd_ref expected shape: [N, 3, 4, H]

    Return:
        xf_ref:  [N,4,3]
        xfd_ref: [N,4,3]
    """
    Xf = ref["Xf_ref"]
    Xfd = ref["Xfd_ref"]
    H = Xf.shape[-1]
    k = min(max(k, 0), H - 1)
    xf_ref = Xf[:, :, :, k].permute(0, 2, 1).contiguous()
    xfd_ref = Xfd[:, :, :, k].permute(0, 2, 1).contiguous()
    return xf_ref, xfd_ref


def extract_masks(ref: Dict[str, torch.Tensor], k: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    S = ref["S"]
    H = S.shape[-1]
    k = min(max(k, 0), H - 1)
    stance = (S[:, :, k] > 0.5).to(S.dtype)
    swing = 1.0 - stance
    return stance, swing


def make_swing_task_force(
    robot,
    ref: Dict[str, torch.Tensor],
    k: int = 0,
    kp_swing_xyz=(30.0, 30.0, 45.0),
    kd_swing_xyz=(2.0, 2.0, 3.0),
    max_swing_force: float = 6.0,
    max_pos_err: float = 0.08,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Build task-space force-like term for swing foot tracking.

    Only swing legs receive f_swing.
    """
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype
    N = robot.data.joint_pos.shape[0]

    foot_indices = get_foot_indices(robot)
    xf = robot.data.body_pos_w[:, foot_indices, :]
    xfd = robot.data.body_lin_vel_w[:, foot_indices, :]

    xf_ref, xfd_ref = extract_ref_foot_targets(ref, k=k)
    stance, swing = extract_masks(ref, k=k)

    kp = torch.tensor(kp_swing_xyz, device=device, dtype=dtype).view(1, 1, 3)
    kd = torch.tensor(kd_swing_xyz, device=device, dtype=dtype).view(1, 1, 3)

    pos_err = torch.clamp(xf_ref - xf, -max_pos_err, max_pos_err)
    vel_err = xfd_ref - xfd

    f_swing = kp * pos_err + kd * vel_err
    f_swing = torch.clamp(f_swing, -max_swing_force, max_swing_force)
    f_swing = f_swing * swing.unsqueeze(-1)

    info = {
        "swing_mask_env0": swing[0].detach().cpu().tolist(),
        "xf_env0": xf[0].detach().cpu().tolist(),
        "xf_ref_env0": xf_ref[0].detach().cpu().tolist(),
        "xfd_env0": xfd[0].detach().cpu().tolist(),
        "xfd_ref_env0": xfd_ref[0].detach().cpu().tolist(),
        "swing_pos_err_env0": pos_err[0].detach().cpu().tolist(),
        "swing_vel_err_env0": vel_err[0].detach().cpu().tolist(),
        "f_swing_env0": f_swing[0].detach().cpu().tolist(),
        "kp_swing_xyz": list(kp_swing_xyz),
        "kd_swing_xyz": list(kd_swing_xyz),
        "max_swing_force": max_swing_force,
        "max_pos_err": max_pos_err,
        "f_swing_mean_abs": float(f_swing.abs().mean().detach().cpu()),
        "f_swing_max_abs": float(f_swing.abs().max().detach().cpu()),
    }

    return f_swing, info


def make_posture_regularization_torque(
    robot,
    kp_posture: float = 1.0,
    kd_posture: float = 0.05,
    max_posture_tau: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype
    N = robot.data.joint_pos.shape[0]

    q = robot.data.joint_pos
    dq = robot.data.joint_vel
    q_nom = default_spot_joint_pose(device, dtype).unsqueeze(0).repeat(N, 1)

    tau = kp_posture * (q_nom - q) - kd_posture * dq
    tau = torch.clamp(tau, -max_posture_tau, max_posture_tau)

    info = {
        "kp_posture": kp_posture,
        "kd_posture": kd_posture,
        "max_posture_tau": max_posture_tau,
        "tau_posture_env0": tau[0].detach().cpu().tolist(),
        "tau_posture_mean_abs": float(tau.abs().mean().detach().cpu()),
        "tau_posture_max_abs": float(tau.abs().max().detach().cpu()),
    }
    return tau, info


def make_simple_wbc_ls_torque(
    robot,
    ref: Dict[str, torch.Tensor],
    f_ref: torch.Tensor,
    k: int = 0,
    linear_rows: str = "0_3",
    kp_swing_xyz=(30.0, 30.0, 45.0),
    kd_swing_xyz=(2.0, 2.0, 3.0),
    max_swing_force: float = 6.0,
    max_pos_err: float = 0.08,
    swing_task_scale: float = 0.25,
    stance_force_scale: float = 1.0,
    enable_posture: bool = True,
    kp_posture: float = 1.0,
    kd_posture: float = 0.05,
    max_posture_tau: float = 0.25,
    max_total_tau: float = 3.0,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Compute simple WBC-LS style residual torque.
    """
    Jv_feet, j_info = extract_foot_jacobians_action_order(robot=robot, linear_rows=linear_rows)

    tau_stance = compute_tau_jtf(Jv_feet, f_ref) * stance_force_scale

    f_swing, swing_info = make_swing_task_force(
        robot=robot,
        ref=ref,
        k=k,
        kp_swing_xyz=kp_swing_xyz,
        kd_swing_xyz=kd_swing_xyz,
        max_swing_force=max_swing_force,
        max_pos_err=max_pos_err,
    )
    tau_swing = compute_tau_jtf(Jv_feet, f_swing) * swing_task_scale

    if enable_posture:
        tau_posture, posture_info = make_posture_regularization_torque(
            robot=robot,
            kp_posture=kp_posture,
            kd_posture=kd_posture,
            max_posture_tau=max_posture_tau,
        )
    else:
        tau_posture = torch.zeros_like(tau_stance)
        posture_info = {
            "kp_posture": kp_posture,
            "kd_posture": kd_posture,
            "max_posture_tau": max_posture_tau,
            "tau_posture_mean_abs": 0.0,
            "tau_posture_max_abs": 0.0,
        }

    tau_total = tau_stance + tau_swing + tau_posture
    tau_total = torch.clamp(tau_total, -max_total_tau, max_total_tau)

    info: Dict[str, object] = {
        "linear_rows": linear_rows,
        "stance_force_scale": stance_force_scale,
        "swing_task_scale": swing_task_scale,
        "enable_posture": enable_posture,
        "max_total_tau": max_total_tau,
        "tau_stance_env0": tau_stance[0].detach().cpu().tolist(),
        "tau_swing_env0": tau_swing[0].detach().cpu().tolist(),
        "tau_total_env0": tau_total[0].detach().cpu().tolist(),
        "tau_stance_mean_abs": float(tau_stance.abs().mean().detach().cpu()),
        "tau_stance_max_abs": float(tau_stance.abs().max().detach().cpu()),
        "tau_swing_mean_abs": float(tau_swing.abs().mean().detach().cpu()),
        "tau_swing_max_abs": float(tau_swing.abs().max().detach().cpu()),
        "tau_total_mean_abs": float(tau_total.abs().mean().detach().cpu()),
        "tau_total_max_abs": float(tau_total.abs().max().detach().cpu()),
    }
    info.update(j_info)
    info.update(swing_info)
    info.update(posture_info)

    return tau_total, info
