# isaaclab_carto/lowlevel/swing_tracking_effort.py
#
# B7-b: swing tracking as residual effort.
#
# This is a bridge between:
#   B7-a visual swing joint-position offsets
# and:
#   future WBC swing foot tracking.
#
# It computes small swing-leg joint targets from Ref.S / Ref.phase,
# then converts them to effort residual:
#
#   tau_swing = Kp * (q_des - q) - Kd * dq
#
# Only swing HY/KN joints receive this residual.
# Stance legs receive zero swing residual.
#
# This is not full IK/WBC yet.

from __future__ import annotations

from typing import Dict, Tuple

import torch


# Native action/joint order observed in Isaac Spot:
# [fl_hx, fr_hx, hl_hx, hr_hx,
#  fl_hy, fr_hy, hl_hy, hr_hy,
#  fl_kn, fr_kn, hl_kn, hr_kn]
HX_IDX = torch.tensor([0, 1, 2, 3], dtype=torch.long)
HY_IDX = torch.tensor([4, 5, 6, 7], dtype=torch.long)
KN_IDX = torch.tensor([8, 9, 10, 11], dtype=torch.long)


def default_spot_joint_pose(device, dtype) -> torch.Tensor:
    q = torch.zeros((12,), device=device, dtype=dtype)
    q[4:8] = 0.65
    q[8:12] = -1.20
    return q


def compute_swing_phase(ref: Dict[str, torch.Tensor], theta, k: int = 0):
    S = ref["S"]
    H = S.shape[-1]
    k = min(max(k, 0), H - 1)

    phase = ref["phase"][:, :, k]
    duty = theta.gait["duty_i"]

    swing_mask = (S[:, :, k] < 0.5).to(S.dtype)

    denom = torch.clamp(1.0 - duty, min=1e-5)
    progress = torch.clamp((phase - duty) / denom, min=0.0, max=1.0)
    progress = progress * swing_mask

    lift = torch.sin(torch.pi * progress) * swing_mask
    return swing_mask, progress, lift


def make_swing_joint_target(
    ref: Dict[str, torch.Tensor],
    theta,
    k: int = 0,
    hy_lift_delta: float = -0.035,
    kn_lift_delta: float = 0.08,
    hy_sweep_delta: float = 0.010,
    lift_sign: float = 1.0,
    knee_sign: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Build desired joint target for swing legs.

    Returns:
        q_des: [N,12]
        info
    """
    device = ref["S"].device
    dtype = ref["S"].dtype
    N = ref["S"].shape[0]

    q_nom = default_spot_joint_pose(device, dtype).unsqueeze(0).repeat(N, 1)

    swing_mask, progress, lift = compute_swing_phase(ref, theta, k=k)

    sweep = (2.0 * progress - 1.0) * swing_mask

    hy_offset = lift_sign * hy_lift_delta * lift + hy_sweep_delta * sweep
    kn_offset = knee_sign * kn_lift_delta * lift

    hy_idx = HY_IDX.to(device=device)
    kn_idx = KN_IDX.to(device=device)

    q_des = q_nom.clone()
    q_des[:, hy_idx] = q_nom[:, hy_idx] + hy_offset
    q_des[:, kn_idx] = q_nom[:, kn_idx] + kn_offset

    info = {
        "swing_mask_env0": swing_mask[0].detach().cpu().tolist(),
        "swing_progress_env0": progress[0].detach().cpu().tolist(),
        "lift_profile_env0": lift[0].detach().cpu().tolist(),
        "hy_offset_env0": hy_offset[0].detach().cpu().tolist(),
        "kn_offset_env0": kn_offset[0].detach().cpu().tolist(),
        "q_des_env0": q_des[0].detach().cpu().tolist(),
        "hy_lift_delta": hy_lift_delta,
        "kn_lift_delta": kn_lift_delta,
        "hy_sweep_delta": hy_sweep_delta,
        "lift_sign": lift_sign,
        "knee_sign": knee_sign,
    }
    return q_des, info


def make_swing_tracking_torque(
    robot,
    ref: Dict[str, torch.Tensor],
    theta,
    k: int = 0,
    kp_swing: float = 12.0,
    kd_swing: float = 0.8,
    max_swing_tau: float = 1.0,
    hy_lift_delta: float = -0.035,
    kn_lift_delta: float = 0.08,
    hy_sweep_delta: float = 0.010,
    lift_sign: float = 1.0,
    knee_sign: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Make residual torque for swing tracking.

    Only HY and KN joints of swing legs are actuated by this term.
    """
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype
    N = robot.data.joint_pos.shape[0]

    q = robot.data.joint_pos
    dq = robot.data.joint_vel

    q_des, info = make_swing_joint_target(
        ref=ref,
        theta=theta,
        k=k,
        hy_lift_delta=hy_lift_delta,
        kn_lift_delta=kn_lift_delta,
        hy_sweep_delta=hy_sweep_delta,
        lift_sign=lift_sign,
        knee_sign=knee_sign,
    )

    swing_mask = torch.tensor(info["swing_mask_env0"], device=device, dtype=dtype).unsqueeze(0)
    if N > 1:
        # Recompute full mask for all envs.
        swing_mask, _progress, _lift = compute_swing_phase(ref, theta, k=k)

    tau = torch.zeros((N, 12), device=device, dtype=dtype)

    hy_idx = HY_IDX.to(device=device)
    kn_idx = KN_IDX.to(device=device)

    q_err = q_des - q

    # Per-leg mask assigned to corresponding HY/KN joints.
    tau[:, hy_idx] = kp_swing * q_err[:, hy_idx] - kd_swing * dq[:, hy_idx]
    tau[:, kn_idx] = kp_swing * q_err[:, kn_idx] - kd_swing * dq[:, kn_idx]

    tau[:, hy_idx] = tau[:, hy_idx] * swing_mask
    tau[:, kn_idx] = tau[:, kn_idx] * swing_mask

    tau = torch.clamp(tau, -max_swing_tau, max_swing_tau)

    info.update({
        "kp_swing": kp_swing,
        "kd_swing": kd_swing,
        "max_swing_tau": max_swing_tau,
        "q_env0": q[0].detach().cpu().tolist(),
        "dq_env0": dq[0].detach().cpu().tolist(),
        "q_err_env0": q_err[0].detach().cpu().tolist(),
        "tau_swing_env0": tau[0].detach().cpu().tolist(),
        "tau_swing_mean_abs": float(tau.abs().mean().detach().cpu()),
        "tau_swing_max_abs": float(tau.abs().max().detach().cpu()),
    })

    return tau, info
