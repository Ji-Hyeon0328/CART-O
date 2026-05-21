# isaaclab_carto/lowlevel/balance_augmented_wbc.py
#
# B8-g: Balance-augmented QPS-WBC bridge.
#
# Motivation from B8-f:
#   - reduced implicit PD gives residual authority
#   - too much reduction causes sit-down / pitch collapse
#
# This module adds a simple balance augmentation to the previous QPS-WBC bridge.
#
# It is still NOT full floating-base WBC-QP.
#
# Main addition:
#   A base height/pitch feedback creates a desired "balance wrench":
#
#       Fz_balance   = Kp_h (h_ref - h) + Kd_h (0 - vz)
#       My_balance   = Kp_pitch (pitch_ref - pitch) + Kd_pitch (0 - pitch_rate)
#
#   Then we distribute that desired vertical force / pitch moment over stance legs:
#
#       sum_i dz_i = Fz_balance
#       sum_i (x_i - x_com) dz_i = My_balance
#
#   This produces f_balance on stance legs, which is added to forceMPC f_ref.
#
# Goal:
#   Prevent sit-down / forward pitch collapse under reduced implicit PD.

from __future__ import annotations

from typing import Dict, Tuple

import torch

from isaaclab_carto.lowlevel.qps_wbc_bridge import (
    QPSWBCBridgeState,
    make_qps_wbc_bridge_torque,
)


FOOT_NAMES = ["fl_foot", "fr_foot", "hl_foot", "hr_foot"]


def get_foot_indices(robot):
    name_to_idx = {name: i for i, name in enumerate(robot.body_names)}
    return [name_to_idx[name] for name in FOOT_NAMES]


def extract_stance(ref: Dict[str, torch.Tensor], k: int = 0) -> torch.Tensor:
    S = ref["S"]
    H = S.shape[-1]
    k = min(max(k, 0), H - 1)
    return (S[:, :, k] > 0.5).to(S.dtype)


def compute_balance_force_bias(
    robot,
    ref: Dict[str, torch.Tensor],
    x_hat: torch.Tensor,
    k: int = 0,
    h_ref: float = 0.67,
    pitch_ref: float = 0.0,
    kp_h: float = 120.0,
    kd_h: float = 20.0,
    kp_pitch: float = 35.0,
    kd_pitch: float = 6.0,
    max_extra_fz_per_leg: float = 8.0,
    max_remove_fz_per_leg: float = 4.0,
    max_pitch_moment: float = 8.0,
    min_stance_legs: int = 2,
    front_unload_gain: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Compute vertical force bias on stance legs.

    Args:
        x_hat: [N,12] = [x y z roll pitch yaw vx vy vz wx wy wz]
        front_unload_gain:
            Optional extra heuristic to reduce front-leg load when pitch is negative.
            0 disables it.

    Returns:
        f_bias: [N,4,3], only z component nonzero
    """
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype
    N = robot.data.joint_pos.shape[0]

    stance = extract_stance(ref, k=k).to(device=device, dtype=dtype)
    num_stance = torch.clamp(stance.sum(dim=1, keepdim=True), min=float(min_stance_legs))

    foot_indices = get_foot_indices(robot)
    foot_pos = robot.data.body_pos_w[:, foot_indices, :]  # [N,4,3]
    base_pos = robot.data.root_pos_w[:, 0:3]              # [N,3]

    x_rel = foot_pos[:, :, 0] - base_pos[:, 0:1]

    h = x_hat[:, 2]
    pitch = x_hat[:, 4]
    vz = x_hat[:, 8]
    pitch_rate = x_hat[:, 10]  # wy

    h_err = torch.full_like(h, h_ref) - h
    pitch_err = torch.full_like(pitch, pitch_ref) - pitch

    # Positive Fz means "add upward support".
    Fz_balance = kp_h * h_err + kd_h * (0.0 - vz)

    # Positive My convention is approximate here.
    # We use it as a pitch-restoring vertical load redistribution target.
    My_balance = kp_pitch * pitch_err + kd_pitch * (0.0 - pitch_rate)
    My_balance = torch.clamp(My_balance, -max_pitch_moment, max_pitch_moment)

    fz_bias = torch.zeros((N, 4), device=device, dtype=dtype)

    for env_id in range(N):
        mask = stance[env_id] > 0.5
        if mask.sum() < min_stance_legs:
            continue

        xr = x_rel[env_id, mask]
        ones = torch.ones_like(xr)

        # Solve least-squares:
        #   [1 1 ...] dz = Fz_balance
        #   [x x ...] dz = My_balance
        A = torch.stack([ones, xr], dim=0)  # [2,ns]
        b = torch.stack([Fz_balance[env_id], My_balance[env_id]], dim=0)  # [2]

        # Minimum-norm solution dz = A^T (A A^T)^-1 b
        H = A @ A.T + 1e-6 * torch.eye(2, device=device, dtype=dtype)
        dz = A.T @ torch.linalg.solve(H, b)

        full = torch.zeros((4,), device=device, dtype=dtype)
        full[mask] = dz
        fz_bias[env_id] = full

    # Optional heuristic:
    # If pitch < 0 means nose-down in current convention, unload front legs slightly
    # and compensate on rear legs. This is intentionally weak and optional.
    if front_unload_gain > 0.0:
        front = torch.tensor([1.0, 1.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 4)
        rear = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device, dtype=dtype).view(1, 4)
        nose_down = torch.clamp(-pitch, min=0.0).view(N, 1)
        unload = front_unload_gain * nose_down
        fz_bias = fz_bias - unload * front * stance
        rear_count = torch.clamp((rear * stance).sum(dim=1, keepdim=True), min=1.0)
        fz_bias = fz_bias + unload * rear * stance * (2.0 / rear_count)

    fz_bias = torch.clamp(fz_bias, -max_remove_fz_per_leg, max_extra_fz_per_leg)
    fz_bias = fz_bias * stance

    f_bias = torch.zeros((N, 4, 3), device=device, dtype=dtype)
    f_bias[:, :, 2] = fz_bias

    info = {
        "balance_stance_env0": stance[0].detach().cpu().tolist(),
        "balance_num_stance_env0": float(stance[0].sum().detach().cpu()),
        "balance_h_env0": float(h[0].detach().cpu()),
        "balance_h_err_env0": float(h_err[0].detach().cpu()),
        "balance_pitch_env0": float(pitch[0].detach().cpu()),
        "balance_pitch_err_env0": float(pitch_err[0].detach().cpu()),
        "balance_vz_env0": float(vz[0].detach().cpu()),
        "balance_pitch_rate_env0": float(pitch_rate[0].detach().cpu()),
        "Fz_balance_env0": float(Fz_balance[0].detach().cpu()),
        "My_balance_env0": float(My_balance[0].detach().cpu()),
        "x_rel_env0": x_rel[0].detach().cpu().tolist(),
        "fz_bias_env0": fz_bias[0].detach().cpu().tolist(),
        "f_bias_env0": f_bias[0].detach().cpu().tolist(),
        "max_extra_fz_per_leg": max_extra_fz_per_leg,
        "max_remove_fz_per_leg": max_remove_fz_per_leg,
        "front_unload_gain": front_unload_gain,
    }

    return f_bias, info


def make_balance_augmented_qps_wbc_torque(
    robot,
    ref: Dict[str, torch.Tensor],
    x_hat: torch.Tensor,
    f_ref: torch.Tensor,
    state: QPSWBCBridgeState | None = None,
    k: int = 0,
    linear_rows: str = "0_3",
    # Balance terms
    h_ref_balance: float = 0.67,
    pitch_ref: float = 0.0,
    kp_h_balance: float = 120.0,
    kd_h_balance: float = 20.0,
    kp_pitch_balance: float = 35.0,
    kd_pitch_balance: float = 6.0,
    max_extra_fz_per_leg: float = 8.0,
    max_remove_fz_per_leg: float = 4.0,
    max_pitch_moment: float = 8.0,
    front_unload_gain: float = 0.0,
    balance_scale: float = 1.0,
    # QPS-WBC bridge terms
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
) -> Tuple[torch.Tensor, Dict[str, object], torch.Tensor]:
    """
    Add balance force bias to f_ref, then pass to QPS-WBC bridge.
    """
    f_bias, balance_info = compute_balance_force_bias(
        robot=robot,
        ref=ref,
        x_hat=x_hat,
        k=k,
        h_ref=h_ref_balance,
        pitch_ref=pitch_ref,
        kp_h=kp_h_balance,
        kd_h=kd_h_balance,
        kp_pitch=kp_pitch_balance,
        kd_pitch=kd_pitch_balance,
        max_extra_fz_per_leg=max_extra_fz_per_leg,
        max_remove_fz_per_leg=max_remove_fz_per_leg,
        max_pitch_moment=max_pitch_moment,
        front_unload_gain=front_unload_gain,
    )

    f_aug = f_ref + balance_scale * f_bias

    tau, wbc_info = make_qps_wbc_bridge_torque(
        robot=robot,
        ref=ref,
        f_ref=f_aug,
        state=state,
        k=k,
        linear_rows=linear_rows,
        kp_swing_xyz=kp_swing_xyz,
        kd_swing_xyz=kd_swing_xyz,
        max_task_cmd=max_task_cmd,
        max_pos_err=max_pos_err,
        kp_posture=kp_posture,
        kd_posture=kd_posture,
        max_posture_tau=max_posture_tau,
        w_force=w_force,
        w_swing=w_swing,
        w_posture=w_posture,
        w_reg=w_reg,
        w_rate=w_rate,
        max_total_tau=max_total_tau,
    )

    info: Dict[str, object] = {
        "balance_scale": balance_scale,
        "f_ref_env0": f_ref[0].detach().cpu().tolist(),
        "f_aug_env0": f_aug[0].detach().cpu().tolist(),
        "f_bias_norm_env0": float(torch.linalg.norm(f_bias[0]).detach().cpu()),
        "f_aug_norm_env0": float(torch.linalg.norm(f_aug[0]).detach().cpu()),
    }
    info.update(balance_info)
    info.update(wbc_info)

    return tau, info, f_aug
