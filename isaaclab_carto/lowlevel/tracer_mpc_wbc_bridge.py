# isaaclab_carto/lowlevel/tracer_mpc_wbc_bridge.py
#
# B8-y: Design-aligned MPC -> WBC bridge for CARTO/TRACER.
#
# This file intentionally returns to the intended low-level controller diagram:
#
#   Ref.S / contact schedule
#   -> MPC-like GRF planner
#   -> WBC-like task combiner
#   -> tau_nominal
#   -> residual later
#
# It is still a bridge implementation, not a full rigid-body QP WBC.
# The purpose is to make the software interface match the architecture:
#
#   f_mpc:
#     stance legs: nonzero GRF
#     swing legs: exactly zero GRF
#
#   tau_wbc:
#     stance force tracking       J_stance^T f_mpc
#     base/posture stabilization  joint posture residual
#     swing foot task             J_swing^T f_swing_task
#
# Later replacement:
#   - replace distribute_grf_ls with forceMPC horizon QP
#   - replace compose_tau_bridge with full WBC QP
#   - add adaptive residual after tau_wbc

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class MpcWbcBridgeConfig:
    # Base force gains
    kp_xy: float = 20.0
    kd_xy: float = 14.0
    kp_z: float = 180.0
    kd_z: float = 35.0

    # Optional roll/pitch moment gains for vertical force redistribution
    kp_roll: float = 40.0
    kd_roll: float = 8.0
    kp_pitch: float = 40.0
    kd_pitch: float = 8.0
    use_rp_moment: bool = False

    # Contact force constraints
    mass: float = 32.0
    gravity: float = 9.81
    mu: float = 0.70
    min_fz: float = 5.0
    max_fz: float = 180.0

    # WBC bridge scaling
    tau_sign: float = -1.0
    tau_force_scale: float = 0.22

    # Swing task as WBC-like Cartesian foot task
    kp_swing_z: float = 140.0
    kd_swing_z: float = 18.0
    kp_swing_xy: float = 0.0
    kd_swing_xy: float = 0.0
    max_swing_force: float = 80.0
    tau_swing_scale: float = 1.0

    # Joint posture/nullspace
    kp_posture: float = 8.0
    kd_posture: float = 0.8
    max_posture_tau: float = 8.0

    # Final torque
    max_total_tau: float = 18.0


def distribute_grf_ls(
    base_pos_w: torch.Tensor,
    base_rpy_w: torch.Tensor,
    base_lin_vel_w: torch.Tensor,
    base_ang_vel_w: torch.Tensor,
    base_ref: torch.Tensor,
    foot_pos_w: torch.Tensor,
    stance_mask: torch.Tensor,
    cfg: MpcWbcBridgeConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MPC-like one-step GRF planner.

    Args:
        base_pos_w: [N,3]
        base_rpy_w: [N,3]
        base_lin_vel_w: [N,3]
        base_ang_vel_w: [N,3]
        base_ref: [N,6] = x,y,z,roll,pitch,yaw
        foot_pos_w: [N,4,3]
        stance_mask: [N,4], 1 stance, 0 swing

    Returns:
        f_mpc: [N,4,3], swing legs exactly zero
        desired_wrench: [N,6] = Fx,Fy,Fz,Mx,My,Mz
    """
    device = base_pos_w.device
    dtype = base_pos_w.dtype
    n = base_pos_w.shape[0]

    pos_err = base_ref[:, 0:3] - base_pos_w
    rpy_err = base_ref[:, 3:6] - base_rpy_w

    F_des = torch.zeros((n, 3), device=device, dtype=dtype)
    M_des = torch.zeros((n, 3), device=device, dtype=dtype)

    F_des[:, 0] = cfg.kp_xy * pos_err[:, 0] - cfg.kd_xy * base_lin_vel_w[:, 0]
    F_des[:, 1] = cfg.kp_xy * pos_err[:, 1] - cfg.kd_xy * base_lin_vel_w[:, 1]
    F_des[:, 2] = cfg.mass * cfg.gravity + cfg.kp_z * pos_err[:, 2] - cfg.kd_z * base_lin_vel_w[:, 2]

    if cfg.use_rp_moment:
        M_des[:, 0] = cfg.kp_roll * rpy_err[:, 0] - cfg.kd_roll * base_ang_vel_w[:, 0]
        M_des[:, 1] = cfg.kp_pitch * rpy_err[:, 1] - cfg.kd_pitch * base_ang_vel_w[:, 1]

    f_mpc = torch.zeros((n, 4, 3), device=device, dtype=dtype)

    for e in range(n):
        stance_ids = torch.where(stance_mask[e] > 0.5)[0]
        ns = int(stance_ids.numel())
        if ns <= 0:
            continue

        # Horizontal force equally distributed first.
        fx_i = F_des[e, 0] / ns
        fy_i = F_des[e, 1] / ns

        # Vertical force distribution.
        if cfg.use_rp_moment and ns >= 3:
            r = foot_pos_w[e, stance_ids, :] - base_pos_w[e].unsqueeze(0)
            A = torch.stack(
                [
                    torch.ones((ns,), device=device, dtype=dtype),
                    r[:, 1],
                    -r[:, 0],
                ],
                dim=0,
            )
            b = torch.stack([F_des[e, 2], M_des[e, 0], M_des[e, 1]], dim=0).unsqueeze(1)
            reg = 1.0e-4 * torch.eye(3, device=device, dtype=dtype)
            try:
                fz = (A.T @ torch.linalg.solve(A @ A.T + reg, b)).squeeze(1)
            except Exception:
                fz = torch.ones((ns,), device=device, dtype=dtype) * (F_des[e, 2] / ns)
        else:
            fz = torch.ones((ns,), device=device, dtype=dtype) * (F_des[e, 2] / ns)

        fz = torch.clamp(fz, cfg.min_fz, cfg.max_fz)

        # Friction cone clamp.
        fxy_norm = torch.sqrt(fx_i * fx_i + fy_i * fy_i).clamp_min(1.0e-9)
        max_fxy = cfg.mu * fz
        scale = torch.clamp(max_fxy / fxy_norm, max=1.0)

        f_mpc[e, stance_ids, 0] = fx_i * scale
        f_mpc[e, stance_ids, 1] = fy_i * scale
        f_mpc[e, stance_ids, 2] = fz

    desired_wrench = torch.cat([F_des, M_des], dim=1)
    return f_mpc, desired_wrench


def tau_from_grf(jv_feet: torch.Tensor, f_mpc: torch.Tensor, cfg: MpcWbcBridgeConfig) -> torch.Tensor:
    """Map foot GRF to joint torques.

    jv_feet: [N,4,3,12]
    f_mpc: [N,4,3]
    """
    tau = torch.einsum("nfij,nfi->nj", jv_feet, f_mpc)
    return cfg.tau_sign * cfg.tau_force_scale * tau


def posture_tau(
    q: torch.Tensor,
    qd: torch.Tensor,
    q_nom: torch.Tensor,
    swing_mask: torch.Tensor,
    cfg: MpcWbcBridgeConfig,
) -> torch.Tensor:
    """Joint posture/nullspace torque.

    swing_mask: [N,4], 1 swing, 0 stance. The swing leg is excluded from
    posture so the swing task is not fought by nullspace posture.
    """
    tau = cfg.kp_posture * (q_nom - q) - cfg.kd_posture * qd
    tau = torch.clamp(tau, -cfg.max_posture_tau, cfg.max_posture_tau)

    # Native joint order:
    # [fl_hx, fr_hx, hl_hx, hr_hx,
    #  fl_hy, fr_hy, hl_hy, hr_hy,
    #  fl_kn, fr_kn, hl_kn, hr_kn]
    hx = [0, 1, 2, 3]
    hy = [4, 5, 6, 7]
    kn = [8, 9, 10, 11]
    for e in range(q.shape[0]):
        for leg in range(4):
            if swing_mask[e, leg] > 0.5:
                tau[e, hx[leg]] = 0.0
                tau[e, hy[leg]] = 0.0
                tau[e, kn[leg]] = 0.0
    return tau


def swing_task_tau(
    jv_feet: torch.Tensor,
    foot_pos_w: torch.Tensor,
    foot_vel_w: torch.Tensor,
    swing_target_pos_w: torch.Tensor,
    stance_mask: torch.Tensor,
    cfg: MpcWbcBridgeConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """WBC-like Cartesian swing task mapped as J_swing^T f_task.

    This is not final QP WBC, but it keeps the correct task separation:
    swing legs receive zero GRF from MPC, but receive Cartesian swing task force.
    """
    device = foot_pos_w.device
    dtype = foot_pos_w.dtype
    n = foot_pos_w.shape[0]
    swing_mask = 1.0 - stance_mask

    f_task = torch.zeros((n, 4, 3), device=device, dtype=dtype)
    pos_err = swing_target_pos_w - foot_pos_w
    vel_err = -foot_vel_w

    f_task[:, :, 0] = cfg.kp_swing_xy * pos_err[:, :, 0] + cfg.kd_swing_xy * vel_err[:, :, 0]
    f_task[:, :, 1] = cfg.kp_swing_xy * pos_err[:, :, 1] + cfg.kd_swing_xy * vel_err[:, :, 1]
    f_task[:, :, 2] = cfg.kp_swing_z * pos_err[:, :, 2] + cfg.kd_swing_z * vel_err[:, :, 2]

    f_task = f_task * swing_mask.unsqueeze(-1)

    # Clamp vector norm per foot.
    norm = torch.linalg.norm(f_task, dim=-1, keepdim=True).clamp_min(1.0e-9)
    scale = torch.clamp(cfg.max_swing_force / norm, max=1.0)
    f_task = f_task * scale

    tau = torch.einsum("nfij,nfi->nj", jv_feet, f_task)
    tau = cfg.tau_swing_scale * tau
    return tau, f_task


def compose_tau_bridge(
    jv_feet: torch.Tensor,
    f_mpc: torch.Tensor,
    q: torch.Tensor,
    qd: torch.Tensor,
    q_nom: torch.Tensor,
    foot_pos_w: torch.Tensor,
    foot_vel_w: torch.Tensor,
    swing_target_pos_w: torch.Tensor,
    stance_mask: torch.Tensor,
    cfg: MpcWbcBridgeConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    swing_mask = 1.0 - stance_mask
    tau_force = tau_from_grf(jv_feet, f_mpc, cfg)
    tau_post = posture_tau(q, qd, q_nom, swing_mask, cfg)
    tau_swing, f_swing_task = swing_task_tau(
        jv_feet=jv_feet,
        foot_pos_w=foot_pos_w,
        foot_vel_w=foot_vel_w,
        swing_target_pos_w=swing_target_pos_w,
        stance_mask=stance_mask,
        cfg=cfg,
    )

    tau = tau_force + tau_post + tau_swing
    tau = torch.clamp(tau, -cfg.max_total_tau, cfg.max_total_tau)

    debug = {
        "tau_force": tau_force,
        "tau_posture": tau_post,
        "tau_swing": tau_swing,
        "f_swing_task": f_swing_task,
        "swing_mask": swing_mask,
    }
    return tau, debug
