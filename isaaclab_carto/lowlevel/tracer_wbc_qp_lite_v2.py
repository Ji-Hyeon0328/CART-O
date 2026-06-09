# isaaclab_carto/lowlevel/tracer_wbc_qp_lite_v2.py
#
# B8-aa: WBC-QP-lite v2 for CARTO/TRACER.
#
# Motivation from B8-z:
#   - tau_output_sign = +1 dog-sits / collapses.
#   - tau_output_sign = -1 keeps standing.
#   - contact schedule and f_mpc/f_qp masking are correct.
#   - swing foot still does not lift.
#
# B8-aa keeps the stable sign convention and introduces a firmer WBC-lite
# formulation with explicit qdd:
#
#   decision x = [qdd(12), tau(12), f(12)]
#
# soft dynamics:
#   qdd ~= Minv (tau - J^T f)
#
# task objectives:
#   stance foot acceleration -> 0
#   swing foot acceleration -> a_swing_des
#   f -> f_mpc
#   tau -> posture
#
# This is still actuated-joint WBC-lite, not full floating-base WBC.
# But it is a cleaner bridge toward the final WBC-QP interface.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class WbcQpLiteV2Config:
    # Dynamics and task weights
    w_dyn: float = 5.0
    w_force_track: float = 1.0
    w_stance_acc: float = 10.0
    w_swing_acc: float = 25.0
    w_tau_posture: float = 0.10
    w_tau_reg: float = 0.02
    w_qdd_reg: float = 0.005

    # Swing task gains
    kp_swing_xy: float = 0.0
    kd_swing_xy: float = 0.0
    kp_swing_z: float = 140.0
    kd_swing_z: float = 16.0
    max_swing_acc: float = 12.0
    swing_acc_sign: float = 1.0

    # Optional direct joint-space bias on swing joints
    # This is not the main controller; it is a small nullspace-like bias
    # to help test whether commanded qdd can influence the swing leg.
    use_swing_joint_bias: bool = False
    w_swing_joint_bias: float = 0.5
    swing_hy_qdd: float = 0.0
    swing_kn_qdd: float = 0.0

    # Posture task
    kp_posture: float = 8.0
    kd_posture: float = 0.8
    max_posture_tau: float = 8.0

    # Contact and torque projection
    min_fz: float = 0.0
    max_fz: float = 180.0
    mu: float = 0.70
    max_tau: float = 24.0

    # Numerical
    reg: float = 1.0e-4
    inv_mass_fallback: float = 1.0

    # Stable sign convention found in B8-z
    tau_output_sign: float = -1.0


HX = [0, 1, 2, 3]
HY = [4, 5, 6, 7]
KN = [8, 9, 10, 11]


def make_joint_mass_inverse(mass_matrix: Optional[torch.Tensor], num_joints: int, device, dtype, cfg: WbcQpLiteV2Config):
    if mass_matrix is None:
        return torch.eye(num_joints, device=device, dtype=dtype).unsqueeze(0) * cfg.inv_mass_fallback

    M = mass_matrix
    if M.dim() == 2:
        M = M.unsqueeze(0)

    if M.shape[-1] >= num_joints + 6:
        Mj = M[..., 6:6 + num_joints, 6:6 + num_joints]
    elif M.shape[-1] >= num_joints:
        Mj = M[..., -num_joints:, -num_joints:]
    else:
        n = M.shape[0]
        return torch.eye(num_joints, device=device, dtype=dtype).unsqueeze(0).repeat(n, 1, 1) * cfg.inv_mass_fallback

    n = Mj.shape[0]
    eye = torch.eye(num_joints, device=device, dtype=dtype).unsqueeze(0).repeat(n, 1, 1)
    try:
        return torch.linalg.inv(Mj + cfg.reg * eye)
    except Exception:
        return eye * cfg.inv_mass_fallback


def posture_tau(q, qd, q_nom, stance_mask, cfg: WbcQpLiteV2Config):
    tau = cfg.kp_posture * (q_nom - q) - cfg.kd_posture * qd
    tau = torch.clamp(tau, -cfg.max_posture_tau, cfg.max_posture_tau)

    swing_mask = 1.0 - stance_mask
    for e in range(q.shape[0]):
        for leg in range(4):
            if swing_mask[e, leg] > 0.5:
                tau[e, HX[leg]] = 0.0
                tau[e, HY[leg]] = 0.0
                tau[e, KN[leg]] = 0.0
    return tau


def desired_swing_acc(foot_pos_w, foot_vel_w, swing_target_pos_w, stance_mask, cfg: WbcQpLiteV2Config):
    pos_err = swing_target_pos_w - foot_pos_w
    vel_err = -foot_vel_w

    acc = torch.zeros_like(foot_pos_w)
    acc[:, :, 0] = cfg.kp_swing_xy * pos_err[:, :, 0] + cfg.kd_swing_xy * vel_err[:, :, 0]
    acc[:, :, 1] = cfg.kp_swing_xy * pos_err[:, :, 1] + cfg.kd_swing_xy * vel_err[:, :, 1]
    acc[:, :, 2] = cfg.kp_swing_z * pos_err[:, :, 2] + cfg.kd_swing_z * vel_err[:, :, 2]

    acc = cfg.swing_acc_sign * acc
    acc = acc * (1.0 - stance_mask).unsqueeze(-1)
    acc = torch.clamp(acc, -cfg.max_swing_acc, cfg.max_swing_acc)
    return acc


def project_contact_forces(f, stance_mask, cfg: WbcQpLiteV2Config):
    out = f.clone()
    out = out * stance_mask.unsqueeze(-1)

    fz = torch.clamp(out[:, :, 2], cfg.min_fz, cfg.max_fz)
    fx = out[:, :, 0]
    fy = out[:, :, 1]

    fxy = torch.sqrt(fx * fx + fy * fy).clamp_min(1.0e-9)
    max_fxy = cfg.mu * fz
    scale = torch.clamp(max_fxy / fxy, max=1.0)

    out[:, :, 0] = fx * scale
    out[:, :, 1] = fy * scale
    out[:, :, 2] = fz
    out = out * stance_mask.unsqueeze(-1)
    return out


def solve_wbc_qp_lite_v2(
    jv_feet: torch.Tensor,
    f_mpc: torch.Tensor,
    q: torch.Tensor,
    qd: torch.Tensor,
    q_nom: torch.Tensor,
    foot_pos_w: torch.Tensor,
    foot_vel_w: torch.Tensor,
    swing_target_pos_w: torch.Tensor,
    stance_mask: torch.Tensor,
    mass_matrix: Optional[torch.Tensor],
    cfg: WbcQpLiteV2Config,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device, dtype = q.device, q.dtype
    n, num_joints = q.shape
    num_feet = 4
    num_f = 12

    # x = [qdd(12), tau(12), f(12)]
    nq = num_joints
    nt = num_joints
    nf = num_f
    nx = nq + nt + nf

    Minv = make_joint_mass_inverse(mass_matrix, num_joints, device, dtype, cfg)
    if Minv.shape[0] == 1 and n > 1:
        Minv = Minv.repeat(n, 1, 1)

    tau_post = posture_tau(q, qd, q_nom, stance_mask, cfg)
    a_swing = desired_swing_acc(foot_pos_w, foot_vel_w, swing_target_pos_w, stance_mask, cfg)

    qdd_sol = torch.zeros((n, num_joints), device=device, dtype=dtype)
    tau_sol = torch.zeros((n, num_joints), device=device, dtype=dtype)
    f_sol = torch.zeros((n, num_feet, 3), device=device, dtype=dtype)
    residual_norm = torch.zeros((n,), device=device, dtype=dtype)

    eye_q = torch.eye(nq, device=device, dtype=dtype)
    eye_t = torch.eye(nt, device=device, dtype=dtype)
    eye_f = torch.eye(nf, device=device, dtype=dtype)

    for e in range(n):
        rows = []
        rhs = []

        J_all = jv_feet[e].reshape(num_f, num_joints)
        Minv_e = Minv[e]

        # 1) Soft actuated dynamics:
        # qdd - Minv tau + Minv J^T f = 0
        A_dyn = torch.zeros((nq, nx), device=device, dtype=dtype)
        A_dyn[:, 0:nq] = eye_q
        A_dyn[:, nq:nq+nt] = -Minv_e
        A_dyn[:, nq+nt:] = Minv_e @ J_all.T
        rows.append(cfg.w_dyn * A_dyn)
        rhs.append(torch.zeros((nq,), device=device, dtype=dtype))

        # 2) Force tracking f ~= f_mpc.
        A_f = torch.zeros((nf, nx), device=device, dtype=dtype)
        A_f[:, nq+nt:] = eye_f
        rows.append(cfg.w_force_track * A_f)
        rhs.append(cfg.w_force_track * f_mpc[e].reshape(nf))

        # 3) Foot acceleration tasks, using J qdd.
        for leg in range(num_feet):
            J = jv_feet[e, leg]
            A_acc = torch.zeros((3, nx), device=device, dtype=dtype)
            A_acc[:, 0:nq] = J

            if stance_mask[e, leg] > 0.5:
                rows.append(cfg.w_stance_acc * A_acc)
                rhs.append(torch.zeros((3,), device=device, dtype=dtype))
            else:
                rows.append(cfg.w_swing_acc * A_acc)
                rhs.append(cfg.w_swing_acc * a_swing[e, leg])

        # 4) Posture torque target.
        A_tau = torch.zeros((nt, nx), device=device, dtype=dtype)
        A_tau[:, nq:nq+nt] = eye_t
        rows.append(cfg.w_tau_posture * A_tau)
        rhs.append(cfg.w_tau_posture * tau_post[e])

        # 5) Torque regularization.
        rows.append(cfg.w_tau_reg * A_tau)
        rhs.append(torch.zeros((nt,), device=device, dtype=dtype))

        # 6) qdd regularization.
        A_qdd = torch.zeros((nq, nx), device=device, dtype=dtype)
        A_qdd[:, 0:nq] = eye_q
        rows.append(cfg.w_qdd_reg * A_qdd)
        rhs.append(torch.zeros((nq,), device=device, dtype=dtype))

        # Optional swing joint qdd bias.
        if cfg.use_swing_joint_bias:
            swing_mask = 1.0 - stance_mask[e]
            A_bias_rows = []
            b_bias_rows = []
            for leg in range(4):
                if swing_mask[leg] > 0.5:
                    row_hy = torch.zeros((nx,), device=device, dtype=dtype)
                    row_kn = torch.zeros((nx,), device=device, dtype=dtype)
                    row_hy[HY[leg]] = 1.0
                    row_kn[KN[leg]] = 1.0
                    A_bias_rows.extend([row_hy, row_kn])
                    b_bias_rows.extend([
                        torch.tensor(cfg.swing_hy_qdd, device=device, dtype=dtype),
                        torch.tensor(cfg.swing_kn_qdd, device=device, dtype=dtype),
                    ])
            if A_bias_rows:
                A_bias = torch.stack(A_bias_rows, dim=0)
                b_bias = torch.stack(b_bias_rows, dim=0)
                rows.append(cfg.w_swing_joint_bias * A_bias)
                rhs.append(cfg.w_swing_joint_bias * b_bias)

        A_big = torch.cat(rows, dim=0)
        b_big = torch.cat(rhs, dim=0)

        H = A_big.T @ A_big + cfg.reg * torch.eye(nx, device=device, dtype=dtype)
        g = A_big.T @ b_big

        try:
            x = torch.linalg.solve(H, g)
        except Exception:
            x = torch.linalg.lstsq(A_big, b_big).solution

        qdd_e = x[0:nq]
        tau_e = x[nq:nq+nt]
        f_e = x[nq+nt:].reshape(num_feet, 3)

        f_proj = project_contact_forces(f_e.unsqueeze(0), stance_mask[e:e+1], cfg)[0]
        tau_e = torch.clamp(tau_e, -cfg.max_tau, cfg.max_tau)

        qdd_sol[e] = qdd_e
        tau_sol[e] = cfg.tau_output_sign * tau_e
        f_sol[e] = f_proj
        residual_norm[e] = torch.linalg.norm(A_big @ x - b_big)

    debug = {
        "qdd_sol": qdd_sol,
        "tau_raw": cfg.tau_output_sign * tau_sol if cfg.tau_output_sign != 0 else tau_sol,
        "tau_posture_target": tau_post,
        "a_swing_des": a_swing,
        "f_qp": f_sol,
        "residual_norm": residual_norm,
        "Minv_diag_mean": torch.diagonal(Minv, dim1=-2, dim2=-1).mean(dim=-1),
    }
    return tau_sol, debug
