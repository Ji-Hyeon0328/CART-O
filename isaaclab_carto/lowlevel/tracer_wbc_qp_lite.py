# isaaclab_carto/lowlevel/tracer_wbc_qp_lite.py
#
# B8-z: WBC-QP-lite solver for CARTO/TRACER low-level controller.
#
# Purpose:
#   Move from the B8-y "task-composition bridge"
#
#       tau = J^T f_mpc + J^T f_swing_task + posture
#
#   to a firmer WBC-style least-squares/QP interface with explicit decision
#   variables:
#
#       x = [tau(12), f_contact(12)]
#
#   and explicit objectives:
#
#       1. MPC GRF tracking:
#            f_contact -> f_mpc, with swing-leg f_mpc = 0
#
#       2. stance foot acceleration constraint:
#            J_stance qdd ~= 0
#
#       3. swing foot acceleration tracking:
#            J_swing qdd ~= a_swing_des
#
#       4. posture / regularized torque:
#            tau ~= tau_posture
#
# Approximate joint-space dynamics:
#
#       qdd ~= Mj^{-1} (tau - J^T f_contact)
#
# where Mj is the actuated-joint block of the articulated mass matrix.
#
# This is not the final floating-base WBC QP. It is a reasonable intermediate
# implementation that has the right software interface and can later be
# upgraded to:
#
#       variables: qdd, tau, f
#       constraints: full floating-base dynamics, stance accel=0, friction cone
#       objectives: base accel, swing accel, MPC GRF, posture
#
# Important:
#   - swing leg f_mpc remains exactly zero at the MPC layer.
#   - no adaptive residual is added here.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class WbcQpLiteConfig:
    # Tracking weights
    w_force_track: float = 1.0
    w_stance_acc: float = 8.0
    w_swing_acc: float = 12.0
    w_tau_posture: float = 0.20
    w_tau_reg: float = 0.02

    # Swing task gains
    kp_swing_xy: float = 0.0
    kd_swing_xy: float = 0.0
    kp_swing_z: float = 80.0
    kd_swing_z: float = 12.0
    max_swing_acc: float = 8.0

    # Posture task
    kp_posture: float = 8.0
    kd_posture: float = 0.8
    max_posture_tau: float = 8.0

    # Force and torque limits
    min_fz: float = 0.0
    max_fz: float = 180.0
    mu: float = 0.70
    max_tau: float = 24.0

    # Numerical
    reg: float = 1.0e-4
    inv_mass_fallback: float = 1.0

    # Mapping sign to effort action
    tau_output_sign: float = 1.0


def make_joint_mass_inverse(
    mass_matrix: Optional[torch.Tensor],
    num_joints: int,
    device,
    dtype,
    cfg: WbcQpLiteConfig,
) -> torch.Tensor:
    """Return [N,12,12] approximate inverse joint mass matrix.

    Isaac often exposes full floating-base mass matrix [N, 18, 18].
    We use the actuated joint block as an intermediate approximation.
    If unavailable, use scaled identity.
    """
    if mass_matrix is None:
        return torch.eye(num_joints, device=device, dtype=dtype).unsqueeze(0) * cfg.inv_mass_fallback

    M = mass_matrix
    if M.dim() == 2:
        M = M.unsqueeze(0)

    # Try to take the joint block.
    if M.shape[-1] >= num_joints + 6:
        Mj = M[..., 6:6 + num_joints, 6:6 + num_joints]
    elif M.shape[-1] >= num_joints:
        Mj = M[..., -num_joints:, -num_joints:]
    else:
        n = M.shape[0]
        return torch.eye(num_joints, device=device, dtype=dtype).unsqueeze(0).repeat(n, 1, 1) * cfg.inv_mass_fallback

    # Robust inverse.
    n = Mj.shape[0]
    eye = torch.eye(num_joints, device=device, dtype=dtype).unsqueeze(0).repeat(n, 1, 1)
    try:
        Minv = torch.linalg.inv(Mj + cfg.reg * eye)
    except Exception:
        Minv = eye * cfg.inv_mass_fallback
    return Minv


def posture_tau(q, qd, q_nom, stance_mask, cfg: WbcQpLiteConfig):
    tau = cfg.kp_posture * (q_nom - q) - cfg.kd_posture * qd
    tau = torch.clamp(tau, -cfg.max_posture_tau, cfg.max_posture_tau)

    # Do not force posture on swing leg strongly.
    swing_mask = 1.0 - stance_mask
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


def desired_swing_acc(
    foot_pos_w: torch.Tensor,
    foot_vel_w: torch.Tensor,
    swing_target_pos_w: torch.Tensor,
    stance_mask: torch.Tensor,
    cfg: WbcQpLiteConfig,
) -> torch.Tensor:
    pos_err = swing_target_pos_w - foot_pos_w
    vel_err = -foot_vel_w

    acc = torch.zeros_like(foot_pos_w)
    acc[:, :, 0] = cfg.kp_swing_xy * pos_err[:, :, 0] + cfg.kd_swing_xy * vel_err[:, :, 0]
    acc[:, :, 1] = cfg.kp_swing_xy * pos_err[:, :, 1] + cfg.kd_swing_xy * vel_err[:, :, 1]
    acc[:, :, 2] = cfg.kp_swing_z * pos_err[:, :, 2] + cfg.kd_swing_z * vel_err[:, :, 2]

    swing_mask = 1.0 - stance_mask
    acc = acc * swing_mask.unsqueeze(-1)
    acc = torch.clamp(acc, -cfg.max_swing_acc, cfg.max_swing_acc)
    return acc


def project_contact_forces(f: torch.Tensor, stance_mask: torch.Tensor, cfg: WbcQpLiteConfig) -> torch.Tensor:
    """Project decision contact forces to simple unilateral/friction limits.

    This projection is mainly diagnostic and for debug printing. The final
    torque returned by this QP-lite currently uses solved tau directly.
    """
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


def solve_wbc_qp_lite(
    jv_feet: torch.Tensor,          # [N,4,3,12]
    f_mpc: torch.Tensor,            # [N,4,3]
    q: torch.Tensor,                # [N,12]
    qd: torch.Tensor,               # [N,12]
    q_nom: torch.Tensor,            # [N,12]
    foot_pos_w: torch.Tensor,       # [N,4,3]
    foot_vel_w: torch.Tensor,       # [N,4,3]
    swing_target_pos_w: torch.Tensor, # [N,4,3]
    stance_mask: torch.Tensor,      # [N,4]
    mass_matrix: Optional[torch.Tensor],
    cfg: WbcQpLiteConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Solve WBC-QP-lite independently for each environment.

    Decision:
        x = [tau_12, f_12]
    """
    device, dtype = q.device, q.dtype
    n, num_joints = q.shape
    num_feet = 4
    num_f = 12
    num_x = num_joints + num_f

    Minv = make_joint_mass_inverse(mass_matrix, num_joints, device, dtype, cfg)
    if Minv.shape[0] == 1 and n > 1:
        Minv = Minv.repeat(n, 1, 1)

    tau_post = posture_tau(q, qd, q_nom, stance_mask, cfg)
    a_swing_des = desired_swing_acc(foot_pos_w, foot_vel_w, swing_target_pos_w, stance_mask, cfg)

    tau_sol = torch.zeros((n, num_joints), device=device, dtype=dtype)
    f_sol = torch.zeros((n, num_feet, 3), device=device, dtype=dtype)
    qdd_pred = torch.zeros((n, num_joints), device=device, dtype=dtype)
    residual_norm = torch.zeros((n,), device=device, dtype=dtype)

    eye_tau = torch.eye(num_joints, device=device, dtype=dtype)
    eye_f = torch.eye(num_f, device=device, dtype=dtype)

    for e in range(n):
        rows = []
        rhs = []

        # Helper matrices.
        Minv_e = Minv[e]
        J_all = jv_feet[e].reshape(num_f, num_joints)  # [12,12], foot-major
        B_tau = Minv_e                               # qdd contribution from tau
        B_f = -Minv_e @ J_all.T                      # qdd contribution from contact force decision

        # 1) MPC force tracking: f ~= f_mpc.
        A = torch.zeros((num_f, num_x), device=device, dtype=dtype)
        A[:, num_joints:] = eye_f
        rows.append(cfg.w_force_track * A)
        rhs.append(cfg.w_force_track * f_mpc[e].reshape(num_f))

        # 2) Stance foot acceleration ~= 0.
        for leg in range(num_feet):
            J = jv_feet[e, leg]  # [3,12]
            A_acc = torch.zeros((3, num_x), device=device, dtype=dtype)
            A_acc[:, :num_joints] = J @ B_tau
            A_acc[:, num_joints:] = J @ B_f

            if stance_mask[e, leg] > 0.5:
                rows.append(cfg.w_stance_acc * A_acc)
                rhs.append(torch.zeros((3,), device=device, dtype=dtype))
            else:
                # 3) Swing foot acceleration tracking.
                rows.append(cfg.w_swing_acc * A_acc)
                rhs.append(cfg.w_swing_acc * a_swing_des[e, leg])

        # 4) Posture torque target.
        A_tau = torch.zeros((num_joints, num_x), device=device, dtype=dtype)
        A_tau[:, :num_joints] = eye_tau
        rows.append(cfg.w_tau_posture * A_tau)
        rhs.append(cfg.w_tau_posture * tau_post[e])

        # 5) Torque regularization.
        rows.append(cfg.w_tau_reg * A_tau)
        rhs.append(torch.zeros((num_joints,), device=device, dtype=dtype))

        A_big = torch.cat(rows, dim=0)
        b_big = torch.cat(rhs, dim=0)

        H = A_big.T @ A_big + cfg.reg * torch.eye(num_x, device=device, dtype=dtype)
        g = A_big.T @ b_big

        try:
            x = torch.linalg.solve(H, g)
        except Exception:
            x = torch.linalg.lstsq(A_big, b_big).solution

        tau_e = x[:num_joints]
        f_e = x[num_joints:].reshape(num_feet, 3)

        tau_e = torch.clamp(tau_e, -cfg.max_tau, cfg.max_tau)
        f_proj = project_contact_forces(f_e.unsqueeze(0), stance_mask[e:e+1], cfg)[0]

        tau_sol[e] = cfg.tau_output_sign * tau_e
        f_sol[e] = f_proj
        qdd_pred[e] = Minv_e @ (tau_e - J_all.T @ f_proj.reshape(num_f))
        residual_norm[e] = torch.linalg.norm(A_big @ x - b_big)

    debug = {
        "tau_posture_target": tau_post,
        "a_swing_des": a_swing_des,
        "f_qp": f_sol,
        "qdd_pred": qdd_pred,
        "residual_norm": residual_norm,
        "Minv_diag_mean": torch.diagonal(Minv, dim1=-2, dim2=-1).mean(dim=-1),
    }
    return tau_sol, debug
