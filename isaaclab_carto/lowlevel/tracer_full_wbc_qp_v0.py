# isaaclab_carto/lowlevel/tracer_full_wbc_qp_v0.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch

@dataclass
class FullWbcQpV0Config:
    w_dyn: float = 20.0
    w_base_acc: float = 5.0
    w_stance_acc: float = 30.0
    w_swing_acc: float = 40.0
    w_force_track: float = 1.0
    w_swing_force_zero: float = 50.0
    w_tau_reg: float = 0.02
    w_qdd_reg: float = 0.002
    w_tau_posture: float = 0.10
    kp_base_xy: float = 12.0
    kd_base_xy: float = 8.0
    kp_base_z: float = 40.0
    kd_base_z: float = 10.0
    kp_base_rp: float = 30.0
    kd_base_rp: float = 6.0
    kp_base_yaw: float = 10.0
    kd_base_yaw: float = 3.0
    max_base_acc_lin: float = 6.0
    max_base_acc_ang: float = 8.0
    kp_swing_xy: float = 0.0
    kd_swing_xy: float = 0.0
    kp_swing_z: float = 180.0
    kd_swing_z: float = 20.0
    max_swing_acc: float = 15.0
    kp_posture: float = 8.0
    kd_posture: float = 0.8
    max_posture_tau: float = 8.0
    mu: float = 0.70
    min_fz: float = 0.0
    max_fz: float = 180.0
    max_tau: float = 28.0
    reg: float = 1.0e-4
    tau_output_sign: float = -1.0

def make_selection_matrix(num_joints: int, device, dtype):
    S = torch.zeros((num_joints, 6 + num_joints), device=device, dtype=dtype)
    S[:, 6:] = torch.eye(num_joints, device=device, dtype=dtype)
    return S

def make_h_full(num_envs, num_joints, device, dtype, gravity_forces, coriolis_forces):
    h = torch.zeros((num_envs, 6 + num_joints), device=device, dtype=dtype)
    hj = torch.zeros((num_envs, num_joints), device=device, dtype=dtype)
    if gravity_forces is not None and gravity_forces.shape[-1] == num_joints:
        hj = hj + gravity_forces.to(device=device, dtype=dtype)
    if coriolis_forces is not None and coriolis_forces.shape[-1] == num_joints:
        hj = hj + coriolis_forces.to(device=device, dtype=dtype)
    h[:, 6:] = hj
    return h

def posture_tau(q, qd, q_nom, stance_mask, cfg):
    tau = cfg.kp_posture * (q_nom - q) - cfg.kd_posture * qd
    tau = torch.clamp(tau, -cfg.max_posture_tau, cfg.max_posture_tau)
    hx, hy, kn = [0,1,2,3], [4,5,6,7], [8,9,10,11]
    swing_mask = 1.0 - stance_mask
    for e in range(q.shape[0]):
        for leg in range(4):
            if swing_mask[e, leg] > 0.5:
                tau[e, hx[leg]] = 0.0
                tau[e, hy[leg]] = 0.0
                tau[e, kn[leg]] = 0.0
    return tau

def desired_base_acc(x_hat, base_ref, cfg):
    pos_err = base_ref[:, 0:3] - x_hat[:, 0:3]
    rpy_err = base_ref[:, 3:6] - x_hat[:, 3:6]
    lin_vel = x_hat[:, 6:9]
    ang_vel = x_hat[:, 9:12]
    acc = torch.zeros((x_hat.shape[0], 6), device=x_hat.device, dtype=x_hat.dtype)
    acc[:, 0] = cfg.kp_base_xy * pos_err[:, 0] - cfg.kd_base_xy * lin_vel[:, 0]
    acc[:, 1] = cfg.kp_base_xy * pos_err[:, 1] - cfg.kd_base_xy * lin_vel[:, 1]
    acc[:, 2] = cfg.kp_base_z * pos_err[:, 2] - cfg.kd_base_z * lin_vel[:, 2]
    acc[:, 3] = cfg.kp_base_rp * rpy_err[:, 0] - cfg.kd_base_rp * ang_vel[:, 0]
    acc[:, 4] = cfg.kp_base_rp * rpy_err[:, 1] - cfg.kd_base_rp * ang_vel[:, 1]
    acc[:, 5] = cfg.kp_base_yaw * rpy_err[:, 2] - cfg.kd_base_yaw * ang_vel[:, 2]
    acc[:, 0:3] = torch.clamp(acc[:, 0:3], -cfg.max_base_acc_lin, cfg.max_base_acc_lin)
    acc[:, 3:6] = torch.clamp(acc[:, 3:6], -cfg.max_base_acc_ang, cfg.max_base_acc_ang)
    return acc

def desired_swing_acc(foot_pos_w, foot_vel_w, swing_target_pos_w, stance_mask, cfg):
    pos_err = swing_target_pos_w - foot_pos_w
    vel_err = -foot_vel_w
    acc = torch.zeros_like(foot_pos_w)
    acc[:, :, 0] = cfg.kp_swing_xy * pos_err[:, :, 0] + cfg.kd_swing_xy * vel_err[:, :, 0]
    acc[:, :, 1] = cfg.kp_swing_xy * pos_err[:, :, 1] + cfg.kd_swing_xy * vel_err[:, :, 1]
    acc[:, :, 2] = cfg.kp_swing_z * pos_err[:, :, 2] + cfg.kd_swing_z * vel_err[:, :, 2]
    acc = acc * (1.0 - stance_mask).unsqueeze(-1)
    return torch.clamp(acc, -cfg.max_swing_acc, cfg.max_swing_acc)

def project_contact_forces(f, stance_mask, cfg):
    out = f.clone() * stance_mask.unsqueeze(-1)
    fz = torch.clamp(out[:, :, 2], cfg.min_fz, cfg.max_fz)
    fx, fy = out[:, :, 0], out[:, :, 1]
    fxy = torch.sqrt(fx * fx + fy * fy).clamp_min(1.0e-9)
    max_fxy = cfg.mu * fz
    scale = torch.clamp(max_fxy / fxy, max=1.0)
    out[:, :, 0] = fx * scale
    out[:, :, 1] = fy * scale
    out[:, :, 2] = fz
    return out * stance_mask.unsqueeze(-1)

def solve_full_wbc_qp_v0(M_full, Jfeet_full, f_mpc, q, qd, q_nom, x_hat, base_ref,
                         foot_pos_w, foot_vel_w, swing_target_pos_w, stance_mask,
                         gravity_forces, coriolis_forces, cfg):
    device, dtype = q.device, q.dtype
    n, num_joints = q.shape
    nq_full, ntau, nf = 6 + num_joints, num_joints, 12
    nx = nq_full + ntau + nf
    S = make_selection_matrix(num_joints, device, dtype)
    ST = S.T
    h_full = make_h_full(n, num_joints, device, dtype, gravity_forces, coriolis_forces)
    base_acc_des = desired_base_acc(x_hat, base_ref, cfg)
    swing_acc_des = desired_swing_acc(foot_pos_w, foot_vel_w, swing_target_pos_w, stance_mask, cfg)
    tau_post = posture_tau(q, qd, q_nom, stance_mask, cfg)

    tau_sol = torch.zeros((n, ntau), device=device, dtype=dtype)
    qdd_sol = torch.zeros((n, nq_full), device=device, dtype=dtype)
    f_sol = torch.zeros((n, 4, 3), device=device, dtype=dtype)
    residual_norm = torch.zeros((n,), device=device, dtype=dtype)

    I_qdd = torch.eye(nq_full, device=device, dtype=dtype)
    I_tau = torch.eye(ntau, device=device, dtype=dtype)
    I_f = torch.eye(nf, device=device, dtype=dtype)

    for e in range(n):
        rows, rhs = [], []
        M = M_full[e]
        J_all = Jfeet_full[e].reshape(nf, nq_full)

        # Full dynamics: M qdd - S^T tau - J^T f = -h
        A_dyn = torch.zeros((nq_full, nx), device=device, dtype=dtype)
        A_dyn[:, 0:nq_full] = M
        A_dyn[:, nq_full:nq_full+ntau] = -ST
        A_dyn[:, nq_full+ntau:] = -J_all.T
        rows.append(cfg.w_dyn * A_dyn)
        rhs.append(cfg.w_dyn * (-h_full[e]))

        # Base acceleration task
        A_base = torch.zeros((6, nx), device=device, dtype=dtype)
        A_base[:, 0:6] = torch.eye(6, device=device, dtype=dtype)
        rows.append(cfg.w_base_acc * A_base)
        rhs.append(cfg.w_base_acc * base_acc_des[e])

        # Foot acceleration tasks
        for leg in range(4):
            J = Jfeet_full[e, leg]
            A_acc = torch.zeros((3, nx), device=device, dtype=dtype)
            A_acc[:, 0:nq_full] = J
            if stance_mask[e, leg] > 0.5:
                rows.append(cfg.w_stance_acc * A_acc)
                rhs.append(torch.zeros((3,), device=device, dtype=dtype))
            else:
                rows.append(cfg.w_swing_acc * A_acc)
                rhs.append(cfg.w_swing_acc * swing_acc_des[e, leg])

        # MPC force tracking
        A_f = torch.zeros((nf, nx), device=device, dtype=dtype)
        A_f[:, nq_full+ntau:] = I_f
        rows.append(cfg.w_force_track * A_f)
        rhs.append(cfg.w_force_track * f_mpc[e].reshape(nf))

        # Swing force zero
        swing_force_rows = []
        for leg in range(4):
            if stance_mask[e, leg] < 0.5:
                for ax in range(3):
                    row = torch.zeros((nx,), device=device, dtype=dtype)
                    row[nq_full + ntau + leg * 3 + ax] = 1.0
                    swing_force_rows.append(row)
        if swing_force_rows:
            A_sw = torch.stack(swing_force_rows, dim=0)
            rows.append(cfg.w_swing_force_zero * A_sw)
            rhs.append(torch.zeros((A_sw.shape[0],), device=device, dtype=dtype))

        # Posture torque and regularization
        A_tau = torch.zeros((ntau, nx), device=device, dtype=dtype)
        A_tau[:, nq_full:nq_full+ntau] = I_tau
        rows.append(cfg.w_tau_posture * A_tau); rhs.append(cfg.w_tau_posture * tau_post[e])
        rows.append(cfg.w_tau_reg * A_tau); rhs.append(torch.zeros((ntau,), device=device, dtype=dtype))

        A_qdd = torch.zeros((nq_full, nx), device=device, dtype=dtype)
        A_qdd[:, 0:nq_full] = I_qdd
        rows.append(cfg.w_qdd_reg * A_qdd); rhs.append(torch.zeros((nq_full,), device=device, dtype=dtype))

        A = torch.cat(rows, dim=0)
        b = torch.cat(rhs, dim=0)
        H = A.T @ A + cfg.reg * torch.eye(nx, device=device, dtype=dtype)
        g = A.T @ b
        try:
            x = torch.linalg.solve(H, g)
        except Exception:
            x = torch.linalg.lstsq(A, b).solution

        qdd_e = x[0:nq_full]
        tau_e = torch.clamp(x[nq_full:nq_full+ntau], -cfg.max_tau, cfg.max_tau)
        f_e = x[nq_full+ntau:].reshape(4, 3)
        f_proj = project_contact_forces(f_e.unsqueeze(0), stance_mask[e:e+1], cfg)[0]

        qdd_sol[e] = qdd_e
        f_sol[e] = f_proj
        tau_sol[e] = cfg.tau_output_sign * tau_e
        residual_norm[e] = torch.linalg.norm(A @ x - b)

    debug = {
        "qdd_full": qdd_sol, "f_qp": f_sol, "h_full": h_full,
        "base_acc_des": base_acc_des, "swing_acc_des": swing_acc_des,
        "tau_posture_target": tau_post, "residual_norm": residual_norm,
    }
    return tau_sol, debug
