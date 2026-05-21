# isaaclab_carto/lowlevel/projected_force_mpc.py
#
# B5-c: projected forceMPC.
#
# This extends B5 skeleton by adding explicit force projection:
#
#   1. solve unconstrained centroidal LS force planner
#   2. apply stance/swing mask
#   3. enforce fz bounds
#   4. enforce friction pyramid:
#        |fx| <= mu * fz
#        |fy| <= mu * fz
#   5. apply optional force-rate limit
#   6. smooth force
#   7. tau = J_foot^T f
#
# This is still not a true QP with inequalities inside the solve.
# It is a stable debug bridge before adding a real constrained QP solver.

from __future__ import annotations

from typing import Dict, Tuple

import torch

from isaaclab_carto.lowlevel.support_force_control import (
    extract_foot_jacobians_action_order,
    compute_tau_jtf,
)


def normalize_beta(beta_t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    beta_t = torch.clamp(beta_t, min=eps)
    return beta_t / torch.clamp(beta_t.sum(dim=1, keepdim=True), min=eps)


def extract_stance_mask(ref: Dict[str, torch.Tensor], use_k: int = 0, min_stance_legs: int = 2) -> torch.Tensor:
    S = ref["S"]
    H = S.shape[-1]
    k = min(max(use_k, 0), H - 1)
    stance = (S[:, :, k] > 0.5).to(S.dtype)

    # safety guard for debug
    unsafe = stance.sum(dim=1) < float(min_stance_legs)
    if torch.any(unsafe):
        stance[unsafe, :] = 1.0

    return stance


def skew(r: torch.Tensor) -> torch.Tensor:
    N = r.shape[0]
    S = torch.zeros((N, 3, 3), device=r.device, dtype=r.dtype)
    S[:, 0, 1] = -r[:, 2]
    S[:, 0, 2] = r[:, 1]
    S[:, 1, 0] = r[:, 2]
    S[:, 1, 2] = -r[:, 0]
    S[:, 2, 0] = -r[:, 1]
    S[:, 2, 1] = r[:, 0]
    return S


def build_centroidal_force_map(robot, mass: float = 32.5) -> Tuple[torch.Tensor, Dict[str, object]]:
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype
    N = robot.data.joint_pos.shape[0]

    body_names = robot.body_names
    name_to_idx = {name: i for i, name in enumerate(body_names)}
    foot_names = ["fl_foot", "fr_foot", "hl_foot", "hr_foot"]
    foot_indices = [name_to_idx[name] for name in foot_names]

    foot_pos_w = robot.data.body_pos_w[:, foot_indices, :]
    base_pos_w = robot.data.root_pos_w[:, None, :]
    r = foot_pos_w - base_pos_w

    # Approximate inertia for debug only.
    I_diag = torch.tensor([1.0, 2.0, 2.2], device=device, dtype=dtype)
    I_inv = torch.diag(1.0 / I_diag).unsqueeze(0).repeat(N, 1, 1)

    G = torch.zeros((N, 6, 12), device=device, dtype=dtype)
    for leg in range(4):
        col = slice(3 * leg, 3 * leg + 3)
        G[:, 0:3, col] = torch.eye(3, device=device, dtype=dtype).unsqueeze(0) / mass
        G[:, 3:6, col] = torch.bmm(I_inv, skew(r[:, leg, :]))

    info = {
        "foot_names": foot_names,
        "foot_indices": foot_indices,
        "mass": mass,
        "inertia_diag_debug": I_diag.detach().cpu().tolist(),
    }
    return G, info


def beta_to_weights(beta_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    beta = [height/stability, velocity, energy/smoothness]

    Returns:
        w_acc_z, w_acc_xy, w_force, rate_scale
    """
    beta = normalize_beta(beta_t)
    beta_h = beta[:, 0]
    beta_v = beta[:, 1]
    beta_e = beta[:, 2]

    w_acc_z = 5.0 + 20.0 * beta_h
    w_acc_xy = 2.0 + 15.0 * beta_v
    w_force = 1e-3 + 6e-3 * beta_e
    rate_scale = 1.0 - 0.45 * beta_e  # energy beta -> smaller allowed force change
    return w_acc_z, w_acc_xy, w_force, rate_scale


def project_forces(
    f: torch.Tensor,
    stance: torch.Tensor,
    mu: float = 0.6,
    fz_min: float = 0.0,
    fz_max: float = 8.0,
    fxy_abs_max: float = 3.0,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Project forces to simple constraints.

    Args:
        f: [N,4,3]
        stance: [N,4]

    Projection:
        swing: f=0
        stance: fz in [fz_min, fz_max]
        |fx| <= min(fxy_abs_max, mu*fz)
        |fy| <= min(fxy_abs_max, mu*fz)
    """
    f_proj = f.clone()
    f_proj = f_proj * stance.unsqueeze(-1)

    fz_before = f_proj[:, :, 2].clone()
    f_proj[:, :, 2] = torch.clamp(f_proj[:, :, 2], min=fz_min, max=fz_max)

    lim = torch.clamp(mu * f_proj[:, :, 2], min=0.0, max=fxy_abs_max)
    fx_before = f_proj[:, :, 0].clone()
    fy_before = f_proj[:, :, 1].clone()
    f_proj[:, :, 0] = torch.clamp(f_proj[:, :, 0], -lim, lim)
    f_proj[:, :, 1] = torch.clamp(f_proj[:, :, 1], -lim, lim)

    # keep swing zero exactly
    f_proj = f_proj * stance.unsqueeze(-1)

    violations: Dict[str, object] = {
        "fz_clamp_count": int(((fz_before < fz_min) | (fz_before > fz_max)).sum().detach().cpu()),
        "fx_friction_clamp_count": int((fx_before.abs() > lim).sum().detach().cpu()),
        "fy_friction_clamp_count": int((fy_before.abs() > lim).sum().detach().cpu()),
    }
    # violations = {
    #     "fz_clamp_count": int(((fz_before < fz_min) | (fz_before > fz_max)).sum().detach().cpu()),
    #     "fx_friction_clamp_count": int((fx_before.abs() > lim).sum().detach().cpu()),
    #     "fy_friction_clamp_count": int((fy_before.abs() > lim).sum().detach().cpu()),
    # }
    return f_proj, violations


def apply_force_rate_limit(
    f: torch.Tensor,
    prev_f: torch.Tensor | None,
    max_delta_f: float,
    stance: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    if prev_f is None:
        return f, {"force_rate_clamp_count": 0, "max_force_delta_before": 0.0}

    delta = f - prev_f.to(device=f.device, dtype=f.dtype)
    max_before = float(delta.abs().max().detach().cpu())
    delta_clamped = torch.clamp(delta, -max_delta_f, max_delta_f)
    f_limited = prev_f.to(device=f.device, dtype=f.dtype) + delta_clamped
    f_limited = f_limited * stance.unsqueeze(-1)

    count = int((delta.abs() > max_delta_f).sum().detach().cpu())
    return f_limited, {
        "force_rate_clamp_count": count,
        "max_force_delta_before": max_before,
    }


class ProjectedForceMPCState:
    def __init__(self):
        self.prev_f = None

    def reset(self):
        self.prev_f = None


def plan_projected_force_mpc(
    robot,
    ref: Dict[str, torch.Tensor],
    x_hat: torch.Tensor,
    u_cmd: torch.Tensor,
    beta_t: torch.Tensor,
    planner_state: ProjectedForceMPCState | None = None,
    h_ref: float = 0.67,
    mass: float = 32.5,
    gravity: float = 9.81,
    base_kp_h: float = 45.0,
    base_kd_h: float = 10.0,
    base_kp_vxy: float = 4.0,
    residual_ratio: float = 0.02,
    mu: float = 0.6,
    fz_min: float = 0.0,
    fz_max: float = 8.0,
    fxy_abs_max: float = 3.0,
    max_delta_f: float = 1.0,
    smoothing_alpha: float = 0.75,
    use_k: int = 0,
    min_stance_legs: int = 2,
    force_sign: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Projected forceMPC planner.

    Returns:
        f_feet: [N,4,3]
        info
    """
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype
    N = robot.data.joint_pos.shape[0]

    beta_t = normalize_beta(beta_t.to(device=device, dtype=dtype))
    beta_h = beta_t[:, 0]
    beta_v = beta_t[:, 1]
    beta_e = beta_t[:, 2]

    stance = extract_stance_mask(ref, use_k=use_k, min_stance_legs=min_stance_legs).to(device=device, dtype=dtype)
    stance_cols = stance.repeat_interleave(3, dim=1)

    h = x_hat[:, 2]
    v_world_xy = x_hat[:, 6:8]
    vz = x_hat[:, 8]

    v_des_xy = u_cmd[:, 0:2]
    v_err_xy = v_des_xy - v_world_xy
    h_err = torch.full_like(h, h_ref) - h

    kp_h = base_kp_h * (0.5 + 1.5 * beta_h)
    kd_h = base_kd_h * (0.5 + 1.5 * beta_h)
    kp_vxy = base_kp_vxy * (0.5 + 1.5 * beta_v)

    a_des = torch.zeros((N, 6), device=device, dtype=dtype)
    a_des[:, 0:2] = kp_vxy.unsqueeze(1) * v_err_xy
    a_des[:, 2] = gravity + kp_h * h_err + kd_h * (0.0 - vz)
    a_des[:, 3:6] = 0.0

    G, map_info = build_centroidal_force_map(robot, mass=mass)
    Gm = G * stance_cols.unsqueeze(1)

    w_acc_z, w_acc_xy, w_force, rate_scale = beta_to_weights(beta_t)
    W = torch.zeros((N, 6), device=device, dtype=dtype)
    W[:, 0] = w_acc_xy
    W[:, 1] = w_acc_xy
    W[:, 2] = w_acc_z
    W[:, 3:6] = 0.05

    eye12 = torch.eye(12, device=device, dtype=dtype)
    f_raw_list = []
    for i in range(N):
        Wi = torch.diag(W[i])
        H = Gm[i].T @ Wi @ Gm[i] + w_force[i] * eye12
        b = Gm[i].T @ Wi @ a_des[i]
        try:
            fi = torch.linalg.solve(H, b)
        except RuntimeError:
            fi = torch.linalg.lstsq(H, b).solution
        f_raw_list.append(fi)

    f_raw = torch.stack(f_raw_list, dim=0).view(N, 4, 3)
    f_raw = force_sign * residual_ratio * f_raw
    f_raw = f_raw * stance.unsqueeze(-1)

    # Projection pass 1.
    f_proj, proj_info_1 = project_forces(
        f=f_raw,
        stance=stance,
        mu=mu,
        fz_min=fz_min,
        fz_max=fz_max,
        fxy_abs_max=fxy_abs_max,
    )

    # Force-rate limit.
    prev_f = planner_state.prev_f if planner_state is not None else None
    prev=None
    max_delta = max_delta_f * rate_scale.view(N, 1, 1)
    # Per-env max_delta requires custom clamp.
    if prev_f is not None:
        prev = prev_f.to(device=device, dtype=dtype)
        delta = f_proj - prev
        max_before = float(delta.abs().max().detach().cpu())
        delta_clamped = torch.maximum(torch.minimum(delta, max_delta), -max_delta)
        f_rate = prev + delta_clamped
        f_rate = f_rate * stance.unsqueeze(-1)
        rate_info = {
            "force_rate_clamp_count": int((delta.abs() > max_delta).sum().detach().cpu()),
            "max_force_delta_before": max_before,
            "max_delta_f_env0": float(max_delta[0, 0, 0].detach().cpu()),
        }
    else:
        f_rate = f_proj
        rate_info = {
            "force_rate_clamp_count": 0,
            "max_force_delta_before": 0.0,
            "max_delta_f_env0": float(max_delta[0, 0, 0].detach().cpu()),
        }

    # Projection pass 2 after rate limit.
    f_rate, proj_info_2 = project_forces(
        f=f_rate,
        stance=stance,
        mu=mu,
        fz_min=fz_min,
        fz_max=fz_max,
        fxy_abs_max=fxy_abs_max,
    )

    # Smoothing.
    if prev_f is not None:
        alpha = torch.clamp(
            torch.full((N,), smoothing_alpha, device=device, dtype=dtype) + 0.10 * beta_e,
            min=0.0,
            max=0.95,
        )
        f = alpha.view(N, 1, 1) * prev + (1.0 - alpha).view(N, 1, 1) * f_rate
        f = f * stance.unsqueeze(-1)
        # final projection to preserve constraints
        f, proj_info_3 = project_forces(
            f=f,
            stance=stance,
            mu=mu,
            fz_min=fz_min,
            fz_max=fz_max,
            fxy_abs_max=fxy_abs_max,
        )
    else:
        alpha = torch.full((N,), smoothing_alpha, device=device, dtype=dtype)
        f = f_rate
        proj_info_3 = {"fz_clamp_count": 0, "fx_friction_clamp_count": 0, "fy_friction_clamp_count": 0}

    if planner_state is not None:
        planner_state.prev_f = f.detach().clone()

    info = {
        "beta_env0": beta_t[0].detach().cpu().tolist(),
        "kp_h_env0": float(kp_h[0].detach().cpu()),
        "kd_h_env0": float(kd_h[0].detach().cpu()),
        "kp_vxy_env0": float(kp_vxy[0].detach().cpu()),
        "w_acc_z_env0": float(w_acc_z[0].detach().cpu()),
        "w_acc_xy_env0": float(w_acc_xy[0].detach().cpu()),
        "w_force_env0": float(w_force[0].detach().cpu()),
        "rate_scale_env0": float(rate_scale[0].detach().cpu()),
        "smoothing_alpha_env0": float(alpha[0].detach().cpu()),
        "stance_mask_env0": stance[0].detach().cpu().tolist(),
        "num_stance_env0": float(stance[0].sum().detach().cpu()),
        "h_ref": h_ref,
        "h_mean": float(h.mean().detach().cpu()),
        "vz_mean": float(vz.mean().detach().cpu()),
        "h_err_mean": float(h_err.mean().detach().cpu()),
        "v_des_xy_env0": v_des_xy[0].detach().cpu().tolist(),
        "v_world_xy_env0": v_world_xy[0].detach().cpu().tolist(),
        "v_err_xy_env0": v_err_xy[0].detach().cpu().tolist(),
        "a_des_env0": a_des[0].detach().cpu().tolist(),
        "f_raw_env0": f_raw[0].detach().cpu().tolist(),
        "f_projected_env0": f[0].detach().cpu().tolist(),
        "f_norm_env0": float(torch.linalg.norm(f[0]).detach().cpu()),
        "mu": mu,
        "fz_min": fz_min,
        "fz_max": fz_max,
        "fxy_abs_max": fxy_abs_max,
        "force_sign": force_sign,
        "projection_pass1": proj_info_1,
        "projection_pass2": proj_info_2,
        "projection_pass3": proj_info_3,
    }
    info.update(rate_info)
    info.update(map_info)

    return f, info


def make_projected_force_mpc_torque(
    robot,
    ref: Dict[str, torch.Tensor],
    x_hat: torch.Tensor,
    u_cmd: torch.Tensor,
    beta_t: torch.Tensor,
    planner_state: ProjectedForceMPCState | None = None,
    h_ref: float = 0.67,
    mass: float = 32.5,
    gravity: float = 9.81,
    residual_ratio: float = 0.02,
    mu: float = 0.6,
    fz_min: float = 0.0,
    fz_max: float = 8.0,
    fxy_abs_max: float = 3.0,
    max_delta_f: float = 1.0,
    smoothing_alpha: float = 0.75,
    tau_scale: float = 1.0,
    max_tau: float = 3.0,
    linear_rows: str = "0_3",
    force_sign: float = 1.0,
    use_k: int = 0,
    min_stance_legs: int = 2,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    f_feet, info = plan_projected_force_mpc(
        robot=robot,
        ref=ref,
        x_hat=x_hat,
        u_cmd=u_cmd,
        beta_t=beta_t,
        planner_state=planner_state,
        h_ref=h_ref,
        mass=mass,
        gravity=gravity,
        residual_ratio=residual_ratio,
        mu=mu,
        fz_min=fz_min,
        fz_max=fz_max,
        fxy_abs_max=fxy_abs_max,
        max_delta_f=max_delta_f,
        smoothing_alpha=smoothing_alpha,
        use_k=use_k,
        min_stance_legs=min_stance_legs,
        force_sign=force_sign,
    )

    Jv_feet, j_info = extract_foot_jacobians_action_order(robot=robot, linear_rows=linear_rows)
    tau = compute_tau_jtf(Jv_feet, f_feet)
    tau = tau_scale * tau
    tau = torch.clamp(tau, -max_tau, max_tau)

    info.update(j_info)
    info.update({
        "tau_scale": tau_scale,
        "max_tau": max_tau,
        "linear_rows": linear_rows,
        "tau_mean_abs": float(tau.abs().mean().detach().cpu()),
        "tau_max_abs": float(tau.abs().max().detach().cpu()),
    })

    return tau, info
