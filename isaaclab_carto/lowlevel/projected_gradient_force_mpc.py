# isaaclab_carto/lowlevel/projected_gradient_force_mpc.py
#
# B5-d: projected-gradient QP-style forceMPC.
#
# Difference from B5-c:
#
#   B5-c:
#       unconstrained LS solve
#       → project once/several times
#
#   B5-d:
#       build quadratic objective
#       minimize 0.5 f^T H f - b^T f
#       using projected-gradient iterations:
#           f <- f - alpha * grad
#           project f to:
#             - swing force = 0
#             - fz bounds
#             - friction pyramid
#
# This is still not a production QP solver, but it is much closer to the
# constrained forceMPC structure than one-shot LS + projection.

from __future__ import annotations

#from typing import Dict, Tuple
from typing import Dict, Tuple, Any, cast

import torch

from isaaclab_carto.lowlevel.support_force_control import (
    extract_foot_jacobians_action_order,
    compute_tau_jtf,
)

from isaaclab_carto.lowlevel.projected_force_mpc import (
    normalize_beta,
    extract_stance_mask,
    build_centroidal_force_map,
    beta_to_weights,
    project_forces,
)


class PGForceMPCState:
    def __init__(self):
        self.prev_f = None

    def reset(self):
        self.prev_f = None


def _objective_value(H: torch.Tensor, b: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    # 0.5 f^T H f - b^T f
    return 0.5 * torch.dot(f, H @ f) - torch.dot(b, f)


def solve_projected_gradient_qp_batch(
    H_batch: torch.Tensor,
    b_batch: torch.Tensor,
    stance: torch.Tensor,
    mu: float,
    fz_min: float,
    fz_max: float,
    fxy_abs_max: float,
    init_f: torch.Tensor | None = None,
    num_iters: int = 20,
    step_size: float = 0.08,
    use_bb_step: bool = False,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Batch projected-gradient solve.

    Args:
        H_batch: [N,12,12]
        b_batch: [N,12]
        stance: [N,4]
        init_f: optional [N,4,3]

    Returns:
        f: [N,4,3]
    """
    N = H_batch.shape[0]
    device = H_batch.device
    dtype = H_batch.dtype

    if init_f is None:
        f = torch.zeros((N, 4, 3), device=device, dtype=dtype)
    else:
        f = init_f.to(device=device, dtype=dtype).clone()
        f = f * stance.unsqueeze(-1)

    # Initial projection.
    f, proj0 = project_forces(
        f=f, stance=stance, mu=mu, fz_min=fz_min, fz_max=fz_max, fxy_abs_max=fxy_abs_max
    )

    obj0 = []
    obj_last = []
    grad_norm_last = []
    total_proj_counts: Dict[str, int] = {
        "fz_clamp_count": cast(int, proj0["fz_clamp_count"]),
        "fx_friction_clamp_count": cast(int, proj0["fx_friction_clamp_count"]),
        "fy_friction_clamp_count": cast(int, proj0["fy_friction_clamp_count"]),
    }
    # total_proj_counts = {
    #     "fz_clamp_count": int(proj0["fz_clamp_count"]),
    #     "fx_friction_clamp_count": int(proj0["fx_friction_clamp_count"]),
    #     "fy_friction_clamp_count": int(proj0["fy_friction_clamp_count"]),
    # }

    f_flat = f.reshape(N, 12)

    for env_id in range(N):
        obj0.append(float(_objective_value(H_batch[env_id], b_batch[env_id], f_flat[env_id]).detach().cpu()))

    prev_grad = None
    prev_f_flat = None

    for _it in range(num_iters):
        grad = torch.bmm(H_batch, f_flat.unsqueeze(-1)).squeeze(-1) - b_batch

        if use_bb_step and prev_grad is not None and prev_f_flat is not None:
            # Conservative Barzilai-Borwein-like scalar step per batch.
            s = f_flat - prev_f_flat
            y = grad - prev_grad
            sy = (s * y).sum(dim=1)
            yy = (y * y).sum(dim=1)
            alpha = torch.clamp(sy / torch.clamp(yy, min=1e-8), min=1e-4, max=step_size)
            f_flat_next = f_flat - alpha.unsqueeze(1) * grad
        else:
            f_flat_next = f_flat - step_size * grad

        f_next = f_flat_next.view(N, 4, 3)

        f_next, proj_info = project_forces(
            f=f_next,
            stance=stance,
            mu=mu,
            fz_min=fz_min,
            fz_max=fz_max,
            fxy_abs_max=fxy_abs_max,
        )

        total_proj_counts["fz_clamp_count"] += int(proj_info["fz_clamp_count"]) # type: ignore
        total_proj_counts["fx_friction_clamp_count"] += int(proj_info["fx_friction_clamp_count"])
        total_proj_counts["fy_friction_clamp_count"] += int(proj_info["fy_friction_clamp_count"])

        prev_f_flat = f_flat
        prev_grad = grad
        f_flat = f_next.reshape(N, 12)

    for env_id in range(N):
        obj_last.append(float(_objective_value(H_batch[env_id], b_batch[env_id], f_flat[env_id]).detach().cpu()))
        g = H_batch[env_id] @ f_flat[env_id] - b_batch[env_id]
        grad_norm_last.append(float(torch.linalg.norm(g).detach().cpu()))

    info = {
        "pg_num_iters": num_iters,
        "pg_step_size": step_size,
        "pg_use_bb_step": use_bb_step,
        "pg_obj0_env0": obj0[0],
        "pg_obj_last_env0": obj_last[0],
        "pg_obj_delta_env0": obj_last[0] - obj0[0],
        "pg_grad_norm_last_env0": grad_norm_last[0],
        "pg_projection_total": total_proj_counts,
    }

    return f_flat.view(N, 4, 3), info


def plan_projected_gradient_force_mpc(
    robot,
    ref: Dict[str, torch.Tensor],
    x_hat: torch.Tensor,
    u_cmd: torch.Tensor,
    beta_t: torch.Tensor,
    planner_state: PGForceMPCState | None = None,
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
    smoothing_alpha: float = 0.70,
    pg_iters: int = 20,
    pg_step_size: float = 0.08,
    pg_init: str = "prev",  # zero | prev
    use_k: int = 0,
    min_stance_legs: int = 2,
    force_sign: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Projected-gradient QP-style forceMPC planner.

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

    eye12 = torch.eye(12, device=device, dtype=dtype).unsqueeze(0).repeat(N, 1, 1)
    H_batch = torch.zeros((N, 12, 12), device=device, dtype=dtype)
    b_batch = torch.zeros((N, 12), device=device, dtype=dtype)

    for i in range(N):
        Wi = torch.diag(W[i])
        H = Gm[i].T @ Wi @ Gm[i] + w_force[i] * eye12[i]
        b = Gm[i].T @ Wi @ a_des[i]
        # Scale objective because we apply residual_ratio after the solve in older versions.
        # Here we optimize directly for the residual force scale:
        # G f_residual ~= residual_ratio * a_des.
        b = residual_ratio * b
        H_batch[i] = H
        b_batch[i] = b

    init_f = None
    if pg_init == "prev" and planner_state is not None and planner_state.prev_f is not None:
        init_f = planner_state.prev_f
    elif pg_init == "zero":
        init_f = None

    # Force sign is applied through b direction.
    if force_sign < 0:
        b_batch = -b_batch

    f_pg, pg_info = solve_projected_gradient_qp_batch(
        H_batch=H_batch,
        b_batch=b_batch,
        stance=stance,
        mu=mu,
        fz_min=fz_min,
        fz_max=fz_max,
        fxy_abs_max=fxy_abs_max,
        init_f=init_f,
        num_iters=pg_iters,
        step_size=pg_step_size,
        use_bb_step=False,
    )

    # Force-rate limit relative to previous output.
    prev_f = planner_state.prev_f if planner_state is not None else None
    max_delta = max_delta_f * rate_scale.view(N, 1, 1)

    if prev_f is not None:
        prev = prev_f.to(device=device, dtype=dtype)
        delta = f_pg - prev
        max_before = float(delta.abs().max().detach().cpu())
        delta_clamped = torch.maximum(torch.minimum(delta, max_delta), -max_delta)
        f_rate = prev + delta_clamped
        f_rate = f_rate * stance.unsqueeze(-1)
        force_rate_clamp_count = int((delta.abs() > max_delta).sum().detach().cpu())
    else:
        prev = None
        f_rate = f_pg
        max_before = 0.0
        force_rate_clamp_count = 0

    f_rate, proj_after_rate = project_forces(
        f=f_rate,
        stance=stance,
        mu=mu,
        fz_min=fz_min,
        fz_max=fz_max,
        fxy_abs_max=fxy_abs_max,
    )

    # Smoothing.
    if prev is not None:
        alpha = torch.clamp(
            torch.full((N,), smoothing_alpha, device=device, dtype=dtype) + 0.10 * beta_e,
            min=0.0,
            max=0.95,
        )
        f = alpha.view(N, 1, 1) * prev + (1.0 - alpha).view(N, 1, 1) * f_rate
        f = f * stance.unsqueeze(-1)
        f, proj_after_smooth = project_forces(
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
        proj_after_smooth = {"fz_clamp_count": 0, "fx_friction_clamp_count": 0, "fy_friction_clamp_count": 0}

    if planner_state is not None:
        planner_state.prev_f = f.detach().clone()

    info: Dict[str, object] = {
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
        "f_pg_env0": f_pg[0].detach().cpu().tolist(),
        "f_projected_env0": f[0].detach().cpu().tolist(),
        "f_norm_env0": float(torch.linalg.norm(f[0]).detach().cpu()),
        "mu": mu,
        "fz_min": fz_min,
        "fz_max": fz_max,
        "fxy_abs_max": fxy_abs_max,
        "force_sign": force_sign,
        "pg_init": pg_init,
        "force_rate_clamp_count": force_rate_clamp_count,
        "max_force_delta_before": max_before,
        "max_delta_f_env0": float(max_delta[0, 0, 0].detach().cpu()),
        "projection_after_rate": proj_after_rate,
        "projection_after_smooth": proj_after_smooth,
    }
    info.update(pg_info)
    info.update(map_info)

    return f, info

def make_projected_gradient_force_mpc_torque(
    robot,
    ref: Dict[str, torch.Tensor],
    x_hat: torch.Tensor,
    u_cmd: torch.Tensor,
    beta_t: torch.Tensor,
    planner_state: PGForceMPCState | None = None,
    h_ref: float = 0.67,
    mass: float = 32.5,
    gravity: float = 9.81,
    residual_ratio: float = 0.02,
    mu: float = 0.6,
    fz_min: float = 0.0,
    fz_max: float = 8.0,
    fxy_abs_max: float = 3.0,
    max_delta_f: float = 1.0,
    smoothing_alpha: float = 0.70,
    pg_iters: int = 20,
    pg_step_size: float = 0.08,
    pg_init: str = "prev",
    tau_scale: float = 1.0,
    max_tau: float = 3.0,
    linear_rows: str = "0_3",
    force_sign: float = 1.0,
    use_k: int = 0,
    min_stance_legs: int = 2,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    f_feet, info = plan_projected_gradient_force_mpc(
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
        pg_iters=pg_iters,
        pg_step_size=pg_step_size,
        pg_init=pg_init,
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
