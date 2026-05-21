# isaaclab_carto/lowlevel/projected_gradient_force_mpc_v2.py
#
# B5-e version of B5-d.
#
# Main changes:
#   - beta objective mapping is centralized in force_mpc_objectives.py
#   - PG solve can be run with a stronger residual target scale
#   - debug output clearly reports beta semantics
#
# This file intentionally reuses projection/Jacobian utilities from prior steps.

from __future__ import annotations

from typing import Dict, Tuple

import torch

from isaaclab_carto.lowlevel.support_force_control import (
    extract_foot_jacobians_action_order,
    compute_tau_jtf,
)
from isaaclab_carto.lowlevel.projected_force_mpc import (
    extract_stance_mask,
    build_centroidal_force_map,
    project_forces,
)
from isaaclab_carto.lowlevel.force_mpc_objectives import (
    ForceMPCObjectiveConfig,
    normalize_beta,
    build_accel_target,
    build_weight_vector,
    beta_to_objective_terms,
)


class PGForceMPCV2State:
    def __init__(self):
        self.prev_f = None

    def reset(self):
        self.prev_f = None


def objective_value(H: torch.Tensor, b: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.dot(f, H @ f) - torch.dot(b, f)


def solve_pg_projected(
    H_batch: torch.Tensor,
    b_batch: torch.Tensor,
    stance: torch.Tensor,
    mu: float,
    fz_min: float,
    fz_max: float,
    fxy_abs_max: float,
    init_f: torch.Tensor | None,
    pg_iters: int,
    pg_step_size: float,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    N = H_batch.shape[0]
    device = H_batch.device
    dtype = H_batch.dtype

    if init_f is None:
        f = torch.zeros((N, 4, 3), device=device, dtype=dtype)
    else:
        f = init_f.to(device=device, dtype=dtype).clone()

    f, proj0 = project_forces(f, stance, mu, fz_min, fz_max, fxy_abs_max)
    f_flat = f.reshape(N, 12)

    obj0 = [
        float(objective_value(H_batch[i], b_batch[i], f_flat[i]).detach().cpu())
        for i in range(N)
    ]

    projection_total: Dict[str, int] = {
        "fz_clamp_count": int(proj0.get("fz_clamp_count", 0)),
        "fx_friction_clamp_count": int(proj0.get("fx_friction_clamp_count", 0)),
        "fy_friction_clamp_count": int(proj0.get("fy_friction_clamp_count", 0)),
    }

    for _ in range(pg_iters):
        grad = torch.bmm(H_batch, f_flat.unsqueeze(-1)).squeeze(-1) - b_batch
        f_next = (f_flat - pg_step_size * grad).view(N, 4, 3)
        f_next, proj = project_forces(f_next, stance, mu, fz_min, fz_max, fxy_abs_max)

        projection_total["fz_clamp_count"] += int(proj.get("fz_clamp_count", 0))
        projection_total["fx_friction_clamp_count"] += int(proj.get("fx_friction_clamp_count", 0))
        projection_total["fy_friction_clamp_count"] += int(proj.get("fy_friction_clamp_count", 0))

        f_flat = f_next.reshape(N, 12)

    obj_last = [
        float(objective_value(H_batch[i], b_batch[i], f_flat[i]).detach().cpu())
        for i in range(N)
    ]

    grad_last = torch.bmm(H_batch, f_flat.unsqueeze(-1)).squeeze(-1) - b_batch

    info: Dict[str, object] = {
        "pg_obj0_env0": obj0[0],
        "pg_obj_last_env0": obj_last[0],
        "pg_obj_delta_env0": obj_last[0] - obj0[0],
        "pg_grad_norm_last_env0": float(torch.linalg.norm(grad_last[0]).detach().cpu()),
        "pg_projection_total": projection_total,
    }

    return f_flat.view(N, 4, 3), info


def plan_pg_force_mpc_v2(
    robot,
    ref: Dict[str, torch.Tensor],
    x_hat: torch.Tensor,
    u_cmd: torch.Tensor,
    beta_t: torch.Tensor,
    planner_state: PGForceMPCV2State | None = None,
    obj_cfg: ForceMPCObjectiveConfig | None = None,
    h_ref: float = 0.67,
    mass: float = 32.5,
    gravity: float = 9.81,
    residual_ratio: float = 0.05,
    target_scale: float = 1.0,
    mu: float = 0.6,
    fz_min: float = 0.0,
    fz_max: float = 8.0,
    fxy_abs_max: float = 3.0,
    max_delta_f: float = 1.0,
    smoothing_alpha: float = 0.70,
    pg_iters: int = 40,
    pg_step_size: float = 0.06,
    pg_init: str = "prev",
    use_k: int = 0,
    min_stance_legs: int = 2,
    force_sign: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype
    N = robot.data.joint_pos.shape[0]

    beta_t = normalize_beta(beta_t.to(device=device, dtype=dtype))

    stance = extract_stance_mask(ref, use_k=use_k, min_stance_legs=min_stance_legs).to(device=device, dtype=dtype)
    stance_cols = stance.repeat_interleave(3, dim=1)

    a_des, aux = build_accel_target(
        x_hat=x_hat,
        u_cmd=u_cmd,
        beta_t=beta_t,
        h_ref=h_ref,
        gravity=gravity,
        cfg=obj_cfg,
    )
    W_vec, terms = build_weight_vector(beta_t=beta_t, cfg=obj_cfg)

    G, map_info = build_centroidal_force_map(robot, mass=mass)
    Gm = G * stance_cols.unsqueeze(1)

    eye12 = torch.eye(12, device=device, dtype=dtype)
    H_batch = torch.zeros((N, 12, 12), device=device, dtype=dtype)
    b_batch = torch.zeros((N, 12), device=device, dtype=dtype)

    for i in range(N):
        Wi = torch.diag(W_vec[i])
        H_batch[i] = Gm[i].T @ Wi @ Gm[i] + terms["w_force"][i] * eye12
        b_batch[i] = force_sign * residual_ratio * target_scale * (Gm[i].T @ Wi @ a_des[i])

    init_f = None
    if pg_init == "prev" and planner_state is not None and planner_state.prev_f is not None:
        init_f = planner_state.prev_f

    f_pg, pg_info = solve_pg_projected(
        H_batch=H_batch,
        b_batch=b_batch,
        stance=stance,
        mu=mu,
        fz_min=fz_min,
        fz_max=fz_max,
        fxy_abs_max=fxy_abs_max,
        init_f=init_f,
        pg_iters=pg_iters,
        pg_step_size=pg_step_size,
    )

    beta_terms = beta_to_objective_terms(beta_t, obj_cfg)
    rate_scale = beta_terms["rate_scale"]

    prev_f = planner_state.prev_f if planner_state is not None else None
    max_delta = max_delta_f * rate_scale.view(N, 1, 1)

    if prev_f is not None:
        prev = prev_f.to(device=device, dtype=dtype)
        delta = f_pg - prev
        max_before = float(delta.abs().max().detach().cpu())
        delta_clamped = torch.maximum(torch.minimum(delta, max_delta), -max_delta)
        f_rate = (prev + delta_clamped) * stance.unsqueeze(-1)
        force_rate_clamp_count = int((delta.abs() > max_delta).sum().detach().cpu())
    else:
        prev = None
        f_rate = f_pg
        max_before = 0.0
        force_rate_clamp_count = 0

    f_rate, proj_rate = project_forces(f_rate, stance, mu, fz_min, fz_max, fxy_abs_max)

    if prev is not None:
        alpha = torch.clamp(
            torch.full((N,), smoothing_alpha, device=device, dtype=dtype) + beta_terms["smoothing_extra"],
            min=0.0,
            max=0.95,
        )
        f = alpha.view(N, 1, 1) * prev + (1.0 - alpha).view(N, 1, 1) * f_rate
        f = f * stance.unsqueeze(-1)
        f, proj_smooth = project_forces(f, stance, mu, fz_min, fz_max, fxy_abs_max)
    else:
        alpha = torch.full((N,), smoothing_alpha, device=device, dtype=dtype)
        f = f_rate
        proj_smooth = {"fz_clamp_count": 0, "fx_friction_clamp_count": 0, "fy_friction_clamp_count": 0}

    if planner_state is not None:
        planner_state.prev_f = f.detach().clone()

    info: Dict[str, object] = {
        "beta_env0": beta_t[0].detach().cpu().tolist(),
        "beta_semantics": "[height/stability, velocity, energy/smoothness]",
        "kp_h_env0": float(beta_terms["kp_h"][0].detach().cpu()),
        "kd_h_env0": float(beta_terms["kd_h"][0].detach().cpu()),
        "kp_vxy_env0": float(beta_terms["kp_vxy"][0].detach().cpu()),
        "w_acc_z_env0": float(beta_terms["w_acc_z"][0].detach().cpu()),
        "w_acc_xy_env0": float(beta_terms["w_acc_xy"][0].detach().cpu()),
        "w_force_env0": float(beta_terms["w_force"][0].detach().cpu()),
        "rate_scale_env0": float(rate_scale[0].detach().cpu()),
        "smoothing_alpha_env0": float(alpha[0].detach().cpu()),
        "stance_mask_env0": stance[0].detach().cpu().tolist(),
        "num_stance_env0": float(stance[0].sum().detach().cpu()),
        "a_des_env0": a_des[0].detach().cpu().tolist(),
        "v_err_xy_env0": aux["v_err_xy"][0].detach().cpu().tolist(),
        "h_err_mean": float(aux["h_err"].mean().detach().cpu()),
        "f_pg_env0": f_pg[0].detach().cpu().tolist(),
        "f_projected_env0": f[0].detach().cpu().tolist(),
        "f_norm_env0": float(torch.linalg.norm(f[0]).detach().cpu()),
        "residual_ratio": residual_ratio,
        "target_scale": target_scale,
        "mu": mu,
        "fz_min": fz_min,
        "fz_max": fz_max,
        "fxy_abs_max": fxy_abs_max,
        "pg_iters": pg_iters,
        "pg_step_size": pg_step_size,
        "pg_init": pg_init,
        "force_rate_clamp_count": force_rate_clamp_count,
        "max_force_delta_before": max_before,
        "max_delta_f_env0": float(max_delta[0, 0, 0].detach().cpu()),
        "projection_after_rate": proj_rate,
        "projection_after_smooth": proj_smooth,
    }
    info.update(pg_info)
    info.update(map_info)

    return f, info


def make_pg_force_mpc_v2_torque(
    robot,
    ref: Dict[str, torch.Tensor],
    x_hat: torch.Tensor,
    u_cmd: torch.Tensor,
    beta_t: torch.Tensor,
    planner_state: PGForceMPCV2State | None = None,
    obj_cfg: ForceMPCObjectiveConfig | None = None,
    h_ref: float = 0.67,
    mass: float = 32.5,
    gravity: float = 9.81,
    residual_ratio: float = 0.05,
    target_scale: float = 1.0,
    mu: float = 0.6,
    fz_min: float = 0.0,
    fz_max: float = 8.0,
    fxy_abs_max: float = 3.0,
    max_delta_f: float = 1.0,
    smoothing_alpha: float = 0.70,
    pg_iters: int = 40,
    pg_step_size: float = 0.06,
    pg_init: str = "prev",
    tau_scale: float = 1.0,
    max_tau: float = 3.0,
    linear_rows: str = "0_3",
    force_sign: float = 1.0,
    use_k: int = 0,
    min_stance_legs: int = 2,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    f_feet, info = plan_pg_force_mpc_v2(
        robot=robot,
        ref=ref,
        x_hat=x_hat,
        u_cmd=u_cmd,
        beta_t=beta_t,
        planner_state=planner_state,
        obj_cfg=obj_cfg,
        h_ref=h_ref,
        mass=mass,
        gravity=gravity,
        residual_ratio=residual_ratio,
        target_scale=target_scale,
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
