# isaaclab_carto/lowlevel/simple_force_planner.py
#
# B4-b: simple MPC-shaped force planner.
#
# This is NOT the full MATLAB forceMPC port yet.
# It prepares the same interface:
#   x_hat, u_cmd, Ref.S, beta_t -> f_feet_now -> tau = J_foot^T f_feet_now

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
    S = ref['S']
    H = S.shape[-1]
    k = min(max(use_k, 0), H - 1)
    stance = (S[:, :, k] > 0.5).to(S.dtype)
    num_stance = stance.sum(dim=1)
    unsafe = num_stance < float(min_stance_legs)
    if torch.any(unsafe):
        stance[unsafe, :] = 1.0
    return stance


def yaw_to_rot2d(yaw: torch.Tensor) -> torch.Tensor:
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    R = torch.zeros((yaw.shape[0], 2, 2), device=yaw.device, dtype=yaw.dtype)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    return R


def make_default_beta(num_envs: int, device, dtype) -> torch.Tensor:
    beta = torch.tensor([0.45, 0.35, 0.20], device=device, dtype=dtype)
    return beta.unsqueeze(0).repeat(num_envs, 1)


class SimpleForcePlannerState:
    def __init__(self):
        self.prev_f_feet = None

    def reset(self):
        self.prev_f_feet = None


def plan_simple_forces(
    robot,
    ref: Dict[str, torch.Tensor],
    x_hat: torch.Tensor,
    u_cmd: torch.Tensor,
    beta_t: torch.Tensor | None = None,
    planner_state: SimpleForcePlannerState | None = None,
    h_ref: float = 0.67,
    mass: float = 32.5,
    gravity: float = 9.81,
    base_kp_h: float = 40.0,
    base_kd_h: float = 8.0,
    base_kp_vxy: float = 10.0,
    residual_ratio: float = 0.02,
    max_fz_per_foot: float = 8.0,
    max_fxy_per_foot: float = 2.0,
    smoothing_alpha: float = 0.80,
    use_k: int = 0,
    min_stance_legs: int = 2,
    force_sign: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype
    num_envs = robot.data.joint_pos.shape[0]

    if beta_t is None:
        beta_t = make_default_beta(num_envs, device=device, dtype=dtype)
    beta_t = normalize_beta(beta_t.to(device=device, dtype=dtype))

    beta_h = beta_t[:, 0]
    beta_v = beta_t[:, 1]
    beta_e = beta_t[:, 2]

    stance = extract_stance_mask(ref=ref, use_k=use_k, min_stance_legs=min_stance_legs).to(device=device, dtype=dtype)
    num_stance = torch.clamp(stance.sum(dim=1), min=1.0)

    h = x_hat[:, 2]
    yaw = x_hat[:, 5]
    v_world_xy = x_hat[:, 6:8]
    vz = x_hat[:, 8]

    R = yaw_to_rot2d(yaw)
    v_des_world_xy = torch.bmm(R, u_cmd[:, 0:2].unsqueeze(-1)).squeeze(-1)
    v_err_xy = v_des_world_xy - v_world_xy

    kp_h = base_kp_h * (0.5 + 1.5 * beta_h)
    kd_h = base_kd_h * (0.5 + 1.5 * beta_h)
    kp_vxy = base_kp_vxy * (0.5 + 1.5 * beta_v)

    energy_scale = torch.clamp(1.0 - 0.5 * beta_e, min=0.3, max=1.0)
    alpha = torch.clamp(
        torch.full((num_envs,), smoothing_alpha, device=device, dtype=dtype) + 0.10 * beta_e,
        min=0.0,
        max=0.95,
    )

    h_ref_t = torch.full_like(h, h_ref)
    h_err = h_ref_t - h

    fz_total = mass * gravity + kp_h * h_err + kd_h * (0.0 - vz)
    fz_residual_total = residual_ratio * energy_scale * fz_total
    fz_per_stance = fz_residual_total / num_stance
    fz_per_stance = torch.clamp(fz_per_stance, -max_fz_per_foot, max_fz_per_foot)

    fxy_total = residual_ratio * energy_scale.unsqueeze(1) * kp_vxy.unsqueeze(1) * v_err_xy
    fxy_per_stance = fxy_total / num_stance.unsqueeze(1)
    fxy_per_stance = torch.clamp(fxy_per_stance, -max_fxy_per_foot, max_fxy_per_foot)

    f_raw = torch.zeros((num_envs, 4, 3), device=device, dtype=dtype)
    f_raw[:, :, 0] = fxy_per_stance[:, 0].unsqueeze(1) * stance
    f_raw[:, :, 1] = fxy_per_stance[:, 1].unsqueeze(1) * stance
    f_raw[:, :, 2] = fz_per_stance.unsqueeze(1) * stance
    f_raw = force_sign * f_raw

    if planner_state is not None and planner_state.prev_f_feet is not None:
        prev = planner_state.prev_f_feet.to(device=device, dtype=dtype)
        f_feet = alpha.view(num_envs, 1, 1) * prev + (1.0 - alpha).view(num_envs, 1, 1) * f_raw
    else:
        f_feet = f_raw

    # Keep swing legs force-free even with smoothing.
    f_feet = f_feet * stance.unsqueeze(-1)

    if planner_state is not None:
        planner_state.prev_f_feet = f_feet.detach().clone()

    info = {
        'beta_env0': beta_t[0].detach().cpu().tolist(),
        'beta_h_env0': float(beta_h[0].detach().cpu()),
        'beta_v_env0': float(beta_v[0].detach().cpu()),
        'beta_e_env0': float(beta_e[0].detach().cpu()),
        'kp_h_env0': float(kp_h[0].detach().cpu()),
        'kd_h_env0': float(kd_h[0].detach().cpu()),
        'kp_vxy_env0': float(kp_vxy[0].detach().cpu()),
        'energy_scale_env0': float(energy_scale[0].detach().cpu()),
        'smoothing_alpha_env0': float(alpha[0].detach().cpu()),
        'stance_mask_env0': stance[0].detach().cpu().tolist(),
        'num_stance_env0': float(num_stance[0].detach().cpu()),
        'h_ref': h_ref,
        'h_mean': float(h.mean().detach().cpu()),
        'vz_mean': float(vz.mean().detach().cpu()),
        'h_err_mean': float(h_err.mean().detach().cpu()),
        'v_des_world_xy_env0': v_des_world_xy[0].detach().cpu().tolist(),
        'v_world_xy_env0': v_world_xy[0].detach().cpu().tolist(),
        'v_err_xy_env0': v_err_xy[0].detach().cpu().tolist(),
        'fz_total_mean': float(fz_total.mean().detach().cpu()),
        'fz_residual_total_mean': float(fz_residual_total.mean().detach().cpu()),
        'fz_per_stance_mean': float(fz_per_stance.mean().detach().cpu()),
        'fxy_per_stance_env0': fxy_per_stance[0].detach().cpu().tolist(),
        'f_feet_env0': f_feet[0].detach().cpu().tolist(),
        'force_sign': force_sign,
        'use_k': use_k,
    }
    return f_feet, info


def make_simple_force_planner_torque(
    robot,
    ref: Dict[str, torch.Tensor],
    x_hat: torch.Tensor,
    u_cmd: torch.Tensor,
    beta_t: torch.Tensor | None = None,
    planner_state: SimpleForcePlannerState | None = None,
    h_ref: float = 0.67,
    mass: float = 32.5,
    gravity: float = 9.81,
    base_kp_h: float = 40.0,
    base_kd_h: float = 8.0,
    base_kp_vxy: float = 10.0,
    residual_ratio: float = 0.02,
    max_fz_per_foot: float = 8.0,
    max_fxy_per_foot: float = 2.0,
    smoothing_alpha: float = 0.80,
    tau_scale: float = 1.0,
    max_tau: float = 3.0,
    linear_rows: str = '0_3',
    force_sign: float = 1.0,
    use_k: int = 0,
    min_stance_legs: int = 2,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    f_feet, f_info = plan_simple_forces(
        robot=robot,
        ref=ref,
        x_hat=x_hat,
        u_cmd=u_cmd,
        beta_t=beta_t,
        planner_state=planner_state,
        h_ref=h_ref,
        mass=mass,
        gravity=gravity,
        base_kp_h=base_kp_h,
        base_kd_h=base_kd_h,
        base_kp_vxy=base_kp_vxy,
        residual_ratio=residual_ratio,
        max_fz_per_foot=max_fz_per_foot,
        max_fxy_per_foot=max_fxy_per_foot,
        smoothing_alpha=smoothing_alpha,
        use_k=use_k,
        min_stance_legs=min_stance_legs,
        force_sign=force_sign,
    )
    Jv_feet, j_info = extract_foot_jacobians_action_order(robot=robot, linear_rows=linear_rows)
    tau = compute_tau_jtf(Jv_feet, f_feet)
    tau = tau_scale * tau
    tau = torch.clamp(tau, -max_tau, max_tau)

    info: Dict[str, object] = {}
    info.update(j_info)
    info.update(f_info)
    info.update({
        'tau_scale': tau_scale,
        'max_tau': max_tau,
        'linear_rows': linear_rows,
        'tau_mean_abs': float(tau.abs().mean().detach().cpu()),
        'tau_max_abs': float(tau.abs().max().detach().cpu()),
    })
    return tau, info
