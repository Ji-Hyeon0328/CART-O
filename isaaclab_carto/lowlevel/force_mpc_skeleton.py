# isaaclab_carto/lowlevel/force_mpc_skeleton.py
#
# B5-a/B5-b: Python forceMPC skeleton.
#
# This is a first Python/PyTorch port of the MATLAB forceMPC idea.
# It is intentionally simplified:
#
#   - centroidal single-step/horizon linear model
#   - unconstrained least-squares force planner
#   - Ref.S swing legs are forced to zero by masking
#   - no friction cone yet
#   - no QP inequality constraints yet
#
# Interface:
#
#   x_hat, Ref, beta_t
#     → planned f_feet_now [N, 4, 3]
#     → tau = J_foot^T f_feet_now
#
# Next versions:
#   B5-c: equality masking inside optimizer more explicitly
#   B5-d: fz bounds / friction cone
#   B5-e: full QP-style solve

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
    unsafe = stance.sum(dim=1) < float(min_stance_legs)
    if torch.any(unsafe):
        stance[unsafe, :] = 1.0
    return stance


def skew(r: torch.Tensor) -> torch.Tensor:
    """
    r: [N, 3]
    return [N, 3, 3] skew matrix such that skew(r) f = r x f
    """
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
    """
    Build a simple map from foot forces to base acceleration.

    For each leg force f_i in world frame:
        linear accel contribution: a = sum f_i / m
        angular accel proxy: alpha = I^{-1} sum r_i x f_i

    This version uses approximate diagonal inertia for Spot.
    It is enough for debugging force planner structure.

    Returns:
        G: [N, 6, 12], maps stacked forces [fl,fr,hl,hr] xyz to [a_xyz, alpha_xyz].
    """
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype
    N = robot.data.joint_pos.shape[0]

    # Foot positions in world, from body data.
    body_names = robot.body_names
    name_to_idx = {name: i for i, name in enumerate(body_names)}
    foot_names = ["fl_foot", "fr_foot", "hl_foot", "hr_foot"]
    foot_indices = [name_to_idx[name] for name in foot_names]

    foot_pos_w = robot.data.body_pos_w[:, foot_indices, :]  # [N,4,3]
    base_pos_w = robot.data.root_pos_w[:, None, :]           # [N,1,3]
    r = foot_pos_w - base_pos_w                              # [N,4,3]

    # Approx Spot inertia. This is not exact USD inertia.
    I_diag = torch.tensor([1.0, 2.0, 2.2], device=device, dtype=dtype)
    I_inv = torch.diag(1.0 / I_diag).unsqueeze(0).repeat(N, 1, 1)

    G = torch.zeros((N, 6, 12), device=device, dtype=dtype)

    for leg in range(4):
        col = slice(3 * leg, 3 * leg + 3)
        G[:, 0:3, col] = torch.eye(3, device=device, dtype=dtype).unsqueeze(0) / mass

        S = skew(r[:, leg, :])
        G[:, 3:6, col] = torch.bmm(I_inv, S)

    info = {
        "foot_names": foot_names,
        "foot_indices": foot_indices,
        "mass": mass,
        "inertia_diag": I_diag.detach().cpu().tolist(),
    }
    return G, info


def beta_to_weights(beta_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    beta_t = [height/stability, velocity, energy/smoothness]

    Returns:
        w_acc_z, w_acc_xy, w_force
    """
    beta = normalize_beta(beta_t)
    beta_h = beta[:, 0]
    beta_v = beta[:, 1]
    beta_e = beta[:, 2]

    w_acc_z = 5.0 + 20.0 * beta_h
    w_acc_xy = 2.0 + 15.0 * beta_v
    w_force = 1e-3 + 5e-3 * beta_e
    return w_acc_z, w_acc_xy, w_force


class ForceMPCSkeletonState:
    def __init__(self):
        self.prev_f = None

    def reset(self):
        self.prev_f = None


def plan_unconstrained_force_mpc(
    robot,
    ref: Dict[str, torch.Tensor],
    x_hat: torch.Tensor,
    u_cmd: torch.Tensor,
    beta_t: torch.Tensor,
    planner_state: ForceMPCSkeletonState | None = None,
    h_ref: float = 0.67,
    mass: float = 32.5,
    gravity: float = 9.81,
    dt: float = 0.02,
    base_kp_h: float = 45.0,
    base_kd_h: float = 10.0,
    base_kp_vxy: float = 4.0,
    residual_ratio: float = 0.02,
    smoothing_alpha: float = 0.80,
    max_fz_per_foot: float = 8.0,
    max_fxy_per_foot: float = 2.0,
    use_k: int = 0,
    min_stance_legs: int = 2,
    force_sign: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Unconstrained least-squares force planner.

    It chooses f to make centroidal acceleration close to desired acceleration:
        minimize || W (G f - a_des) ||^2 + w_force ||f||^2

    Then masks swing legs and clamps force.

    Output:
        f_feet: [N,4,3]
    """
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype
    N = robot.data.joint_pos.shape[0]

    beta_t = normalize_beta(beta_t.to(device=device, dtype=dtype))
    beta_h = beta_t[:, 0]
    beta_v = beta_t[:, 1]
    beta_e = beta_t[:, 2]

    stance = extract_stance_mask(ref, use_k=use_k, min_stance_legs=min_stance_legs).to(device=device, dtype=dtype)
    stance_cols = stance.repeat_interleave(3, dim=1)  # [N,12]

    # State.
    h = x_hat[:, 2]
    v_world_xy = x_hat[:, 6:8]
    vz = x_hat[:, 8]

    # Desired accelerations.
    # Horizontal command is treated as desired world velocity approximately because yaw is small in these tests.
    v_des_xy = u_cmd[:, 0:2]
    v_err_xy = v_des_xy - v_world_xy
    h_err = torch.full_like(h, h_ref) - h

    kp_h = base_kp_h * (0.5 + 1.5 * beta_h)
    kd_h = base_kd_h * (0.5 + 1.5 * beta_h)
    kp_vxy = base_kp_vxy * (0.5 + 1.5 * beta_v)

    a_des = torch.zeros((N, 6), device=device, dtype=dtype)
    a_des[:, 0:2] = kp_vxy.unsqueeze(1) * v_err_xy
    a_des[:, 2] = gravity + kp_h * h_err + kd_h * (0.0 - vz)
    # Angular acceleration target is zero for now.

    G, map_info = build_centroidal_force_map(robot, mass=mass)

    # Mask swing columns: optimizer only uses stance columns.
    Gm = G * stance_cols.unsqueeze(1)

    w_acc_z, w_acc_xy, w_force = beta_to_weights(beta_t)

    # Build W diagonal for [ax, ay, az, alphax, alphay, alphaz].
    W = torch.zeros((N, 6), device=device, dtype=dtype)
    W[:, 0] = w_acc_xy
    W[:, 1] = w_acc_xy
    W[:, 2] = w_acc_z
    W[:, 3:6] = 0.05  # weak angular regularization for now

    # Solve per env:
    # (G^T W G + lambda I) f = G^T W a_des
    f_list = []
    eye12 = torch.eye(12, device=device, dtype=dtype)
    for i in range(N):
        Wi = torch.diag(W[i])
        H = Gm[i].T @ Wi @ Gm[i] + w_force[i] * eye12
        b = Gm[i].T @ Wi @ a_des[i]
        try:
            fi = torch.linalg.solve(H, b)
        except RuntimeError:
            fi = torch.linalg.lstsq(H, b).solution
        f_list.append(fi)

    f = torch.stack(f_list, dim=0)  # [N,12]
    f = residual_ratio * f

    # Apply force sign, mask, clamp.
    f = force_sign * f * stance_cols
    f = f.view(N, 4, 3)

    f[:, :, 0:2] = torch.clamp(f[:, :, 0:2], -max_fxy_per_foot, max_fxy_per_foot)
    f[:, :, 2] = torch.clamp(f[:, :, 2], 0.0, max_fz_per_foot)

    # Smooth but keep swing zero.
    if planner_state is not None and planner_state.prev_f is not None:
        alpha = torch.clamp(torch.full((N,), smoothing_alpha, device=device, dtype=dtype) + 0.10 * beta_e, 0.0, 0.95)
        prev = planner_state.prev_f.to(device=device, dtype=dtype)
        f = alpha.view(N, 1, 1) * prev + (1.0 - alpha).view(N, 1, 1) * f
        f = f * stance.unsqueeze(-1)
    else:
        alpha = torch.full((N,), smoothing_alpha, device=device, dtype=dtype)

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
        "f_feet_env0": f[0].detach().cpu().tolist(),
        "f_norm_env0": float(torch.linalg.norm(f[0]).detach().cpu()),
        "force_sign": force_sign,
    }
    info.update(map_info)

    return f, info


def make_force_mpc_skeleton_torque(
    robot,
    ref: Dict[str, torch.Tensor],
    x_hat: torch.Tensor,
    u_cmd: torch.Tensor,
    beta_t: torch.Tensor,
    planner_state: ForceMPCSkeletonState | None = None,
    h_ref: float = 0.67,
    mass: float = 32.5,
    gravity: float = 9.81,
    dt: float = 0.02,
    residual_ratio: float = 0.02,
    smoothing_alpha: float = 0.80,
    max_fz_per_foot: float = 8.0,
    max_fxy_per_foot: float = 2.0,
    tau_scale: float = 1.0,
    max_tau: float = 3.0,
    linear_rows: str = "0_3",
    force_sign: float = 1.0,
    use_k: int = 0,
    min_stance_legs: int = 2,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    f_feet, info = plan_unconstrained_force_mpc(
        robot=robot,
        ref=ref,
        x_hat=x_hat,
        u_cmd=u_cmd,
        beta_t=beta_t,
        planner_state=planner_state,
        h_ref=h_ref,
        mass=mass,
        gravity=gravity,
        dt=dt,
        residual_ratio=residual_ratio,
        smoothing_alpha=smoothing_alpha,
        max_fz_per_foot=max_fz_per_foot,
        max_fxy_per_foot=max_fxy_per_foot,
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
