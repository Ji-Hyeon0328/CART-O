# B6: TRACER/CARTO low-level stack wrapper.
# Organizes verified modules into one callable residual stack:
# z_t/a_HL/beta/u_cmd -> thetaDecoder -> thetaRefMapper -> projected forceMPC -> tau=J^T f.
# Current limitation: this is residual effort injection, not visible walking yet.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from isaaclab_carto.lowlevel.theta_decoder import theta_decoder
from isaaclab_carto.lowlevel.theta_ref_mapper import theta_ref_mapper
from isaaclab_carto.lowlevel.projected_force_mpc import (
    ProjectedForceMPCState,
    make_projected_force_mpc_torque,
)


@dataclass
class TracerLowLevelConfig:
    robot_name: str = "spot"
    h_ref: float = 0.67
    mass: float = 32.5
    gravity: float = 9.81
    residual_ratio: float = 0.02
    mu: float = 0.6
    fz_min: float = 0.0
    fz_max: float = 8.0
    fxy_abs_max: float = 3.0
    max_delta_f: float = 1.0
    smoothing_alpha: float = 0.75
    tau_scale: float = 1.0
    max_tau: float = 3.0
    linear_rows: str = "0_3"
    force_sign: float = 1.0
    ref_k: int = 0
    min_stance_legs: int = 2
    control_dt: float = 0.02


def normalize_beta(beta_t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    beta_t = torch.clamp(beta_t, min=eps)
    return beta_t / torch.clamp(beta_t.sum(dim=1, keepdim=True), min=eps)


def make_beta_from_preset(num_envs: int, device, dtype, preset: str = "balanced") -> torch.Tensor:
    presets = {
        "height": [0.70, 0.20, 0.10],
        "velocity": [0.25, 0.65, 0.10],
        "energy": [0.25, 0.20, 0.55],
        "balanced": [0.45, 0.35, 0.20],
    }
    if preset not in presets:
        raise ValueError(f"Unknown beta preset: {preset}")
    beta = torch.tensor(presets[preset], device=device, dtype=dtype)
    return beta.unsqueeze(0).repeat(num_envs, 1)


def make_z_from_mode(num_envs: int, device, dtype, z_mode: str = "conservative") -> torch.Tensor:
    if z_mode == "conservative":
        z_value = 0.0
    elif z_mode == "aggressive":
        z_value = 1.0
    else:
        raise ValueError(f"Unknown z_mode: {z_mode}")
    return torch.full((num_envs,), z_value, device=device, dtype=dtype)


def make_default_a_hl(num_envs: int, device, dtype) -> torch.Tensor:
    a_hl = torch.tensor([0.20, -0.20, 0.30, -0.10], device=device, dtype=dtype)
    return a_hl.unsqueeze(0).repeat(num_envs, 1)


def make_command(num_envs: int, device, dtype, vx: float, vy: float, wz: float) -> torch.Tensor:
    u_cmd = torch.zeros((num_envs, 3), device=device, dtype=dtype)
    u_cmd[:, 0] = vx
    u_cmd[:, 1] = vy
    u_cmd[:, 2] = wz
    return u_cmd


def advance_theta_phase_with_clock(theta, step: int, dt: float):
    T = theta.gait["T"]
    phase_i = theta.gait["phase_i"]
    phase_advance = (step * dt) / torch.clamp(T, min=1e-6)
    theta.gait["phase_i"] = torch.remainder(phase_i + phase_advance.unsqueeze(1), 1.0)
    return theta


class TracerLowLevelStack:
    def __init__(self, cfg: Optional[TracerLowLevelConfig] = None):
        self.cfg = cfg if cfg is not None else TracerLowLevelConfig()
        self.force_state = ProjectedForceMPCState()

    def reset(self) -> None:
        self.force_state.reset()

    def compute(
        self,
        robot,
        x_hat: torch.Tensor,
        ref_params: Dict[str, torch.Tensor],
        z_t: torch.Tensor,
        a_hl: torch.Tensor,
        beta_t: torch.Tensor,
        u_cmd: torch.Tensor,
        step: int,
        use_gait_clock: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        cfg = self.cfg

        theta = theta_decoder(z_t=z_t, a_HL=a_hl, x_hat=x_hat, u_cmd=u_cmd, robot_name=cfg.robot_name)
        if use_gait_clock:
            theta = advance_theta_phase_with_clock(theta, step=step, dt=cfg.control_dt)

        ref = theta_ref_mapper(theta=theta, x_hat=x_hat, u_cmd=u_cmd, params=ref_params)

        beta_norm = normalize_beta(beta_t)
        tau, force_info = make_projected_force_mpc_torque(
            robot=robot,
            ref=ref,
            x_hat=x_hat,
            u_cmd=u_cmd,
            beta_t=beta_norm,
            planner_state=self.force_state,
            h_ref=cfg.h_ref,
            mass=cfg.mass,
            gravity=cfg.gravity,
            residual_ratio=cfg.residual_ratio,
            mu=cfg.mu,
            fz_min=cfg.fz_min,
            fz_max=cfg.fz_max,
            fxy_abs_max=cfg.fxy_abs_max,
            max_delta_f=cfg.max_delta_f,
            smoothing_alpha=cfg.smoothing_alpha,
            tau_scale=cfg.tau_scale,
            max_tau=cfg.max_tau,
            linear_rows=cfg.linear_rows,
            force_sign=cfg.force_sign,
            use_k=cfg.ref_k,
            min_stance_legs=cfg.min_stance_legs,
        )

        env0 = 0
        k = cfg.ref_k
        info: Dict[str, object] = {
            "z_t_env0": float(z_t[env0].detach().cpu()),
            "a_hl_env0": a_hl[env0].detach().cpu().tolist(),
            "beta_env0": beta_norm[env0].detach().cpu().tolist(),
            "u_cmd_env0": u_cmd[env0].detach().cpu().tolist(),
            "theta_T_env0": float(theta.gait["T"][env0].detach().cpu()),
            "theta_phase_i_env0": theta.gait["phase_i"][env0].detach().cpu().tolist(),
            "theta_duty_i_env0": theta.gait["duty_i"][env0].detach().cpu().tolist(),
            "theta_h_body_ref_env0": float(theta.base["h_body_ref"][env0].detach().cpu()),
            "ref_S_env0": ref["S"][env0, :, k].detach().cpu().tolist(),
            "ref_phase_env0": ref["phase"][env0, :, k].detach().cpu().tolist(),
        }
        info.update(force_info)
        return tau, info
