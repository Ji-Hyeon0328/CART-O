# isaaclab_carto/lowlevel/theta_decoder.py
#
# Python/Torch port of thetaDecoder.m.
#
# Input:
#   z_t   : [num_envs] or scalar
#           0 = conservative, 1 = aggressive
#
#   a_HL  : [num_envs, 4]
#           [a_swing, a_body, a_duty, a_imp]
#
#   x_hat : [num_envs, 12]
#           [x y z roll pitch yaw vx vy vz wx wy wz]
#
#   u_cmd : [num_envs, 3]
#           [vx_cmd vy_cmd wz_cmd]
#
# Output:
#   Theta dataclass:
#       Theta.gait["phase_i"]       [N, 4]
#       Theta.gait["T"]             [N]
#       Theta.gait["duty_i"]        [N, 4]
#       Theta.foot["h_swing_i"]     [N, 4]
#       Theta.foot["delta_p_td_i"]  [N, 3, 4]
#       Theta.foot["delta_t_td_i"]  [N, 4]
#       Theta.base["h_body_ref"]    [N]
#       Theta.base["roll_ref"]      [N]
#       Theta.base["pitch_ref"]     [N]
#       Theta.ctrl["k_des"]         [N]
#       Theta.ctrl["mu_exp"]        [N]

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class Theta:
    gait: Dict[str, torch.Tensor]
    foot: Dict[str, torch.Tensor]
    base: Dict[str, torch.Tensor]
    ctrl: Dict[str, torch.Tensor]


def _as_batch_z(z_t: torch.Tensor | int | float, num_envs: int, device, dtype=torch.long) -> torch.Tensor:
    if not torch.is_tensor(z_t):
        z_t = torch.tensor(z_t, device=device, dtype=dtype)
    z_t = z_t.to(device=device) # type: ignore
    if z_t.ndim == 0: # type: ignore
        z_t = z_t.repeat(num_envs) # type: ignore
    return z_t.reshape(num_envs).long() # type: ignore


def _as_batch_action(a_HL: torch.Tensor, num_envs: int, device, dtype) -> torch.Tensor:
    a_HL = a_HL.to(device=device, dtype=dtype)
    if a_HL.ndim == 1:
        a_HL = a_HL.unsqueeze(0).repeat(num_envs, 1)
    if a_HL.shape != (num_envs, 4):
        raise ValueError(f"a_HL must have shape [{num_envs}, 4], got {tuple(a_HL.shape)}")
    return a_HL


def theta_decoder(
    z_t: torch.Tensor | int,
    a_HL: torch.Tensor,
    x_hat: torch.Tensor,
    u_cmd: torch.Tensor,
    robot_name: str = "spot",
) -> Theta:
    """
    Rule-based Theta decoder.

    This follows the Matlab prototype but adds robot-specific body-height presets.
    For the current Isaac Lab Spot USD, the observed standing base height is about 0.53 m,
    so the Spot preset uses higher nominal body height than the old Matlab toy/Go1 demo.

    Args:
        z_t:
            0 = conservative, 1 = aggressive.
        a_HL:
            [N, 4] = [a_swing, a_body, a_duty, a_imp].
        x_hat:
            [N, 12] = [x y z roll pitch yaw vx vy vz wx wy wz].
        u_cmd:
            [N, 3] = [vx_cmd vy_cmd wz_cmd].
        robot_name:
            "spot", "go1", or "toy25".

    Returns:
        Theta dataclass.
    """
    if x_hat.ndim != 2 or x_hat.shape[-1] != 12:
        raise ValueError(f"x_hat must have shape [num_envs, 12], got {tuple(x_hat.shape)}")
    if u_cmd.ndim != 2 or u_cmd.shape[-1] < 3:
        raise ValueError(f"u_cmd must have shape [num_envs, >=3], got {tuple(u_cmd.shape)}")

    device = x_hat.device
    dtype = x_hat.dtype
    num_envs = x_hat.shape[0]

    z_t = _as_batch_z(z_t, num_envs, device=device)
    a_HL = _as_batch_action(a_HL, num_envs, device=device, dtype=dtype)
    u_cmd = u_cmd[:, :3].to(device=device, dtype=dtype)

    robot_name = robot_name.lower()

    a_swing = a_HL[:, 0]
    a_body = a_HL[:, 1]
    a_duty = a_HL[:, 2]
    a_imp = a_HL[:, 3]

    roll = x_hat[:, 3]
    pitch = x_hat[:, 4]
    vx = x_hat[:, 6]
    vy = x_hat[:, 7]

    vx_cmd = u_cmd[:, 0]
    vy_cmd = u_cmd[:, 1]

    v_cmd_mag = torch.sqrt(vx_cmd * vx_cmd + vy_cmd * vy_cmd)
    v_err_xy = torch.stack(
        [
            vx - vx_cmd,
            vy - vy_cmd,
            torch.zeros_like(vx),
        ],
        dim=-1,
    )

    clip = torch.clamp

    #z_bool = z_t > 0
    z_bool: torch.Tensor = torch.gt(z_t, 0)

    # -------------------------------------------------------------------------
    # Gait nominal values.
    #
    # Leg order:
    #   [LF, RF, LH, RH] == [fl, fr, hl, hr]
    #
    # Conservative:
    #   crawl-ish sequence.
    # Aggressive:
    #   trot-like sequence.
    # -------------------------------------------------------------------------
    phase_cons = torch.tensor([0.00, 0.25, 0.50, 0.75], device=device, dtype=dtype)
    phase_aggr = torch.tensor([0.00, 0.50, 0.50, 0.00], device=device, dtype=dtype)

    phase_cons_b = phase_cons.unsqueeze(0).repeat(num_envs, 1)
    phase_aggr_b = phase_aggr.unsqueeze(0).repeat(num_envs, 1)

    phase_i = torch.where(
        z_bool.unsqueeze(1),
        #phase_aggr[None, :].repeat(num_envs, 1),
        #phase_cons[None, :].repeat(num_envs, 1),
        phase_aggr_b,
        phase_cons_b,
    )

    T_nom = torch.where(
        z_bool,
        torch.full((num_envs,), 0.45, device=device, dtype=dtype),
        torch.full((num_envs,), 0.75, device=device, dtype=dtype),
    )

    duty_nom = torch.where(
        z_bool,
        torch.full((num_envs,), 0.55, device=device, dtype=dtype),
        torch.full((num_envs,), 0.78, device=device, dtype=dtype),
    )

    # -------------------------------------------------------------------------
    # Robot-specific nominal body height.
    #
    # Matlab toy/Go1 demo used 0.36 / 0.43, but Isaac Lab Spot standing height
    # from the debug result is around 0.53 m. So Spot needs a higher reference.
    # -------------------------------------------------------------------------
    if robot_name == "spot":
        h_cons = 0.52
        h_aggr = 0.56
        h_min = 0.45
        h_max = 0.65
        h_swing_cons = 0.08
        h_swing_aggr = 0.12
        mu_cons = 0.55
        mu_aggr = 0.60
        k_imp_cons = 0.80
        k_imp_aggr = 1.15
    elif robot_name == "go1":
        h_cons = 0.30
        h_aggr = 0.34
        h_min = 0.24
        h_max = 0.42
        h_swing_cons = 0.06
        h_swing_aggr = 0.10
        mu_cons = 0.55
        mu_aggr = 0.60
        k_imp_cons = 0.80
        k_imp_aggr = 1.15
    else:
        h_cons = 0.36
        h_aggr = 0.43
        h_min = 0.28
        h_max = 0.60
        h_swing_cons = 0.07
        h_swing_aggr = 0.11
        mu_cons = 0.55
        mu_aggr = 0.60
        k_imp_cons = 0.80
        k_imp_aggr = 1.15

    h_body_nom = torch.where(
        z_bool,
        torch.full((num_envs,), h_aggr, device=device, dtype=dtype),
        torch.full((num_envs,), h_cons, device=device, dtype=dtype),
    )

    h_swing_nom = torch.where(
        z_bool,
        torch.full((num_envs,), h_swing_aggr, device=device, dtype=dtype),
        torch.full((num_envs,), h_swing_cons, device=device, dtype=dtype),
    )

    k_imp_nom = torch.where(
        z_bool,
        torch.full((num_envs,), k_imp_aggr, device=device, dtype=dtype),
        torch.full((num_envs,), k_imp_cons, device=device, dtype=dtype),
    )

    mu_nom = torch.where(
        z_bool,
        torch.full((num_envs,), mu_aggr, device=device, dtype=dtype),
        torch.full((num_envs,), mu_cons, device=device, dtype=dtype),
    )

    # Gains copied from the Matlab prototype idea.
    k_swing = 0.04
    k_body = 0.05
    k_duty = 0.10
    k_imp = 0.25

    duty_state_corr = 0.05 * torch.abs(roll) + 0.05 * torch.abs(pitch)
    body_state_corr = -0.03 * torch.abs(roll) - 0.03 * torch.abs(pitch)

    T_raw = T_nom * (1.0 - 0.25 * torch.tanh(v_cmd_mag))
    T = clip(T_raw, 0.35, 0.95)

    duty = duty_nom + k_duty * torch.tanh(a_duty) + duty_state_corr
    duty = clip(duty, 0.45, 0.85)

    h_body_ref = h_body_nom + k_body * torch.tanh(a_body) + body_state_corr
    h_body_ref = clip(h_body_ref, h_min, h_max)

    h_swing = h_swing_nom + k_swing * torch.tanh(a_swing)
    h_swing = clip(h_swing, 0.04, 0.20)

    k_des = k_imp_nom + k_imp * torch.tanh(a_imp)
    k_des = clip(k_des, 0.50, 1.60)

    # -------------------------------------------------------------------------
    # Foot touchdown correction.
    #
    # Matlab code used Kv_foot * v_err_xy. Here we keep a simple equivalent.
    # Shape: [N, 3, 4].
    # -------------------------------------------------------------------------
    delta_p_td_i = torch.zeros((num_envs, 3, 4), device=device, dtype=dtype)

    # If actual vx is lower than command, move touchdown slightly forward.
    delta_p = -0.08 * v_err_xy
    delta_p_td_i[:, :, :] = delta_p[:, :, None]
    delta_p_td_i = clip(delta_p_td_i, -0.05, 0.05)

    delta_t_td_i = torch.zeros((num_envs, 4), device=device, dtype=dtype)

    duty_i = duty[:, None].repeat(1, 4)
    h_swing_i = h_swing[:, None].repeat(1, 4)

    # Simple posture-stabilizing roll/pitch references.
    roll_ref = clip(-0.20 * roll, -0.15, 0.15)
    pitch_ref = clip(-0.20 * pitch, -0.15, 0.15)

    return Theta(
        gait={
            "phase_i": phase_i,
            "T": T,
            "duty_i": duty_i,
        },
        foot={
            "h_swing_i": h_swing_i,
            "delta_p_td_i": delta_p_td_i,
            "delta_t_td_i": delta_t_td_i,
        },
        base={
            "h_body_ref": h_body_ref,
            "roll_ref": roll_ref,
            "pitch_ref": pitch_ref,
        },
        ctrl={
            "k_des": k_des,
            "mu_exp": mu_nom,
        },
    )