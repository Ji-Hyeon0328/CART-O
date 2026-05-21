# isaaclab_carto/lowlevel/theta_ref_mapper.py
#
# Python/Torch port of thetaRefMapper.m.
#
# Input:
#   Theta
#   x_hat = [x y z roll pitch yaw vx vy vz wx wy wz]
#   u_cmd = [vx_cmd vy_cmd wz_cmd]
#   params:
#       dt
#       N
#       hip_offset_body
#       p_foot_now
#
# Output:
#   Ref dict:
#       Ref["S"]        [Nenv, 4, H]
#       Ref["phase"]    [Nenv, 4, H]
#       Ref["Xb_ref"]   [Nenv, 12, H]
#       Ref["Xf_ref"]   [Nenv, 3, 4, H]
#       Ref["Xfd_ref"]  [Nenv, 3, 4, H]

from __future__ import annotations

from typing import Dict, Any

import torch

from .theta_decoder import Theta


def rotz_2d(yaw: torch.Tensor) -> torch.Tensor:
    """
    Batched 2D yaw rotation.

    Args:
        yaw: [N]

    Returns:
        R: [N, 2, 2]
    """
    c = torch.cos(yaw)
    s = torch.sin(yaw)

    R = torch.zeros((yaw.shape[0], 2, 2), device=yaw.device, dtype=yaw.dtype)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    return R


def _to_batched_3x4(x: torch.Tensor, num_envs: int, device, dtype, name: str) -> torch.Tensor:
    x = x.to(device=device, dtype=dtype)
    if x.ndim == 2:
        if x.shape != (3, 4):
            raise ValueError(f"{name} must have shape [3, 4] or [N, 3, 4], got {tuple(x.shape)}")
        x = x.unsqueeze(0).repeat(num_envs, 1, 1)
    elif x.ndim == 3:
        if x.shape[0] != num_envs or x.shape[1:] != (3, 4):
            raise ValueError(f"{name} must have shape [{num_envs}, 3, 4], got {tuple(x.shape)}")
    else:
        raise ValueError(f"{name} must have shape [3, 4] or [N, 3, 4], got {tuple(x.shape)}")
    return x


def theta_ref_mapper(
    theta: Theta,
    x_hat: torch.Tensor,
    u_cmd: torch.Tensor,
    params: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """
    Deterministic Theta-to-Reference mapper.

    Coordinate convention:
        - x_hat position/orientation: world frame.
        - u_cmd: body-frame command [vx, vy, wz].
        - mapper rotates [vx, vy] by yaw to update world-frame base reference.

    Args:
        theta:
            Output of theta_decoder().
        x_hat:
            [Nenv, 12] = [x y z roll pitch yaw vx vy vz wx wy wz].
        u_cmd:
            [Nenv, 3] = [vx_cmd vy_cmd wz_cmd].
        params:
            dict with:
                "dt": float
                "N": int
                "hip_offset_body": torch.Tensor [3, 4] or [Nenv, 3, 4]
                "p_foot_now": torch.Tensor [3, 4] or [Nenv, 3, 4]

    Returns:
        Ref dict.
    """
    if x_hat.ndim != 2 or x_hat.shape[-1] != 12:
        raise ValueError(f"x_hat must have shape [Nenv, 12], got {tuple(x_hat.shape)}")
    if u_cmd.ndim != 2 or u_cmd.shape[-1] < 3:
        raise ValueError(f"u_cmd must have shape [Nenv, >=3], got {tuple(u_cmd.shape)}")

    device = x_hat.device
    dtype = x_hat.dtype
    num_envs = x_hat.shape[0]

    dt = float(params.get("dt", 0.02))
    H = int(params.get("N", 20))

    hip_offset_body = _to_batched_3x4(
        params["hip_offset_body"], num_envs, device, dtype, "hip_offset_body"
    )
    p_foot_now = _to_batched_3x4(
        params["p_foot_now"], num_envs, device, dtype, "p_foot_now"
    )

    u_cmd = u_cmd[:, :3].to(device=device, dtype=dtype)

    p0 = x_hat[:, 0:3]
    eta0 = x_hat[:, 3:6]
    yaw0 = eta0[:, 2]

    vx_cmd = u_cmd[:, 0]
    vy_cmd = u_cmd[:, 1]
    wz_cmd = u_cmd[:, 2]

    phase_i = theta.gait["phase_i"].to(device=device, dtype=dtype)
    T = theta.gait["T"].to(device=device, dtype=dtype)
    duty_i = theta.gait["duty_i"].to(device=device, dtype=dtype)

    h_body_ref = theta.base["h_body_ref"].to(device=device, dtype=dtype)
    roll_ref = theta.base["roll_ref"].to(device=device, dtype=dtype)
    pitch_ref = theta.base["pitch_ref"].to(device=device, dtype=dtype)

    h_swing_i = theta.foot["h_swing_i"].to(device=device, dtype=dtype)
    delta_p_td_i = theta.foot["delta_p_td_i"].to(device=device, dtype=dtype)

    # Outputs.
    S = torch.zeros((num_envs, 4, H), device=device, dtype=dtype)
    phase = torch.zeros((num_envs, 4, H), device=device, dtype=dtype)

    Xb_ref = torch.zeros((num_envs, 12, H), device=device, dtype=dtype)
    Xf_ref = torch.zeros((num_envs, 3, 4, H), device=device, dtype=dtype)
    Xfd_ref = torch.zeros((num_envs, 3, 4, H), device=device, dtype=dtype)

    # -------------------------------------------------------------------------
    # Base reference sequence.
    # -------------------------------------------------------------------------
    p_ref = p0.clone()
    yaw_ref = yaw0.clone()

    for k in range(H):
        Xb_ref[:, 0, k] = p_ref[:, 0]
        Xb_ref[:, 1, k] = p_ref[:, 1]
        Xb_ref[:, 2, k] = h_body_ref

        Xb_ref[:, 3, k] = roll_ref
        Xb_ref[:, 4, k] = pitch_ref
        Xb_ref[:, 5, k] = yaw_ref

        Xb_ref[:, 6, k] = vx_cmd
        Xb_ref[:, 7, k] = vy_cmd
        Xb_ref[:, 8, k] = 0.0

        Xb_ref[:, 9, k] = 0.0
        Xb_ref[:, 10, k] = 0.0
        Xb_ref[:, 11, k] = wz_cmd

        Rz = rotz_2d(yaw_ref)
        v_body_xy = torch.stack([vx_cmd, vy_cmd], dim=-1).unsqueeze(-1)
        v_world_xy = torch.bmm(Rz, v_body_xy).squeeze(-1)

        p_ref[:, 0] = p_ref[:, 0] + dt * v_world_xy[:, 0]
        p_ref[:, 1] = p_ref[:, 1] + dt * v_world_xy[:, 1]
        p_ref[:, 2] = h_body_ref

        yaw_ref = yaw_ref + dt * wz_cmd

    # -------------------------------------------------------------------------
    # Foot reference sequence and contact schedule.
    # -------------------------------------------------------------------------
    for k in range(H):
        time_ahead = k * dt

        p_base_k = Xb_ref[:, 0:3, k]
        yaw_base_k = Xb_ref[:, 5, k]
        Rz_k = rotz_2d(yaw_base_k)

        for leg in range(4):
            # Phase in [0, 1).
            phi = torch.remainder(phase_i[:, leg] + time_ahead / torch.clamp(T, min=1e-6), 1.0)
            phase[:, leg, k] = phi

            is_stance = (phi < duty_i[:, leg]).to(dtype)
            S[:, leg, k] = is_stance

            # Nominal foot position from base + rotated hip offset.
            hip_xy_body = hip_offset_body[:, 0:2, leg].unsqueeze(-1)
            hip_xy_world = torch.bmm(Rz_k, hip_xy_body).squeeze(-1)

            p_nom = torch.zeros((num_envs, 3), device=device, dtype=dtype)
            p_nom[:, 0:2] = p_base_k[:, 0:2] + hip_xy_world

            # In this prototype, ground height is assumed 0 in local terrain frame.
            # Since Isaac Lab world env origins can be far away, x/y can be large.
            # z=0 here is not base-relative; it is a terrain-plane reference placeholder.
            p_nom[:, 2] = 0.0

            p_td = p_nom + delta_p_td_i[:, :, leg]

            # Swing progress.
            swing_denom = torch.clamp(1.0 - duty_i[:, leg], min=1e-4)
            s_sw = torch.clamp((phi - duty_i[:, leg]) / swing_denom, 0.0, 1.0)

            z_lift = h_swing_i[:, leg] * torch.sin(torch.pi * s_sw)

            foot_ref = p_td.clone()
            foot_ref[:, 2] = z_lift * (1.0 - is_stance)

            # During stance, optionally keep current foot now.
            # For now, keep nominal p_td to preserve deterministic behavior.
            _ = p_foot_now

            Xf_ref[:, :, leg, k] = foot_ref

    # Foot velocity by finite difference.
    if H >= 2:
        Xfd_ref[:, :, :, :-1] = (Xf_ref[:, :, :, 1:] - Xf_ref[:, :, :, :-1]) / dt
        Xfd_ref[:, :, :, -1] = Xfd_ref[:, :, :, -2]

    return {
        "S": S,
        "phase": phase,
        "Xb_ref": Xb_ref,
        "Xf_ref": Xf_ref,
        "Xfd_ref": Xfd_ref,
    }