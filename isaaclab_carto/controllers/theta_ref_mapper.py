"""Batched PyTorch port of thetaRefMapper.m."""
from __future__ import annotations
import torch

def _rotz2d(yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)], dim=-2)

def theta_ref_mapper(theta: dict, x_hat: torch.Tensor, u_cmd: torch.Tensor, params: dict) -> dict:
    device, dtype = x_hat.device, x_hat.dtype
    B = x_hat.shape[0]
    N = int(params.get("N", 20))
    dt = float(params.get("dt", 0.02))
    hip_offset_body = params["hip_offset_body"].to(device=device, dtype=dtype)
    p_foot_now = params["p_foot_now"].to(device=device, dtype=dtype)

    p0 = x_hat[:, 0:3]
    yaw0 = x_hat[:, 5]
    vx_cmd, vy_cmd, wz_cmd = u_cmd[:, 0], u_cmd[:, 1], u_cmd[:, 2]
    phase_i = theta["gait"]["phase_i"]
    T = theta["gait"]["T"].clamp_min(1e-6)
    duty_i = theta["gait"]["duty_i"]
    h_body_ref = theta["base"]["h_body_ref"]
    roll_ref = theta["base"]["roll_ref"]
    pitch_ref = theta["base"]["pitch_ref"]
    h_swing_i = theta["foot"]["h_swing_i"]
    delta_p_td_i = theta["foot"]["delta_p_td_i"]

    S = torch.zeros((B, 4, N), device=device, dtype=dtype)
    phase = torch.zeros((B, 4, N), device=device, dtype=dtype)
    Xb_ref = torch.zeros((B, 12, N), device=device, dtype=dtype)
    Xf_ref = torch.zeros((B, 3, 4, N), device=device, dtype=dtype)
    Xfd_ref = torch.zeros((B, 3, 4, N), device=device, dtype=dtype)

    p_ref = p0.clone()
    yaw_ref = yaw0.clone()
    for k in range(N):
        Xb_ref[:, 0, k] = p_ref[:, 0]
        Xb_ref[:, 1, k] = p_ref[:, 1]
        Xb_ref[:, 2, k] = h_body_ref
        Xb_ref[:, 3, k] = roll_ref
        Xb_ref[:, 4, k] = pitch_ref
        Xb_ref[:, 5, k] = yaw_ref
        Xb_ref[:, 6, k] = vx_cmd
        Xb_ref[:, 7, k] = vy_cmd
        Xb_ref[:, 11, k] = wz_cmd
        Rz = _rotz2d(yaw_ref)
        v_world_xy = torch.bmm(Rz, torch.stack([vx_cmd, vy_cmd], dim=-1).unsqueeze(-1)).squeeze(-1)
        p_ref[:, 0] += dt * v_world_xy[:, 0]
        p_ref[:, 1] += dt * v_world_xy[:, 1]
        p_ref[:, 2] = h_body_ref
        yaw_ref += dt * wz_cmd

    for k in range(N):
        time_ahead = k * dt
        p_base_k = Xb_ref[:, 0:3, k]
        yaw_base_k = Xb_ref[:, 5, k]
        Rz_k = _rotz2d(yaw_base_k)
        phi = torch.remainder(phase_i + time_ahead / T[:, None], 1.0)
        phase[:, :, k] = phi
        S[:, :, k] = (phi < duty_i).to(dtype)
        for leg in range(4):
            hip_xy_body = hip_offset_body[0:2, leg].view(1, 2, 1).expand(B, 2, 1)
            hip_xy = p_base_k[:, 0:2] + torch.bmm(Rz_k, hip_xy_body).squeeze(-1)
            hip_z = p_base_k[:, 2] + hip_offset_body[2, leg]
            p_hip_world = torch.cat([hip_xy, hip_z[:, None]], dim=-1)
            stance_mask = S[:, leg, k] > 0.5
            p_stance = p_foot_now[:, leg].view(1, 3).expand(B, 3)
            swing_denom = torch.clamp(1.0 - duty_i[:, leg], min=1e-6)
            swing_phase = torch.clamp((phi[:, leg] - duty_i[:, leg]) / swing_denom, 0.0, 1.0)
            step_preview = torch.stack([0.18 * vx_cmd, 0.10 * vy_cmd, torch.zeros_like(vx_cmd)], dim=-1)
            p_td = p_hip_world + step_preview + delta_p_td_i[:, :, leg]
            p_sw = (1.0 - swing_phase[:, None]) * p_stance + swing_phase[:, None] * p_td
            p_sw[:, 2] += h_swing_i[:, leg] * torch.sin(torch.pi * swing_phase)
            Xf_ref[:, :, leg, k] = torch.where(stance_mask[:, None], p_stance, p_sw)
    if N > 1:
        Xfd_ref[:, :, :, 0:N-1] = (Xf_ref[:, :, :, 1:N] - Xf_ref[:, :, :, 0:N-1]) / dt
        Xfd_ref[:, :, :, N-1] = Xfd_ref[:, :, :, N-2]
    return {"S": S, "phase": phase, "Xb_ref": Xb_ref, "Xf_ref": Xf_ref, "Xfd_ref": Xfd_ref}
