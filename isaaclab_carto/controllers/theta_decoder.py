"""Batched PyTorch port of thetaDecoder.m.

Rule-based first version:
- z_t = 0: conservative gait
- z_t = 1: aggressive/trot-like gait
"""
from __future__ import annotations
import torch

def theta_decoder(z_t: torch.Tensor, a_HL: torch.Tensor, x_hat: torch.Tensor, u_cmd: torch.Tensor) -> dict:
    device, dtype = x_hat.device, x_hat.dtype
    B = x_hat.shape[0]
    z_bool = (z_t.reshape(-1).to(device=device) > 0.5)

    a_swing, a_body, a_duty, a_imp = a_HL[:, 0], a_HL[:, 1], a_HL[:, 2], a_HL[:, 3]
    roll, pitch = x_hat[:, 3], x_hat[:, 4]
    vx_cmd, vy_cmd = u_cmd[:, 0], u_cmd[:, 1]
    v_cmd_mag = torch.sqrt(vx_cmd * vx_cmd + vy_cmd * vy_cmd + 1.0e-9)

    phase_cons = torch.tensor([0.00, 0.25, 0.50, 0.75], device=device, dtype=dtype)
    phase_agg = torch.tensor([0.00, 0.50, 0.50, 0.00], device=device, dtype=dtype)
    phase_nom = torch.where(z_bool[:, None], phase_agg[None, :].expand(B, 4), phase_cons[None, :].expand(B, 4))

    def choose(cons, agg):
        return torch.where(z_bool, torch.full((B,), agg, device=device, dtype=dtype), torch.full((B,), cons, device=device, dtype=dtype))

    T_nom = choose(0.75, 0.45)
    duty_nom = choose(0.78, 0.55)
    h_body_nom = choose(0.36, 0.43)
    h_swing_nom = choose(0.07, 0.11)
    k_imp_nom = choose(0.80, 1.15)
    mu_nom = choose(0.55, 0.60)

    duty_state_corr = 0.05 * torch.abs(roll) + 0.05 * torch.abs(pitch)
    body_state_corr = -0.03 * torch.abs(roll) - 0.03 * torch.abs(pitch)

    T = torch.clamp(T_nom * (1.0 - 0.25 * torch.tanh(v_cmd_mag)), 0.30, 1.00)
    duty = torch.clamp(duty_nom + 0.10 * a_duty + duty_state_corr, 0.35, 0.90)
    h_body = torch.clamp(h_body_nom + 0.05 * a_body + body_state_corr, 0.25, 0.55)
    h_swing = torch.clamp(h_swing_nom + 0.04 * a_swing + 0.02 * torch.tanh(v_cmd_mag), 0.03, 0.18)
    k_des = torch.clamp(k_imp_nom + 0.25 * a_imp, 0.40, 1.60)
    mu_exp = torch.clamp(mu_nom, 0.20, 1.00)

    return {
        "gait": {"phase_i": phase_nom, "T": T, "duty_i": duty[:, None].expand(B, 4)},
        "base": {
            "h_body_ref": h_body,
            "roll_ref": torch.zeros((B,), device=device, dtype=dtype),
            "pitch_ref": torch.zeros((B,), device=device, dtype=dtype),
        },
        "foot": {
            "h_swing_i": h_swing[:, None].expand(B, 4),
            "delta_p_td_i": torch.zeros((B, 3, 4), device=device, dtype=dtype),
            "delta_t_td_i": torch.zeros((B, 4), device=device, dtype=dtype),
        },
        "ctrl": {"k_des": k_des, "mu_exp": mu_exp},
    }
