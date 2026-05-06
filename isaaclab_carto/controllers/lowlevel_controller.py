"""First Go1 low-level controller for Isaac Lab.

Current stage:
- z_t/a_HL -> theta_decoder -> theta_ref_mapper
- simple phase-based joint target generation
- torque PD realization

MPC/WBC/residual are intentionally added later after this loop is stable.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from .robot_presets import get_go1_preset
from .theta_decoder import theta_decoder
from .theta_ref_mapper import theta_ref_mapper

@dataclass
class LowLevelInfo:
    theta: dict
    ref: dict
    q_des: torch.Tensor
    tau: torch.Tensor

class Go1LowLevelController:
    def __init__(self, num_envs: int, device: str | torch.device, dt: float = 0.02, horizon: int = 20, torque_scale: float = 1.0):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.dt = dt
        self.horizon = horizon
        self.torque_scale = torque_scale
        self.preset = get_go1_preset(self.device, self.dtype)
        self.params = {"dt": dt, "N": horizon, "hip_offset_body": self.preset.hip_offset_body, "p_foot_now": self.preset.nominal_foot_pos}
        self.q_nom = self.preset.nominal_joint_pos.unsqueeze(0).repeat(num_envs, 1)
        self.tau_limit = self.preset.torque_limit.unsqueeze(0).repeat(num_envs, 1)
        self.kp = torch.tensor([18.0, 22.0, 25.0] * 4, device=self.device).unsqueeze(0).repeat(num_envs, 1)
        self.kd = torch.tensor([0.8, 1.0, 1.2] * 4, device=self.device).unsqueeze(0).repeat(num_envs, 1)
        self.phase_state = torch.zeros((num_envs, 4), device=self.device)
        self.last_tau = torch.zeros((num_envs, 12), device=self.device)

    def build_x_hat(self, robot) -> torch.Tensor:
        data = robot.data
        pos = data.root_pos_w
        quat = data.root_quat_w
        lin_vel = data.root_lin_vel_b
        ang_vel = data.root_ang_vel_b
        rpy = quat_wxyz_to_rpy(quat)
        return torch.cat([pos, rpy, lin_vel, ang_vel], dim=-1)

    def make_scripted_high_level(self, mode: str = "conservative"):
        if mode == "aggressive":
            z = torch.ones((self.num_envs,), device=self.device)
            a = torch.tensor([0.60, 0.10, -0.10, 0.20], device=self.device).repeat(self.num_envs, 1)
            beta = torch.tensor([0.25, 0.45, 0.30], device=self.device).repeat(self.num_envs, 1)
            u = torch.tensor([0.35, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        elif mode == "high_clearance":
            z = torch.zeros((self.num_envs,), device=self.device)
            a = torch.tensor([0.90, 0.00, 0.25, 0.20], device=self.device).repeat(self.num_envs, 1)
            beta = torch.tensor([0.40, 0.25, 0.35], device=self.device).repeat(self.num_envs, 1)
            u = torch.tensor([0.20, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        else:
            z = torch.zeros((self.num_envs,), device=self.device)
            a = torch.tensor([0.25, -0.10, 0.35, -0.05], device=self.device).repeat(self.num_envs, 1)
            beta = torch.tensor([0.35, 0.35, 0.30], device=self.device).repeat(self.num_envs, 1)
            u = torch.tensor([0.15, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        return z, a, beta, u

    def step(self, robot, mode: str = "conservative"):
        q = robot.data.joint_pos
        dq = robot.data.joint_vel
        x_hat = self.build_x_hat(robot)
        z_t, a_HL, beta_t, u_cmd = self.make_scripted_high_level(mode)
        theta = theta_decoder(z_t, a_HL, x_hat, u_cmd)
        self.phase_state = torch.remainder(self.phase_state + self.dt / theta["gait"]["T"].clamp_min(1e-6).unsqueeze(-1), 1.0)
        theta["gait"]["phase_i"] = self.phase_state
        ref = theta_ref_mapper(theta, x_hat, u_cmd, self.params)
        q_des = self.phase_to_joint_targets(theta, ref)
        k_des = theta["ctrl"]["k_des"].unsqueeze(-1)
        kp = self.kp * k_des
        kd = self.kd * torch.sqrt(torch.clamp(k_des, min=0.1))
        tau = kp * (q_des - q) - kd * dq
        tau = self.torque_scale * torch.clamp(tau, -self.tau_limit, self.tau_limit)
        self.last_tau = tau
        return tau, LowLevelInfo(theta=theta, ref=ref, q_des=q_des, tau=tau)

    def phase_to_joint_targets(self, theta: dict, ref: dict) -> torch.Tensor:
        q_des = self.q_nom.clone()
        S0 = ref["S"][:, :, 0]
        phase0 = ref["phase"][:, :, 0]
        duty = theta["gait"]["duty_i"]
        swing = (S0 < 0.5).to(q_des.dtype)
        swing_denom = torch.clamp(1.0 - duty, min=1e-6)
        swing_phase = torch.clamp((phase0 - duty) / swing_denom, 0.0, 1.0)
        lift = torch.sin(torch.pi * swing_phase) * swing
        h_swing = theta["foot"]["h_swing_i"]
        amp = torch.clamp(h_swing / 0.10, 0.3, 1.8)
        for leg in range(4):
            hip_idx = leg
            thigh_idx = 4 + leg
            calf_idx = 8 + leg

            q_des[:, hip_idx] = self.q_nom[:, hip_idx]
            q_des[:, thigh_idx] = self.q_nom[:, thigh_idx] + 0.20 * amp[:, leg] * lift[:, leg]
            q_des[:, calf_idx] = self.q_nom[:, calf_idx] - 0.30 * amp[:, leg] * lift[:, leg]
            # j0 = 3 * leg
            # q_des[:, j0 + 0] = self.q_nom[:, j0 + 0]
            # q_des[:, j0 + 1] = self.q_nom[:, j0 + 1] + 0.25 * amp[:, leg] * lift[:, leg]
            # q_des[:, j0 + 2] = self.q_nom[:, j0 + 2] - 0.40 * amp[:, leg] * lift[:, leg]
        return q_des

def quat_wxyz_to_rpy(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack([roll, pitch, yaw], dim=-1)
