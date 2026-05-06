"""Robot preset utilities for the Go1 low-level Isaac Lab prototype.

Python counterpart of the MATLAB applyRobotPresetV3.m idea, starting with Go1.
Leg order: [LF, RF, LH, RH].
Joint order assumption:
    [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
     RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]
Verify Isaac joint order in the terminal printout before trusting control results.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class Go1Preset:
    name: str
    mass: float
    gravity: float
    ibody: torch.Tensor
    nominal_body_height: float
    hip_offset_body: torch.Tensor
    nominal_foot_pos: torch.Tensor
    nominal_joint_pos: torch.Tensor
    torque_limit: torch.Tensor

def get_go1_preset(device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> Go1Preset:
    ibody = torch.tensor([
        [0.0168128557, -0.0002296769, -0.0002945293],
        [-0.0002296769, 0.0630095650, -0.0000418731],
        [-0.0002945293, -0.0000418731, 0.0716547275],
    ], device=device, dtype=dtype)
    hip_offset_body = torch.tensor([
        [0.11215, 0.11215, -0.11215, -0.11215],
        [0.04675, -0.04675, 0.04675, -0.04675],
        [0.0, 0.0, 0.0, 0.0],
    ], device=device, dtype=dtype)
    nominal_foot_pos = torch.tensor([
        [0.18, 0.18, -0.18, -0.18],
        [0.11, -0.11, 0.11, -0.11],
        [0.0, 0.0, 0.0, 0.0],
    ], device=device, dtype=dtype)
    nominal_joint_pos = torch.tensor([
        0.0, 0.80, -1.50, 0.0, 0.80, -1.50,
        0.0, 0.80, -1.50, 0.0, 0.80, -1.50,
    ], device=device, dtype=dtype)
    # nominal_joint_pos = [
    #     FL_hip, FR_hip, RL_hip, RR_hip,
    #     FL_thigh, FR_thigh, RL_thigh, RR_thigh,
    #     FL_calf, FR_calf, RL_calf, RR_calf
    # ]
    torque_limit = torch.tensor([23.7, 23.7, 35.55] * 4, device=device, dtype=dtype)
    return Go1Preset(
        name="go1", mass=12.0, gravity=9.81, ibody=ibody,
        nominal_body_height=0.30,
        hip_offset_body=hip_offset_body,
        nominal_foot_pos=nominal_foot_pos,
        nominal_joint_pos=nominal_joint_pos,
        torque_limit=torque_limit,
    )
