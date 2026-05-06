"""Go1 J^T f support stand diagnostic.

Place at:
source/isaaclab_carto/isaaclab_carto/scripts/run_go1_jtf_support_stand.py
"""

from __future__ import annotations

import argparse
import os
import sys
import torch
from isaaclab.app import AppLauncher

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if SOURCE_DIR not in sys.path:
    sys.path.append(SOURCE_DIR)


def as_torch(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, device=device, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Go1 J^T f support stand diagnostic.")
    parser.add_argument("--support_scale", type=float, default=3.0)
    parser.add_argument("--lin_rows", type=int, default=0, choices=[0, 3])
    parser.add_argument("--fz_sign", type=float, default=1.0, choices=[1.0, -1.0])
    parser.add_argument("--pd_scale", type=float, default=0.35)
    parser.add_argument("--num_envs", type=int, default=4)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    launcher = AppLauncher(args)
    simulation_app = launcher.app

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_carto.envs.go1_lowlevel_env_cfg import Go1LowLevelEnvCfg

    env_cfg = Go1LowLevelEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    robot = env.scene["robot"]
    device = env.device
    num_envs = env.num_envs

    print("-" * 80)
    print("[INFO] Go1 J^T f support stand diagnostic")
    print("[INFO] joint names:", robot.data.joint_names)
    print("[INFO] body names:", robot.data.body_names)
    print("[INFO] default_joint_pos:", robot.data.default_joint_pos[0])
    print(f"[INFO] support_scale={args.support_scale}, lin_rows={args.lin_rows}:{args.lin_rows+3}, fz_sign={args.fz_sign}, pd_scale={args.pd_scale}")
    print("-" * 80)

    body_names = list(robot.data.body_names)
    foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    foot_body_ids = torch.tensor([body_names.index(n) for n in foot_names], device=device, dtype=torch.long)
    print("[INFO] foot body ids:", foot_body_ids.detach().cpu().tolist())

    tau_limit = torch.tensor(
        [23.7, 23.7, 23.7, 23.7,
         23.7, 23.7, 23.7, 23.7,
         35.55, 35.55, 35.55, 35.55],
        device=device,
    ).unsqueeze(0).repeat(num_envs, 1)

    q_nom = robot.data.default_joint_pos.clone()

    kp = args.pd_scale * torch.tensor(
        [8.0, 8.0, 8.0, 8.0,
         14.0, 14.0, 14.0, 14.0,
         14.0, 14.0, 14.0, 14.0],
        device=device,
    ).unsqueeze(0).repeat(num_envs, 1)
    kd = args.pd_scale * torch.tensor(
        [0.5, 0.5, 0.5, 0.5,
         1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 1.0, 1.0],
        device=device,
    ).unsqueeze(0).repeat(num_envs, 1)

    mass = 12.0
    g = 9.81
    h_ref=0.32

    base_z=robot.data.root_pos_w[:,2]
    base_vz = robot.data.root_lin_vel_w[:,2]

    kh=400.0
    dh=50.0

    fz_total=mass*g+kh*(h_ref-base_z)-dh*base_vz
    fz_total = torch.clamp(fz_total,0.5*mass*g,3.0*mass*g)

    fz_per_foot = args.fz_sign*args.support_scale*fz_total/4.0

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            q = robot.data.joint_pos
            dq = robot.data.joint_vel

            J_all = as_torch(robot.root_physx_view.get_jacobians(), device)
            J_foot = J_all[:, foot_body_ids, :, :]
            J_foot_joint = J_foot[:, :, :, 6:]
            J_lin = J_foot_joint[:, :, args.lin_rows:args.lin_rows + 3, :]

            f_support = torch.zeros((num_envs, 4, 3), device=device)
            f_support[:, :, 2] = fz_per_foot[:,None]

            tau_support = torch.einsum("efcj,efc->ej", J_lin, f_support)
            tau_pd = kp * (q_nom - q) - kd * dq

            ramp = min(step / 100.0, 1.0)
            ramp_support = min(max((step-50)/100.0,0.0),1.0)
            ramp_pd = min(step/20.0,1.0)
            #tau = ramp * (tau_support + tau_pd)
            tau = ramp_support*tau_support + ramp_pd*tau_pd

            tau = torch.clamp(tau, -tau_limit, tau_limit)

            env.step(tau)

            if step % 100 == 0:
                base_z = robot.data.root_pos_w[:, 2].mean().item()
                max_tau = tau.abs().max().item()
                print(f"[step {step:05d}] base z={base_z:.3f}, max|tau|={max_tau:.2f}")
                print("[tau_support env0]", tau_support[0].detach().cpu())
                print("[tau_pd env0]", tau_pd[0].detach().cpu())

            step += 1

    env.close()


if __name__ == "__main__":
    main()
