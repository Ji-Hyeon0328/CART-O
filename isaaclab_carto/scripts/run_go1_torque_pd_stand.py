"""Step 1: Go1 torque PD stand test.

Run:
python source/isaaclab_carto/isaaclab_carto/scripts/run_go1_torque_pd_stand.py --headless
"""
from __future__ import annotations
import argparse, os, sys
import torch
from isaaclab.app import AppLauncher

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if SOURCE_DIR not in sys.path:
    sys.path.append(SOURCE_DIR)

def main():
    parser = argparse.ArgumentParser(description="Go1 torque PD stand test.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    simulation_app = launcher.app

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_carto.envs.go1_lowlevel_env_cfg import Go1LowLevelEnvCfg
    from isaaclab_carto.controllers.robot_presets import get_go1_preset

    env_cfg = Go1LowLevelEnvCfg()
    env_cfg.scene.num_envs = 4
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()
    robot = env.scene["robot"]
    device = env.device
    preset = get_go1_preset(device=device)
    #q_nom = preset.nominal_joint_pos.unsqueeze(0).repeat(env.num_envs, 1)
    #tau_limit = preset.torque_limit.unsqueeze(0).repeat(env.num_envs, 1)

    # Use Isaac asset's own default joint position first.
    # This avoids joint-order/sign mismatch from our hand-written nominal pose.
    q_nom = robot.data.default_joint_pos.clone()
    print("[INFO] joint names:")
    print(robot.data.joint_names)

    print("[INFO] default_joint_pos:")
    print(robot.data.default_joint_pos[0])

    q_nom[:, 0:4] = 0.0       # hips
    q_nom[:, 4:8] = 0.75      # thighs
    q_nom[:, 8:12] = -1.40    # calves

    foot_body_ids = torch.tensor([13, 14, 15, 16], device=device, dtype=torch.long)
    mass = 12.0
    g = 9.81
    fz_per_foot = mass * g / 4.0

    tau_limit = torch.tensor(
        [23.7, 23.7, 23.7, 23.7,        # hips
        23.7, 23.7, 23.7, 23.7,        # thighs
        35.55, 35.55, 35.55, 35.55],   # calves
        device=device,
    ).unsqueeze(0).repeat(env.num_envs, 1)

    ## Isaac Go1 joint order:
    ## [FL_hip, FR_hip, RL_hip, RR_hip,
    ##  FL_thigh, FR_thigh, RL_thigh, RR_thigh,
    ##  FL_calf, FR_calf, RL_calf, RR_calf]
    kp = torch.tensor(
        [5.0, 5.0, 5.0, 5.0,      # hips
        10.0, 10.0, 10.0, 10.0,      # thighs
        10.0, 10.0, 10.0, 10.0],     # calves
        device=device,
    ).unsqueeze(0).repeat(env.num_envs, 1)

    kd = torch.tensor(
        [0.3, 0.3, 0.3, 0.3,          # hips
        0.8, 0.8, 0.8, 0.8,          # thighs
        0.8, 0.8, 0.8, 0.8],         # calves
        device=device,
    ).unsqueeze(0).repeat(env.num_envs, 1)
    torque_scale=1.00
    print("-" * 80)
    print("[INFO] Go1 torque PD stand test")
    try:
        print("[INFO] Joint names:", robot.data.joint_names)
    except Exception:
        print("[WARN] Could not print joint names from robot.data.")
    print("-" * 80)
    print("[INFO] root_physx_view gravity/jacobian methods:")
    print([m for m in dir(robot.root_physx_view) if "gravity" in m.lower() or "jacobian" in m.lower() or "force" in m.lower()])
    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            with torch.inference_mode():
                q = robot.data.joint_pos
                dq = robot.data.joint_vel

                # Jacobian: [num_envs, num_bodies, 6, 18]
                J_all = robot.root_physx_view.get_jacobians()
                if not isinstance(J_all, torch.Tensor):
                    J_all = torch.tensor(J_all, device=device, dtype=torch.float32)
                else:
                    J_all = J_all.to(device)

                # foot Jacobian: [num_envs, 4, 6, 18]
                J_foot = J_all[:, foot_body_ids, :, :]

                # joint part only: [num_envs, 4, 6, 12]
                J_foot_joint = J_foot[:, :, :, 6:]

                # World vertical support force for each foot.
                # First try assuming translational Jacobian rows are 0:3.
                f_support = torch.zeros((env.num_envs, 4, 3), device=device)
                f_support[:, :, 2] = fz_per_foot

                # Candidate A: linear rows = 0:3
                J_lin = J_foot_joint[:, :, 3:6, :]  # [N, 4, 3, 12]

                tau_support = torch.einsum("efcj,efc->ej", J_lin, f_support)

                tau = torch.clamp(tau_support, -tau_limit, tau_limit)

                obs, rew, terminated, truncated, info = env.step(tau)
            
            if step % 100 == 0:
                base_z = robot.data.root_pos_w[:, 2].mean().item()
                print(f"[step {step:05d}] base z={base_z:.3f}, max|tau|={tau.abs().max().item():.2f}")
                print("[tau_support env0]", tau_support[0].detach().cpu())
            step += 1
    env.close()

if __name__ == "__main__":
    main()
