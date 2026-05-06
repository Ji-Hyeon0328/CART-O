"""Step 2/3: Go1 theta-decoder gait demo with torque PD realization.

Run:
python source/isaaclab_carto/isaaclab_carto/scripts/run_go1_theta_gait_demo.py --headless --mode conservative
python source/isaaclab_carto/isaaclab_carto/scripts/run_go1_theta_gait_demo.py --headless --mode aggressive
python source/isaaclab_carto/isaaclab_carto/scripts/run_go1_theta_gait_demo.py --headless --mode high_clearance
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
    parser = argparse.ArgumentParser(description="Go1 theta-decoder gait demo.")
    parser.add_argument("--mode", type=str, default="conservative", choices=["conservative", "aggressive", "high_clearance"])
    parser.add_argument("--torque_scale", type=float, default=0.45)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    simulation_app = launcher.app

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_carto.envs.go1_lowlevel_env_cfg import Go1LowLevelEnvCfg
    from isaaclab_carto.controllers.lowlevel_controller import Go1LowLevelController

    env_cfg = Go1LowLevelEnvCfg()
    env_cfg.scene.num_envs = 4
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()
    robot = env.scene["robot"]
    controller = Go1LowLevelController(
        num_envs=env.num_envs,
        device=env.device,
        dt=env_cfg.sim.dt * env_cfg.decimation,
        horizon=20,
        torque_scale=args.torque_scale,
    )
    print("-" * 80)
    print(f"[INFO] Go1 theta gait demo | mode={args.mode}")
    print("[INFO] Current stage: decoder/mapper + simple torque PD realization")
    print("-" * 80)
    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            tau, info = controller.step(robot, mode=args.mode)
            env.step(tau)
            if step % 100 == 0:
                duty = info.theta["gait"]["duty_i"][0].detach().cpu().numpy()
                phase = info.ref["phase"][0, :, 0].detach().cpu().numpy()
                contact = info.ref["S"][0, :, 0].detach().cpu().numpy()
                print(f"[step {step:05d}] z={robot.data.root_pos_w[:,2].mean().item():.3f}, max|tau|={tau.abs().max().item():.2f}, duty={duty.round(2)}, phase={phase.round(2)}, S={contact.astype(int)}")
            step += 1
    env.close()

if __name__ == "__main__":
    main()
