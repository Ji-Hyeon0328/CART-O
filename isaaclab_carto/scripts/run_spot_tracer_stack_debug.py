# B6: Integrated TRACER/CARTO low-level stack debug.
# This is not yet visible walking; it is integrated residual stack verification.

import os
import sys
import argparse
from typing import Any

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CARTO/TRACER integrated low-level stack debug")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--print_every", type=int, default=25)
parser.add_argument("--z_mode", type=str, default="conservative", choices=["conservative", "aggressive"])
parser.add_argument("--beta_preset", type=str, default="balanced", choices=["height", "velocity", "energy", "balanced"])
parser.add_argument("--vx", type=float, default=0.10)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--h_ref", type=float, default=0.67)
parser.add_argument("--mass", type=float, default=32.5)
parser.add_argument("--gravity", type=float, default=9.81)
parser.add_argument("--residual_ratio", type=float, default=0.02)
parser.add_argument("--mu", type=float, default=0.6)
parser.add_argument("--fz_min", type=float, default=0.0)
parser.add_argument("--fz_max", type=float, default=8.0)
parser.add_argument("--fxy_abs_max", type=float, default=3.0)
parser.add_argument("--max_delta_f", type=float, default=1.0)
parser.add_argument("--smoothing_alpha", type=float, default=0.75)
parser.add_argument("--tau_scale", type=float, default=1.0)
parser.add_argument("--max_tau", type=float, default=3.0)
parser.add_argument("--force_sign", type=float, default=1.0, choices=[1.0, -1.0])
parser.add_argument("--linear_rows", type=str, default="0_3", choices=["0_3", "3_6"])
parser.add_argument("--ref_k", type=int, default=0)
parser.add_argument("--min_stance_legs", type=int, default=2)
parser.add_argument("--control_dt", type=float, default=0.02)
parser.add_argument("--spawn_z", type=float, default=0.60)
parser.add_argument("--no_gait_clock", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if SOURCE_DIR not in sys.path:
    sys.path.append(SOURCE_DIR)

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg  # noqa: E402
from isaaclab_carto.envs.carto_effort_env_cfg import CartoEffortEnvCfg  # noqa: E402
from isaaclab_carto.lowlevel.spot_state import make_x_hat, build_spot_ref_params, print_robot_debug_info  # noqa: E402
from isaaclab_carto.lowlevel.effort_control import summarize_torque  # noqa: E402
from isaaclab_carto.lowlevel.tracer_lowlevel_stack import (  # noqa: E402
    TracerLowLevelConfig,
    TracerLowLevelStack,
    make_beta_from_preset,
    make_z_from_mode,
    make_default_a_hl,
    make_command,
)


def patch_flat_safe_env(env_cfg: Any) -> None:
    try:
        env_cfg.scene.terrain.max_init_terrain_level = 0
    except Exception as exc:
        print(f"[WARN] Could not set max_init_terrain_level: {exc}")
    try:
        tg = env_cfg.scene.terrain.terrain_generator
        tg.num_rows = 1
        tg.num_cols = 1
        tg.size = (8.0, 8.0)
        tg.sub_terrains = {"flat": MeshPlaneTerrainCfg(proportion=1.0)}
        print("[INFO] Patched terrain generator to flat-only.")
    except Exception as exc:
        print(f"[WARN] Could not patch terrain generator to flat-only: {exc}")
    try:
        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, args.spawn_z)
        print(f"[INFO] Patched robot spawn z to {args.spawn_z}.")
    except Exception as exc:
        print(f"[WARN] Could not patch robot spawn z: {exc}")


def print_debug(step, robot, tau_cmd, info):
    env_id = 0
    x_hat_after = make_x_hat(robot, velocity_frame="world")
    print("\n" + "=" * 120)
    print(f"[TRACER LOWLEVEL STACK DEBUG] step={step}")
    print("=" * 120)
    print("[x_hat after env.step env0]")
    print("pos xyz      :", x_hat_after[env_id, 0:3].detach().cpu().numpy())
    print("rpy          :", x_hat_after[env_id, 3:6].detach().cpu().numpy())
    print("lin vel      :", x_hat_after[env_id, 6:9].detach().cpu().numpy())
    print("ang vel      :", x_hat_after[env_id, 9:12].detach().cpu().numpy())

    print("\n[high-level interface env0]")
    for key in ["z_t_env0", "a_hl_env0", "beta_env0", "u_cmd_env0"]:
        print(f"{key}: {info.get(key)}")

    print("\n[theta/ref env0]")
    for key in ["theta_T_env0", "theta_phase_i_env0", "theta_duty_i_env0", "theta_h_body_ref_env0", "ref_S_env0", "ref_phase_env0"]:
        print(f"{key}: {info.get(key)}")

    print("\n[projected forceMPC env0]")
    keys = [
        "stance_mask_env0", "num_stance_env0", "a_des_env0", "f_raw_env0", "f_projected_env0",
        "projection_pass1", "projection_pass2", "projection_pass3", "force_rate_clamp_count",
        "max_force_delta_before", "w_acc_z_env0", "w_acc_xy_env0", "w_force_env0", "tau_mean_abs", "tau_max_abs",
    ]
    for key in keys:
        if key in info:
            print(f"{key}: {info[key]}")

    print("\n[tau_stack env0, action order]")
    print(tau_cmd[env_id].detach().cpu().numpy())
    print("tau stats:", summarize_torque(tau_cmd))

    if hasattr(robot.data, "applied_torque"):
        applied = robot.data.applied_torque
        print("\n[applied_torque env0 after env.step]")
        print(applied[env_id].detach().cpu().numpy())
        print("applied stats:", summarize_torque(applied))
    print("=" * 120 + "\n")


def main() -> None:
    env_cfg = CartoEffortEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    patch_flat_safe_env(env_cfg)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    _obs_dict, _ = env.reset()
    robot = env.scene["robot"]
    print_robot_debug_info(robot)

    dtype = robot.data.joint_pos.dtype
    device = robot.data.joint_pos.device
    ref_params = build_spot_ref_params(device=device, dtype=dtype, dt=args.control_dt, horizon=20)

    stack_cfg = TracerLowLevelConfig(
        h_ref=args.h_ref, mass=args.mass, gravity=args.gravity, residual_ratio=args.residual_ratio,
        mu=args.mu, fz_min=args.fz_min, fz_max=args.fz_max, fxy_abs_max=args.fxy_abs_max,
        max_delta_f=args.max_delta_f, smoothing_alpha=args.smoothing_alpha, tau_scale=args.tau_scale,
        max_tau=args.max_tau, linear_rows=args.linear_rows, force_sign=args.force_sign,
        ref_k=args.ref_k, min_stance_legs=args.min_stance_legs, control_dt=args.control_dt,
    )
    stack = TracerLowLevelStack(stack_cfg)

    print("\n" + "=" * 120)
    print("[INFO] Starting integrated TRACER/CARTO low-level stack debug")
    print("=" * 120)
    print(f"z_mode={args.z_mode}, beta_preset={args.beta_preset}, u_cmd=[{args.vx}, {args.vy}, {args.wz}]")
    print("NOTE: this is integrated residual stack debug, not yet full physical walking.")
    print("=" * 120 + "\n")

    for step in range(args.steps):
        if not simulation_app.is_running():
            break
        x_hat = make_x_hat(robot, velocity_frame="world")
        z_t = make_z_from_mode(env.num_envs, device=device, dtype=dtype, z_mode=args.z_mode)
        a_hl = make_default_a_hl(env.num_envs, device=device, dtype=dtype)
        beta_t = make_beta_from_preset(env.num_envs, device=device, dtype=dtype, preset=args.beta_preset)
        u_cmd = make_command(env.num_envs, device=device, dtype=dtype, vx=args.vx, vy=args.vy, wz=args.wz)

        tau_cmd, info = stack.compute(
            robot=robot, x_hat=x_hat, ref_params=ref_params, z_t=z_t, a_hl=a_hl,
            beta_t=beta_t, u_cmd=u_cmd, step=step, use_gait_clock=not args.no_gait_clock,
        )
        _obs_dict, _rewards, terminated, truncated, _extras = env.step(tau_cmd)

        if step % args.print_every == 0:
            print_debug(step, robot, tau_cmd, info)
        if torch.any(terminated) or torch.any(truncated):
            print(f"[WARN] terminated/truncated at step={step}")
            print("terminated:", terminated.detach().cpu().numpy())
            print("truncated :", truncated.detach().cpu().numpy())

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
