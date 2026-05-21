# isaaclab_carto/scripts/run_spot_pg_force_mpc_v2_debug.py
#
# B5-e: beta objective cleanup debug.
#
# Runs PG forceMPC v2 and prints clear beta/objective mapping.

import os
import sys
import argparse
from typing import Any

import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="CARTO Spot PG forceMPC v2 beta objective debug")

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--print_every", type=int, default=25)

parser.add_argument("--z_mode", type=str, default="conservative", choices=["conservative", "aggressive"])
parser.add_argument("--vx", type=float, default=0.10)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--beta_preset", type=str, default="balanced", choices=["height", "velocity", "energy", "balanced"])

parser.add_argument("--h_ref", type=float, default=0.67)
parser.add_argument("--mass", type=float, default=32.5)
parser.add_argument("--gravity", type=float, default=9.81)
parser.add_argument("--residual_ratio", type=float, default=0.05)
parser.add_argument("--target_scale", type=float, default=1.0)

parser.add_argument("--mu", type=float, default=0.6)
parser.add_argument("--fz_min", type=float, default=0.0)
parser.add_argument("--fz_max", type=float, default=8.0)
parser.add_argument("--fxy_abs_max", type=float, default=3.0)
parser.add_argument("--max_delta_f", type=float, default=1.0)
parser.add_argument("--smoothing_alpha", type=float, default=0.70)

parser.add_argument("--pg_iters", type=int, default=40)
parser.add_argument("--pg_step_size", type=float, default=0.06)
parser.add_argument("--pg_init", type=str, default="prev", choices=["zero", "prev"])

parser.add_argument("--max_tau", type=float, default=3.0)
parser.add_argument("--tau_scale", type=float, default=1.0)
parser.add_argument("--force_sign", type=float, default=1.0, choices=[1.0, -1.0])
parser.add_argument("--linear_rows", type=str, default="0_3", choices=["0_3", "3_6"])

parser.add_argument("--ref_k", type=int, default=0)
parser.add_argument("--min_stance_legs", type=int, default=2)
parser.add_argument("--control_dt", type=float, default=0.02)
parser.add_argument("--spawn_z", type=float, default=0.60)

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
from isaaclab_carto.lowlevel.theta_decoder import theta_decoder  # noqa: E402
from isaaclab_carto.lowlevel.theta_ref_mapper import theta_ref_mapper  # noqa: E402
from isaaclab_carto.lowlevel.effort_control import summarize_torque  # noqa: E402
from isaaclab_carto.lowlevel.force_mpc_objectives import make_beta_preset  # noqa: E402
from isaaclab_carto.lowlevel.projected_gradient_force_mpc_v2 import (  # noqa: E402
    PGForceMPCV2State,
    make_pg_force_mpc_v2_torque,
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


def make_fixed_command(num_envs: int, device, dtype) -> torch.Tensor:
    u_cmd = torch.zeros((num_envs, 3), device=device, dtype=dtype)
    u_cmd[:, 0] = args.vx
    u_cmd[:, 1] = args.vy
    u_cmd[:, 2] = args.wz
    return u_cmd


def make_fake_highlevel(num_envs: int, device, dtype):
    z_value = 0.0 if args.z_mode == "conservative" else 1.0
    z_t = torch.full((num_envs,), z_value, device=device, dtype=dtype)
    a_hl = torch.tensor([0.20, -0.20, 0.30, -0.10], device=device, dtype=dtype)
    return z_t, a_hl.unsqueeze(0).repeat(num_envs, 1)


def advance_theta_phase_with_clock(theta, step: int, dt: float):
    T = theta.gait["T"]
    phase_i = theta.gait["phase_i"]
    phase_advance = (step * dt) / torch.clamp(T, min=1e-6)
    theta.gait["phase_i"] = torch.remainder(phase_i + phase_advance.unsqueeze(1), 1.0)
    return theta


def print_debug(step, robot, u_cmd, beta_t, theta, ref, tau_cmd, info):
    env_id = 0
    x_hat_after = make_x_hat(robot, velocity_frame="world")

    print("\n" + "-" * 132)
    print(f"[PG FORCE MPC V2 / BETA OBJECTIVE DEBUG] step={step}")
    print("-" * 132)

    print("[x_hat after env.step env0]")
    print("pos xyz      :", x_hat_after[env_id, 0:3].detach().cpu().numpy())
    print("rpy          :", x_hat_after[env_id, 3:6].detach().cpu().numpy())
    print("lin vel      :", x_hat_after[env_id, 6:9].detach().cpu().numpy())
    print("ang vel      :", x_hat_after[env_id, 9:12].detach().cpu().numpy())

    print("\n[u_cmd env0]")
    print(u_cmd[env_id].detach().cpu().numpy())

    print("\n[beta env0]")
    print(beta_t[env_id].detach().cpu().numpy(), f"preset={args.beta_preset}")
    print("semantics:", info.get("beta_semantics"))

    print("\n[Theta / Ref env0]")
    print("z_mode       :", args.z_mode)
    print("T            :", float(theta.gait["T"][env_id].detach().cpu()))
    print("phase_i      :", theta.gait["phase_i"][env_id].detach().cpu().numpy())
    print("duty_i       :", theta.gait["duty_i"][env_id].detach().cpu().numpy())
    k = args.ref_k
    print("S[:,k]       :", ref["S"][env_id, :, k].detach().cpu().numpy())

    print("\n[objective terms]")
    for key in [
        "kp_h_env0", "kd_h_env0", "kp_vxy_env0",
        "w_acc_z_env0", "w_acc_xy_env0", "w_force_env0",
        "rate_scale_env0", "smoothing_alpha_env0",
        "residual_ratio", "target_scale",
    ]:
        print(f"{key}: {info.get(key)}")

    print("\n[PG-QP forceMPC v2 info]")
    keys = [
        "stance_mask_env0", "num_stance_env0",
        "a_des_env0", "v_err_xy_env0", "h_err_mean",
        "f_pg_env0", "f_projected_env0",
        "pg_iters", "pg_step_size", "pg_init",
        "pg_obj0_env0", "pg_obj_last_env0", "pg_obj_delta_env0",
        "pg_grad_norm_last_env0", "pg_projection_total",
        "projection_after_rate", "projection_after_smooth",
        "force_rate_clamp_count", "max_force_delta_before",
        "tau_mean_abs", "tau_max_abs",
    ]
    for key in keys:
        if key in info:
            print(f"{key}: {info[key]}")

    print("\n[tau_pg_force_mpc_v2 env0, action order]")
    print(tau_cmd[env_id].detach().cpu().numpy())
    print("tau stats:", summarize_torque(tau_cmd))

    if hasattr(robot.data, "applied_torque"):
        applied = robot.data.applied_torque
        print("\n[applied_torque env0 after env.step]")
        print(applied[env_id].detach().cpu().numpy())
        print("applied stats:", summarize_torque(applied))

    print("-" * 132 + "\n")


def main():
    env_cfg = CartoEffortEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    patch_flat_safe_env(env_cfg)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    _obs_dict, _ = env.reset()

    robot = env.scene["robot"]
    print_robot_debug_info(robot)

    dtype = robot.data.joint_pos.dtype
    device = robot.data.joint_pos.device
    params = build_spot_ref_params(device=device, dtype=dtype, dt=args.control_dt, horizon=20)
    planner_state = PGForceMPCV2State()

    print("\n" + "=" * 132)
    print("[INFO] Starting Spot B5-e PG forceMPC v2 beta objective debug")
    print("=" * 132)
    print(f"z_mode={args.z_mode}, beta_preset={args.beta_preset}, u_cmd=[{args.vx},{args.vy},{args.wz}]")
    print(f"residual_ratio={args.residual_ratio}, target_scale={args.target_scale}")
    print("=" * 132 + "\n")

    for step in range(args.steps):
        if not simulation_app.is_running():
            break

        x_hat = make_x_hat(robot, velocity_frame="world")
        u_cmd = make_fixed_command(env.num_envs, device=device, dtype=dtype)
        z_t, a_hl = make_fake_highlevel(env.num_envs, device=device, dtype=dtype)
        beta_t = make_beta_preset(env.num_envs, device=device, dtype=dtype, preset=args.beta_preset)

        theta = theta_decoder(z_t=z_t, a_HL=a_hl, x_hat=x_hat, u_cmd=u_cmd, robot_name="spot")
        theta = advance_theta_phase_with_clock(theta, step=step, dt=args.control_dt)
        ref = theta_ref_mapper(theta=theta, x_hat=x_hat, u_cmd=u_cmd, params=params)

        tau_cmd, info = make_pg_force_mpc_v2_torque(
            robot=robot,
            ref=ref,
            x_hat=x_hat,
            u_cmd=u_cmd,
            beta_t=beta_t,
            planner_state=planner_state,
            h_ref=args.h_ref,
            mass=args.mass,
            gravity=args.gravity,
            residual_ratio=args.residual_ratio,
            target_scale=args.target_scale,
            mu=args.mu,
            fz_min=args.fz_min,
            fz_max=args.fz_max,
            fxy_abs_max=args.fxy_abs_max,
            max_delta_f=args.max_delta_f,
            smoothing_alpha=args.smoothing_alpha,
            pg_iters=args.pg_iters,
            pg_step_size=args.pg_step_size,
            pg_init=args.pg_init,
            tau_scale=args.tau_scale,
            max_tau=args.max_tau,
            linear_rows=args.linear_rows,
            force_sign=args.force_sign,
            use_k=args.ref_k,
            min_stance_legs=args.min_stance_legs,
        )

        _obs_dict, _rewards, terminated, truncated, _extras = env.step(tau_cmd)

        if step % args.print_every == 0:
            print_debug(step, robot, u_cmd, beta_t, theta, ref, tau_cmd, info)

        if torch.any(terminated) or torch.any(truncated):
            print(f"[WARN] terminated/truncated at step={step}")
            print("terminated:", terminated.detach().cpu().numpy())
            print("truncated :", truncated.detach().cpu().numpy())

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
