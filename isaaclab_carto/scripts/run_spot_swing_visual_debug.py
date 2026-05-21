# isaaclab_carto/scripts/run_spot_swing_visual_debug.py
#
# B7-a: visual swing gait check.
#
# This script checks:
#   Ref.S / Ref.phase
#   → small swing-leg joint position offsets
#   → visible leg lifting motion
#
# It intentionally uses the original JointPositionAction environment.
# Zero action should correspond to nominal standing posture.
#
# This is NOT physical walking yet.
# It is a visual bridge from gait references to leg motion.

import os
import sys
import argparse
from typing import Any

import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="CARTO/TRACER Spot swing visual debug")

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--print_every", type=int, default=25)

parser.add_argument("--z_mode", type=str, default="conservative", choices=["conservative", "aggressive"])
parser.add_argument("--vx", type=float, default=0.05)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)

parser.add_argument("--hy_lift_delta", type=float, default=-0.06)
parser.add_argument("--kn_lift_delta", type=float, default=0.14)
parser.add_argument("--hy_sweep_delta", type=float, default=0.02)
parser.add_argument("--max_action_abs", type=float, default=0.22)
parser.add_argument("--lift_sign", type=float, default=1.0, choices=[1.0, -1.0])
parser.add_argument("--knee_sign", type=float, default=1.0, choices=[1.0, -1.0])

parser.add_argument("--control_dt", type=float, default=0.02)
parser.add_argument("--ref_k", type=int, default=0)
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

from isaaclab_carto.envs.carto_env_cfg import CartoEnvCfg  # noqa: E402
from isaaclab_carto.lowlevel.spot_state import make_x_hat, build_spot_ref_params, print_robot_debug_info  # noqa: E402
from isaaclab_carto.lowlevel.theta_decoder import theta_decoder  # noqa: E402
from isaaclab_carto.lowlevel.theta_ref_mapper import theta_ref_mapper  # noqa: E402
from isaaclab_carto.lowlevel.spot_swing_visual import make_swing_visual_joint_action  # noqa: E402


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


def make_command(num_envs: int, device, dtype) -> torch.Tensor:
    u_cmd = torch.zeros((num_envs, 3), device=device, dtype=dtype)
    u_cmd[:, 0] = args.vx
    u_cmd[:, 1] = args.vy
    u_cmd[:, 2] = args.wz
    return u_cmd


def make_fake_highlevel(num_envs: int, device, dtype):
    z_value = 0.0 if args.z_mode == "conservative" else 1.0
    z_t = torch.full((num_envs,), z_value, device=device, dtype=dtype)

    # Keep this mild for visual test.
    a_hl = torch.tensor([0.10, -0.25, 0.20, -0.10], device=device, dtype=dtype)
    return z_t, a_hl.unsqueeze(0).repeat(num_envs, 1)


def advance_theta_phase_with_clock(theta, step: int, dt: float):
    T = theta.gait["T"]
    phase_i = theta.gait["phase_i"]
    phase_advance = (step * dt) / torch.clamp(T, min=1e-6)
    theta.gait["phase_i"] = torch.remainder(phase_i + phase_advance.unsqueeze(1), 1.0)
    return theta


def print_debug(step, robot, x_hat_before, u_cmd, theta, ref, action, swing_info):
    env_id = 0
    x_hat_after = make_x_hat(robot, velocity_frame="world")

    print("\n" + "=" * 120)
    print(f"[SWING VISUAL DEBUG] step={step}")
    print("=" * 120)

    print("[x_hat before env.step env0]")
    print("pos xyz      :", x_hat_before[env_id, 0:3].detach().cpu().numpy())
    print("rpy          :", x_hat_before[env_id, 3:6].detach().cpu().numpy())
    print("lin vel      :", x_hat_before[env_id, 6:9].detach().cpu().numpy())
    print("ang vel      :", x_hat_before[env_id, 9:12].detach().cpu().numpy())

    print("\n[x_hat after env.step env0]")
    print("pos xyz      :", x_hat_after[env_id, 0:3].detach().cpu().numpy())
    print("rpy          :", x_hat_after[env_id, 3:6].detach().cpu().numpy())
    print("lin vel      :", x_hat_after[env_id, 6:9].detach().cpu().numpy())
    print("ang vel      :", x_hat_after[env_id, 9:12].detach().cpu().numpy())

    print("\n[u_cmd env0]")
    print(u_cmd[env_id].detach().cpu().numpy())

    print("\n[Theta env0]")
    print("z_mode       :", args.z_mode)
    print("T            :", float(theta.gait["T"][env_id].detach().cpu()))
    print("phase_i      :", theta.gait["phase_i"][env_id].detach().cpu().numpy())
    print("duty_i       :", theta.gait["duty_i"][env_id].detach().cpu().numpy())

    k = args.ref_k
    print(f"\n[Ref env0, k={k}]")
    print("S[:,k]       :", ref["S"][env_id, :, k].detach().cpu().numpy())
    print("phase[:,k]   :", ref["phase"][env_id, :, k].detach().cpu().numpy())

    print("\n[swing visual info]")
    keys = [
        "swing_mask_env0",
        "swing_progress_env0",
        "lift_profile_env0",
        "hy_offset_env0",
        "kn_offset_env0",
        "hy_lift_delta",
        "kn_lift_delta",
        "hy_sweep_delta",
        "lift_sign",
        "knee_sign",
        "max_action_abs",
    ]
    for key in keys:
        if key in swing_info:
            print(f"{key}: {swing_info[key]}")

    print("\n[joint position action env0]")
    print(action[env_id].detach().cpu().numpy())

    if hasattr(robot.data, "joint_pos"):
        print("\n[joint_pos env0]")
        print(robot.data.joint_pos[env_id].detach().cpu().numpy())

    print("=" * 120 + "\n")


def main() -> None:
    env_cfg = CartoEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    patch_flat_safe_env(env_cfg)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    _obs_dict, _ = env.reset()

    robot = env.scene["robot"]
    print_robot_debug_info(robot)

    dtype = robot.data.joint_pos.dtype
    device = robot.data.joint_pos.device

    params = build_spot_ref_params(
        device=device,
        dtype=dtype,
        dt=args.control_dt,
        horizon=20,
    )

    print("\n" + "=" * 120)
    print("[INFO] Starting Spot B7-a swing visual debug")
    print("=" * 120)
    print(f"z_mode={args.z_mode}, u_cmd=[{args.vx}, {args.vy}, {args.wz}]")
    print(f"hy_lift_delta={args.hy_lift_delta}, kn_lift_delta={args.kn_lift_delta}, hy_sweep_delta={args.hy_sweep_delta}")
    print("NOTE: This is visual swing-reference check, not physical walking.")
    print("=" * 120 + "\n")

    for step in range(args.steps):
        if not simulation_app.is_running():
            break

        x_hat = make_x_hat(robot, velocity_frame="world")
        u_cmd = make_command(env.num_envs, device=device, dtype=dtype)
        z_t, a_hl = make_fake_highlevel(env.num_envs, device=device, dtype=dtype)

        theta = theta_decoder(
            z_t=z_t,
            a_HL=a_hl,
            x_hat=x_hat,
            u_cmd=u_cmd,
            robot_name="spot",
        )

        if not args.no_gait_clock:
            theta = advance_theta_phase_with_clock(theta, step=step, dt=args.control_dt)

        ref = theta_ref_mapper(
            theta=theta,
            x_hat=x_hat,
            u_cmd=u_cmd,
            params=params,
        )

        action, swing_info = make_swing_visual_joint_action(
            ref=ref,
            theta=theta,
            k=args.ref_k,
            hy_lift_delta=args.hy_lift_delta,
            kn_lift_delta=args.kn_lift_delta,
            hy_sweep_delta=args.hy_sweep_delta,
            max_action_abs=args.max_action_abs,
            lift_sign=args.lift_sign,
            knee_sign=args.knee_sign,
        )

        _obs_dict, _rewards, terminated, truncated, _extras = env.step(action)

        if step % args.print_every == 0:
            print_debug(
                step=step,
                robot=robot,
                x_hat_before=x_hat,
                u_cmd=u_cmd,
                theta=theta,
                ref=ref,
                action=action,
                swing_info=swing_info,
            )

        if torch.any(terminated) or torch.any(truncated):
            print(f"[WARN] terminated/truncated at step={step}")
            print("terminated:", terminated.detach().cpu().numpy())
            print("truncated :", truncated.detach().cpu().numpy())

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
