# isaaclab_carto/scripts/run_spot_effort_flat_debug.py
#
# Safer B-0.5 effort debug.
#
# Why this script exists:
#   The first B0/B1/B2 logs showed that the robot can spawn on different
#   terrain cells/heights and that implicit PD can dominate applied torque.
#   This script forces a flat-only terrain setup as much as possible and
#   prints applied_torque AFTER env.step() so tau_cmd/applied_torque are easier
#   to interpret.
#
# Recommended runs:
#
# B0.5 zero effort, implicit PD kept:
#   python source/isaaclab_carto/isaaclab_carto/scripts/run_spot_effort_flat_debug.py \
#     --num_envs 1 --torque_mode zero --steps 300 --print_every 25
#
# B1.5 small joint PD effort, implicit PD kept:
#   python source/isaaclab_carto/isaaclab_carto/scripts/run_spot_effort_flat_debug.py \
#     --num_envs 1 --torque_mode joint_pd --kp 5 --kd 0.5 --max_tau 5 --steps 300 --print_every 25
#
# Probe:
#   python source/isaaclab_carto/isaaclab_carto/scripts/run_spot_effort_flat_debug.py \
#     --num_envs 1 --torque_mode zero --probe_joint fl_hy --probe_amp 0.5 --probe_freq 0.5 \
#     --steps 300 --print_every 25

import os
import sys
import argparse
from typing import Any

import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="CARTO Spot safer flat effort debug")

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--print_every", type=int, default=25)

parser.add_argument(
    "--torque_mode",
    type=str,
    default="zero",
    choices=["zero", "joint_pd"],
)
parser.add_argument("--kp", type=float, default=5.0)
parser.add_argument("--kd", type=float, default=0.5)
parser.add_argument("--max_tau", type=float, default=5.0)

parser.add_argument("--probe_joint", type=str, default="fl_hy")
parser.add_argument("--probe_amp", type=float, default=0.0)
parser.add_argument("--probe_freq", type=float, default=0.5)

parser.add_argument(
    "--disable_implicit_pd",
    action="store_true",
    help="Set actuator stiffness/damping to zero. This may fall quickly.",
)

parser.add_argument("--spawn_z", type=float, default=0.60)
parser.add_argument("--control_dt", type=float, default=0.02)

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
from isaaclab_carto.lowlevel.spot_state import make_x_hat, print_robot_debug_info  # noqa: E402
from isaaclab_carto.lowlevel.effort_control import (  # noqa: E402
    make_zero_torque,
    make_joint_pd_torque,
    add_sine_probe_torque,
    summarize_torque,
)


def patch_flat_safe_env(env_cfg: Any) -> None:
    """
    Best-effort patch to make the terrain flat and spawn height predictable.
    This avoids debugging torque on random rough/stair/slope cells.
    """
    # Use one flat terrain cell.
    try:
        env_cfg.scene.terrain.max_init_terrain_level = 0
    except Exception as exc:
        print(f"[WARN] Could not set max_init_terrain_level: {exc}")

    try:
        tg = env_cfg.scene.terrain.terrain_generator
        tg.num_rows = 1
        tg.num_cols = 1
        tg.size = (8.0, 8.0)
        tg.sub_terrains = {
            "flat": MeshPlaneTerrainCfg(proportion=1.0),
        }
        print("[INFO] Patched terrain generator to flat-only.")
    except Exception as exc:
        print(f"[WARN] Could not patch terrain generator to flat-only: {exc}")

    # Avoid spawning high above terrain.
    try:
        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, args.spawn_z)
        print(f"[INFO] Patched robot spawn z to {args.spawn_z}.")
    except Exception as exc:
        print(f"[WARN] Could not patch robot spawn z: {exc}")

    # Optional pure effort.
    if args.disable_implicit_pd:
        try:
            for name, actuator_cfg in env_cfg.scene.robot.actuators.items():
                print(f"[INFO] Disabling implicit PD for actuator group: {name}")
                actuator_cfg.stiffness = 0.0
                actuator_cfg.damping = 0.0
        except Exception as exc:
            print(f"[WARN] Failed to disable implicit PD: {exc}")


def make_torque_action(robot, step: int) -> torch.Tensor:
    if args.torque_mode == "zero":
        tau = make_zero_torque(robot)
    elif args.torque_mode == "joint_pd":
        tau = make_joint_pd_torque(
            robot=robot,
            kp=args.kp,
            kd=args.kd,
            max_tau=args.max_tau,
        )
    else:
        raise ValueError(args.torque_mode)

    tau = add_sine_probe_torque(
        tau=tau,
        robot=robot,
        step=step,
        dt=args.control_dt,
        joint_name=args.probe_joint,
        amp=args.probe_amp,
        freq=args.probe_freq,
    )

    return torch.clamp(tau, -args.max_tau, args.max_tau)


def print_debug(step: int, robot, x_hat_before: torch.Tensor, tau_cmd: torch.Tensor) -> None:
    env_id = 0

    x_hat_after = make_x_hat(robot, velocity_frame="world")

    print("\n" + "-" * 90)
    print(f"[EFFORT FLAT DEBUG] step={step}")
    print("-" * 90)

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

    print("\n[joint env0 after env.step]")
    print("joint_pos    :", robot.data.joint_pos[env_id].detach().cpu().numpy())
    print("joint_vel    :", robot.data.joint_vel[env_id].detach().cpu().numpy())

    print("\n[tau_cmd env0]")
    print(tau_cmd[env_id].detach().cpu().numpy())
    print("tau stats    :", summarize_torque(tau_cmd))

    if hasattr(robot.data, "applied_torque"):
        applied = robot.data.applied_torque
        print("\n[applied_torque env0 after env.step]")
        print(applied[env_id].detach().cpu().numpy())
        print("applied stats:", summarize_torque(applied))

    print("-" * 90 + "\n")


def main() -> None:
    env_cfg = CartoEffortEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    patch_flat_safe_env(env_cfg)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    _obs_dict, _ = env.reset()

    robot = env.scene["robot"]
    print_robot_debug_info(robot)

    print("\n" + "=" * 90)
    print("[INFO] Starting safer flat Spot effort debug")
    print("=" * 90)
    print(f"num_envs            : {env.num_envs}")
    print(f"steps               : {args.steps}")
    print(f"print_every         : {args.print_every}")
    print(f"torque_mode         : {args.torque_mode}")
    print(f"kp, kd, max_tau     : {args.kp}, {args.kd}, {args.max_tau}")
    print(f"disable_implicit_pd : {args.disable_implicit_pd}")
    print(f"spawn_z             : {args.spawn_z}")
    print(f"probe               : {args.probe_joint}, amp={args.probe_amp}, freq={args.probe_freq}")
    print("=" * 90 + "\n")

    for step in range(args.steps):
        if not simulation_app.is_running():
            break

        x_hat_before = make_x_hat(robot, velocity_frame="world")
        tau_cmd = make_torque_action(robot, step)

        _obs_dict, _rewards, terminated, truncated, _extras = env.step(tau_cmd)

        if step % args.print_every == 0:
            print_debug(step=step, robot=robot, x_hat_before=x_hat_before, tau_cmd=tau_cmd)

        if torch.any(terminated) or torch.any(truncated):
            print(f"[WARN] terminated/truncated at step={step}")
            print("terminated:", terminated.detach().cpu().numpy())
            print("truncated :", truncated.detach().cpu().numpy())

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
