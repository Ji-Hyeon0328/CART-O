# isaaclab_carto/scripts/run_spot_b8v_position_stance_calibration.py
#
# B8-v: position-action stance calibration.
#
# Why:
#   B8-u failed before the swing phase. With zero position action, the robot
#   started falling backward during warmup. So the position-action environment
#   does not currently provide a stable upright nominal stance.
#
# Goal:
#   Find a static position-action stance offset that can hold the robot upright
#   before trying swing again.
#
# This script does NOT attempt walking.
# It only applies constant all-leg or front/hind stance offsets from the first
# simulation step and prints base height/attitude.
#
# Once a stable stance offset is found, we will reuse it as q_stance/action_stance
# for the next support-shifted swing test.

import os
import sys
import argparse
from typing import Any

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CARTO/TRACER B8-v position stance calibration")

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=360)
parser.add_argument("--print_every", type=int, default=20)

# Env / actuator
parser.add_argument("--spawn_z", type=float, default=0.60)
parser.add_argument("--pd_scale", type=float, default=0.80)

# Stance offsets in native action coordinates.
# Native action order:
# [fl_hx, fr_hx, hl_hx, hr_hx,
#  fl_hy, fr_hy, hl_hy, hr_hy,
#  fl_kn, fr_kn, hl_kn, hr_kn]
parser.add_argument("--hx_all", type=float, default=0.0)
parser.add_argument("--hy_all", type=float, default=0.0)
parser.add_argument("--kn_all", type=float, default=0.0)

parser.add_argument("--hy_front", type=float, default=None)
parser.add_argument("--hy_hind", type=float, default=None)
parser.add_argument("--kn_front", type=float, default=None)
parser.add_argument("--kn_hind", type=float, default=None)

parser.add_argument("--action_scale", type=float, default=1.0)
parser.add_argument("--max_action_abs", type=float, default=0.35)

# Optional small base-height style compensation through knee ramp.
parser.add_argument("--ramp_steps", type=int, default=80)

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
from isaaclab_carto.lowlevel.spot_state import make_x_hat, print_robot_debug_info  # noqa: E402


FOOT_NAMES = ["fl_foot", "fr_foot", "hl_foot", "hr_foot"]
HX_IDX = torch.tensor([0, 1, 2, 3], dtype=torch.long)
HY_IDX = torch.tensor([4, 5, 6, 7], dtype=torch.long)
KN_IDX = torch.tensor([8, 9, 10, 11], dtype=torch.long)


def smooth01(s: float) -> float:
    s = max(0.0, min(1.0, s))
    return float(0.5 - 0.5 * torch.cos(torch.tensor(torch.pi * s)).item())


def patch_flat_safe_env(env_cfg: Any) -> None:
    try:
        env_cfg.scene.terrain.max_init_terrain_level = 0
    except Exception:
        pass
    try:
        tg = env_cfg.scene.terrain.terrain_generator
        tg.num_rows = 1
        tg.num_cols = 1
        tg.size = (8.0, 8.0)
        tg.sub_terrains = {"flat": MeshPlaneTerrainCfg(proportion=1.0)}
        print("[INFO] Patched terrain generator to flat-only.")
    except Exception as exc:
        print(f"[WARN] Could not patch terrain generator: {exc}")

    try:
        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, args.spawn_z)
        print(f"[INFO] Patched robot spawn z to {args.spawn_z}.")
    except Exception as exc:
        print(f"[WARN] Could not patch robot spawn z: {exc}")

    try:
        for actuator in env_cfg.scene.robot.actuators.values():
            actuator.stiffness = float(actuator.stiffness) * args.pd_scale
            actuator.damping = float(actuator.damping) * args.pd_scale
        print(f"[INFO] Scaled implicit actuator stiffness/damping by {args.pd_scale}.")
    except Exception as exc:
        print(f"[WARN] Could not scale implicit actuator gains: {exc}")


def get_foot_indices(robot):
    name_to_idx = {name: i for i, name in enumerate(robot.body_names)}
    return [name_to_idx[name] for name in FOOT_NAMES]


def make_action(num_envs, device, dtype, step):
    action = torch.zeros((num_envs, 12), device=device, dtype=dtype)

    ramp = smooth01(step / max(args.ramp_steps, 1))

    hx = args.hx_all
    hy_f = args.hy_all if args.hy_front is None else args.hy_front
    hy_h = args.hy_all if args.hy_hind is None else args.hy_hind
    kn_f = args.kn_all if args.kn_front is None else args.kn_front
    kn_h = args.kn_all if args.kn_hind is None else args.kn_hind

    # HX all legs
    action[:, 0:4] = hx * ramp

    # HY: front LF/RF, hind LH/RH
    action[:, 4] = hy_f * ramp
    action[:, 5] = hy_f * ramp
    action[:, 6] = hy_h * ramp
    action[:, 7] = hy_h * ramp

    # KN: front LF/RF, hind LH/RH
    action[:, 8] = kn_f * ramp
    action[:, 9] = kn_f * ramp
    action[:, 10] = kn_h * ramp
    action[:, 11] = kn_h * ramp

    action = args.action_scale * action
    action = torch.clamp(action, -args.max_action_abs, args.max_action_abs)
    return action, ramp


def print_debug(step, ramp, action, x_hat, foot_pos, foot0, robot):
    foot_delta = foot_pos - foot0
    print("\n" + "=" * 132)
    print(f"[B8-v POSITION STANCE CALIBRATION] step={step}")
    print("=" * 132)
    print("ramp:", ramp)
    print("action env0:", action[0].detach().cpu().numpy())
    print("\n[base]")
    print("base pos xyz:", x_hat[0, 0:3].detach().cpu().numpy())
    print("base rpy:", x_hat[0, 3:6].detach().cpu().numpy())
    print("base vel:", x_hat[0, 6:9].detach().cpu().numpy())
    print("\n[feet]")
    print("foot_delta env0:")
    print(foot_delta[0].detach().cpu().numpy())
    print("foot_delta z:", foot_delta[0, :, 2].detach().cpu().numpy())
    print("min/max foot_delta_z:", float(foot_delta[0, :, 2].min().detach().cpu()), float(foot_delta[0, :, 2].max().detach().cpu()))
    try:
        print("\n[torque]")
        print("applied_torque max_abs:", float(robot.data.applied_torque.abs().max().detach().cpu()))
    except Exception:
        pass
    print("=" * 132 + "\n")


def main():
    env_cfg = CartoEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    patch_flat_safe_env(env_cfg)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()
    robot = env.scene["robot"]
    print_robot_debug_info(robot)

    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype
    foot_indices = get_foot_indices(robot)
    foot0 = robot.data.body_pos_w[:, foot_indices, :].detach().clone()

    print("\n" + "=" * 132)
    print("[INFO] Starting B8-v position stance calibration")
    print("hx_all:", args.hx_all)
    print("hy_all:", args.hy_all, "hy_front:", args.hy_front, "hy_hind:", args.hy_hind)
    print("kn_all:", args.kn_all, "kn_front:", args.kn_front, "kn_hind:", args.kn_hind)
    print("=" * 132)

    for step in range(args.steps):
        if not simulation_app.is_running():
            break

        x_hat = make_x_hat(robot, velocity_frame="world")
        foot_pos = robot.data.body_pos_w[:, foot_indices, :]
        action, ramp = make_action(args.num_envs, device, dtype, step)

        env.step(action)

        if step % max(args.print_every, 1) == 0:
            print_debug(step, ramp, action, x_hat, foot_pos, foot0, robot)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
