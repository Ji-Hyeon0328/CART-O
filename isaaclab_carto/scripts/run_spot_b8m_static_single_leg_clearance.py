# isaaclab_carto/scripts/run_spot_b8m_static_single_leg_clearance.py
#
# B8-m: static single-leg clearance probe with position action.
#
# Why:
#   B8-l showed position action can make large motion, but the cyclic crawl
#   destabilizes the base. Before trying full gait again, isolate one leg:
#
#       stand -> lift one selected leg slowly -> hold -> lower
#
# Goal:
#   Verify true visible clearance while the other three legs remain stance.
#
# This is not walking. It is the bridge between:
#   B8-l: cyclic crawl falls
#   next: CoM/support-shifted crawl

import os
import sys
import argparse
from typing import Any

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CARTO/TRACER B8-m static single-leg clearance")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle_steps", type=int, default=160)
parser.add_argument("--probe_steps", type=int, default=260)
parser.add_argument("--print_every", type=int, default=20)

parser.add_argument("--test_leg", type=str, default="RF", choices=["LF", "RF", "LH", "RH"])
parser.add_argument("--hy_lift_delta", type=float, default=-0.04)
parser.add_argument("--kn_lift_delta", type=float, default=-0.12)
parser.add_argument("--hx_delta", type=float, default=0.0)
parser.add_argument("--action_scale", type=float, default=1.0)
parser.add_argument("--max_action_abs", type=float, default=0.25)

# phase lengths in sim/control steps
parser.add_argument("--ramp_up_steps", type=int, default=80)
parser.add_argument("--hold_steps", type=int, default=80)
parser.add_argument("--ramp_down_steps", type=int, default=80)

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
from isaaclab_carto.envs.carto_env_cfg import CartoEnvCfg  # noqa: E402
from isaaclab_carto.lowlevel.spot_state import make_x_hat, print_robot_debug_info  # noqa: E402

LEG_NAMES = ["LF", "RF", "LH", "RH"]
FOOT_NAMES = ["fl_foot", "fr_foot", "hl_foot", "hr_foot"]
LEG_TO_ID = {"LF": 0, "RF": 1, "LH": 2, "RH": 3}

# [fl_hx, fr_hx, hl_hx, hr_hx, fl_hy, fr_hy, hl_hy, hr_hy, fl_kn, fr_kn, hl_kn, hr_kn]
HX_IDX = torch.tensor([0, 1, 2, 3], dtype=torch.long)
HY_IDX = torch.tensor([4, 5, 6, 7], dtype=torch.long)
KN_IDX = torch.tensor([8, 9, 10, 11], dtype=torch.long)


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


def get_foot_indices(robot):
    name_to_idx = {name: i for i, name in enumerate(robot.body_names)}
    return [name_to_idx[name] for name in FOOT_NAMES]


def smooth_profile(step: int) -> float:
    ru = args.ramp_up_steps
    hold = args.hold_steps
    rd = args.ramp_down_steps

    if step < ru:
        s = step / max(ru, 1)
        return 0.5 - 0.5 * torch.cos(torch.tensor(torch.pi * s)).item()
    if step < ru + hold:
        return 1.0
    if step < ru + hold + rd:
        s = (step - ru - hold) / max(rd, 1)
        return 0.5 + 0.5 * torch.cos(torch.tensor(torch.pi * s)).item()
    return 0.0


def make_action(num_envs, device, dtype, step):
    leg = LEG_TO_ID[args.test_leg]
    prof = smooth_profile(step)

    action = torch.zeros((num_envs, 12), device=device, dtype=dtype)
    hx_idx = int(HX_IDX[leg])
    hy_idx = int(HY_IDX[leg])
    kn_idx = int(KN_IDX[leg])

    action[:, hx_idx] = args.hx_delta * prof
    action[:, hy_idx] = args.hy_lift_delta * prof
    action[:, kn_idx] = args.kn_lift_delta * prof

    action = args.action_scale * action
    action = torch.clamp(action, -args.max_action_abs, args.max_action_abs)

    info = {
        "profile": prof,
        "test_leg": args.test_leg,
        "hx_cmd": args.hx_delta * prof,
        "hy_cmd": args.hy_lift_delta * prof,
        "kn_cmd": args.kn_lift_delta * prof,
    }
    return action, info


def print_debug(step, robot, action, info, base_x0, foot0):
    leg = LEG_TO_ID[args.test_leg]
    foot_indices = get_foot_indices(robot)

    x = make_x_hat(robot, velocity_frame="world")
    foot = robot.data.body_pos_w[:, foot_indices, :]
    foot_delta = foot - foot0
    base_delta = x[:, 0:6] - base_x0

    print("\n" + "=" * 120)
    print(f"[B8-m STATIC SINGLE-LEG CLEARANCE] step={step}")
    print("=" * 120)
    print("test_leg:", args.test_leg, "profile:", info["profile"])
    print("cmd hx/hy/kn:", info["hx_cmd"], info["hy_cmd"], info["kn_cmd"])
    print("action env0:", action[0].detach().cpu().numpy())
    print("base_delta xyz+rpy:", base_delta[0].detach().cpu().numpy())
    print("test foot_delta xyz:", foot_delta[0, leg].detach().cpu().numpy())
    print("all foot_delta env0:", foot_delta[0].detach().cpu().numpy())
    print("clearance_z_for_test_leg:", float(foot_delta[0, leg, 2].detach().cpu()))
    print("foot_delta max_abs:", float(foot_delta.abs().max().detach().cpu()))
    print("=" * 120 + "\n")


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

    print("\n" + "=" * 120)
    print("[INFO] Starting B8-m static single-leg clearance")
    print("test_leg:", args.test_leg)
    print("hy_lift_delta:", args.hy_lift_delta, "kn_lift_delta:", args.kn_lift_delta, "hx_delta:", args.hx_delta)
    print("=" * 120)

    zero = torch.zeros((args.num_envs, 12), device=device, dtype=dtype)
    for step in range(args.settle_steps):
        if not simulation_app.is_running():
            break
        env.step(zero)
        if step % max(args.print_every, 1) == 0:
            x = make_x_hat(robot, velocity_frame="world")
            print(f"[SETTLE] step={step} pos={x[0,0:3].detach().cpu().numpy()} rpy={x[0,3:6].detach().cpu().numpy()}")

    foot_indices = get_foot_indices(robot)
    foot0 = robot.data.body_pos_w[:, foot_indices, :].detach().clone()
    base_x0 = make_x_hat(robot, velocity_frame="world")[:, 0:6].detach().clone()

    print("\n[INFO] baseline saved")
    print("base_x0 env0:", base_x0[0].detach().cpu().numpy())
    print("foot0 env0:", foot0[0].detach().cpu().numpy())

    for step in range(args.probe_steps):
        if not simulation_app.is_running():
            break
        action, info = make_action(args.num_envs, device, dtype, step)
        env.step(action)

        if step % max(args.print_every, 1) == 0:
            print_debug(step, robot, action, info, base_x0, foot0)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
