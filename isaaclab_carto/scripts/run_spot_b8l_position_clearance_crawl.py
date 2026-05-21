# isaaclab_carto/scripts/run_spot_b8l_position_clearance_crawl.py
#
# B8-l: position-action clearance crawl.
#
# Motivation:
#   B8-k showed an important mismatch:
#     - Jacobian predicts HY-/KN- should increase foot z.
#     - But effort residual probes barely move the joints/feet because the
#       implicit actuator remains dominant.
#
# So this script temporarily switches back to the JointPositionAction path
# and directly sends scheduled swing-leg joint target offsets.
#
# Goal:
#   Verify which sign/offset produces visible swing-foot clearance when the
#   action channel is a position target, not a small residual effort.
#
# This is not final WBC. It is a bridge diagnostic:
#
#   gait schedule -> position action -> visible clearance
#
# After we find a reliable clearance pattern, we can reintroduce balance/WBC
# around it.

import os
import sys
import argparse
from typing import Any

import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="CARTO/TRACER B8-l position-action clearance crawl")

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle_steps", type=int, default=120)
parser.add_argument("--probe_steps", type=int, default=360)
parser.add_argument("--print_every", type=int, default=20)

# Gait timing
parser.add_argument("--T", type=float, default=1.80)
parser.add_argument("--duty", type=float, default=0.78)
parser.add_argument("--phase_mode", type=str, default="crawl", choices=["crawl", "trot"])
parser.add_argument("--control_dt", type=float, default=0.02)

# Position action offsets
parser.add_argument("--hy_lift_delta", type=float, default=-0.08)
parser.add_argument("--kn_lift_delta", type=float, default=-0.22)
parser.add_argument("--hx_sweep_delta", type=float, default=0.0)
parser.add_argument("--hy_sweep_delta", type=float, default=0.0)

parser.add_argument("--action_scale", type=float, default=1.0)
parser.add_argument("--max_action_abs", type=float, default=0.35)

# Safety / env
parser.add_argument("--spawn_z", type=float, default=0.60)
parser.add_argument("--zero_settle", action="store_true")

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

# Native action order:
# [fl_hx, fr_hx, hl_hx, hr_hx,
#  fl_hy, fr_hy, hl_hy, hr_hy,
#  fl_kn, fr_kn, hl_kn, hr_kn]
HX_IDX = torch.tensor([0, 1, 2, 3], dtype=torch.long)
HY_IDX = torch.tensor([4, 5, 6, 7], dtype=torch.long)
KN_IDX = torch.tensor([8, 9, 10, 11], dtype=torch.long)


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


def get_foot_indices(robot):
    name_to_idx = {name: i for i, name in enumerate(robot.body_names)}
    return [name_to_idx[name] for name in FOOT_NAMES]


def gait_phase(step: int, device, dtype):
    if args.phase_mode == "crawl":
        base = torch.tensor([0.00, 0.25, 0.50, 0.75], device=device, dtype=dtype)
    else:
        base = torch.tensor([0.00, 0.50, 0.50, 0.00], device=device, dtype=dtype)

    phase_advance = (step * args.control_dt) / max(args.T, 1e-6)
    phase = torch.remainder(base + phase_advance, 1.0)
    S = (phase < args.duty).to(dtype)
    swing_mask = 1.0 - S

    denom = max(1.0 - args.duty, 1e-6)
    progress = torch.clamp((phase - args.duty) / denom, min=0.0, max=1.0) * swing_mask
    lift = torch.sin(torch.pi * progress) * swing_mask
    sweep = (2.0 * progress - 1.0) * swing_mask
    return phase, S, swing_mask, progress, lift, sweep


def make_position_action(num_envs: int, device, dtype, step: int):
    phase, S, swing_mask, progress, lift, sweep = gait_phase(step, device=device, dtype=dtype)

    action = torch.zeros((num_envs, 12), device=device, dtype=dtype)
    hx_idx = HX_IDX.to(device=device)
    hy_idx = HY_IDX.to(device=device)
    kn_idx = KN_IDX.to(device=device)

    # The actual offsets sent to the action path.
    hx_offset = args.hx_sweep_delta * sweep
    hy_offset = args.hy_lift_delta * lift + args.hy_sweep_delta * sweep
    kn_offset = args.kn_lift_delta * lift

    action[:, hx_idx] = hx_offset.unsqueeze(0).repeat(num_envs, 1)
    action[:, hy_idx] = hy_offset.unsqueeze(0).repeat(num_envs, 1)
    action[:, kn_idx] = kn_offset.unsqueeze(0).repeat(num_envs, 1)

    action = args.action_scale * action
    action = torch.clamp(action, -args.max_action_abs, args.max_action_abs)

    info = {
        "phase": phase.detach().cpu().tolist(),
        "S": S.detach().cpu().tolist(),
        "swing_mask": swing_mask.detach().cpu().tolist(),
        "progress": progress.detach().cpu().tolist(),
        "lift": lift.detach().cpu().tolist(),
        "sweep": sweep.detach().cpu().tolist(),
        "hx_offset": hx_offset.detach().cpu().tolist(),
        "hy_offset": hy_offset.detach().cpu().tolist(),
        "kn_offset": kn_offset.detach().cpu().tolist(),
    }
    return action, info


def print_debug(step, robot, action, info, base_x0, foot0):
    foot_indices = get_foot_indices(robot)
    x = make_x_hat(robot, velocity_frame="world")
    foot = robot.data.body_pos_w[:, foot_indices, :]
    foot_delta = foot - foot0
    base_delta = x[:, 0:6] - base_x0

    swing_mask = torch.tensor(info["swing_mask"], device=foot_delta.device, dtype=foot_delta.dtype).view(1, 4, 1)
    swing_foot_delta = foot_delta * swing_mask

    print("\n" + "=" * 132)
    print(f"[B8-l POSITION CLEARANCE CRAWL] step={step}")
    print("=" * 132)
    print("[x_hat env0]")
    print("pos xyz:", x[0, 0:3].detach().cpu().numpy())
    print("rpy    :", x[0, 3:6].detach().cpu().numpy())
    print("[base_delta xyz+rpy]")
    print(base_delta[0].detach().cpu().numpy())

    print("\n[gait]")
    print("T:", args.T, "duty:", args.duty, "swing_time:", args.T * (1.0 - args.duty))
    print("phase      :", info["phase"])
    print("S          :", info["S"])
    print("swing_mask :", info["swing_mask"])
    print("progress   :", info["progress"])
    print("lift       :", info["lift"])

    print("\n[action offsets before scale]")
    print("hx_offset:", info["hx_offset"])
    print("hy_offset:", info["hy_offset"])
    print("kn_offset:", info["kn_offset"])

    print("\n[action env0]")
    print(action[0].detach().cpu().numpy())
    print("action max_abs:", float(action.abs().max().detach().cpu()))

    print("\n[foot delta from settled baseline env0]")
    print(foot_delta[0].detach().cpu().numpy())
    print("foot_delta max_abs:", float(foot_delta.abs().max().detach().cpu()))
    print("swing_foot_delta max_abs:", float(swing_foot_delta.abs().max().detach().cpu()))
    print("swing_foot_delta z max:", float(swing_foot_delta[:, :, 2].max().detach().cpu()))
    print("swing_foot_delta z min:", float(swing_foot_delta[:, :, 2].min().detach().cpu()))

    print("=" * 132 + "\n")


def main():
    env_cfg = CartoEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    patch_flat_safe_env(env_cfg)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    robot = env.scene["robot"]
    print_robot_debug_info(robot)

    dtype = robot.data.joint_pos.dtype
    device = robot.data.joint_pos.device

    print("\n" + "=" * 132)
    print("[INFO] Starting B8-l position-action clearance crawl")
    print("=" * 132)
    print("T:", args.T, "duty:", args.duty, "swing_time:", args.T * (1.0 - args.duty))
    print("hy_lift_delta:", args.hy_lift_delta, "kn_lift_delta:", args.kn_lift_delta)
    print("action_scale:", args.action_scale, "max_action_abs:", args.max_action_abs)
    print("=" * 132 + "\n")

    # Settle with zero position action.
    zero_action = torch.zeros((args.num_envs, 12), device=device, dtype=dtype)
    for step in range(args.settle_steps):
        if not simulation_app.is_running():
            break
        env.step(zero_action)
        if step % max(args.print_every, 1) == 0:
            x = make_x_hat(robot, velocity_frame="world")
            print(f"[SETTLE] step={step} pos={x[0,0:3].detach().cpu().numpy()} rpy={x[0,3:6].detach().cpu().numpy()}")

    foot_indices = get_foot_indices(robot)
    foot0 = robot.data.body_pos_w[:, foot_indices, :].detach().clone()
    base_x0 = make_x_hat(robot, velocity_frame="world")[:, 0:6].detach().clone()

    print("\n" + "-" * 132)
    print("[INFO] settled baseline saved")
    print("base_x0 env0:", base_x0[0].detach().cpu().numpy())
    print("foot0 env0:", foot0[0].detach().cpu().numpy())
    print("-" * 132 + "\n")

    for step in range(args.probe_steps):
        if not simulation_app.is_running():
            break

        action, info = make_position_action(args.num_envs, device=device, dtype=dtype, step=step)
        env.step(action)

        if step % max(args.print_every, 1) == 0:
            print_debug(step, robot, action, info, base_x0, foot0)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
