# isaaclab_carto/scripts/run_spot_b8o_leg_specific_preshift_crawl.py
#
# B8-o: leg-specific pre-shift crawl.
#
# Motivation:
#   B8-n showed that pre-shift can keep the robot standing for RF single-leg
#   experiments, but clearance is still small. Also, the good shift direction
#   depends on which leg is about to swing.
#
# This script implements a simple rule-based pre-shift map:
#
#   for each upcoming swing leg:
#       pre-shift stance legs by a leg-specific HX sign
#       hold shift
#       lift that leg
#       lower that leg
#       move to next leg
#
# This is still NOT final MPC/WBC. It is a bridge experiment:
#
#   contact schedule -> leg-specific base preparation -> swing action
#
# After this, the hand-tuned pre-shift map should be replaced by a
# support-region / feasible CoM reference module.

import os
import sys
import argparse
from typing import Any, Dict

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CARTO/TRACER B8-o leg-specific pre-shift crawl")

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle_steps", type=int, default=160)
parser.add_argument("--cycles", type=int, default=2)
parser.add_argument("--print_every", type=int, default=20)

# Leg sequence
parser.add_argument("--leg_sequence", type=str, default="RF,LH,LF,RH")

# Shift
parser.add_argument("--hx_shift_mag", type=float, default=0.025)
parser.add_argument("--shift_ramp_steps", type=int, default=100)
parser.add_argument("--shift_hold_steps", type=int, default=80)
parser.add_argument("--release_shift_each_step", action="store_true")

# Lift
parser.add_argument("--hy_lift_delta", type=float, default=-0.020)
parser.add_argument("--kn_lift_delta", type=float, default=-0.060)
parser.add_argument("--hx_lift_delta", type=float, default=0.0)
parser.add_argument("--lift_ramp_steps", type=int, default=100)
parser.add_argument("--lift_hold_steps", type=int, default=50)
parser.add_argument("--lower_steps", type=int, default=80)

# Action limits
parser.add_argument("--action_scale", type=float, default=1.0)
parser.add_argument("--max_action_abs", type=float, default=0.20)

# Shift signs.
# From B8-n:
#   RF with hx_shift_sign +1 and -1 both stood; +1 looked stable but small clearance.
#   LF with +1 fell, so LF should likely use -1.
# For hind legs, start with analogous lateral rule and then inspect.
parser.add_argument("--shift_sign_RF", type=float, default=1.0, choices=[1.0, -1.0])
parser.add_argument("--shift_sign_LF", type=float, default=-1.0, choices=[1.0, -1.0])
parser.add_argument("--shift_sign_RH", type=float, default=1.0, choices=[1.0, -1.0])
parser.add_argument("--shift_sign_LH", type=float, default=-1.0, choices=[1.0, -1.0])

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
ID_TO_LEG = {v: k for k, v in LEG_TO_ID.items()}

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


def get_foot_indices(robot):
    name_to_idx = {name: i for i, name in enumerate(robot.body_names)}
    return [name_to_idx[name] for name in FOOT_NAMES]


def parse_leg_sequence():
    seq = [s.strip().upper() for s in args.leg_sequence.split(",") if s.strip()]
    for s in seq:
        if s not in LEG_TO_ID:
            raise ValueError(f"Invalid leg in sequence: {s}")
    return seq


def shift_sign_for_leg(leg_name: str) -> float:
    return {
        "RF": args.shift_sign_RF,
        "LF": args.shift_sign_LF,
        "RH": args.shift_sign_RH,
        "LH": args.shift_sign_LH,
    }[leg_name]


def step_phase(local_step: int):
    sr = args.shift_ramp_steps
    sh = args.shift_hold_steps
    lr = args.lift_ramp_steps
    lh = args.lift_hold_steps
    low = args.lower_steps

    if local_step < sr:
        return "shift_ramp", smooth01(local_step / max(sr, 1)), 0.0
    if local_step < sr + sh:
        return "shift_hold", 1.0, 0.0
    if local_step < sr + sh + lr:
        return "lift_ramp", 1.0, smooth01((local_step - sr - sh) / max(lr, 1))
    if local_step < sr + sh + lr + lh:
        return "lift_hold", 1.0, 1.0
    if local_step < sr + sh + lr + lh + low:
        return "lower", 1.0, smooth01(1.0 - (local_step - sr - sh - lr - lh) / max(low, 1))

    release_len = sr if args.release_shift_each_step else 1
    rel = local_step - (sr + sh + lr + lh + low)
    return "release_shift", smooth01(1.0 - rel / max(release_len, 1)), 0.0


def episode_lengths():
    core = args.shift_ramp_steps + args.shift_hold_steps + args.lift_ramp_steps + args.lift_hold_steps + args.lower_steps
    if args.release_shift_each_step:
        return core + args.shift_ramp_steps
    return core + 1


def make_action(num_envs, device, dtype, leg_name: str, local_step: int):
    leg = LEG_TO_ID[leg_name]
    phase, shift_p, lift_p = step_phase(local_step)
    sign = shift_sign_for_leg(leg_name)

    action = torch.zeros((num_envs, 12), device=device, dtype=dtype)
    hx_idx = HX_IDX.to(device=device)
    hy_idx = HY_IDX.to(device=device)
    kn_idx = KN_IDX.to(device=device)

    # Shift stance legs only. Swing leg is excluded from the pre-shift.
    for i in range(4):
        if i != leg:
            action[:, hx_idx[i]] += sign * args.hx_shift_mag * shift_p

    # Lift the selected leg.
    action[:, hx_idx[leg]] += args.hx_lift_delta * lift_p
    action[:, hy_idx[leg]] += args.hy_lift_delta * lift_p
    action[:, kn_idx[leg]] += args.kn_lift_delta * lift_p

    action = args.action_scale * action
    action = torch.clamp(action, -args.max_action_abs, args.max_action_abs)

    info = {
        "leg_name": leg_name,
        "leg_id": leg,
        "phase": phase,
        "shift_profile": shift_p,
        "lift_profile": lift_p,
        "shift_sign": sign,
        "hx_shift_cmd": sign * args.hx_shift_mag * shift_p,
        "hx_lift_cmd": args.hx_lift_delta * lift_p,
        "hy_lift_cmd": args.hy_lift_delta * lift_p,
        "kn_lift_cmd": args.kn_lift_delta * lift_p,
    }
    return action, info


def print_debug(global_step, local_step, robot, action, info, base_x0, foot0):
    leg = info["leg_id"]
    foot_indices = get_foot_indices(robot)

    x = make_x_hat(robot, velocity_frame="world")
    foot = robot.data.body_pos_w[:, foot_indices, :]
    foot_delta = foot - foot0
    base_delta = x[:, 0:6] - base_x0

    print("\n" + "=" * 132)
    print(f"[B8-o LEG-SPECIFIC PRESHIFT CRAWL] global_step={global_step} local_step={local_step}")
    print("=" * 132)
    print("leg:", info["leg_name"], "phase:", info["phase"])
    print("shift_sign:", info["shift_sign"], "shift_profile:", info["shift_profile"], "lift_profile:", info["lift_profile"])
    print("shift/lift cmd:", info["hx_shift_cmd"], info["hx_lift_cmd"], info["hy_lift_cmd"], info["kn_lift_cmd"])
    print("action env0:", action[0].detach().cpu().numpy())

    print("\n[base_delta xyz+rpy]")
    print(base_delta[0].detach().cpu().numpy())
    print("base_delta_y:", float(base_delta[0, 1].detach().cpu()))
    print("base_delta_roll:", float(base_delta[0, 3].detach().cpu()))
    print("base_delta_pitch:", float(base_delta[0, 4].detach().cpu()))
    print("base_delta_z:", float(base_delta[0, 2].detach().cpu()))

    print("\n[foot delta]")
    print("test foot_delta xyz:", foot_delta[0, leg].detach().cpu().numpy())
    print("clearance_z_for_test_leg:", float(foot_delta[0, leg, 2].detach().cpu()))
    print("all foot_delta env0:", foot_delta[0].detach().cpu().numpy())
    print("foot_delta max_abs:", float(foot_delta.abs().max().detach().cpu()))
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

    seq = parse_leg_sequence()
    per_leg_len = episode_lengths()

    print("\n" + "=" * 132)
    print("[INFO] Starting B8-o leg-specific pre-shift crawl")
    print("leg_sequence:", seq)
    print("per_leg_len:", per_leg_len, "cycles:", args.cycles)
    print("shift signs RF/LF/RH/LH:", args.shift_sign_RF, args.shift_sign_LF, args.shift_sign_RH, args.shift_sign_LH)
    print("=" * 132)

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

    total_steps = args.cycles * len(seq) * per_leg_len
    for gstep in range(total_steps):
        if not simulation_app.is_running():
            break

        leg_idx_in_seq = (gstep // per_leg_len) % len(seq)
        leg_name = seq[leg_idx_in_seq]
        local_step = gstep % per_leg_len

        action, info = make_action(args.num_envs, device, dtype, leg_name, local_step)
        env.step(action)

        if gstep % max(args.print_every, 1) == 0:
            print_debug(gstep, local_step, robot, action, info, base_x0, foot0)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
