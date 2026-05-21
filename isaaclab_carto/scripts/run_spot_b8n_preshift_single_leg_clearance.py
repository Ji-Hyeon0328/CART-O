# isaaclab_carto/scripts/run_spot_b8n_preshift_single_leg_clearance.py
#
# B8-n: pre-shifted single-leg clearance with position action.
#
# B8-m showed:
#   - single-leg command makes the robot slowly fall sideways
#   - selected test foot z does not get reliable clearance
#
# Interpretation:
#   We are lifting a leg without first shifting the body/support condition.
#
# B8-n adds a quasi-static pre-shift phase:
#
#   settle
#   -> apply small HX offset on stance legs only
#   -> hold the shift
#   -> lift selected leg
#   -> lower selected leg
#   -> release shift
#
# This is still a diagnostic. It tests which HX shift sign gives a better
# base lateral/roll response before trying cyclic gait again.

import os
import sys
import argparse
from typing import Any

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CARTO/TRACER B8-n pre-shifted single-leg clearance")

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle_steps", type=int, default=160)
parser.add_argument("--probe_steps", type=int, default=360)
parser.add_argument("--print_every", type=int, default=20)

parser.add_argument("--test_leg", type=str, default="RF", choices=["LF", "RF", "LH", "RH"])

# Pre-shift by stance-leg HX offsets
parser.add_argument("--hx_shift_mag", type=float, default=0.04)
parser.add_argument("--hx_shift_sign", type=float, default=1.0, choices=[1.0, -1.0])
parser.add_argument("--shift_swing_leg", action="store_true")

# Lift offsets
parser.add_argument("--hy_lift_delta", type=float, default=-0.03)
parser.add_argument("--kn_lift_delta", type=float, default=-0.09)
parser.add_argument("--hx_lift_delta", type=float, default=0.0)

parser.add_argument("--action_scale", type=float, default=1.0)
parser.add_argument("--max_action_abs", type=float, default=0.25)

# phase lengths
parser.add_argument("--shift_ramp_steps", type=int, default=80)
parser.add_argument("--shift_hold_steps", type=int, default=60)
parser.add_argument("--lift_ramp_steps", type=int, default=80)
parser.add_argument("--lift_hold_steps", type=int, default=80)
parser.add_argument("--lower_steps", type=int, default=80)

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


def profiles(step: int):
    # shift profile: ramp -> hold through lift/lower -> release after lower
    sr = args.shift_ramp_steps
    sh = args.shift_hold_steps
    lr = args.lift_ramp_steps
    lh = args.lift_hold_steps
    low = args.lower_steps

    if step < sr:
        shift_p = smooth01(step / max(sr, 1))
        lift_p = 0.0
        phase = "shift_ramp"
    elif step < sr + sh:
        shift_p = 1.0
        lift_p = 0.0
        phase = "shift_hold"
    elif step < sr + sh + lr:
        shift_p = 1.0
        lift_p = smooth01((step - sr - sh) / max(lr, 1))
        phase = "lift_ramp"
    elif step < sr + sh + lr + lh:
        shift_p = 1.0
        lift_p = 1.0
        phase = "lift_hold"
    elif step < sr + sh + lr + lh + low:
        shift_p = 1.0
        lift_p = smooth01(1.0 - (step - sr - sh - lr - lh) / max(low, 1))
        phase = "lower"
    else:
        # release shift after lift completes
        rel_step = step - (sr + sh + lr + lh + low)
        shift_p = smooth01(1.0 - rel_step / max(sr, 1))
        lift_p = 0.0
        phase = "release_shift"
    return phase, shift_p, lift_p


def make_action(num_envs, device, dtype, step):
    leg = LEG_TO_ID[args.test_leg]
    phase, shift_p, lift_p = profiles(step)

    action = torch.zeros((num_envs, 12), device=device, dtype=dtype)

    hx_idx = HX_IDX.to(device=device)
    hy_idx = HY_IDX.to(device=device)
    kn_idx = KN_IDX.to(device=device)

    # stance pre-shift: apply HX offset on stance legs, optionally also swing leg
    for i in range(4):
        if i != leg or args.shift_swing_leg:
            action[:, hx_idx[i]] += args.hx_shift_sign * args.hx_shift_mag * shift_p

    # selected leg lift
    action[:, hx_idx[leg]] += args.hx_lift_delta * lift_p
    action[:, hy_idx[leg]] += args.hy_lift_delta * lift_p
    action[:, kn_idx[leg]] += args.kn_lift_delta * lift_p

    action = args.action_scale * action
    action = torch.clamp(action, -args.max_action_abs, args.max_action_abs)

    info = {
        "phase": phase,
        "shift_profile": shift_p,
        "lift_profile": lift_p,
        "test_leg": args.test_leg,
        "hx_shift_cmd": args.hx_shift_sign * args.hx_shift_mag * shift_p,
        "hx_lift_cmd": args.hx_lift_delta * lift_p,
        "hy_lift_cmd": args.hy_lift_delta * lift_p,
        "kn_lift_cmd": args.kn_lift_delta * lift_p,
    }
    return action, info


def print_debug(step, robot, action, info, base_x0, foot0):
    leg = LEG_TO_ID[args.test_leg]
    foot_indices = get_foot_indices(robot)

    x = make_x_hat(robot, velocity_frame="world")
    foot = robot.data.body_pos_w[:, foot_indices, :]
    foot_delta = foot - foot0
    base_delta = x[:, 0:6] - base_x0

    print("\n" + "=" * 130)
    print(f"[B8-n PRESHIFT SINGLE-LEG CLEARANCE] step={step}")
    print("=" * 130)
    print("phase:", info["phase"])
    print("test_leg:", args.test_leg)
    print("shift_profile:", info["shift_profile"], "lift_profile:", info["lift_profile"])
    print("hx_shift_sign:", args.hx_shift_sign, "hx_shift_cmd:", info["hx_shift_cmd"])
    print("lift cmd hx/hy/kn:", info["hx_lift_cmd"], info["hy_lift_cmd"], info["kn_lift_cmd"])
    print("action env0:", action[0].detach().cpu().numpy())

    print("\n[base_delta xyz+rpy]")
    print(base_delta[0].detach().cpu().numpy())
    print("base_delta_y:", float(base_delta[0, 1].detach().cpu()))
    print("base_delta_roll:", float(base_delta[0, 3].detach().cpu()))
    print("base_delta_pitch:", float(base_delta[0, 4].detach().cpu()))
    print("base_delta_z:", float(base_delta[0, 2].detach().cpu()))

    print("\n[foot delta]")
    print("test foot_delta xyz:", foot_delta[0, leg].detach().cpu().numpy())
    print("all foot_delta env0:", foot_delta[0].detach().cpu().numpy())
    print("clearance_z_for_test_leg:", float(foot_delta[0, leg, 2].detach().cpu()))
    print("foot_delta max_abs:", float(foot_delta.abs().max().detach().cpu()))
    print("=" * 130 + "\n")


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

    print("\n" + "=" * 130)
    print("[INFO] Starting B8-n pre-shifted single-leg clearance")
    print("test_leg:", args.test_leg)
    print("hx_shift_mag:", args.hx_shift_mag, "hx_shift_sign:", args.hx_shift_sign)
    print("hy_lift_delta:", args.hy_lift_delta, "kn_lift_delta:", args.kn_lift_delta)
    print("=" * 130)

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
