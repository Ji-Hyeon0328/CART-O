# isaaclab_carto/scripts/run_spot_b8u_position_swing_support_shift.py
#
# B8-u: position-action swing + support-shift reference probe.
#
# Context:
#   B8-q2/B8-r/B8-s/B8-t showed that:
#     - support-region base reference can keep the robot upright,
#     - 3-leg support-shifted standing is possible,
#     - additive effort swing and direct implicit target inside an effort env
#       do not produce reliable visible swing clearance.
#
# B8-u switches back to a JointPositionAction environment for the swing task.
# It uses a support-shift phase before the swing and then sends position-action
# offsets to the selected swing leg.
#
# This is a diagnostic bridge:
#   support-region shift idea + position-action swing authority
#
# It is not final MPC/WBC. If this creates visible clearance, the final design
# should represent swing as position/IK/WBC task, while stance/base support is
# handled by the nominal controller and residual torque only corrects mismatch.

import os
import sys
import argparse
from typing import Any

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CARTO/TRACER B8-u position swing with support shift")

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--warmup_steps", type=int, default=160)
parser.add_argument("--shift_steps", type=int, default=220)
parser.add_argument("--lift_steps", type=int, default=180)
parser.add_argument("--hold_steps", type=int, default=100)
parser.add_argument("--lower_steps", type=int, default=140)
parser.add_argument("--print_every", type=int, default=20)

parser.add_argument("--test_leg", type=str, default="RF", choices=["LF", "RF", "LH", "RH"])

# Env / implicit actuator scaling
parser.add_argument("--spawn_z", type=float, default=0.60)
parser.add_argument("--pd_scale", type=float, default=0.70)

# Support shift as position-action offset on stance legs.
# This is still hand-tuned but leg-specific.
parser.add_argument("--hx_shift_mag", type=float, default=0.020)
parser.add_argument("--shift_sign_RF", type=float, default=1.0, choices=[1.0, -1.0])
parser.add_argument("--shift_sign_LF", type=float, default=-1.0, choices=[1.0, -1.0])
parser.add_argument("--shift_sign_RH", type=float, default=1.0, choices=[1.0, -1.0])
parser.add_argument("--shift_sign_LH", type=float, default=-1.0, choices=[1.0, -1.0])

# Swing position-action offsets
parser.add_argument("--hy_lift_delta", type=float, default=-0.030)
parser.add_argument("--kn_lift_delta", type=float, default=-0.090)
parser.add_argument("--hx_lift_delta", type=float, default=0.0)

# Optional sweep component during swing, usually start at zero
parser.add_argument("--hx_sweep_delta", type=float, default=0.0)
parser.add_argument("--hy_sweep_delta", type=float, default=0.0)

# Action scaling / limits
parser.add_argument("--action_scale", type=float, default=1.0)
parser.add_argument("--max_action_abs", type=float, default=0.25)

# Safety
parser.add_argument("--max_pitch_for_lift", type=float, default=0.25)
parser.add_argument("--max_roll_for_lift", type=float, default=0.20)
parser.add_argument("--disable_lift_gate", action="store_true")

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
LEG_TO_ID = {"LF": 0, "RF": 1, "LH": 2, "RH": 3}

# Native action order confirmed earlier:
# [fl_hx, fr_hx, hl_hx, hr_hx, fl_hy, fr_hy, hl_hy, hr_hy, fl_kn, fr_kn, hl_kn, hr_kn]
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


def get_phase(step: int):
    w, s, l, h, lo = args.warmup_steps, args.shift_steps, args.lift_steps, args.hold_steps, args.lower_steps
    if step < w:
        return "warmup", 0.0, 0.0
    t = step - w
    if t < s:
        return "shift", smooth01(t / max(s, 1)), 0.0
    t -= s
    if t < l:
        return "lift", 1.0, smooth01(t / max(l, 1))
    t -= l
    if t < h:
        return "hold_lift", 1.0, 1.0
    t -= h
    if t < lo:
        return "lower", 1.0, smooth01(1.0 - t / max(lo, 1))
    return "done", 0.0, 0.0


def shift_sign_for_leg(leg_name: str) -> float:
    return {
        "RF": args.shift_sign_RF,
        "LF": args.shift_sign_LF,
        "RH": args.shift_sign_RH,
        "LH": args.shift_sign_LH,
    }[leg_name]


def safe_to_lift(x_hat):
    if args.disable_lift_gate:
        return torch.ones((x_hat.shape[0],), dtype=torch.bool, device=x_hat.device)
    roll = torch.abs(x_hat[:, 3])
    pitch = torch.abs(x_hat[:, 4])
    return torch.logical_and(roll < args.max_roll_for_lift, pitch < args.max_pitch_for_lift)


def make_action(num_envs, device, dtype, phase, shift_profile, lift_profile, x_hat):
    action = torch.zeros((num_envs, 12), device=device, dtype=dtype)

    leg = LEG_TO_ID[args.test_leg]
    sign = shift_sign_for_leg(args.test_leg)

    hx_idx = HX_IDX.to(device=device)
    hy_idx = HY_IDX.to(device=device)
    kn_idx = KN_IDX.to(device=device)

    # 1) Support pre-shift: apply HX offset to stance legs only.
    if phase in ["shift", "lift", "hold_lift", "lower"]:
        for i in range(4):
            if i != leg:
                action[:, hx_idx[i]] += sign * args.hx_shift_mag * shift_profile

    # 2) Swing position offset on selected leg.
    lift_enabled = safe_to_lift(x_hat)
    profile = lift_profile
    if phase in ["lift", "hold_lift", "lower"] and not bool(lift_enabled[0].detach().cpu()):
        profile = 0.0

    if phase in ["lift", "hold_lift", "lower"] and profile > 0.0:
        action[:, hx_idx[leg]] += args.hx_lift_delta * profile
        action[:, hy_idx[leg]] += args.hy_lift_delta * profile
        action[:, kn_idx[leg]] += args.kn_lift_delta * profile

        # Optional fore-aft/lateral sweep. Starts disabled.
        sweep = torch.sin(torch.tensor(torch.pi * profile, device=device, dtype=dtype))
        action[:, hx_idx[leg]] += args.hx_sweep_delta * sweep
        action[:, hy_idx[leg]] += args.hy_sweep_delta * sweep

    action = args.action_scale * action
    action = torch.clamp(action, -args.max_action_abs, args.max_action_abs)

    info = {
        "shift_sign": sign,
        "lift_enabled": lift_enabled,
        "effective_lift_profile": profile,
    }
    return action, info


def print_debug(step, phase, shift_profile, lift_profile, action, info, x_hat, foot_pos, foot0, foot_shift0, robot):
    leg = LEG_TO_ID[args.test_leg]
    foot_delta_start = foot_pos - foot0
    foot_delta_shift = foot_pos - foot_shift0 if foot_shift0 is not None else torch.zeros_like(foot_pos)

    print("\n" + "=" * 132)
    print(f"[B8-u POSITION SWING SUPPORT SHIFT] step={step}")
    print("=" * 132)
    print("phase:", phase, "test_leg:", args.test_leg)
    print("shift_profile:", shift_profile, "lift_profile:", lift_profile, "effective_lift_profile:", info["effective_lift_profile"])
    print("shift_sign:", info["shift_sign"], "lift_enabled:", bool(info["lift_enabled"][0].detach().cpu()))
    print("action env0:", action[0].detach().cpu().numpy())

    print("\n[base]")
    print("base pos xyz:", x_hat[0, 0:3].detach().cpu().numpy())
    print("base rpy:", x_hat[0, 3:6].detach().cpu().numpy())
    print("base vel:", x_hat[0, 6:9].detach().cpu().numpy())

    print("\n[foot]")
    print("test foot_delta_from_start xyz:", foot_delta_start[0, leg].detach().cpu().numpy())
    print("test foot_delta_from_shift_start xyz:", foot_delta_shift[0, leg].detach().cpu().numpy())
    print("clearance_z_from_start:", float(foot_delta_start[0, leg, 2].detach().cpu()))
    print("clearance_z_from_shift_start:", float(foot_delta_shift[0, leg, 2].detach().cpu()))
    print("all foot_delta_from_shift_start env0:", foot_delta_shift[0].detach().cpu().numpy())

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

    total_steps = args.warmup_steps + args.shift_steps + args.lift_steps + args.hold_steps + args.lower_steps
    foot0 = None
    foot_shift0 = None

    print("\n" + "=" * 132)
    print("[INFO] Starting B8-u position-action support-shifted swing probe")
    print("test_leg:", args.test_leg)
    print("shift signs RF/LF/RH/LH:", args.shift_sign_RF, args.shift_sign_LF, args.shift_sign_RH, args.shift_sign_LH)
    print("swing deltas hx/hy/kn:", args.hx_lift_delta, args.hy_lift_delta, args.kn_lift_delta)
    print("=" * 132)

    for step in range(total_steps):
        if not simulation_app.is_running():
            break

        x_hat = make_x_hat(robot, velocity_frame="world")
        foot_pos = robot.data.body_pos_w[:, foot_indices, :]

        if foot0 is None:
            foot0 = foot_pos.detach().clone()
        if step == args.warmup_steps + args.shift_steps:
            foot_shift0 = foot_pos.detach().clone()

        phase, shift_profile, lift_profile = get_phase(step)
        action, info = make_action(args.num_envs, device, dtype, phase, shift_profile, lift_profile, x_hat)

        env.step(action)

        if step % max(args.print_every, 1) == 0:
            print_debug(step, phase, shift_profile, lift_profile, action, info, x_hat, foot_pos, foot0, foot_shift0, robot)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
