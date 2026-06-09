# isaaclab_carto/scripts/run_spot_b8x_position_joint_response_sweep.py
#
# B8-x: position-action joint response / foot-lift direction sweep.
#
# Why:
#   B8-w kept the robot standing but still produced almost zero clearance.
#   The action changed during lift, but the selected foot did not move upward.
#
# B8-x checks the missing diagnostic:
#
#   Did the selected joint actually move?
#
# If q_selected follows the position action but foot_z does not change,
# then the issue is swing direction / contact loading / kinematics.
#
# If q_selected does not move, then the action target is not being applied
# as expected or implicit/contact constraints dominate the joint response.
#
# This script:
#   - uses a stable stance preset from B8-v,
#   - sequentially tests several HY/KN combinations on one selected leg,
#   - logs action command, actual joint delta, and foot delta.

import os
import sys
import argparse
from typing import Any, List, Tuple

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CARTO/TRACER B8-x position joint response sweep")

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--warmup_steps", type=int, default=180)
parser.add_argument("--ramp_steps", type=int, default=120)
parser.add_argument("--hold_steps", type=int, default=80)
parser.add_argument("--return_steps", type=int, default=80)
parser.add_argument("--between_steps", type=int, default=80)
parser.add_argument("--print_every", type=int, default=20)

parser.add_argument("--test_leg", type=str, default="RF", choices=["LF", "RF", "LH", "RH"])
parser.add_argument("--stance_preset", type=str, default="kn_pos", choices=["kn_pos", "hy_kn", "asym", "custom"])

# Env / actuator
parser.add_argument("--spawn_z", type=float, default=0.60)
parser.add_argument("--pd_scale", type=float, default=0.80)

# Custom stance values
parser.add_argument("--hx_all", type=float, default=0.0)
parser.add_argument("--hy_all", type=float, default=0.0)
parser.add_argument("--kn_all", type=float, default=0.10)
parser.add_argument("--hy_front", type=float, default=None)
parser.add_argument("--hy_hind", type=float, default=None)
parser.add_argument("--kn_front", type=float, default=None)
parser.add_argument("--kn_hind", type=float, default=None)

# Sweep magnitude
parser.add_argument("--hy_mag", type=float, default=0.08)
parser.add_argument("--kn_mag", type=float, default=0.18)
parser.add_argument("--hx_mag", type=float, default=0.00)

# Action limit
parser.add_argument("--max_action_abs", type=float, default=0.40)
parser.add_argument("--action_scale", type=float, default=1.0)
parser.add_argument("--stance_ramp_steps", type=int, default=100)

# Safety
parser.add_argument("--stop_on_fall", action="store_true")
parser.add_argument("--fall_pitch", type=float, default=0.45)
parser.add_argument("--fall_roll", type=float, default=0.35)

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


def get_stance_offsets():
    if args.stance_preset == "kn_pos":
        return 0.0, 0.0, 0.0, 0.10, 0.10
    if args.stance_preset == "hy_kn":
        return 0.0, -0.04, -0.04, 0.10, 0.10
    if args.stance_preset == "asym":
        return 0.0, -0.04, 0.02, 0.12, 0.04

    hy_front = args.hy_all if args.hy_front is None else args.hy_front
    hy_hind = args.hy_all if args.hy_hind is None else args.hy_hind
    kn_front = args.kn_all if args.kn_front is None else args.kn_front
    kn_hind = args.kn_all if args.kn_hind is None else args.kn_hind
    return args.hx_all, hy_front, hy_hind, kn_front, kn_hind


def make_stance_action(num_envs, device, dtype, stance_ramp):
    hx_all, hy_front, hy_hind, kn_front, kn_hind = get_stance_offsets()
    action = torch.zeros((num_envs, 12), device=device, dtype=dtype)
    action[:, 0:4] = hx_all * stance_ramp
    action[:, 4] = hy_front * stance_ramp
    action[:, 5] = hy_front * stance_ramp
    action[:, 6] = hy_hind * stance_ramp
    action[:, 7] = hy_hind * stance_ramp
    action[:, 8] = kn_front * stance_ramp
    action[:, 9] = kn_front * stance_ramp
    action[:, 10] = kn_hind * stance_ramp
    action[:, 11] = kn_hind * stance_ramp
    return action


def test_sequence() -> List[Tuple[str, float, float, float]]:
    return [
        ("HY- KN-", 0.0, -args.hy_mag, -args.kn_mag),
        ("HY- KN+", 0.0, -args.hy_mag, args.kn_mag),
        ("HY+ KN-", 0.0, args.hy_mag, -args.kn_mag),
        ("HY+ KN+", 0.0, args.hy_mag, args.kn_mag),
        ("KN+ only", 0.0, 0.0, args.kn_mag),
        ("KN- only", 0.0, 0.0, -args.kn_mag),
        ("HY- only", 0.0, -args.hy_mag, 0.0),
        ("HY+ only", 0.0, args.hy_mag, 0.0),
    ]


def get_stage(global_step: int):
    if global_step < args.warmup_steps:
        return "warmup", -1, 0.0, "none"

    local = global_step - args.warmup_steps
    block = args.ramp_steps + args.hold_steps + args.return_steps + args.between_steps
    seq = test_sequence()
    test_id = local // block
    within = local % block

    if test_id >= len(seq):
        return "done", len(seq), 0.0, "done"

    label = seq[test_id][0]
    if within < args.ramp_steps:
        return "ramp", int(test_id), smooth01(within / max(args.ramp_steps, 1)), label
    within -= args.ramp_steps
    if within < args.hold_steps:
        return "hold", int(test_id), 1.0, label
    within -= args.hold_steps
    if within < args.return_steps:
        return "return", int(test_id), smooth01(1.0 - within / max(args.return_steps, 1)), label
    return "between", int(test_id), 0.0, label


def make_action(num_envs, device, dtype, step):
    stance_ramp = smooth01(step / max(args.stance_ramp_steps, 1))
    action = make_stance_action(num_envs, device, dtype, stance_ramp)

    stage, test_id, profile, label = get_stage(step)

    if test_id >= 0 and test_id < len(test_sequence()) and stage in ["ramp", "hold", "return"]:
        _, hx_delta, hy_delta, kn_delta = test_sequence()[test_id]
        leg = LEG_TO_ID[args.test_leg]
        action[:, int(HX_IDX[leg])] += hx_delta * profile
        action[:, int(HY_IDX[leg])] += hy_delta * profile
        action[:, int(KN_IDX[leg])] += kn_delta * profile

    action = args.action_scale * action
    action = torch.clamp(action, -args.max_action_abs, args.max_action_abs)
    return action, stage, test_id, profile, label, stance_ramp


def selected_joint_values(tensor, leg):
    return torch.stack([
        tensor[:, int(HX_IDX[leg])],
        tensor[:, int(HY_IDX[leg])],
        tensor[:, int(KN_IDX[leg])],
    ], dim=1)


def print_debug(step, action, stage, test_id, profile, label, stance_ramp, x_hat, robot, foot_pos, foot0, foot_test0, q0, q_test0):
    leg = LEG_TO_ID[args.test_leg]
    fd_start = foot_pos - foot0
    fd_test = foot_pos - foot_test0 if foot_test0 is not None else torch.zeros_like(foot_pos)

    q = robot.data.joint_pos
    qd = robot.data.joint_vel

    q_delta_start = q - q0
    q_delta_test = q - q_test0 if q_test0 is not None else torch.zeros_like(q)

    print("\n" + "=" * 132)
    print(f"[B8-x POSITION JOINT RESPONSE SWEEP] step={step}")
    print("=" * 132)
    print("stage:", stage, "test_id:", test_id, "label:", label, "profile:", profile, "test_leg:", args.test_leg)
    print("stance_preset:", args.stance_preset, "stance_ramp:", stance_ramp)
    print("action env0:", action[0].detach().cpu().numpy())

    print("\n[base]")
    print("base pos xyz:", x_hat[0, 0:3].detach().cpu().numpy())
    print("base rpy:", x_hat[0, 3:6].detach().cpu().numpy())
    print("base vel:", x_hat[0, 6:9].detach().cpu().numpy())

    print("\n[selected joint actual response]")
    print("q_selected_delta_from_start [hx,hy,kn]:", selected_joint_values(q_delta_start, leg)[0].detach().cpu().numpy())
    print("q_selected_delta_from_test_start [hx,hy,kn]:", selected_joint_values(q_delta_test, leg)[0].detach().cpu().numpy())
    print("qd_selected [hx,hy,kn]:", selected_joint_values(qd, leg)[0].detach().cpu().numpy())

    print("\n[foot response]")
    print("test foot_delta_from_start xyz:", fd_start[0, leg].detach().cpu().numpy())
    print("test foot_delta_from_test_start xyz:", fd_test[0, leg].detach().cpu().numpy())
    print("clearance_z_from_start:", float(fd_start[0, leg, 2].detach().cpu()))
    print("clearance_z_from_test_start:", float(fd_test[0, leg, 2].detach().cpu()))

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

    seq = test_sequence()
    total_steps = args.warmup_steps + len(seq) * (args.ramp_steps + args.hold_steps + args.return_steps + args.between_steps)

    foot0 = robot.data.body_pos_w[:, foot_indices, :].detach().clone()
    q0 = robot.data.joint_pos.detach().clone()
    foot_test0 = None
    q_test0 = None
    prev_test_id = -999

    print("\n" + "=" * 132)
    print("[INFO] Starting B8-x position joint response sweep")
    print("test_leg:", args.test_leg, "stance_preset:", args.stance_preset)
    print("hy_mag:", args.hy_mag, "kn_mag:", args.kn_mag)
    print("sequence:", [s[0] for s in seq])
    print("=" * 132)

    for step in range(total_steps):
        if not simulation_app.is_running():
            break

        x_hat = make_x_hat(robot, velocity_frame="world")
        foot_pos = robot.data.body_pos_w[:, foot_indices, :]

        action, stage, test_id, profile, label, stance_ramp = make_action(args.num_envs, device, dtype, step)

        if test_id != prev_test_id and test_id >= 0:
            foot_test0 = foot_pos.detach().clone()
            q_test0 = robot.data.joint_pos.detach().clone()
            prev_test_id = test_id

        env.step(action)

        if step % max(args.print_every, 1) == 0:
            print_debug(step, action, stage, test_id, profile, label, stance_ramp, x_hat, robot, foot_pos, foot0, foot_test0, q0, q_test0)

        if args.stop_on_fall:
            if abs(float(x_hat[0, 3].detach().cpu())) > args.fall_roll or abs(float(x_hat[0, 4].detach().cpu())) > args.fall_pitch:
                print("[STOP] fall threshold reached.")
                break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
