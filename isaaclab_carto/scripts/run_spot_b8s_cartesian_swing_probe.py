# isaaclab_carto/scripts/run_spot_b8s_cartesian_swing_probe.py
#
# B8-s: Cartesian swing-force probe with support-shifted 3-leg stance.
#
# Why:
#   B8-r kept the robot standing in 3-leg support, but joint-PD swing torque
#   did not produce visible clearance. Even with tau_swing saturated, the foot
#   stayed near the settled ground height.
#
# Change in B8-s:
#   Instead of joint-space swing PD, apply a Cartesian upward swing-foot force:
#
#       tau_swing = J_swing^T * [0, 0, F_lift]
#
#   while keeping:
#       support-region base_ref
#       stance-leg vertical support
#       posture torque
#
# This is still a diagnostic, not final WBC.
# It checks whether a WBC-like swing-foot task has authority.

import os
import sys
import argparse
from typing import Any

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CARTO/TRACER B8-s Cartesian swing-force probe")

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--warmup_steps", type=int, default=120)
parser.add_argument("--shift_steps", type=int, default=180)
parser.add_argument("--lift_steps", type=int, default=160)
parser.add_argument("--hold_steps", type=int, default=120)
parser.add_argument("--lower_steps", type=int, default=120)
parser.add_argument("--print_every", type=int, default=20)

parser.add_argument("--test_leg", type=str, default="RF", choices=["LF", "RF", "LH", "RH"])

# Env / actuator
parser.add_argument("--spawn_z", type=float, default=0.60)
parser.add_argument("--pd_scale", type=float, default=0.65)

# Support/base ref
parser.add_argument("--height_ref", type=float, default=0.665)
parser.add_argument("--alpha", type=float, default=0.12)
parser.add_argument("--margin", type=float, default=0.030)
parser.add_argument("--max_shift_per_step", type=float, default=0.0005)

# Base force gains
parser.add_argument("--kp_xy", type=float, default=20.0)
parser.add_argument("--kd_xy", type=float, default=14.0)
parser.add_argument("--kp_z", type=float, default=180.0)
parser.add_argument("--kd_z", type=float, default=35.0)

# Joint posture stabilizer
parser.add_argument("--kp_posture", type=float, default=8.0)
parser.add_argument("--kd_posture", type=float, default=0.8)
parser.add_argument("--max_posture_tau", type=float, default=8.0)

# Cartesian swing force
parser.add_argument("--swing_fz", type=float, default=45.0)
parser.add_argument("--swing_fx", type=float, default=0.0)
parser.add_argument("--swing_fy", type=float, default=0.0)
parser.add_argument("--swing_cart_sign", type=float, default=1.0, choices=[1.0, -1.0])
parser.add_argument("--max_swing_tau", type=float, default=18.0)

# Force limits
parser.add_argument("--mass_override", type=float, default=32.0)
parser.add_argument("--mu", type=float, default=0.7)
parser.add_argument("--min_fz", type=float, default=10.0)
parser.add_argument("--max_fz", type=float, default=150.0)

# Torque scales
parser.add_argument("--tau_force_scale", type=float, default=0.22)
parser.add_argument("--tau_posture_scale", type=float, default=1.0)
parser.add_argument("--tau_swing_scale", type=float, default=1.0)
parser.add_argument("--tau_sign", type=float, default=-1.0, choices=[1.0, -1.0])
parser.add_argument("--max_total_tau", type=float, default=20.0)

# Lift gate
parser.add_argument("--require_margin", action="store_true")
parser.add_argument("--max_pitch_for_lift", type=float, default=0.20)
parser.add_argument("--max_roll_for_lift", type=float, default=0.15)

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
from isaaclab.utils import configclass  # noqa: E402

try:
    from isaaclab.envs.mdp import JointEffortActionCfg  # noqa: E402
except Exception:
    from isaaclab.envs.mdp.actions.actions_cfg import JointEffortActionCfg  # noqa: E402

from isaaclab_carto.envs.carto_env_cfg import CartoEnvCfg  # noqa: E402
from isaaclab_carto.lowlevel.spot_state import make_x_hat, print_robot_debug_info  # noqa: E402
from isaaclab_carto.lowlevel.support_region_ref import SupportRegionRefConfig, compute_support_region_ref  # noqa: E402


FOOT_NAMES = ["fl_foot", "fr_foot", "hl_foot", "hr_foot"]
LEG_TO_ID = {"LF": 0, "RF": 1, "LH": 2, "RH": 3}
HX_IDX = torch.tensor([0, 1, 2, 3], dtype=torch.long)
HY_IDX = torch.tensor([4, 5, 6, 7], dtype=torch.long)
KN_IDX = torch.tensor([8, 9, 10, 11], dtype=torch.long)


@configclass
class CartoEffortActionsCfg:
    joint_effort = JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0)


@configclass
class CartoEffortEnvCfg(CartoEnvCfg):
    actions: CartoEffortActionsCfg = CartoEffortActionsCfg()


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


def get_mass(robot):
    try:
        return float(robot.root_physx_view.get_masses().sum().detach().cpu())
    except Exception:
        return args.mass_override


def get_jacobian_feet(robot, foot_indices):
    jac = robot.root_physx_view.get_jacobians()
    if jac.shape[-1] == robot.num_joints + 6:
        jac = jac[..., 6:]
    return jac[:, foot_indices, 0:3, :]


def get_phase(step: int):
    w, s, l, h, lo = args.warmup_steps, args.shift_steps, args.lift_steps, args.hold_steps, args.lower_steps
    if step < w:
        return "warmup", 0.0
    t = step - w
    if t < s:
        return "shift", 0.0
    t -= s
    if t < l:
        return "lift", smooth01(t / max(l, 1))
    t -= l
    if t < h:
        return "hold_lift", 1.0
    t -= h
    if t < lo:
        return "lower", smooth01(1.0 - t / max(lo, 1))
    return "done", 0.0


def compute_stance_mask(num_envs, device, dtype, phase):
    stance = torch.ones((num_envs, 4), device=device, dtype=dtype)
    if phase in ["shift", "lift", "hold_lift", "lower"]:
        stance[:, LEG_TO_ID[args.test_leg]] = 0.0
    return stance


def make_base_ref(phase, x_hat, foot_pos, stance_mask, prev_base_ref, cfg):
    base_pos = x_hat[:, 0:3]
    base_rpy = x_hat[:, 3:6]
    out = compute_support_region_ref(
        foot_pos_w=foot_pos,
        base_pos_w=base_pos,
        base_rpy_w=base_rpy,
        stance_mask=stance_mask,
        prev_base_ref=prev_base_ref,
        cfg=cfg,
    )
    base_ref = out.base_ref.detach().clone()
    if phase == "warmup":
        base_ref[:, 0:2] = base_pos[:, 0:2]
    base_ref[:, 2] = args.height_ref
    base_ref[:, 3:5] = 0.0
    return base_ref, out


def compute_foot_forces(x_hat, base_ref, foot_pos, stance_mask, mass):
    device, dtype = x_hat.device, x_hat.dtype
    n = x_hat.shape[0]
    g = 9.81
    base_pos, base_vel = x_hat[:, 0:3], x_hat[:, 6:9]
    pos_err = base_ref[:, 0:3] - base_pos

    F_des = torch.zeros((n, 3), device=device, dtype=dtype)
    F_des[:, 0] = args.kp_xy * pos_err[:, 0] - args.kd_xy * base_vel[:, 0]
    F_des[:, 1] = args.kp_xy * pos_err[:, 1] - args.kd_xy * base_vel[:, 1]
    F_des[:, 2] = mass * g + args.kp_z * pos_err[:, 2] - args.kd_z * base_vel[:, 2]

    foot_forces = torch.zeros((n, 4, 3), device=device, dtype=dtype)
    for e in range(n):
        stance_ids = torch.where(stance_mask[e] > 0.5)[0]
        ns = max(int(stance_ids.numel()), 1)
        fz = torch.clamp(torch.ones((ns,), device=device, dtype=dtype) * (F_des[e, 2] / ns), args.min_fz, args.max_fz)
        fx_i = F_des[e, 0] / ns
        fy_i = F_des[e, 1] / ns
        fxy_norm = torch.sqrt(fx_i * fx_i + fy_i * fy_i).clamp_min(1.0e-9)
        scale = torch.clamp((args.mu * fz) / fxy_norm, max=1.0)
        foot_forces[e, stance_ids, 0] = fx_i * scale
        foot_forces[e, stance_ids, 1] = fy_i * scale
        foot_forces[e, stance_ids, 2] = fz
    wrench = torch.cat([F_des, torch.zeros((n, 3), device=device, dtype=dtype)], dim=1)
    return foot_forces, wrench


def jt_force_to_tau(jv_feet, foot_forces):
    return torch.einsum("nfij,nfi->nj", jv_feet, foot_forces)


def posture_tau(robot, q_nom, swing_leg=None):
    q, qd = robot.data.joint_pos, robot.data.joint_vel
    tau = args.kp_posture * (q_nom - q) - args.kd_posture * qd
    tau = torch.clamp(tau, -args.max_posture_tau, args.max_posture_tau)
    if swing_leg is not None:
        leg = LEG_TO_ID[swing_leg]
        idxs = [int(HX_IDX[leg]), int(HY_IDX[leg]), int(KN_IDX[leg])]
        tau[:, idxs] = 0.0
    return tau


def swing_cart_tau(jv_feet, profile, num_envs, device, dtype):
    tau = torch.zeros((num_envs, 12), device=device, dtype=dtype)
    if profile <= 0.0:
        return tau
    leg = LEG_TO_ID[args.test_leg]
    f = torch.zeros((num_envs, 3), device=device, dtype=dtype)
    f[:, 0] = args.swing_fx * profile
    f[:, 1] = args.swing_fy * profile
    f[:, 2] = args.swing_fz * profile
    tau_leg = torch.einsum("nij,ni->nj", jv_feet[:, leg, :, :], f)
    tau = args.swing_cart_sign * tau_leg
    return torch.clamp(tau, -args.max_swing_tau, args.max_swing_tau)


def safe_to_lift(x_hat, out):
    roll, pitch = torch.abs(x_hat[:, 3]), torch.abs(x_hat[:, 4])
    ok = torch.logical_and(roll < args.max_roll_for_lift, pitch < args.max_pitch_for_lift)
    if args.require_margin:
        ok = torch.logical_and(ok, out.swing_allowed)
    return ok


def print_debug(step, phase, profile, x_hat, base_ref, out, foot_pos, foot0, foot_shift0, foot_forces, wrench, tau_force, tau_post, tau_swing, action, robot, lift_enabled):
    leg = LEG_TO_ID[args.test_leg]
    foot_delta_start = foot_pos - foot0
    foot_delta_shift = foot_pos - foot_shift0 if foot_shift0 is not None else torch.zeros_like(foot_pos)

    print("\n" + "=" * 132)
    print(f"[B8-s CARTESIAN SWING PROBE] step={step}")
    print("=" * 132)
    print("phase:", phase, "test_leg:", args.test_leg, "swing_profile:", profile, "lift_enabled:", bool(lift_enabled[0].detach().cpu()))
    print("[support-region]")
    print("support_center_xy:", out.support_center_xy[0].detach().cpu().numpy())
    print("current_xy:", out.current_xy[0].detach().cpu().numpy())
    print("target_xy:", out.target_xy[0].detach().cpu().numpy())
    print("margin_to_edge:", float(out.margin_to_edge[0].detach().cpu()))
    print("swing_allowed:", bool(out.swing_allowed[0].detach().cpu()))
    print("stance_count:", int(out.stance_count[0].detach().cpu()))

    print("\n[base]")
    print("base pos xyz:", x_hat[0, 0:3].detach().cpu().numpy())
    print("base rpy:", x_hat[0, 3:6].detach().cpu().numpy())
    print("base ref:", base_ref[0].detach().cpu().numpy())
    print("pos err:", (base_ref[0, 0:3] - x_hat[0, 0:3]).detach().cpu().numpy())

    print("\n[foot]")
    print("test foot_delta_from_start xyz:", foot_delta_start[0, leg].detach().cpu().numpy())
    print("test foot_delta_from_shift_start xyz:", foot_delta_shift[0, leg].detach().cpu().numpy())
    print("clearance_z_from_start:", float(foot_delta_start[0, leg, 2].detach().cpu()))
    print("clearance_z_from_shift_start:", float(foot_delta_shift[0, leg, 2].detach().cpu()))

    print("\n[force/tau]")
    print("desired wrench:", wrench[0].detach().cpu().numpy())
    print("foot forces:", foot_forces[0].detach().cpu().numpy())
    print("tau_force max_abs:", float(tau_force.abs().max().detach().cpu()))
    print("tau_posture max_abs:", float(tau_post.abs().max().detach().cpu()))
    print("tau_swing_cart max_abs:", float(tau_swing.abs().max().detach().cpu()))
    print("action max_abs:", float(action.abs().max().detach().cpu()))
    try:
        print("applied_torque max_abs:", float(robot.data.applied_torque.abs().max().detach().cpu()))
    except Exception:
        pass
    print("=" * 132 + "\n")


def main():
    env_cfg = CartoEffortEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    patch_flat_safe_env(env_cfg)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()
    robot = env.scene["robot"]
    print_robot_debug_info(robot)

    device, dtype = robot.data.joint_pos.device, robot.data.joint_pos.dtype
    foot_indices = get_foot_indices(robot)
    mass = get_mass(robot)
    q_nom = robot.data.joint_pos.detach().clone()

    cfg = SupportRegionRefConfig(
        alpha=args.alpha,
        margin=args.margin,
        max_shift_per_step=args.max_shift_per_step,
        height_ref=args.height_ref,
    )

    total_steps = args.warmup_steps + args.shift_steps + args.lift_steps + args.hold_steps + args.lower_steps
    prev_base_ref = None
    foot0 = None
    foot_shift0 = None

    print("\n" + "=" * 132)
    print("[INFO] Starting B8-s Cartesian swing-force probe")
    print("test_leg:", args.test_leg, "mass:", mass, "cfg:", cfg)
    print("swing force:", args.swing_fx, args.swing_fy, args.swing_fz, "swing_cart_sign:", args.swing_cart_sign)
    print("=" * 132)

    for step in range(total_steps):
        if not simulation_app.is_running():
            break

        phase, raw_profile = get_phase(step)
        x_hat = make_x_hat(robot, velocity_frame="world")
        foot_pos = robot.data.body_pos_w[:, foot_indices, :]

        if foot0 is None:
            foot0 = foot_pos.detach().clone()
        if step == args.warmup_steps + args.shift_steps:
            foot_shift0 = foot_pos.detach().clone()

        stance_mask = compute_stance_mask(args.num_envs, device, dtype, phase)
        base_ref, out = make_base_ref(phase, x_hat, foot_pos, stance_mask, prev_base_ref, cfg)
        prev_base_ref = base_ref.detach().clone()

        lift_enabled = safe_to_lift(x_hat, out)
        profile = raw_profile
        if phase in ["lift", "hold_lift", "lower"] and not bool(lift_enabled[0].detach().cpu()):
            profile = 0.0

        foot_forces, wrench = compute_foot_forces(x_hat, base_ref, foot_pos, stance_mask, mass)
        jv_feet = get_jacobian_feet(robot, foot_indices)

        tau_force = args.tau_sign * args.tau_force_scale * jt_force_to_tau(jv_feet, foot_forces)
        swing_active = args.test_leg if phase in ["lift", "hold_lift", "lower"] else None
        tau_post = args.tau_posture_scale * posture_tau(robot, q_nom, swing_leg=swing_active)
        tau_swing = args.tau_swing_scale * swing_cart_tau(jv_feet, profile, args.num_envs, device, dtype)

        action = torch.clamp(tau_force + tau_post + tau_swing, -args.max_total_tau, args.max_total_tau)
        env.step(action)

        if step % max(args.print_every, 1) == 0:
            print_debug(step, phase, profile, x_hat, base_ref, out, foot_pos, foot0, foot_shift0, foot_forces, wrench, tau_force, tau_post, tau_swing, action, robot, lift_enabled)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
