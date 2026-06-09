# isaaclab_carto/scripts/run_spot_b8af_wbc_actuator_target_alignment.py
# B8-af: Full-WBC + actuator target alignment.
# Keeps B8-ad full-WBC torque control, but aligns Isaac implicit actuator targets
# so implicit PD does not fight the WBC torque. During swing, optional small
# HY/KN target assist is applied to the selected swing leg as an actuator-layer probe.

import os
import sys
import argparse
from typing import Any

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CARTO/TRACER B8-af WBC actuator target alignment")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--warmup_steps", type=int, default=120)
parser.add_argument("--shift_steps", type=int, default=220)
parser.add_argument("--lift_steps", type=int, default=160)
parser.add_argument("--hold_steps", type=int, default=100)
parser.add_argument("--lower_steps", type=int, default=120)
parser.add_argument("--print_every", type=int, default=20)
parser.add_argument("--test_leg", type=str, default="RF", choices=["LF", "RF", "LH", "RH"])

# Env / actuator
parser.add_argument("--spawn_z", type=float, default=0.60)
parser.add_argument("--pd_scale", type=float, default=0.35)

# Reference
parser.add_argument("--height_ref", type=float, default=0.625)
parser.add_argument("--alpha", type=float, default=0.12)
parser.add_argument("--margin", type=float, default=0.030)
parser.add_argument("--max_shift_per_step", type=float, default=0.0005)

# MPC GRF
parser.add_argument("--kp_xy", type=float, default=20.0)
parser.add_argument("--kd_xy", type=float, default=14.0)
parser.add_argument("--kp_z", type=float, default=180.0)
parser.add_argument("--kd_z", type=float, default=35.0)
parser.add_argument("--mass_override", type=float, default=32.0)
parser.add_argument("--mu", type=float, default=0.70)
parser.add_argument("--min_fz", type=float, default=8.0)
parser.add_argument("--max_fz", type=float, default=180.0)

# Full WBC QP v1 weights
parser.add_argument("--w_dyn", type=float, default=25.0)
parser.add_argument("--w_base_acc", type=float, default=5.0)
parser.add_argument("--w_stance_acc", type=float, default=30.0)
parser.add_argument("--w_swing_acc", type=float, default=45.0)
parser.add_argument("--w_force_track", type=float, default=1.0)
parser.add_argument("--w_swing_force_zero", type=float, default=80.0)
parser.add_argument("--w_tau_posture", type=float, default=0.08)
parser.add_argument("--w_tau_reg", type=float, default=0.03)
parser.add_argument("--w_qdd_reg", type=float, default=0.01)

# Swing task
parser.add_argument("--swing_clearance", type=float, default=0.060)
parser.add_argument("--kp_swing_z", type=float, default=160.0)
parser.add_argument("--kd_swing_z", type=float, default=18.0)
parser.add_argument("--max_swing_acc", type=float, default=12.0)

# Torque
parser.add_argument("--max_tau", type=float, default=24.0)
parser.add_argument("--tau_output_sign", type=float, default=1.0, choices=[1.0, -1.0])
parser.add_argument("--tau_cmd_scale", type=float, default=1.0)

# Implicit target alignment
parser.add_argument("--align_targets", action="store_true")
parser.add_argument("--target_mode", type=str, default="current", choices=["current", "nominal"])
parser.add_argument("--target_vel_zero", action="store_true")
parser.add_argument("--swing_target_assist", action="store_true")
parser.add_argument("--swing_hy_bias", type=float, default=-0.08)
parser.add_argument("--swing_kn_bias", type=float, default=-0.16)
parser.add_argument("--target_max_delta", type=float, default=0.25)

# Safety gate
parser.add_argument("--require_margin", action="store_true")
parser.add_argument("--max_pitch_for_swing", type=float, default=0.25)
parser.add_argument("--max_roll_for_swing", type=float, default=0.20)

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
from isaaclab_carto.lowlevel.tracer_mpc_wbc_bridge import MpcWbcBridgeConfig, distribute_grf_ls  # noqa: E402
from isaaclab_carto.lowlevel.tracer_full_wbc_qp_v1 import FullWbcQpV1Config, solve_full_wbc_qp_v1  # noqa: E402

FOOT_NAMES = ["fl_foot", "fr_foot", "hl_foot", "hr_foot"]
LEG_TO_ID = {"LF": 0, "RF": 1, "LH": 2, "RH": 3}
HX = [0, 1, 2, 3]
HY = [4, 5, 6, 7]
KN = [8, 9, 10, 11]

@configclass
class CartoEffortActionsCfg:
    joint_effort = JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0)

@configclass
class CartoEffortEnvCfg(CartoEnvCfg):
    actions: CartoEffortActionsCfg = CartoEffortActionsCfg()

def smooth01(s):
    s = max(0.0, min(1.0, s))
    return float(0.5 - 0.5 * torch.cos(torch.tensor(torch.pi * s)).item())

def patch_flat_safe_env(env_cfg: Any):
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
    except Exception as exc:
        print(f"[WARN] terrain patch failed: {exc}")
    try:
        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, args.spawn_z)
    except Exception:
        pass
    try:
        for actuator in env_cfg.scene.robot.actuators.values():
            actuator.stiffness = float(actuator.stiffness) * args.pd_scale
            actuator.damping = float(actuator.damping) * args.pd_scale
        print(f"[INFO] Scaled implicit actuator stiffness/damping by {args.pd_scale}")
    except Exception as exc:
        print(f"[WARN] actuator scale failed: {exc}")

def get_foot_indices(robot):
    name_to_idx = {name: i for i, name in enumerate(robot.body_names)}
    return [name_to_idx[name] for name in FOOT_NAMES]

def get_mass(robot):
    try:
        return float(robot.root_physx_view.get_masses().sum().detach().cpu())
    except Exception:
        return args.mass_override

def get_phase(step):
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

def stance_mask_for_phase(num_envs, device, dtype, phase, swing_enabled=True):
    stance = torch.ones((num_envs, 4), device=device, dtype=dtype)
    if phase in ["shift", "lift", "hold_lift", "lower"] and swing_enabled:
        stance[:, LEG_TO_ID[args.test_leg]] = 0.0
    return stance

def make_base_ref(phase, x_hat, foot_pos, stance_mask, prev_base_ref, cfg):
    out = compute_support_region_ref(
        foot_pos_w=foot_pos,
        base_pos_w=x_hat[:, 0:3],
        base_rpy_w=x_hat[:, 3:6],
        stance_mask=stance_mask,
        prev_base_ref=prev_base_ref,
        cfg=cfg,
    )
    base_ref = out.base_ref.detach().clone()
    if phase == "warmup":
        base_ref[:, 0:2] = x_hat[:, 0:2]
    base_ref[:, 2] = args.height_ref
    base_ref[:, 3:5] = 0.0
    return base_ref, out

def safe_to_swing(x_hat, out):
    ok = (torch.abs(x_hat[:, 3]) < args.max_roll_for_swing) & (torch.abs(x_hat[:, 4]) < args.max_pitch_for_swing)
    if args.require_margin:
        ok = ok & out.swing_allowed
    return ok

def make_swing_target(phase, profile, foot_pos, foot_swing0, swing_enabled):
    target = foot_pos.detach().clone()
    if foot_swing0 is None:
        return target
    leg = LEG_TO_ID[args.test_leg]
    target[:, leg, :] = foot_swing0[:, leg, :]
    if phase in ["lift", "hold_lift", "lower"] and bool(swing_enabled[0].detach().cpu()):
        target[:, leg, 2] = foot_swing0[:, leg, 2] + args.swing_clearance * profile
    return target

def get_foot_vel(robot, foot_indices):
    try:
        return robot.data.body_lin_vel_w[:, foot_indices, :]
    except Exception:
        return torch.zeros_like(robot.data.body_pos_w[:, foot_indices, :])

def get_gravity(robot):
    try:
        return robot.root_physx_view.get_generalized_gravity_forces()
    except Exception:
        return None

def get_coriolis(robot):
    try:
        return robot.root_physx_view.get_coriolis_and_centrifugal_forces()
    except Exception:
        return None

def apply_implicit_target_alignment(robot, q_nom, phase, profile, swing_enabled):
    if not args.align_targets:
        return None
    q = robot.data.joint_pos.detach()
    q_target = q_nom.detach().clone() if args.target_mode == "nominal" else q.detach().clone()
    leg = LEG_TO_ID[args.test_leg]
    if args.swing_target_assist and phase in ["lift", "hold_lift", "lower"] and bool(swing_enabled[0].detach().cpu()):
        q_target[:, HY[leg]] += args.swing_hy_bias * profile
        q_target[:, KN[leg]] += args.swing_kn_bias * profile
    delta = torch.clamp(q_target - q, -args.target_max_delta, args.target_max_delta)
    q_target = q + delta
    try:
        robot.set_joint_position_target(q_target)
    except Exception as exc:
        print(f"[WARN] set_joint_position_target failed: {exc}")
        return None
    if args.target_vel_zero:
        try:
            robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel))
        except Exception as exc:
            print(f"[WARN] set_joint_velocity_target failed: {exc}")
    return q_target

def print_debug(step, phase, profile, x_hat, base_ref, out, stance_mask, f_mpc, foot_pos, foot_swing0, swing_target, tau_cmd, qpd, robot, swing_enabled, q_target):
    leg = LEG_TO_ID[args.test_leg]
    idx = [HX[leg], HY[leg], KN[leg]]
    q = robot.data.joint_pos
    qd = robot.data.joint_vel
    applied = robot.data.applied_torque
    residual = applied - tau_cmd
    ratio = applied.abs() / tau_cmd.abs().clamp_min(1.0)
    foot_delta_swing = foot_pos - foot_swing0 if foot_swing0 is not None else torch.zeros_like(foot_pos)
    target_err = swing_target - foot_pos
    print("\n" + "=" * 140)
    print(f"[B8-af WBC-ACTUATOR-TARGET-ALIGNMENT] step={step}")
    print("=" * 140)
    print("phase:", phase, "test_leg:", args.test_leg, "profile:", profile, "swing_enabled:", bool(swing_enabled[0].detach().cpu()))
    print("height_ref:", args.height_ref, "pd_scale:", args.pd_scale, "align_targets:", args.align_targets, "target_mode:", args.target_mode)
    print("swing_target_assist:", args.swing_target_assist, "swing_hy_bias:", args.swing_hy_bias, "swing_kn_bias:", args.swing_kn_bias)
    print("stance_mask LF/RF/LH/RH:", stance_mask[0].detach().cpu().numpy())
    print("margin_to_edge:", float(out.margin_to_edge[0].detach().cpu()), "swing_allowed:", bool(out.swing_allowed[0].detach().cpu()))
    print("\n[base]")
    print("base pos xyz:", x_hat[0, 0:3].detach().cpu().numpy())
    print("base rpy:", x_hat[0, 3:6].detach().cpu().numpy())
    print("base ref:", base_ref[0].detach().cpu().numpy())
    print("\n[MPC/WBC]")
    print("f_mpc:"); print(f_mpc[0].detach().cpu().numpy())
    print("f_qp:"); print(qpd["f_qp"][0].detach().cpu().numpy())
    print("swing_leg_f_qp:", qpd["f_qp"][0, leg].detach().cpu().numpy())
    print("swing_acc_des selected:", qpd["swing_acc_des"][0, leg].detach().cpu().numpy())
    print("qdd_full selected leg [hx,hy,kn]:", qpd["qdd_full"][0, [6 + leg, 6 + leg + 4, 6 + leg + 8]].detach().cpu().numpy())
    print("h_full base:", qpd["h_full"][0, 0:6].detach().cpu().numpy())
    print("residual_norm:", float(qpd["residual_norm"][0].detach().cpu()))
    print("\n[implicit target alignment: selected leg hx/hy/kn]")
    if q_target is not None:
        print("q_target:", q_target[0, idx].detach().cpu().numpy())
        print("q_target_minus_q:", (q_target - q)[0, idx].detach().cpu().numpy())
    else:
        print("q_target: None")
    print("\n[actuator authority: selected leg hx/hy/kn]")
    print("joint q:", q[0, idx].detach().cpu().numpy())
    print("joint qd:", qd[0, idx].detach().cpu().numpy())
    print("tau_cmd:", tau_cmd[0, idx].detach().cpu().numpy())
    print("applied_torque:", applied[0, idx].detach().cpu().numpy())
    print("applied_minus_cmd:", residual[0, idx].detach().cpu().numpy())
    print("abs(applied)/max(abs(cmd),1):", ratio[0, idx].detach().cpu().numpy())
    print("all tau_cmd max_abs:", float(tau_cmd.abs().max().detach().cpu()))
    print("all applied_torque max_abs:", float(applied.abs().max().detach().cpu()))
    print("all residual max_abs:", float(residual.abs().max().detach().cpu()))
    print("\n[swing foot]")
    print("swing target:", swing_target[0, leg].detach().cpu().numpy())
    print("foot pos:", foot_pos[0, leg].detach().cpu().numpy())
    print("foot target error:", target_err[0, leg].detach().cpu().numpy())
    print("test foot_delta_from_swing_start:", foot_delta_swing[0, leg].detach().cpu().numpy())
    print("clearance_z_from_swing_start:", float(foot_delta_swing[0, leg, 2].detach().cpu()))
    print("=" * 140 + "\n")

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
    ref_cfg = SupportRegionRefConfig(alpha=args.alpha, margin=args.margin, max_shift_per_step=args.max_shift_per_step, height_ref=args.height_ref)
    mpc_cfg = MpcWbcBridgeConfig(kp_xy=args.kp_xy, kd_xy=args.kd_xy, kp_z=args.kp_z, kd_z=args.kd_z, mass=mass, mu=args.mu, min_fz=args.min_fz, max_fz=args.max_fz)
    wbc_cfg = FullWbcQpV1Config(mass=mass, w_dyn=args.w_dyn, w_base_acc=args.w_base_acc, w_stance_acc=args.w_stance_acc, w_swing_acc=args.w_swing_acc, w_force_track=args.w_force_track, w_swing_force_zero=args.w_swing_force_zero, w_tau_posture=args.w_tau_posture, w_tau_reg=args.w_tau_reg, w_qdd_reg=args.w_qdd_reg, kp_swing_z=args.kp_swing_z, kd_swing_z=args.kd_swing_z, max_swing_acc=args.max_swing_acc, mu=args.mu, max_fz=args.max_fz, max_tau=args.max_tau, tau_output_sign=args.tau_output_sign)
    q_nom = robot.data.joint_pos.detach().clone()
    prev_base_ref = None
    foot_swing0 = None
    swing_start_step = args.warmup_steps + args.shift_steps
    total_steps = args.warmup_steps + args.shift_steps + args.lift_steps + args.hold_steps + args.lower_steps
    print("\n" + "=" * 140)
    print("[INFO] Starting B8-af WBC actuator target alignment")
    print("test_leg:", args.test_leg, "mass:", mass, "pd_scale:", args.pd_scale, "height_ref:", args.height_ref)
    print("wbc_cfg:", wbc_cfg)
    print("=" * 140)
    for step in range(total_steps):
        if not simulation_app.is_running():
            break
        x_hat = make_x_hat(robot, velocity_frame="world")
        foot_pos = robot.data.body_pos_w[:, foot_indices, :]
        foot_vel = get_foot_vel(robot, foot_indices)
        M = robot.root_physx_view.get_generalized_mass_matrices()
        Jfeet_full = robot.root_physx_view.get_jacobians()[:, foot_indices, 0:3, :]
        gravity, coriolis = get_gravity(robot), get_coriolis(robot)
        phase, profile = get_phase(step)
        if step == swing_start_step:
            foot_swing0 = foot_pos.detach().clone()
        stance_trial = stance_mask_for_phase(args.num_envs, device, dtype, phase, True)
        base_ref, out = make_base_ref(phase, x_hat, foot_pos, stance_trial, prev_base_ref, ref_cfg)
        swing_enabled = safe_to_swing(x_hat, out)
        stance_mask = stance_mask_for_phase(args.num_envs, device, dtype, phase, bool(swing_enabled[0].detach().cpu()))
        if not torch.allclose(stance_mask, stance_trial):
            base_ref, out = make_base_ref(phase, x_hat, foot_pos, stance_mask, prev_base_ref, ref_cfg)
        prev_base_ref = base_ref.detach().clone()
        f_mpc, _ = distribute_grf_ls(x_hat[:, 0:3], x_hat[:, 3:6], x_hat[:, 6:9], x_hat[:, 9:12], base_ref, foot_pos, stance_mask, mpc_cfg)
        swing_target = make_swing_target(phase, profile, foot_pos, foot_swing0, swing_enabled)
        tau_cmd, qpd = solve_full_wbc_qp_v1(M, Jfeet_full, f_mpc, robot.data.joint_pos, robot.data.joint_vel, q_nom, x_hat, base_ref, foot_pos, foot_vel, swing_target, stance_mask, gravity, coriolis, wbc_cfg)
        tau_cmd = args.tau_cmd_scale * tau_cmd
        q_target = apply_implicit_target_alignment(robot, q_nom, phase, profile, swing_enabled)
        env.step(tau_cmd)
        if step % max(args.print_every, 1) == 0:
            print_debug(step, phase, profile, x_hat, base_ref, out, stance_mask, f_mpc, foot_pos, foot_swing0, swing_target, tau_cmd, qpd, robot, swing_enabled, q_target)
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
