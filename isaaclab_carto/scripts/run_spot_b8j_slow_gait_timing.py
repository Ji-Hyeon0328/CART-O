# isaaclab_carto/scripts/run_spot_b8j_slow_gait_timing.py
#
# B8-j: slow gait timing / longer swing window.
#
# Motivation:
#   B8-i finally produced visible schedule-driven leg attempts, but the foot
#   tended to drag the toe on the ground. The logs showed:
#
#       T    ~= 0.746 s
#       duty ~= 0.802
#       swing_time ~= T * (1-duty) ~= 0.148 s
#
#   That is a very short swing window. B8-j overrides T and duty after
#   theta_decoder to create a slower, longer swing phase.
#
# Goal:
#   Keep balance-augmented WBC + hybrid swing joint residual, but slow the gait:
#
#       T_override    ~= 1.10 ~ 1.30 s
#       duty_override ~= 0.60 ~ 0.68
#
#   This should turn short "wagging" into slower in-place stepping attempts.
#
# This is still not final walking. It is a timing/clearance diagnostic.

import os
import sys
import argparse
from typing import Any, Dict

import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="CARTO/TRACER B8-j slow gait timing debug")

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle_steps", type=int, default=120)
parser.add_argument("--probe_steps", type=int, default=300)
parser.add_argument("--print_every", type=int, default=20)

parser.add_argument("--z_mode", type=str, default="conservative", choices=["conservative", "aggressive"])
parser.add_argument("--vx", type=float, default=0.0)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--beta_preset", type=str, default="balanced", choices=["height", "velocity", "energy", "balanced"])

# Timing overrides
parser.add_argument("--T_override", type=float, default=1.20)
parser.add_argument("--duty_override", type=float, default=0.65)
parser.add_argument("--phase_mode", type=str, default="crawl", choices=["crawl", "trot"])
parser.add_argument("--freeze_base_x_during_probe", action="store_true")

# Reduced implicit PD
parser.add_argument("--pd_scale", type=float, default=0.50)
parser.add_argument("--stiffness_override", type=float, default=None)
parser.add_argument("--damping_override", type=float, default=None)

# forceMPC
parser.add_argument("--h_ref", type=float, default=0.67)
parser.add_argument("--mass", type=float, default=32.5)
parser.add_argument("--gravity", type=float, default=9.81)
parser.add_argument("--residual_ratio", type=float, default=0.04)
parser.add_argument("--target_scale", type=float, default=1.0)
parser.add_argument("--mu", type=float, default=0.6)
parser.add_argument("--fz_min", type=float, default=0.0)
parser.add_argument("--fz_max", type=float, default=8.0)
parser.add_argument("--fxy_abs_max", type=float, default=3.0)
parser.add_argument("--max_delta_f", type=float, default=1.0)
parser.add_argument("--smoothing_alpha", type=float, default=0.70)
parser.add_argument("--pg_iters", type=int, default=40)
parser.add_argument("--pg_step_size", type=float, default=0.06)
parser.add_argument("--pg_init", type=str, default="prev", choices=["zero", "prev"])

# balance augmentation
parser.add_argument("--h_ref_balance", type=float, default=0.67)
parser.add_argument("--pitch_ref", type=float, default=0.0)
parser.add_argument("--kp_h_balance", type=float, default=180.0)
parser.add_argument("--kd_h_balance", type=float, default=30.0)
parser.add_argument("--kp_pitch_balance", type=float, default=55.0)
parser.add_argument("--kd_pitch_balance", type=float, default=10.0)
parser.add_argument("--max_extra_fz_per_leg", type=float, default=10.0)
parser.add_argument("--max_remove_fz_per_leg", type=float, default=5.0)
parser.add_argument("--max_pitch_moment", type=float, default=12.0)
parser.add_argument("--front_unload_gain", type=float, default=0.0)
parser.add_argument("--balance_scale", type=float, default=1.0)

# QPS-WBC bridge
parser.add_argument("--linear_rows", type=str, default="0_3", choices=["0_3", "3_6"])
parser.add_argument("--force_sign", type=float, default=1.0, choices=[1.0, -1.0])
parser.add_argument("--w_force", type=float, default=1.1)
parser.add_argument("--w_swing", type=float, default=0.08)
parser.add_argument("--w_posture", type=float, default=0.60)
parser.add_argument("--w_reg", type=float, default=0.05)
parser.add_argument("--w_rate", type=float, default=0.35)

parser.add_argument("--kp_swing_xy", type=float, default=6.0)
parser.add_argument("--kp_swing_z", type=float, default=12.0)
parser.add_argument("--kd_swing_xy", type=float, default=1.0)
parser.add_argument("--kd_swing_z", type=float, default=1.5)
parser.add_argument("--max_task_cmd", type=float, default=1.2)
parser.add_argument("--max_pos_err", type=float, default=0.025)

parser.add_argument("--kp_posture", type=float, default=1.4)
parser.add_argument("--kd_posture", type=float, default=0.07)
parser.add_argument("--max_posture_tau", type=float, default=0.35)

# WBC authority
parser.add_argument("--tau_probe_scale", type=float, default=4.0)
parser.add_argument("--max_wbc_tau", type=float, default=8.0)

# Hybrid explicit swing joint torque
parser.add_argument("--enable_hybrid_swing", action="store_true")
parser.add_argument("--swing_joint_scale", type=float, default=1.2)
parser.add_argument("--kp_swing_joint", type=float, default=18.0)
parser.add_argument("--kd_swing_joint", type=float, default=1.0)
parser.add_argument("--max_swing_joint_tau", type=float, default=3.5)
parser.add_argument("--hy_lift_delta", type=float, default=-0.08)
parser.add_argument("--kn_lift_delta", type=float, default=0.20)
parser.add_argument("--hy_sweep_delta", type=float, default=0.002)
parser.add_argument("--lift_sign", type=float, default=1.0, choices=[1.0, -1.0])
parser.add_argument("--knee_sign", type=float, default=1.0, choices=[1.0, -1.0])

parser.add_argument("--max_total_tau", type=float, default=10.0)

# Settling
parser.add_argument("--settle_mode", type=str, default="stance", choices=["zero", "stance"])
parser.add_argument("--settle_residual_ratio", type=float, default=0.01)
parser.add_argument("--settle_max_tau", type=float, default=0.5)

parser.add_argument("--ref_k", type=int, default=0)
parser.add_argument("--min_stance_legs", type=int, default=2)
parser.add_argument("--control_dt", type=float, default=0.02)
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

from isaaclab_carto.envs.carto_effort_env_cfg import CartoEffortEnvCfg  # noqa: E402
from isaaclab_carto.lowlevel.spot_state import make_x_hat, build_spot_ref_params, print_robot_debug_info  # noqa: E402
from isaaclab_carto.lowlevel.theta_decoder import theta_decoder  # noqa: E402
from isaaclab_carto.lowlevel.theta_ref_mapper import theta_ref_mapper  # noqa: E402
from isaaclab_carto.lowlevel.effort_control import summarize_torque  # noqa: E402
from isaaclab_carto.lowlevel.force_mpc_objectives import make_beta_preset  # noqa: E402
from isaaclab_carto.lowlevel.projected_gradient_force_mpc_v2 import PGForceMPCV2State, plan_pg_force_mpc_v2  # noqa: E402
from isaaclab_carto.lowlevel.qps_wbc_bridge import QPSWBCBridgeState  # noqa: E402
from isaaclab_carto.lowlevel.balance_augmented_wbc import make_balance_augmented_qps_wbc_torque  # noqa: E402
from isaaclab_carto.lowlevel.support_force_control import extract_foot_jacobians_action_order, compute_tau_jtf  # noqa: E402


FOOT_NAMES = ["fl_foot", "fr_foot", "hl_foot", "hr_foot"]
HY_IDX = torch.tensor([4, 5, 6, 7], dtype=torch.long)
KN_IDX = torch.tensor([8, 9, 10, 11], dtype=torch.long)


def scale_number_or_mapping(value, scale: float, override):
    if override is not None:
        if isinstance(value, dict):
            return {k: override for k in value.keys()}
        return override
    if isinstance(value, (int, float)):
        return value * scale
    if isinstance(value, dict):
        return {k: scale_number_or_mapping(v, scale, override) for k, v in value.items()}
    return value


def patch_reduced_implicit_pd(env_cfg: Any) -> None:
    robot_cfg = env_cfg.scene.robot
    actuators = getattr(robot_cfg, "actuators", None)

    print("\n[PD PATCH]")
    print("pd_scale:", args.pd_scale)
    print("stiffness_override:", args.stiffness_override)
    print("damping_override:", args.damping_override)

    if actuators is None:
        print("[PD PATCH][WARN] robot cfg has no actuators attribute.")
        return

    items = actuators.items() if isinstance(actuators, dict) else [(str(i), a) for i, a in enumerate(actuators)]

    for name, act in items:
        print(f"[PD PATCH] actuator={name}")
        if hasattr(act, "stiffness"):
            old = getattr(act, "stiffness")
            new = scale_number_or_mapping(old, args.pd_scale, args.stiffness_override)
            setattr(act, "stiffness", new)
            print("  stiffness:", old, "->", new)
        if hasattr(act, "damping"):
            old = getattr(act, "damping")
            new = scale_number_or_mapping(old, args.pd_scale, args.damping_override)
            setattr(act, "damping", new)
            print("  damping  :", old, "->", new)


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


def make_fixed_command(num_envs: int, device, dtype) -> torch.Tensor:
    u_cmd = torch.zeros((num_envs, 3), device=device, dtype=dtype)
    u_cmd[:, 0] = args.vx
    u_cmd[:, 1] = args.vy
    u_cmd[:, 2] = args.wz
    return u_cmd


def make_fake_highlevel(num_envs: int, device, dtype):
    z_value = 0.0 if args.z_mode == "conservative" else 1.0
    z_t = torch.full((num_envs,), z_value, device=device, dtype=dtype)
    a_hl = torch.tensor([0.15, -0.25, 0.20, -0.10], device=device, dtype=dtype)
    return z_t, a_hl.unsqueeze(0).repeat(num_envs, 1)


def apply_timing_override(theta, device, dtype):
    N = theta.gait["T"].shape[0]
    theta.gait["T"] = torch.full((N,), args.T_override, device=device, dtype=dtype)
    duty = torch.full((N, 4), args.duty_override, device=device, dtype=dtype)
    theta.gait["duty_i"] = duty

    if args.phase_mode == "crawl":
        phases = torch.tensor([0.00, 0.25, 0.50, 0.75], device=device, dtype=dtype)
    else:
        phases = torch.tensor([0.00, 0.50, 0.50, 0.00], device=device, dtype=dtype)
    theta.gait["phase_i"] = phases.unsqueeze(0).repeat(N, 1)
    return theta


def advance_theta_phase_with_clock(theta, step: int, dt: float):
    T = theta.gait["T"]
    phase_i = theta.gait["phase_i"]
    phase_advance = (step * dt) / torch.clamp(T, min=1e-6)
    theta.gait["phase_i"] = torch.remainder(phase_i + phase_advance.unsqueeze(1), 1.0)
    return theta


def get_foot_indices(robot):
    name_to_idx = {name: i for i, name in enumerate(robot.body_names)}
    return [name_to_idx[name] for name in FOOT_NAMES]


def build_theta_ref(robot, params, step: int, device, dtype):
    x_hat = make_x_hat(robot, velocity_frame="world")
    u_cmd = make_fixed_command(robot.data.joint_pos.shape[0], device=device, dtype=dtype)
    z_t, a_hl = make_fake_highlevel(robot.data.joint_pos.shape[0], device=device, dtype=dtype)
    beta_t = make_beta_preset(robot.data.joint_pos.shape[0], device=device, dtype=dtype, preset=args.beta_preset)

    theta = theta_decoder(z_t=z_t, a_HL=a_hl, x_hat=x_hat, u_cmd=u_cmd, robot_name="spot")
    theta = apply_timing_override(theta, device=device, dtype=dtype)
    theta = advance_theta_phase_with_clock(theta, step=step, dt=args.control_dt)
    ref = theta_ref_mapper(theta=theta, x_hat=x_hat, u_cmd=u_cmd, params=params)
    return x_hat, u_cmd, beta_t, theta, ref


def make_stance_settle_tau(robot, ref, x_hat, u_cmd, beta_t, state):
    f_ref, _info = plan_pg_force_mpc_v2(
        robot=robot,
        ref=ref,
        x_hat=x_hat,
        u_cmd=u_cmd,
        beta_t=beta_t,
        planner_state=state,
        h_ref=args.h_ref,
        mass=args.mass,
        gravity=args.gravity,
        residual_ratio=args.settle_residual_ratio,
        target_scale=1.0,
        mu=args.mu,
        fz_min=args.fz_min,
        fz_max=args.fz_max,
        fxy_abs_max=args.fxy_abs_max,
        max_delta_f=args.max_delta_f,
        smoothing_alpha=args.smoothing_alpha,
        pg_iters=args.pg_iters,
        pg_step_size=args.pg_step_size,
        pg_init=args.pg_init,
        use_k=args.ref_k,
        min_stance_legs=args.min_stance_legs,
        force_sign=args.force_sign,
    )
    Jv_feet, _ = extract_foot_jacobians_action_order(robot=robot, linear_rows=args.linear_rows)
    tau = compute_tau_jtf(Jv_feet, f_ref)
    return torch.clamp(tau, -args.settle_max_tau, args.settle_max_tau)


def compute_swing_phase(ref: Dict[str, torch.Tensor], theta, k: int):
    S = ref["S"]
    H = S.shape[-1]
    k = min(max(k, 0), H - 1)

    phase = ref["phase"][:, :, k]
    duty = theta.gait["duty_i"]

    swing_mask = (S[:, :, k] < 0.5).to(S.dtype)
    denom = torch.clamp(1.0 - duty, min=1e-5)
    progress = torch.clamp((phase - duty) / denom, min=0.0, max=1.0) * swing_mask
    lift = torch.sin(torch.pi * progress) * swing_mask
    sweep = (2.0 * progress - 1.0) * swing_mask
    return swing_mask, progress, lift, sweep


def make_hybrid_swing_joint_torque(robot, ref, theta, q_swing_base, k: int):
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype
    N = robot.data.joint_pos.shape[0]

    q = robot.data.joint_pos
    dq = robot.data.joint_vel

    hy_idx = HY_IDX.to(device=device)
    kn_idx = KN_IDX.to(device=device)

    swing_mask, progress, lift, sweep = compute_swing_phase(ref, theta, k=k)

    hy_offset = args.lift_sign * args.hy_lift_delta * lift + args.hy_sweep_delta * sweep
    kn_offset = args.knee_sign * args.kn_lift_delta * lift

    q_des = q_swing_base.clone()
    q_des[:, hy_idx] = q_swing_base[:, hy_idx] + hy_offset
    q_des[:, kn_idx] = q_swing_base[:, kn_idx] + kn_offset

    tau = torch.zeros((N, 12), device=device, dtype=dtype)
    q_err = q_des - q

    tau[:, hy_idx] = args.kp_swing_joint * q_err[:, hy_idx] - args.kd_swing_joint * dq[:, hy_idx]
    tau[:, kn_idx] = args.kp_swing_joint * q_err[:, kn_idx] - args.kd_swing_joint * dq[:, kn_idx]

    tau[:, hy_idx] = tau[:, hy_idx] * swing_mask
    tau[:, kn_idx] = tau[:, kn_idx] * swing_mask
    tau = torch.clamp(tau, -args.max_swing_joint_tau, args.max_swing_joint_tau)

    info = {
        "swing_joint_enabled": bool(args.enable_hybrid_swing),
        "swing_mask_env0": swing_mask[0].detach().cpu().tolist(),
        "swing_progress_env0": progress[0].detach().cpu().tolist(),
        "lift_profile_env0": lift[0].detach().cpu().tolist(),
        "hy_offset_env0": hy_offset[0].detach().cpu().tolist(),
        "kn_offset_env0": kn_offset[0].detach().cpu().tolist(),
        "tau_swing_joint_mean_abs": float(tau.abs().mean().detach().cpu()),
        "tau_swing_joint_max_abs": float(tau.abs().max().detach().cpu()),
        "swing_joint_scale": args.swing_joint_scale,
        "kp_swing_joint": args.kp_swing_joint,
        "max_swing_joint_tau": args.max_swing_joint_tau,
        "T_override": args.T_override,
        "duty_override": args.duty_override,
        "swing_time_sec": args.T_override * (1.0 - args.duty_override),
    }
    return tau, info


def compute_swing_leg_delta(foot_delta, swing_mask):
    device = foot_delta.device
    mask = torch.tensor(swing_mask, device=device, dtype=foot_delta.dtype).view(1, 4, 1)
    masked = foot_delta.abs() * mask
    return float(masked.max().detach().cpu()), float(masked.mean().detach().cpu())


def print_probe_debug(
    probe_step, robot, u_cmd, beta_t, theta, ref, tau_wbc, tau_swing_joint, tau_cmd,
    wbc_info, swing_info, q_settle, foot_settle, base_settle
):
    env_id = 0
    foot_indices = get_foot_indices(robot)

    x_hat_after = make_x_hat(robot, velocity_frame="world")
    q = robot.data.joint_pos
    foot = robot.data.body_pos_w[:, foot_indices, :]

    q_delta_settle = q - q_settle
    foot_delta_settle = foot - foot_settle
    base_delta_settle = x_hat_after[:, 0:6] - base_settle

    swing_mask = swing_info.get("swing_mask_env0", [0.0, 0.0, 0.0, 0.0])
    swing_delta_max, swing_delta_mean = compute_swing_leg_delta(foot_delta_settle, swing_mask)

    print("\n" + "=" * 132)
    print(f"[B8-j SLOW GAIT TIMING] probe_step={probe_step}")
    print("=" * 132)

    print("[x_hat after env.step env0]")
    print("pos xyz      :", x_hat_after[env_id, 0:3].detach().cpu().numpy())
    print("rpy          :", x_hat_after[env_id, 3:6].detach().cpu().numpy())
    print("lin vel      :", x_hat_after[env_id, 6:9].detach().cpu().numpy())
    print("ang vel      :", x_hat_after[env_id, 9:12].detach().cpu().numpy())

    print("\n[delta from SETTLED baseline env0]")
    print("base_delta xyz+rpy:", base_delta_settle[env_id].detach().cpu().numpy())
    print("q_delta max_abs:", float(q_delta_settle.abs().max().detach().cpu()))
    print("foot_delta max_abs:", float(foot_delta_settle.abs().max().detach().cpu()))
    print("swing_foot_delta max_abs:", swing_delta_max)
    print("swing_foot_delta mean_abs:", swing_delta_mean)
    print("foot_delta env0:", foot_delta_settle[env_id].detach().cpu().numpy())

    print("\n[Theta / Ref env0]")
    print("T            :", float(theta.gait["T"][env_id].detach().cpu()))
    print("duty_i       :", theta.gait["duty_i"][env_id].detach().cpu().numpy())
    print("swing_time   :", float(theta.gait["T"][env_id].detach().cpu()) * (1.0 - float(theta.gait["duty_i"][env_id, 0].detach().cpu())))
    print("phase_i      :", theta.gait["phase_i"][env_id].detach().cpu().numpy())
    k = args.ref_k
    print("S[:,k]       :", ref["S"][env_id, :, k].detach().cpu().numpy())
    print("phase[:,k]   :", ref["phase"][env_id, :, k].detach().cpu().numpy())

    print("\n[balance augmentation]")
    for key in [
        "balance_h_env0", "balance_h_err_env0",
        "balance_pitch_env0", "balance_pitch_err_env0",
        "Fz_balance_env0", "My_balance_env0", "fz_bias_env0",
    ]:
        if key in wbc_info:
            print(f"{key}: {wbc_info[key]}")

    print("\n[hybrid swing joint residual]")
    for key in [
        "swing_joint_enabled", "swing_mask_env0", "swing_progress_env0",
        "lift_profile_env0", "hy_offset_env0", "kn_offset_env0",
        "tau_swing_joint_mean_abs", "tau_swing_joint_max_abs",
        "T_override", "duty_override", "swing_time_sec",
    ]:
        if key in swing_info:
            print(f"{key}: {swing_info[key]}")

    print("\n[torque command]")
    print("tau_wbc stats         :", summarize_torque(tau_wbc))
    print("tau_swing_joint stats :", summarize_torque(tau_swing_joint))
    print("tau_cmd stats         :", summarize_torque(tau_cmd))

    if hasattr(robot.data, "applied_torque"):
        applied = robot.data.applied_torque
        print("\n[applied_torque env0 after env.step]")
        print(applied[env_id].detach().cpu().numpy())
        print("applied stats:", summarize_torque(applied))
        denom = torch.clamp(tau_cmd.abs().mean(), min=1e-6)
        print("applied_mean_abs / cmd_mean_abs:", float(applied.abs().mean().detach().cpu() / denom.detach().cpu()))

    print("=" * 132 + "\n")


def main():
    env_cfg = CartoEffortEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    patch_flat_safe_env(env_cfg)
    patch_reduced_implicit_pd(env_cfg)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    _obs_dict, _ = env.reset()

    robot = env.scene["robot"]
    print_robot_debug_info(robot)

    dtype = robot.data.joint_pos.dtype
    device = robot.data.joint_pos.device
    params = build_spot_ref_params(device=device, dtype=dtype, dt=args.control_dt, horizon=20)

    foot_indices = get_foot_indices(robot)
    settle_force_state = PGForceMPCV2State()

    print("\n" + "=" * 132)
    print("[INFO] Starting B8-j slow gait timing")
    print("=" * 132)
    print(f"T_override={args.T_override}, duty_override={args.duty_override}, swing_time={args.T_override*(1-args.duty_override):.3f}s")
    print(f"pd_scale={args.pd_scale}, enable_hybrid_swing={args.enable_hybrid_swing}")
    print("=" * 132 + "\n")

    # Phase 1: settle.
    for step in range(args.settle_steps):
        if not simulation_app.is_running():
            break

        x_hat, u_cmd, beta_t, theta, ref = build_theta_ref(robot, params, step, device, dtype)

        if args.settle_mode == "zero":
            tau_settle = torch.zeros_like(robot.data.joint_pos)
        else:
            tau_settle = make_stance_settle_tau(robot, ref, x_hat, u_cmd, beta_t, settle_force_state)

        _obs_dict, _rewards, terminated, truncated, _extras = env.step(tau_settle)

        if step % max(args.print_every, 1) == 0:
            x_now = make_x_hat(robot, velocity_frame="world")
            print(f"[SETTLE] step={step} pos={x_now[0,0:3].detach().cpu().numpy()} rpy={x_now[0,3:6].detach().cpu().numpy()} tau={summarize_torque(tau_settle)}")

        if torch.any(terminated) or torch.any(truncated):
            print(f"[WARN] terminated/truncated during settle at step={step}")
            print("terminated:", terminated.detach().cpu().numpy())
            print("truncated :", truncated.detach().cpu().numpy())

    q_settle = robot.data.joint_pos.detach().clone()
    q_swing_base = q_settle.clone()
    foot_settle = robot.data.body_pos_w[:, foot_indices, :].detach().clone()
    base_settle = make_x_hat(robot, velocity_frame="world")[:, 0:6].detach().clone()

    print("\n" + "-" * 132)
    print("[INFO] Settled baseline saved under reduced implicit PD")
    print("base_settle env0:", base_settle[0].detach().cpu().numpy())
    print("q_settle env0   :", q_settle[0].detach().cpu().numpy())
    print("foot_settle env0:", foot_settle[0].detach().cpu().numpy())
    print("-" * 132 + "\n")

    force_state = PGForceMPCV2State()
    wbc_state = QPSWBCBridgeState()

    # Phase 2: probe.
    for probe_step in range(args.probe_steps):
        if not simulation_app.is_running():
            break

        x_hat, u_cmd, beta_t, theta, ref = build_theta_ref(robot, params, probe_step, device, dtype)

        if args.freeze_base_x_during_probe:
            u_cmd[:, 0] = 0.0

        f_ref, _force_info = plan_pg_force_mpc_v2(
            robot=robot,
            ref=ref,
            x_hat=x_hat,
            u_cmd=u_cmd,
            beta_t=beta_t,
            planner_state=force_state,
            h_ref=args.h_ref,
            mass=args.mass,
            gravity=args.gravity,
            residual_ratio=args.residual_ratio,
            target_scale=args.target_scale,
            mu=args.mu,
            fz_min=args.fz_min,
            fz_max=args.fz_max,
            fxy_abs_max=args.fxy_abs_max,
            max_delta_f=args.max_delta_f,
            smoothing_alpha=args.smoothing_alpha,
            pg_iters=args.pg_iters,
            pg_step_size=args.pg_step_size,
            pg_init=args.pg_init,
            use_k=args.ref_k,
            min_stance_legs=args.min_stance_legs,
            force_sign=args.force_sign,
        )

        tau_wbc_raw, wbc_info, _f_aug = make_balance_augmented_qps_wbc_torque(
            robot=robot,
            ref=ref,
            x_hat=x_hat,
            f_ref=f_ref,
            state=wbc_state,
            k=args.ref_k,
            linear_rows=args.linear_rows,
            h_ref_balance=args.h_ref_balance,
            pitch_ref=args.pitch_ref,
            kp_h_balance=args.kp_h_balance,
            kd_h_balance=args.kd_h_balance,
            kp_pitch_balance=args.kp_pitch_balance,
            kd_pitch_balance=args.kd_pitch_balance,
            max_extra_fz_per_leg=args.max_extra_fz_per_leg,
            max_remove_fz_per_leg=args.max_remove_fz_per_leg,
            max_pitch_moment=args.max_pitch_moment,
            front_unload_gain=args.front_unload_gain,
            balance_scale=args.balance_scale,
            kp_swing_xyz=(args.kp_swing_xy, args.kp_swing_xy, args.kp_swing_z),
            kd_swing_xyz=(args.kd_swing_xy, args.kd_swing_xy, args.kd_swing_z),
            max_task_cmd=args.max_task_cmd,
            max_pos_err=args.max_pos_err,
            kp_posture=args.kp_posture,
            kd_posture=args.kd_posture,
            max_posture_tau=args.max_posture_tau,
            w_force=args.w_force,
            w_swing=args.w_swing,
            w_posture=args.w_posture,
            w_reg=args.w_reg,
            w_rate=args.w_rate,
            max_total_tau=args.max_wbc_tau,
        )

        tau_wbc = torch.clamp(args.tau_probe_scale * tau_wbc_raw, -args.max_wbc_tau, args.max_wbc_tau)

        tau_swing_joint, swing_info = make_hybrid_swing_joint_torque(
            robot=robot, ref=ref, theta=theta, q_swing_base=q_swing_base, k=args.ref_k
        )

        if not args.enable_hybrid_swing:
            tau_swing_joint = torch.zeros_like(tau_wbc)

        tau_cmd = tau_wbc + args.swing_joint_scale * tau_swing_joint
        tau_cmd = torch.clamp(tau_cmd, -args.max_total_tau, args.max_total_tau)

        _obs_dict, _rewards, terminated, truncated, _extras = env.step(tau_cmd)

        if probe_step % args.print_every == 0:
            print_probe_debug(
                probe_step=probe_step,
                robot=robot,
                u_cmd=u_cmd,
                beta_t=beta_t,
                theta=theta,
                ref=ref,
                tau_wbc=tau_wbc,
                tau_swing_joint=tau_swing_joint,
                tau_cmd=tau_cmd,
                wbc_info=wbc_info,
                swing_info=swing_info,
                q_settle=q_settle,
                foot_settle=foot_settle,
                base_settle=base_settle,
            )

        if torch.any(terminated) or torch.any(truncated):
            print(f"[WARN] terminated/truncated during probe at probe_step={probe_step}")
            print("terminated:", terminated.detach().cpu().numpy())
            print("truncated :", truncated.detach().cpu().numpy())

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
