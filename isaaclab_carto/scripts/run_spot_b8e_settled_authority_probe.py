# isaaclab_carto/scripts/run_spot_b8e_settled_authority_probe.py
#
# B8-e: settled authority probe.
#
# Why:
#   B8-d showed q_delta ~0.95 rad, but that includes spawn/settling motion.
#   This script separates:
#
#       spawn settling motion
#       vs
#       residual-controller-induced motion
#
# Procedure:
#   Phase 1: settle for N steps with zero residual effort
#   Phase 2: save settled q0 / foot0 / base0
#   Phase 3: apply QPS-WBC residual torque, optionally amplified
#
# This is not a walking controller.
# It is a diagnostic to answer:
#
#   "After the robot is already settled, does residual WBC torque actually
#    move joints/feet/base?"
#
# Interpretation:
#   If settled q_delta / foot_delta grow:
#       residual effort has real post-settle authority.
#   If settled q_delta / foot_delta stay tiny:
#       implicit actuator dominates; reduce implicit PD or use mixed action env.

import os
import sys
import argparse
from typing import Any

import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="CARTO/TRACER B8-e settled authority probe")

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle_steps", type=int, default=120)
parser.add_argument("--probe_steps", type=int, default=160)
parser.add_argument("--print_every", type=int, default=20)

parser.add_argument("--z_mode", type=str, default="conservative", choices=["conservative", "aggressive"])
parser.add_argument("--vx", type=float, default=0.05)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--beta_preset", type=str, default="balanced", choices=["height", "velocity", "energy", "balanced"])

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

# WBC bridge
parser.add_argument("--linear_rows", type=str, default="0_3", choices=["0_3", "3_6"])
parser.add_argument("--force_sign", type=float, default=1.0, choices=[1.0, -1.0])
parser.add_argument("--w_force", type=float, default=0.8)
parser.add_argument("--w_swing", type=float, default=0.35)
parser.add_argument("--w_posture", type=float, default=0.30)
parser.add_argument("--w_reg", type=float, default=0.02)
parser.add_argument("--w_rate", type=float, default=0.10)

parser.add_argument("--kp_swing_xy", type=float, default=25.0)
parser.add_argument("--kp_swing_z", type=float, default=45.0)
parser.add_argument("--kd_swing_xy", type=float, default=2.0)
parser.add_argument("--kd_swing_z", type=float, default=3.0)
parser.add_argument("--max_task_cmd", type=float, default=5.0)
parser.add_argument("--max_pos_err", type=float, default=0.06)

parser.add_argument("--kp_posture", type=float, default=0.8)
parser.add_argument("--kd_posture", type=float, default=0.04)
parser.add_argument("--max_posture_tau", type=float, default=0.20)

# Authority probe
parser.add_argument("--tau_probe_scale", type=float, default=5.0)
parser.add_argument("--max_total_tau", type=float, default=8.0)

# Settling
parser.add_argument("--settle_mode", type=str, default="zero", choices=["zero", "stance"])
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
from isaaclab_carto.lowlevel.qps_wbc_bridge import QPSWBCBridgeState, make_qps_wbc_bridge_torque  # noqa: E402
from isaaclab_carto.lowlevel.support_force_control import extract_foot_jacobians_action_order, compute_tau_jtf  # noqa: E402


FOOT_NAMES = ["fl_foot", "fr_foot", "hl_foot", "hr_foot"]


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


def print_probe_debug(
    probe_step,
    robot,
    u_cmd,
    beta_t,
    theta,
    ref,
    force_info,
    tau_raw,
    tau_cmd,
    wbc_info,
    q_settle,
    foot_settle,
    base_settle,
    q_spawn,
    foot_spawn,
    base_spawn,
):
    env_id = 0
    foot_indices = get_foot_indices(robot)

    x_hat_after = make_x_hat(robot, velocity_frame="world")
    q = robot.data.joint_pos
    foot = robot.data.body_pos_w[:, foot_indices, :]

    q_delta_settle = q - q_settle
    foot_delta_settle = foot - foot_settle
    base_delta_settle = x_hat_after[:, 0:6] - base_settle

    q_delta_spawn = q - q_spawn
    foot_delta_spawn = foot - foot_spawn
    base_delta_spawn = x_hat_after[:, 0:6] - base_spawn

    print("\n" + "=" * 132)
    print(f"[B8-e SETTLED AUTHORITY PROBE] probe_step={probe_step}")
    print("=" * 132)

    print("[x_hat after env.step env0]")
    print("pos xyz      :", x_hat_after[env_id, 0:3].detach().cpu().numpy())
    print("rpy          :", x_hat_after[env_id, 3:6].detach().cpu().numpy())
    print("lin vel      :", x_hat_after[env_id, 6:9].detach().cpu().numpy())
    print("ang vel      :", x_hat_after[env_id, 9:12].detach().cpu().numpy())

    print("\n[delta from SETTLED baseline env0]")
    print("base_delta xyz+rpy:", base_delta_settle[env_id].detach().cpu().numpy())
    print("q_delta max_abs:", float(q_delta_settle.abs().max().detach().cpu()))
    print("q_delta env0   :", q_delta_settle[env_id].detach().cpu().numpy())
    print("foot_delta max_abs:", float(foot_delta_settle.abs().max().detach().cpu()))
    print("foot_delta env0:", foot_delta_settle[env_id].detach().cpu().numpy())

    print("\n[delta from SPAWN baseline env0]")
    print("base_delta xyz+rpy:", base_delta_spawn[env_id].detach().cpu().numpy())
    print("q_delta max_abs:", float(q_delta_spawn.abs().max().detach().cpu()))
    print("foot_delta max_abs:", float(foot_delta_spawn.abs().max().detach().cpu()))

    print("\n[u_cmd / beta env0]")
    print("u_cmd:", u_cmd[env_id].detach().cpu().numpy())
    print("beta :", beta_t[env_id].detach().cpu().numpy(), f"preset={args.beta_preset}")

    print("\n[Theta / Ref env0]")
    print("z_mode       :", args.z_mode)
    print("T            :", float(theta.gait["T"][env_id].detach().cpu()))
    print("phase_i      :", theta.gait["phase_i"][env_id].detach().cpu().numpy())
    print("duty_i       :", theta.gait["duty_i"][env_id].detach().cpu().numpy())
    k = args.ref_k
    print("S[:,k]       :", ref["S"][env_id, :, k].detach().cpu().numpy())
    print("phase[:,k]   :", ref["phase"][env_id, :, k].detach().cpu().numpy())

    print("\n[forceMPC]")
    for key in ["stance_mask_env0", "num_stance_env0", "f_projected_env0", "pg_obj_delta_env0"]:
        if key in force_info:
            print(f"{key}: {force_info[key]}")

    print("\n[QPS-WBC raw info]")
    for key in [
        "stance_mask_env0", "swing_mask_env0",
        "w_force", "w_swing", "w_posture", "w_reg", "w_rate",
        "y_swing_env0",
        "tau_force_max_abs", "tau_posture_max_abs", "tau_qps_wbc_max_abs",
    ]:
        if key in wbc_info:
            print(f"{key}: {wbc_info[key]}")

    print("\n[authority probe after settled baseline]")
    print("tau_probe_scale:", args.tau_probe_scale)
    print("max_total_tau  :", args.max_total_tau)
    print("tau_raw stats  :", summarize_torque(tau_raw))
    print("tau_cmd stats  :", summarize_torque(tau_cmd))

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

    env = ManagerBasedRLEnv(cfg=env_cfg)
    _obs_dict, _ = env.reset()

    robot = env.scene["robot"]
    print_robot_debug_info(robot)

    dtype = robot.data.joint_pos.dtype
    device = robot.data.joint_pos.device
    params = build_spot_ref_params(device=device, dtype=dtype, dt=args.control_dt, horizon=20)

    foot_indices = get_foot_indices(robot)
    q_spawn = robot.data.joint_pos.detach().clone()
    foot_spawn = robot.data.body_pos_w[:, foot_indices, :].detach().clone()
    base_spawn = make_x_hat(robot, velocity_frame="world")[:, 0:6].detach().clone()

    settle_force_state = PGForceMPCV2State()

    print("\n" + "=" * 132)
    print("[INFO] Starting B8-e settled authority probe")
    print("=" * 132)
    print(f"settle_steps={args.settle_steps}, probe_steps={args.probe_steps}, settle_mode={args.settle_mode}")
    print(f"z_mode={args.z_mode}, beta_preset={args.beta_preset}, tau_probe_scale={args.tau_probe_scale}")
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

    # Save settled baseline.
    q_settle = robot.data.joint_pos.detach().clone()
    foot_settle = robot.data.body_pos_w[:, foot_indices, :].detach().clone()
    base_settle = make_x_hat(robot, velocity_frame="world")[:, 0:6].detach().clone()

    print("\n" + "-" * 132)
    print("[INFO] Settled baseline saved")
    print("base_settle env0:", base_settle[0].detach().cpu().numpy())
    print("q_settle env0   :", q_settle[0].detach().cpu().numpy())
    print("foot_settle env0:", foot_settle[0].detach().cpu().numpy())
    print("-" * 132 + "\n")

    # Reset controllers for probe phase.
    force_state = PGForceMPCV2State()
    wbc_state = QPSWBCBridgeState()

    # Phase 2: probe.
    for probe_step in range(args.probe_steps):
        if not simulation_app.is_running():
            break

        x_hat, u_cmd, beta_t, theta, ref = build_theta_ref(robot, params, probe_step, device, dtype)

        f_ref, force_info = plan_pg_force_mpc_v2(
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

        tau_raw, wbc_info = make_qps_wbc_bridge_torque(
            robot=robot,
            ref=ref,
            f_ref=f_ref,
            state=wbc_state,
            k=args.ref_k,
            linear_rows=args.linear_rows,
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
            max_total_tau=args.max_total_tau,
        )

        tau_cmd = torch.clamp(args.tau_probe_scale * tau_raw, -args.max_total_tau, args.max_total_tau)

        _obs_dict, _rewards, terminated, truncated, _extras = env.step(tau_cmd)

        if probe_step % args.print_every == 0:
            print_probe_debug(
                probe_step=probe_step,
                robot=robot,
                u_cmd=u_cmd,
                beta_t=beta_t,
                theta=theta,
                ref=ref,
                force_info=force_info,
                tau_raw=tau_raw,
                tau_cmd=tau_cmd,
                wbc_info=wbc_info,
                q_settle=q_settle,
                foot_settle=foot_settle,
                base_settle=base_settle,
                q_spawn=q_spawn,
                foot_spawn=foot_spawn,
                base_spawn=base_spawn,
            )

        if torch.any(terminated) or torch.any(truncated):
            print(f"[WARN] terminated/truncated during probe at probe_step={probe_step}")
            print("terminated:", terminated.detach().cpu().numpy())
            print("truncated :", truncated.detach().cpu().numpy())

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
