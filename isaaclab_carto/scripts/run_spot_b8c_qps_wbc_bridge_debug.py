# isaaclab_carto/scripts/run_spot_b8c_qps_wbc_bridge_debug.py
#
# B8-c: QP-style WBC bridge debug.
#
# Pipeline:
#   thetaDecoder / thetaRefMapper
#   → forceMPC v2 f_ref
#   → QP-style WBC bridge:
#       weighted LS over force tracking, swing task, posture, tau reg, tau rate

import os
import sys
import argparse
from typing import Any

import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="CARTO/TRACER B8-c QP-style WBC bridge debug")

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--print_every", type=int, default=25)

parser.add_argument("--z_mode", type=str, default="conservative", choices=["conservative", "aggressive"])
parser.add_argument("--vx", type=float, default=0.05)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--beta_preset", type=str, default="balanced", choices=["height", "velocity", "energy", "balanced"])

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

parser.add_argument("--linear_rows", type=str, default="0_3", choices=["0_3", "3_6"])
parser.add_argument("--force_sign", type=float, default=1.0, choices=[1.0, -1.0])

# QP-style WBC bridge weights
parser.add_argument("--w_force", type=float, default=1.0)
parser.add_argument("--w_swing", type=float, default=0.10)
parser.add_argument("--w_posture", type=float, default=0.40)
parser.add_argument("--w_reg", type=float, default=0.02)
parser.add_argument("--w_rate", type=float, default=0.25)

parser.add_argument("--kp_swing_xy", type=float, default=20.0)
parser.add_argument("--kp_swing_z", type=float, default=35.0)
parser.add_argument("--kd_swing_xy", type=float, default=1.5)
parser.add_argument("--kd_swing_z", type=float, default=2.5)
parser.add_argument("--max_task_cmd", type=float, default=4.0)
parser.add_argument("--max_pos_err", type=float, default=0.05)

parser.add_argument("--kp_posture", type=float, default=0.8)
parser.add_argument("--kd_posture", type=float, default=0.04)
parser.add_argument("--max_posture_tau", type=float, default=0.20)
parser.add_argument("--max_total_tau", type=float, default=3.0)

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


def print_debug(step, robot, u_cmd, beta_t, theta, ref, f_ref, force_info, tau_total, wbc_info):
    env_id = 0
    x_hat_after = make_x_hat(robot, velocity_frame="world")

    print("\n" + "=" * 132)
    print(f"[B8-c QP-STYLE WBC BRIDGE DEBUG] step={step}")
    print("=" * 132)

    print("[x_hat after env.step env0]")
    print("pos xyz      :", x_hat_after[env_id, 0:3].detach().cpu().numpy())
    print("rpy          :", x_hat_after[env_id, 3:6].detach().cpu().numpy())
    print("lin vel      :", x_hat_after[env_id, 6:9].detach().cpu().numpy())
    print("ang vel      :", x_hat_after[env_id, 9:12].detach().cpu().numpy())

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

    print("\n[forceMPC v2]")
    for key in ["stance_mask_env0", "num_stance_env0", "f_projected_env0", "pg_obj_delta_env0"]:
        if key in force_info:
            print(f"{key}: {force_info[key]}")

    print("\n[QP-style WBC bridge]")
    for key in [
        "stance_mask_env0", "swing_mask_env0",
        "w_force", "w_swing", "w_posture", "w_reg", "w_rate",
        "xf_env0", "xf_ref_env0", "swing_pos_err_env0",
        "y_swing_env0",
        "tau_force_mean_abs", "tau_force_max_abs",
        "tau_posture_mean_abs", "tau_posture_max_abs",
        "tau_qps_wbc_mean_abs", "tau_qps_wbc_max_abs",
    ]:
        if key in wbc_info:
            print(f"{key}: {wbc_info[key]}")

    print("\n[tau_total env0, action order]")
    print(tau_total[env_id].detach().cpu().numpy())
    print("tau_total stats:", summarize_torque(tau_total))

    if hasattr(robot.data, "applied_torque"):
        applied = robot.data.applied_torque
        print("\n[applied_torque env0 after env.step]")
        print(applied[env_id].detach().cpu().numpy())
        print("applied stats:", summarize_torque(applied))

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
    force_state = PGForceMPCV2State()
    wbc_state = QPSWBCBridgeState()

    print("\n" + "=" * 132)
    print("[INFO] Starting B8-c QP-style WBC bridge debug")
    print("=" * 132)
    print(f"z_mode={args.z_mode}, beta_preset={args.beta_preset}, u_cmd=[{args.vx},{args.vy},{args.wz}]")
    print("NOTE: Weighted LS bridge, not full floating-base QP WBC yet.")
    print("=" * 132 + "\n")

    for step in range(args.steps):
        if not simulation_app.is_running():
            break

        x_hat = make_x_hat(robot, velocity_frame="world")
        u_cmd = make_fixed_command(env.num_envs, device=device, dtype=dtype)
        z_t, a_hl = make_fake_highlevel(env.num_envs, device=device, dtype=dtype)
        beta_t = make_beta_preset(env.num_envs, device=device, dtype=dtype, preset=args.beta_preset)

        theta = theta_decoder(z_t=z_t, a_HL=a_hl, x_hat=x_hat, u_cmd=u_cmd, robot_name="spot")
        theta = advance_theta_phase_with_clock(theta, step=step, dt=args.control_dt)
        ref = theta_ref_mapper(theta=theta, x_hat=x_hat, u_cmd=u_cmd, params=params)

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

        tau_total, wbc_info = make_qps_wbc_bridge_torque(
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

        _obs_dict, _rewards, terminated, truncated, _extras = env.step(tau_total)

        if step % args.print_every == 0:
            print_debug(step, robot, u_cmd, beta_t, theta, ref, f_ref, force_info, tau_total, wbc_info)

        if torch.any(terminated) or torch.any(truncated):
            print(f"[WARN] terminated/truncated at step={step}")
            print("terminated:", terminated.detach().cpu().numpy())
            print("truncated :", truncated.detach().cpu().numpy())

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
