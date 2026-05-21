# source/isaaclab_carto/isaaclab_carto/scripts/run_spot_lowlevel_debug.py
#
# CARTO / TRACER Spot low-level debug script.
#
# Purpose:
#   - No training.
#   - No MPC/WBC/residual yet.
#   - Check Isaac Lab Spot state -> x_hat -> theta_decoder -> theta_ref_mapper.
#   - Keep Spot standing with JointPositionActionCfg zero/default/current action.
#
# Usage:
#   python source/isaaclab_carto/isaaclab_carto/scripts/run_spot_lowlevel_debug.py --num_envs 1
#
# Optional:
#   python source/isaaclab_carto/isaaclab_carto/scripts/run_spot_lowlevel_debug.py --num_envs 1 --z_mode aggressive
#   python source/isaaclab_carto/isaaclab_carto/scripts/run_spot_lowlevel_debug.py --num_envs 1 --action_mode current
#   python source/isaaclab_carto/isaaclab_carto/scripts/run_spot_lowlevel_debug.py --num_envs 1 --print_every 25

import os
import sys
import argparse
from typing import Dict

import torch

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="CARTO Spot low-level debug script")

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--print_every", type=int, default=50)

parser.add_argument(
    "--action_mode",
    type=str,
    default="zero",
    choices=["zero", "default", "current"],
    help=(
        "zero: send zero action. Usually stable for current JointPositionActionCfg. "
        "default: send default_joint_pos as action. "
        "current: send current joint_pos as action."
    ),
)

parser.add_argument(
    "--z_mode",
    type=str,
    default="conservative",
    choices=["conservative", "aggressive"],
    help="Fake gait mode for theta_decoder.",
)

parser.add_argument(
    "--vx",
    type=float,
    default=0.30,
    help="Fallback command vx if command_manager fails.",
)

parser.add_argument(
    "--vy",
    type=float,
    default=0.00,
    help="Fallback command vy if command_manager fails.",
)

parser.add_argument(
    "--wz",
    type=float,
    default=0.10,
    help="Fallback command yaw rate if command_manager fails.",
)

parser.add_argument(
    "--use_fixed_cmd",
    action="store_true",
    help="Use fixed command from --vx --vy --wz instead of command_manager.",
)

parser.add_argument(
    "--enable_gait_action",
    action="store_true",
    help="Enable small Ref.S-based joint-position modulation for final step A.",
)

parser.add_argument(
    "--gait_ref_k",
    type=int,
    default=10,
    help="Horizon index used for Ref.S when generating gait action.",
)

parser.add_argument(
    "--lift_scale",
    type=float,
    default=0.08,
    help="Small hip pitch action offset for swing legs.",
)

parser.add_argument(
    "--knee_scale",
    type=float,
    default=-0.12,
    help="Small knee action offset for swing legs.",
)

parser.add_argument(
    "--use_gait_clock",
    action="store_true",
    help="Advance gait phase by simulation step so swing legs change over time.",
)

parser.add_argument(
    "--control_dt",
    type=float,
    default=0.02,
    help="Controller dt. Current env uses sim dt 0.005 and decimation 4, so control dt is 0.02.",
)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Project import path
# -----------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))

if SOURCE_DIR not in sys.path:
    sys.path.append(SOURCE_DIR)

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_carto.envs.carto_env_cfg import CartoEnvCfg  # noqa: E402

from isaaclab_carto.lowlevel import (  # noqa: E402
    theta_decoder,
    theta_ref_mapper,
    make_x_hat,
    build_spot_ref_params,
    make_standing_action,
    make_gait_joint_position_action,
    print_robot_debug_info,
)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def get_command_from_env(env: ManagerBasedRLEnv, dtype: torch.dtype) -> torch.Tensor:
    """
    Get base velocity command.

    If --use_fixed_cmd is enabled:
        use fixed [vx, vy, wz].

    Otherwise:
        use Isaac Lab command manager.
    """
    if args.use_fixed_cmd:
        u_cmd = torch.zeros((env.num_envs, 3), device=env.device, dtype=dtype)
        u_cmd[:, 0] = args.vx
        u_cmd[:, 1] = args.vy
        u_cmd[:, 2] = args.wz
        return u_cmd

    try:
        u_cmd = env.command_manager.get_command("base_velocity").clone()
        u_cmd = u_cmd[:, :3].to(device=env.device, dtype=dtype)
    except Exception as exc:
        print(f"[WARN] Failed to get command from command_manager: {exc}")
        u_cmd = torch.zeros((env.num_envs, 3), device=env.device, dtype=dtype)
        u_cmd[:, 0] = args.vx
        u_cmd[:, 1] = args.vy
        u_cmd[:, 2] = args.wz

    return u_cmd


def make_fake_highlevel(
    env: ManagerBasedRLEnv,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fake high-level outputs.

    Returns:
        z_t:    [num_envs]
        a_HL:   [num_envs, 4]
        beta_t: [num_envs, 3]
    """
    if args.z_mode == "aggressive":
        z_value = 1
    else:
        z_value = 0

    z_t = torch.full(
        (env.num_envs,),
        z_value,
        device=env.device,
        dtype=torch.long,
    )

    a_HL_single = torch.tensor(
        [0.20, -0.20, 0.30, -0.10],
        device=env.device,
        dtype=dtype,
    )

    a_HL = a_HL_single.unsqueeze(0).repeat(env.num_envs, 1)

    beta_single = torch.tensor(
        [0.30, 0.40, 0.30],
        device=env.device,
        dtype=dtype,
    )

    beta_t = beta_single.unsqueeze(0).repeat(env.num_envs, 1)

    return z_t, a_HL, beta_t

def advance_theta_phase_with_clock(theta, step: int, dt: float):
    """
    Advance theta.gait['phase_i'] using a simple gait clock.

    Without this, theta_decoder returns the same nominal phase every step,
    and using a fixed Ref.S[:, k] can keep the same leg(s) in swing forever.

    Args:
        theta:
            Theta object from theta_decoder().
        step:
            Current control step.
        dt:
            Control dt.

    Returns:
        theta with updated phase_i.
    """
    T = theta.gait["T"]  # [num_envs]
    phase_i = theta.gait["phase_i"]  # [num_envs, 4]

    phase_advance = (step * dt) / torch.clamp(T, min=1e-6)
    theta.gait["phase_i"] = torch.remainder(
        phase_i + phase_advance.unsqueeze(1),
        1.0,
    )
    return theta

def print_lowlevel_debug(
    step: int,
    x_hat: torch.Tensor,
    u_cmd: torch.Tensor,
    z_t: torch.Tensor,
    a_HL: torch.Tensor,
    beta_t: torch.Tensor,
    theta,
    ref: Dict[str, torch.Tensor],
) -> None:
    """
    Print compact low-level debug info for env 0.
    """
    env_id = 0
    k0 = 0
    kmid = min(10, ref["S"].shape[-1] - 1)

    print("\n" + "-" * 90)
    print(f"[LOWLEVEL DEBUG] step={step}")
    print("-" * 90)

    print("[fake high-level env0]")
    print("z_t          :", int(z_t[env_id].detach().cpu()))
    print("a_HL         :", a_HL[env_id].detach().cpu().numpy())
    print("beta_t       :", beta_t[env_id].detach().cpu().numpy())

    print("\n[x_hat env0]")
    print("pos xyz      :", x_hat[env_id, 0:3].detach().cpu().numpy())
    print("rpy          :", x_hat[env_id, 3:6].detach().cpu().numpy())
    print("lin vel      :", x_hat[env_id, 6:9].detach().cpu().numpy())
    print("ang vel      :", x_hat[env_id, 9:12].detach().cpu().numpy())

    print("\n[u_cmd env0]")
    print("u_cmd        :", u_cmd[env_id].detach().cpu().numpy())

    print("\n[Theta env0]")
    print("T            :", float(theta.gait["T"][env_id].detach().cpu()))
    print("phase_i      :", theta.gait["phase_i"][env_id].detach().cpu().numpy())
    print("duty_i       :", theta.gait["duty_i"][env_id].detach().cpu().numpy())
    print("h_swing_i    :", theta.foot["h_swing_i"][env_id].detach().cpu().numpy())
    print("h_body_ref   :", float(theta.base["h_body_ref"][env_id].detach().cpu()))
    print("roll_ref     :", float(theta.base["roll_ref"][env_id].detach().cpu()))
    print("pitch_ref    :", float(theta.base["pitch_ref"][env_id].detach().cpu()))
    print("k_des        :", float(theta.ctrl["k_des"][env_id].detach().cpu()))
    print("mu_exp       :", float(theta.ctrl["mu_exp"][env_id].detach().cpu()))

    print(f"\n[Ref env0, k={k0}]")
    print("S[:,0]       :", ref["S"][env_id, :, k0].detach().cpu().numpy())
    print("phase[:,0]   :", ref["phase"][env_id, :, k0].detach().cpu().numpy())
    print("Xb_ref[:,0]  :", ref["Xb_ref"][env_id, :, k0].detach().cpu().numpy())
    print("Xf_ref[:,:,0]:")
    print(ref["Xf_ref"][env_id, :, :, k0].detach().cpu().numpy())

    print(f"\n[Ref env0, k={kmid}]")
    print(f"S[:,{kmid}]       :", ref["S"][env_id, :, kmid].detach().cpu().numpy())
    print(f"phase[:,{kmid}]   :", ref["phase"][env_id, :, kmid].detach().cpu().numpy())
    print(f"Xb_ref[:,{kmid}]  :", ref["Xb_ref"][env_id, :, kmid].detach().cpu().numpy())
    print(f"Xf_ref[:,:,{kmid}]:")
    print(ref["Xf_ref"][env_id, :, :, kmid].detach().cpu().numpy())

    print("-" * 90 + "\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    env_cfg = CartoEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env = ManagerBasedRLEnv(cfg=env_cfg)
    _obs_dict, _ = env.reset()

    robot = env.scene["robot"]
    print_robot_debug_info(robot)

    dtype = robot.data.joint_pos.dtype

    # Current env uses sim dt=0.005 and decimation=4, so control dt is 0.02.
    params = build_spot_ref_params(
        device=env.device,
        dtype=dtype,
        dt=0.02,
        horizon=20,
    )

    print("\n" + "=" * 90)
    print("[INFO] Starting Spot low-level debug")
    print("=" * 90)
    print(f"num_envs      : {env.num_envs}")
    print(f"steps         : {args.steps}")
    print(f"print_every   : {args.print_every}")
    print(f"action_mode   : {args.action_mode}")
    print(f"z_mode        : {args.z_mode}")
    print(f"fallback cmd  : vx={args.vx}, vy={args.vy}, wz={args.wz}")
    print("=" * 90 + "\n")

    for step in range(args.steps):
        if not simulation_app.is_running():
            break

        x_hat = make_x_hat(robot, velocity_frame="world")
        u_cmd = get_command_from_env(env, dtype=dtype)

        z_t, a_HL, beta_t = make_fake_highlevel(env, dtype=dtype)

        theta = theta_decoder(
            z_t=z_t,
            a_HL=a_HL,
            x_hat=x_hat,
            u_cmd=u_cmd,
            robot_name="spot",
        )

        if args.use_gait_clock:
            theta = advance_theta_phase_with_clock(
                theta=theta,
                step=step,
                dt=args.control_dt,
            )

        ref = theta_ref_mapper(
            theta=theta,
            x_hat=x_hat,
            u_cmd=u_cmd,
            params=params,
        )

        if step % args.print_every == 0:
            print_lowlevel_debug(
                step=step,
                x_hat=x_hat,
                u_cmd=u_cmd,
                z_t=z_t,
                a_HL=a_HL,
                beta_t=beta_t,
                theta=theta,
                ref=ref,
            )

        #actions = make_standing_action(robot, mode=args.action_mode)
        if args.enable_gait_action:
            actions = make_gait_joint_position_action(
                robot=robot,
                ref=ref,
                lift_scale=args.lift_scale,
                knee_scale=args.knee_scale,
                use_k=args.gait_ref_k,
            )
        else:
            actions = make_standing_action(robot, mode=args.action_mode)

        _obs_dict, _rewards, terminated, truncated, _extras = env.step(actions)

        if torch.any(terminated) or torch.any(truncated):
            print(f"[WARN] terminated/truncated at step={step}")
            print("terminated:", terminated.detach().cpu().numpy())
            print("truncated :", truncated.detach().cpu().numpy())

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()