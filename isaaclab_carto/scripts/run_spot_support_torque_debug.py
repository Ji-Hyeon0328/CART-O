# isaaclab_carto/scripts/run_spot_support_torque_debug.py
#
# B3-b: inject small support torque using tau = J^T f.
#
# Assumptions:
#   - B0.5 effort flat debug is working.
#   - inspect_spot_jacobian.py confirmed raw Jacobian shape (N, 17, 6, 18).
#   - implicit PD remains enabled as safety stabilizer.
#
# This is NOT full MPC/WBC yet.
# It only verifies that J^T f residual torque can be computed and injected safely.
#
# Recommended first run:
#
#   python source/isaaclab_carto/isaaclab_carto/scripts/run_spot_support_torque_debug.py \
#     --num_envs 1 --steps 300 --print_every 25 \
#     --fz_per_foot 1.0 --tau_scale 1.0 --max_tau 2.0
#
# If stable, try:
#
#   --fz_per_foot 2.0 --max_tau 3.0
#
# Compare sign:
#
#   --force_sign -1.0

import os
import sys
import argparse
from typing import Any

import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="CARTO Spot J^T f support torque debug")

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--print_every", type=int, default=25)

parser.add_argument("--fz_per_foot", type=float, default=1.0)
parser.add_argument("--tau_scale", type=float, default=1.0)
parser.add_argument("--max_tau", type=float, default=2.0)
parser.add_argument("--force_sign", type=float, default=1.0, choices=[1.0, -1.0])
parser.add_argument("--linear_rows", type=str, default="0_3", choices=["0_3", "3_6"])

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
from isaaclab_carto.lowlevel.spot_state import make_x_hat, print_robot_debug_info  # noqa: E402
from isaaclab_carto.lowlevel.effort_control import summarize_torque  # noqa: E402
from isaaclab_carto.lowlevel.support_force_control import make_support_torque  # noqa: E402


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
        tg.sub_terrains = {
            "flat": MeshPlaneTerrainCfg(proportion=1.0),
        }
        print("[INFO] Patched terrain generator to flat-only.")
    except Exception as exc:
        print(f"[WARN] Could not patch terrain generator to flat-only: {exc}")

    try:
        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, args.spawn_z)
        print(f"[INFO] Patched robot spawn z to {args.spawn_z}.")
    except Exception as exc:
        print(f"[WARN] Could not patch robot spawn z: {exc}")


def print_debug(step: int, robot, x_hat_before: torch.Tensor, tau_support: torch.Tensor, info: dict) -> None:
    env_id = 0
    x_hat_after = make_x_hat(robot, velocity_frame="world")

    print("\n" + "-" * 100)
    print(f"[SUPPORT TORQUE DEBUG] step={step}")
    print("-" * 100)

    print("[x_hat before env.step env0]")
    print("pos xyz      :", x_hat_before[env_id, 0:3].detach().cpu().numpy())
    print("rpy          :", x_hat_before[env_id, 3:6].detach().cpu().numpy())
    print("lin vel      :", x_hat_before[env_id, 6:9].detach().cpu().numpy())
    print("ang vel      :", x_hat_before[env_id, 9:12].detach().cpu().numpy())

    print("\n[x_hat after env.step env0]")
    print("pos xyz      :", x_hat_after[env_id, 0:3].detach().cpu().numpy())
    print("rpy          :", x_hat_after[env_id, 3:6].detach().cpu().numpy())
    print("lin vel      :", x_hat_after[env_id, 6:9].detach().cpu().numpy())
    print("ang vel      :", x_hat_after[env_id, 9:12].detach().cpu().numpy())

    print("\n[tau_support env0, action order]")
    print(tau_support[env_id].detach().cpu().numpy())
    print("tau_support stats:", summarize_torque(tau_support))

    print("\n[joint names action order]")
    print(list(getattr(robot, "joint_names", [])))

    print("\n[J^T f info]")
    for k, v in info.items():
        print(f"{k}: {v}")

    if hasattr(robot.data, "applied_torque"):
        applied = robot.data.applied_torque
        print("\n[applied_torque env0 after env.step]")
        print(applied[env_id].detach().cpu().numpy())
        print("applied stats:", summarize_torque(applied))

    print("-" * 100 + "\n")


def main() -> None:
    env_cfg = CartoEffortEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    patch_flat_safe_env(env_cfg)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    _obs_dict, _ = env.reset()

    robot = env.scene["robot"]
    print_robot_debug_info(robot)

    print("\n" + "=" * 100)
    print("[INFO] Starting Spot support torque debug")
    print("=" * 100)
    print(f"num_envs     : {env.num_envs}")
    print(f"steps        : {args.steps}")
    print(f"print_every  : {args.print_every}")
    print(f"fz_per_foot  : {args.fz_per_foot}")
    print(f"tau_scale    : {args.tau_scale}")
    print(f"max_tau      : {args.max_tau}")
    print(f"force_sign   : {args.force_sign}")
    print(f"linear_rows  : {args.linear_rows}")
    print("=" * 100 + "\n")

    for step in range(args.steps):
        if not simulation_app.is_running():
            break

        x_hat_before = make_x_hat(robot, velocity_frame="world")

        tau_support, info = make_support_torque(
            robot=robot,
            fz_per_foot=args.fz_per_foot,
            tau_scale=args.tau_scale,
            max_tau=args.max_tau,
            linear_rows=args.linear_rows,
            force_sign=args.force_sign,
        )

        _obs_dict, _rewards, terminated, truncated, _extras = env.step(tau_support)

        if step % args.print_every == 0:
            print_debug(
                step=step,
                robot=robot,
                x_hat_before=x_hat_before,
                tau_support=tau_support,
                info=info,
            )

        if torch.any(terminated) or torch.any(truncated):
            print(f"[WARN] terminated/truncated at step={step}")
            print("terminated:", terminated.detach().cpu().numpy())
            print("truncated :", truncated.detach().cpu().numpy())

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
