# isaaclab_carto/scripts/inspect_spot_jacobian.py
#
# B3-a: inspect whether Isaac Lab exposes Spot foot Jacobians.
#
# This script:
#   - creates flat effort env
#   - prints robot joint/body names
#   - checks root_physx_view methods
#   - calls get_jacobians()
#   - tries to extract foot linear Jacobians
#   - computes a tiny tau = J^T f_support sample
#
# Run:
#   python source/isaaclab_carto/isaaclab_carto/scripts/inspect_spot_jacobian.py \
#     --num_envs 1 --steps 5

import os
import sys
import argparse
from typing import Any

import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Inspect Spot foot Jacobians in Isaac Lab")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=5)
parser.add_argument("--spawn_z", type=float, default=0.60)
parser.add_argument("--support_force_z", type=float, default=50.0)
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
from isaaclab_carto.lowlevel.spot_state import print_robot_debug_info  # noqa: E402
from isaaclab_carto.lowlevel.effort_control import make_zero_torque, summarize_torque  # noqa: E402
from isaaclab_carto.lowlevel.jacobian_utils import (  # noqa: E402
    list_robot_view_methods,
    try_get_root_physx_view,
    try_get_jacobians,
    infer_jacobian_layout,
    extract_spot_foot_linear_jacobians,
    compute_tau_from_foot_forces,
)


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


def print_tensor_sample(name: str, x: torch.Tensor, max_items: int = 12) -> None:
    flat = x.flatten()
    n = min(max_items, flat.numel())
    print(f"{name}: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
    print(f"{name} sample:", flat[:n].detach().cpu().numpy())


def main() -> None:
    env_cfg = CartoEffortEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    patch_flat_safe_env(env_cfg)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    _obs_dict, _ = env.reset()
    robot = env.scene["robot"]

    print_robot_debug_info(robot)

    print("\n" + "=" * 100)
    print("[JACOBIAN INSPECT] root/view method availability")
    print("=" * 100)
    method_info = list_robot_view_methods(robot)
    for k, v in method_info.items():
        print(f"{k}: {v}")

    view = try_get_root_physx_view(robot)
    print("root_physx_view type:", type(view))

    for step in range(args.steps):
        if not simulation_app.is_running():
            break

        # Step with zero torque to let buffers update.
        tau0 = make_zero_torque(robot)
        _obs_dict, _rewards, terminated, truncated, _extras = env.step(tau0)

        print("\n" + "-" * 100)
        print(f"[JACOBIAN INSPECT] step={step}")
        print("-" * 100)

        J = try_get_jacobians(robot)
        if J is None:
            print("[ERROR] Could not obtain Jacobians. Paste this output back.")
            continue

        print_tensor_sample("raw Jacobians", J)

        layout = infer_jacobian_layout(J, robot)
        print("layout info:")
        for k, v in layout.items():
            print(f"  {k}: {v}")

        try:
            Jv_feet, ext_info = extract_spot_foot_linear_jacobians(J, robot)
            print_tensor_sample("Jv_feet rows 0:3 default", Jv_feet)

            print("extraction info:")
            for k, v in ext_info.items():
                print(f"  {k}: {v}")

            # Sample support force.
            f_feet = torch.zeros((env.num_envs, 4, 3), device=J.device, dtype=J.dtype)
            f_feet[:, :, 2] = args.support_force_z

            tau_support = compute_tau_from_foot_forces(Jv_feet, f_feet)
            print_tensor_sample("tau_support = J^T f", tau_support)
            print("tau_support stats:", summarize_torque(tau_support))

        except Exception as exc:
            print(f"[ERROR] Foot Jacobian extraction failed: {exc}")

        if torch.any(terminated) or torch.any(truncated):
            print(f"[WARN] terminated/truncated at step={step}")
            print("terminated:", terminated.detach().cpu().numpy())
            print("truncated :", truncated.detach().cpu().numpy())

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
