# isaaclab_carto/scripts/run_spot_b8k_clearance_axis_probe.py
#
# B8-k: swing clearance axis/sign probe.
#
# Motivation:
#   B8-i/B8-j showed:
#     - gait schedule works
#     - hybrid swing residual is nonzero
#     - timing can be slowed
#     - but actual foot clearance remains tiny
#
# So before continuing gait tuning, we must identify:
#
#   Which joint offset direction actually increases foot z?
#
# This script performs two diagnostics:
#
#   1) Kinematic Jacobian sign probe:
#      At the settled pose, inspect foot linear Jacobian Jv.
#      For each leg, print Jz for HX/HY/KN and predicted dz for candidate offsets.
#
#   2) Optional physical residual probe:
#      Apply joint-space residual torque to one selected leg using a chosen
#      HY/KN offset and measure actual foot delta.
#
# This is not walking. It is sign/axis debugging.

import os
import sys
import argparse
from typing import Any, Dict, List, Tuple

import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="CARTO/TRACER B8-k clearance axis/sign probe")

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle_steps", type=int, default=120)
parser.add_argument("--probe_steps", type=int, default=120)
parser.add_argument("--print_every", type=int, default=20)

# Env / PD
parser.add_argument("--pd_scale", type=float, default=0.55)
parser.add_argument("--stiffness_override", type=float, default=None)
parser.add_argument("--damping_override", type=float, default=None)
parser.add_argument("--spawn_z", type=float, default=0.60)

# Mode
parser.add_argument("--mode", type=str, default="jacobian", choices=["jacobian", "manual", "auto"])
parser.add_argument("--test_leg", type=str, default="RF", choices=["LF", "RF", "LH", "RH"])

# Manual offsets
parser.add_argument("--hy_delta", type=float, default=-0.08)
parser.add_argument("--kn_delta", type=float, default=0.22)
parser.add_argument("--hx_delta", type=float, default=0.0)

# Candidate magnitudes for Jacobian sign probe
parser.add_argument("--hy_mag", type=float, default=0.08)
parser.add_argument("--kn_mag", type=float, default=0.22)
parser.add_argument("--hx_mag", type=float, default=0.04)

# Physical residual torque test
parser.add_argument("--kp_joint", type=float, default=25.0)
parser.add_argument("--kd_joint", type=float, default=1.0)
parser.add_argument("--max_joint_tau", type=float, default=5.0)
parser.add_argument("--tau_scale", type=float, default=1.0)
parser.add_argument("--max_total_tau", type=float, default=8.0)

# Optional weak stance support while probing
parser.add_argument("--enable_stance_hold", action="store_true")
parser.add_argument("--stance_hold_tau", type=float, default=0.0)

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
from isaaclab_carto.lowlevel.support_force_control import extract_foot_jacobians_action_order  # noqa: E402


LEG_NAMES = ["LF", "RF", "LH", "RH"]
FOOT_NAMES = ["fl_foot", "fr_foot", "hl_foot", "hr_foot"]
LEG_TO_ID = {"LF": 0, "RF": 1, "LH": 2, "RH": 3}

# Native action order observed in previous diagnostics:
# [fl_hx, fr_hx, hl_hx, hr_hx,
#  fl_hy, fr_hy, hl_hy, hr_hy,
#  fl_kn, fr_kn, hl_kn, hr_kn]
HX_IDX = torch.tensor([0, 1, 2, 3], dtype=torch.long)
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


def get_foot_indices(robot):
    name_to_idx = {name: i for i, name in enumerate(robot.body_names)}
    return [name_to_idx[name] for name in FOOT_NAMES]


def settle_robot(env, robot):
    for step in range(args.settle_steps):
        if not simulation_app.is_running():
            break
        tau = torch.zeros_like(robot.data.joint_pos)
        env.step(tau)
        if step % max(args.print_every, 1) == 0:
            x = make_x_hat(robot, velocity_frame="world")
            print(f"[SETTLE] step={step} pos={x[0,0:3].detach().cpu().numpy()} rpy={x[0,3:6].detach().cpu().numpy()}")


def get_jacobian_snapshot(robot):
    # Jv_feet: [N, 4, 3, 12], action-order joint columns
    Jv_feet, _ = extract_foot_jacobians_action_order(robot=robot, linear_rows="0_3")
    return Jv_feet.detach().clone()


def candidate_offsets() -> List[Tuple[str, float, float, float]]:
    hm = args.hy_mag
    km = args.kn_mag
    xm = args.hx_mag
    candidates = [
        ("HY+ only", 0.0, +hm, 0.0),
        ("HY- only", 0.0, -hm, 0.0),
        ("KN+ only", 0.0, 0.0, +km),
        ("KN- only", 0.0, 0.0, -km),
        ("HY+ KN+", 0.0, +hm, +km),
        ("HY+ KN-", 0.0, +hm, -km),
        ("HY- KN+", 0.0, -hm, +km),
        ("HY- KN-", 0.0, -hm, -km),
        ("HX+ only", +xm, 0.0, 0.0),
        ("HX- only", -xm, 0.0, 0.0),
    ]
    return candidates


def predict_delta_xyz_for_leg(Jv_feet, leg_id: int, hx: float, hy: float, kn: float):
    J = Jv_feet[0, leg_id]  # [3,12]
    hx_col = int(HX_IDX[leg_id])
    hy_col = int(HY_IDX[leg_id])
    kn_col = int(KN_IDX[leg_id])
    dq = torch.zeros((12,), device=J.device, dtype=J.dtype)
    dq[hx_col] = hx
    dq[hy_col] = hy
    dq[kn_col] = kn
    dxyz = J @ dq
    return dxyz.detach().cpu().tolist()


def print_jacobian_probe(robot, Jv_feet):
    print("\n" + "=" * 132)
    print("[B8-k JACOBIAN CLEARANCE SIGN PROBE]")
    print("=" * 132)
    print("Action order:")
    print("  HX:", HX_IDX.tolist())
    print("  HY:", HY_IDX.tolist())
    print("  KN:", KN_IDX.tolist())
    print("Candidate magnitudes:")
    print("  hx_mag:", args.hx_mag, "hy_mag:", args.hy_mag, "kn_mag:", args.kn_mag)

    best_by_leg: Dict[str, Tuple[str, float, float, float, float]] = {}

    for leg_id, leg_name in enumerate(LEG_NAMES):
        J = Jv_feet[0, leg_id]
        hx_col = int(HX_IDX[leg_id])
        hy_col = int(HY_IDX[leg_id])
        kn_col = int(KN_IDX[leg_id])

        print("\n" + "-" * 100)
        print(f"[{leg_name}] foot={FOOT_NAMES[leg_id]}")
        print("Jz columns:")
        print("  Jz_HX:", float(J[2, hx_col].detach().cpu()))
        print("  Jz_HY:", float(J[2, hy_col].detach().cpu()))
        print("  Jz_KN:", float(J[2, kn_col].detach().cpu()))

        best = None
        for name, hx, hy, kn in candidate_offsets():
            dxyz = predict_delta_xyz_for_leg(Jv_feet, leg_id, hx, hy, kn)
            dz = dxyz[2]
            print(f"  candidate={name:10s} hx={hx:+.4f} hy={hy:+.4f} kn={kn:+.4f} -> pred_dxyz={dxyz}")
            if best is None or dz > best[4]:
                best = (name, hx, hy, kn, dz)

        assert best is not None
        best_by_leg[leg_name] = best
        print(f"  BEST_POSITIVE_Z for {leg_name}: {best[0]}  hx={best[1]:+.4f} hy={best[2]:+.4f} kn={best[3]:+.4f} pred_dz={best[4]:+.6f}")

    print("\n" + "=" * 132)
    print("[SUMMARY: use these signs in hybrid stepping if pred_dz is positive]")
    for leg_name, best in best_by_leg.items():
        print(f"{leg_name}: {best[0]}  hx={best[1]:+.4f}, hy={best[2]:+.4f}, kn={best[3]:+.4f}, pred_dz={best[4]:+.6f}")
    print("=" * 132 + "\n")

    return best_by_leg


def make_manual_tau(robot, q_base, leg_id: int, hx: float, hy: float, kn: float):
    device = robot.data.joint_pos.device
    hx_idx = HX_IDX.to(device=device)
    hy_idx = HY_IDX.to(device=device)
    kn_idx = KN_IDX.to(device=device)

    q = robot.data.joint_pos
    dq = robot.data.joint_vel
    q_des = q_base.clone()

    q_des[:, hx_idx[leg_id]] = q_base[:, hx_idx[leg_id]] + hx
    q_des[:, hy_idx[leg_id]] = q_base[:, hy_idx[leg_id]] + hy
    q_des[:, kn_idx[leg_id]] = q_base[:, kn_idx[leg_id]] + kn

    tau = torch.zeros_like(q)
    selected = torch.tensor(
        [int(hx_idx[leg_id]), int(hy_idx[leg_id]), int(kn_idx[leg_id])],
        device=device,
        dtype=torch.long,
    )
    tau[:, selected] = args.kp_joint * (q_des[:, selected] - q[:, selected]) - args.kd_joint * dq[:, selected]
    tau = torch.clamp(tau, -args.max_joint_tau, args.max_joint_tau)
    tau = args.tau_scale * tau
    tau = torch.clamp(tau, -args.max_total_tau, args.max_total_tau)

    return tau, q_des


def run_physical_probe(env, robot, q_base, foot_base, base_x0, leg_id: int, hx: float, hy: float, kn: float, label: str):
    foot_indices = get_foot_indices(robot)

    print("\n" + "=" * 132)
    print("[B8-k PHYSICAL RESIDUAL PROBE]")
    print("=" * 132)
    print("label:", label)
    print("test_leg:", LEG_NAMES[leg_id], "foot:", FOOT_NAMES[leg_id])
    print("offsets: hx=", hx, "hy=", hy, "kn=", kn)
    print("kp_joint:", args.kp_joint, "kd_joint:", args.kd_joint, "max_joint_tau:", args.max_joint_tau, "tau_scale:", args.tau_scale)
    print("=" * 132)

    for step in range(args.probe_steps):
        if not simulation_app.is_running():
            break

        tau, q_des = make_manual_tau(robot, q_base, leg_id, hx=hx, hy=hy, kn=kn)
        env.step(tau)

        if step % max(args.print_every, 1) == 0:
            x = make_x_hat(robot, velocity_frame="world")
            q = robot.data.joint_pos
            foot = robot.data.body_pos_w[:, foot_indices, :]
            foot_delta = foot - foot_base
            q_delta = q - q_base
            leg_delta = foot_delta[0, leg_id]

            print("\n" + "-" * 100)
            print(f"[PHYS PROBE] step={step}")
            print("base_delta xyz+rpy:", (x[0, 0:6] - base_x0[0]).detach().cpu().numpy())
            print("leg foot_delta xyz:", leg_delta.detach().cpu().numpy())
            print("all foot_delta env0:", foot_delta[0].detach().cpu().numpy())
            print("selected q_delta:",
                  float(q_delta[0, int(HX_IDX[leg_id])].detach().cpu()),
                  float(q_delta[0, int(HY_IDX[leg_id])].detach().cpu()),
                  float(q_delta[0, int(KN_IDX[leg_id])].detach().cpu()))
            print("selected q_des_delta:", hx, hy, kn)
            print("tau stats:", summarize_torque(tau))
            if hasattr(robot.data, "applied_torque"):
                print("applied stats:", summarize_torque(robot.data.applied_torque))
            print("clearance_z_for_test_leg:", float(leg_delta[2].detach().cpu()))
            print("max_abs_foot_delta:", float(foot_delta.abs().max().detach().cpu()))


def main():
    env_cfg = CartoEffortEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    patch_flat_safe_env(env_cfg)
    patch_reduced_implicit_pd(env_cfg)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    robot = env.scene["robot"]
    print_robot_debug_info(robot)

    print("\n" + "=" * 132)
    print("[INFO] Starting B8-k clearance axis/sign probe")
    print("=" * 132)
    print("mode:", args.mode)
    print("=" * 132 + "\n")

    settle_robot(env, robot)

    foot_indices = get_foot_indices(robot)
    q_base = robot.data.joint_pos.detach().clone()
    foot_base = robot.data.body_pos_w[:, foot_indices, :].detach().clone()
    base_x0 = make_x_hat(robot, velocity_frame="world")[:, 0:6].detach().clone()

    print("\n" + "-" * 132)
    print("[INFO] Baseline saved")
    print("base_x0 env0:", base_x0[0].detach().cpu().numpy())
    print("q_base env0:", q_base[0].detach().cpu().numpy())
    print("foot_base env0:", foot_base[0].detach().cpu().numpy())
    print("-" * 132 + "\n")

    Jv_feet = get_jacobian_snapshot(robot)
    best_by_leg = print_jacobian_probe(robot, Jv_feet)

    if args.mode == "jacobian":
        env.close()
        simulation_app.close()
        return

    leg_id = LEG_TO_ID[args.test_leg]

    if args.mode == "auto":
        best = best_by_leg[args.test_leg]
        label, hx, hy, kn, pred_dz = best
        print(f"[AUTO] selected best for {args.test_leg}: {label}, pred_dz={pred_dz}")
    else:
        label = "manual"
        hx, hy, kn = args.hx_delta, args.hy_delta, args.kn_delta

    run_physical_probe(
        env=env,
        robot=robot,
        q_base=q_base,
        foot_base=foot_base,
        base_x0=base_x0,
        leg_id=leg_id,
        hx=hx,
        hy=hy,
        kn=kn,
        label=label,
    )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
