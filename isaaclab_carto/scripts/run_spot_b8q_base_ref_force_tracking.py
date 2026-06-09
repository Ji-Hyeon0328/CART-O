# isaaclab_carto/scripts/run_spot_b8q_base_ref_force_tracking.py
#
# B8-q: base_ref -> stance force distribution -> J^T tau tracking probe.
#
# This is the first closed-loop control step after B8-p.
#
# B8-p only generated:
#   support polygon -> CoM/base reference
#
# B8-q applies a simple force-based controller:
#   base_ref error -> desired body force/wrench
#   -> distribute wrench to stance feet
#   -> tau = J^T f
#   -> effort action
#
# Important:
#   This is not final MPC/WBC.
#   It is a bridge/probe to verify that base_ref can influence the simulated body.
#
# Initial test uses all four legs as stance.
# After base reference tracking works, we can connect gait schedule and swing tasks.

import os
import sys
import argparse
from typing import Any

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CARTO/TRACER B8-q base reference force tracking")

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle_steps", type=int, default=180)
parser.add_argument("--probe_steps", type=int, default=500)
parser.add_argument("--print_every", type=int, default=20)

# Reference
parser.add_argument("--alpha", type=float, default=0.25)
parser.add_argument("--margin", type=float, default=0.030)
parser.add_argument("--max_shift_per_step", type=float, default=0.0015)
parser.add_argument("--height_ref", type=float, default=0.54)

# Controller gains
parser.add_argument("--kp_xy", type=float, default=120.0)
parser.add_argument("--kd_xy", type=float, default=30.0)
parser.add_argument("--kp_z", type=float, default=500.0)
parser.add_argument("--kd_z", type=float, default=80.0)
parser.add_argument("--kp_roll", type=float, default=120.0)
parser.add_argument("--kd_roll", type=float, default=20.0)
parser.add_argument("--kp_pitch", type=float, default=120.0)
parser.add_argument("--kd_pitch", type=float, default=20.0)

# Force/torque limits
parser.add_argument("--mass_override", type=float, default=32.0)
parser.add_argument("--mu", type=float, default=0.7)
parser.add_argument("--min_fz", type=float, default=5.0)
parser.add_argument("--max_fz", type=float, default=220.0)
parser.add_argument("--max_total_tau", type=float, default=18.0)
parser.add_argument("--tau_scale", type=float, default=1.0)
parser.add_argument("--tau_sign", type=float, default=1.0, choices=[1.0, -1.0])

# Implicit actuator scaling
parser.add_argument("--pd_scale", type=float, default=0.45)

# Target mode
parser.add_argument("--target_mode", type=str, default="support_center", choices=["support_center", "manual"])
parser.add_argument("--manual_dx", type=float, default=0.04)
parser.add_argument("--manual_dy", type=float, default=0.00)

# Env
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
from isaaclab.utils import configclass  # noqa: E402

try:
    from isaaclab.envs.mdp import JointEffortActionCfg  # noqa: E402
except Exception:
    from isaaclab.envs.mdp.actions.actions_cfg import JointEffortActionCfg  # noqa: E402

from isaaclab_carto.envs.carto_env_cfg import CartoEnvCfg  # noqa: E402
from isaaclab_carto.lowlevel.spot_state import make_x_hat, print_robot_debug_info  # noqa: E402
from isaaclab_carto.lowlevel.support_region_ref import SupportRegionRefConfig, compute_support_region_ref  # noqa: E402


FOOT_NAMES = ["fl_foot", "fr_foot", "hl_foot", "hr_foot"]
LEG_NAMES = ["LF", "RF", "LH", "RH"]


@configclass
class CartoEffortActionsCfg:
    joint_effort = JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
    )


@configclass
class CartoEffortEnvCfg(CartoEnvCfg):
    actions: CartoEffortActionsCfg = CartoEffortActionsCfg()


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

    # Reduce implicit PD dominance while keeping enough passive stability.
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


def get_mass(robot, device, dtype):
    try:
        masses = robot.root_physx_view.get_masses()
        return float(masses.sum().detach().cpu())
    except Exception:
        return args.mass_override


def get_jacobian_feet(robot, foot_indices):
    """Return translational Jacobian [N,4,3,12] for feet in native joint order."""
    jac = robot.root_physx_view.get_jacobians()
    # Expected shape from previous diagnostics: [N, bodies, 6, dofs_or_dofs+6]
    if jac.shape[-1] == robot.num_joints + 6:
        jac = jac[..., 6:]
    # foot body ids are indices in body list.
    jv = jac[:, foot_indices, 0:3, :]
    return jv


def distribute_wrench_to_feet(base_pos, base_ref, base_vel, base_rpy, base_ang_vel, foot_pos, stance_mask, mass):
    """Simple stance force distribution.

    Args:
        base_pos [N,3], base_ref [N,6], base_vel [N,3],
        base_rpy [N,3], base_ang_vel [N,3],
        foot_pos [N,4,3], stance_mask [N,4]
    Returns:
        foot_forces [N,4,3]
        desired_wrench [N,6] = Fx,Fy,Fz,Mx,My,Mz
    """
    device = base_pos.device
    dtype = base_pos.dtype
    n = base_pos.shape[0]
    g = 9.81

    pos_err = base_ref[:, 0:3] - base_pos
    rpy_err = base_ref[:, 3:6] - base_rpy

    F_des = torch.zeros((n, 3), device=device, dtype=dtype)
    M_des = torch.zeros((n, 3), device=device, dtype=dtype)

    F_des[:, 0] = args.kp_xy * pos_err[:, 0] - args.kd_xy * base_vel[:, 0]
    F_des[:, 1] = args.kp_xy * pos_err[:, 1] - args.kd_xy * base_vel[:, 1]
    F_des[:, 2] = mass * g + args.kp_z * pos_err[:, 2] - args.kd_z * base_vel[:, 2]

    M_des[:, 0] = args.kp_roll * rpy_err[:, 0] - args.kd_roll * base_ang_vel[:, 0]
    M_des[:, 1] = args.kp_pitch * rpy_err[:, 1] - args.kd_pitch * base_ang_vel[:, 1]
    M_des[:, 2] = 0.0

    foot_forces = torch.zeros((n, 4, 3), device=device, dtype=dtype)

    for e in range(n):
        stance_ids = torch.where(stance_mask[e] > 0.5)[0]
        ns = max(int(stance_ids.numel()), 1)

        # horizontal force equally distributed
        fx_i = F_des[e, 0] / ns
        fy_i = F_des[e, 1] / ns

        # vertical forces solve:
        # sum fz = Fz
        # sum r_y fz = Mx
        # sum -r_x fz = My
        r = foot_pos[e, stance_ids, :] - base_pos[e].unsqueeze(0)
        A = torch.stack(
            [
                torch.ones((ns,), device=device, dtype=dtype),
                r[:, 1],
                -r[:, 0],
            ],
            dim=0,
        )  # [3,ns]
        b = torch.stack([F_des[e, 2], M_des[e, 0], M_des[e, 1]], dim=0).unsqueeze(1)  # [3,1]

        # minimum norm solution: fz = A^T (A A^T)^-1 b
        try:
            fz = (A.T @ torch.linalg.solve(A @ A.T + 1.0e-4 * torch.eye(3, device=device, dtype=dtype), b)).squeeze(1)
        except Exception:
            fz = torch.ones((ns,), device=device, dtype=dtype) * (F_des[e, 2] / ns)

        fz = torch.clamp(fz, args.min_fz, args.max_fz)

        # friction clamp for horizontal components
        max_fxy = args.mu * fz
        fxy_norm = torch.sqrt(fx_i * fx_i + fy_i * fy_i).clamp_min(1.0e-9)
        scale = torch.clamp(max_fxy / fxy_norm, max=1.0)
        fx = fx_i * scale
        fy = fy_i * scale

        foot_forces[e, stance_ids, 0] = fx
        foot_forces[e, stance_ids, 1] = fy
        foot_forces[e, stance_ids, 2] = fz

    desired_wrench = torch.cat([F_des, M_des], dim=1)
    return foot_forces, desired_wrench


def jt_force_to_tau(jv_feet, foot_forces):
    # jv_feet [N,4,3,12], foot_forces [N,4,3]
    tau = torch.einsum("nfij,nfi->nj", jv_feet, foot_forces)
    return args.tau_sign * args.tau_scale * tau


def print_debug(step, x_hat, base_ref, out, foot_forces, desired_wrench, tau_cmd, action, robot):
    print("\n" + "=" * 132)
    print(f"[B8-q BASE REF FORCE TRACKING] step={step}")
    print("=" * 132)
    print("[support-region]")
    print("stance_count:", int(out.stance_count[0].detach().cpu()))
    print("support_center_xy:", out.support_center_xy[0].detach().cpu().numpy())
    print("current_xy:", out.current_xy[0].detach().cpu().numpy())
    print("target_xy:", out.target_xy[0].detach().cpu().numpy())
    print("margin_to_edge:", float(out.margin_to_edge[0].detach().cpu()))
    print("swing_allowed:", bool(out.swing_allowed[0].detach().cpu()))

    print("\n[base tracking]")
    print("base pos xyz:", x_hat[0, 0:3].detach().cpu().numpy())
    print("base rpy:", x_hat[0, 3:6].detach().cpu().numpy())
    print("base ref:", base_ref[0].detach().cpu().numpy())
    print("pos err xyz:", (base_ref[0, 0:3] - x_hat[0, 0:3]).detach().cpu().numpy())
    print("rpy err:", (base_ref[0, 3:6] - x_hat[0, 3:6]).detach().cpu().numpy())

    print("\n[force/wrench]")
    print("desired_wrench Fx,Fy,Fz,Mx,My,Mz:", desired_wrench[0].detach().cpu().numpy())
    print("foot_forces env0 LF/RF/LH/RH:")
    print(foot_forces[0].detach().cpu().numpy())

    print("\n[tau/action]")
    print("tau_cmd:", tau_cmd[0].detach().cpu().numpy())
    print("tau max_abs:", float(tau_cmd.abs().max().detach().cpu()))
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

    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype
    foot_indices = get_foot_indices(robot)
    mass = get_mass(robot, device, dtype)

    cfg = SupportRegionRefConfig(
        alpha=args.alpha,
        margin=args.margin,
        max_shift_per_step=args.max_shift_per_step,
        height_ref=args.height_ref,
    )

    print("\n" + "=" * 132)
    print("[INFO] Starting B8-q base reference force tracking")
    print("mass:", mass, "cfg:", cfg)
    print("target_mode:", args.target_mode)
    print("=" * 132)

    zero = torch.zeros((args.num_envs, 12), device=device, dtype=dtype)

    # Settle with zero effort. Implicit PD still stabilizes due to actuator config.
    for step in range(args.settle_steps):
        if not simulation_app.is_running():
            break
        env.step(zero)
        if step % max(args.print_every, 1) == 0:
            x = make_x_hat(robot, velocity_frame="world")
            print(f"[SETTLE] step={step} pos={x[0,0:3].detach().cpu().numpy()} rpy={x[0,3:6].detach().cpu().numpy()}")

    prev_base_ref = None
    stance_mask = torch.ones((args.num_envs, 4), device=device, dtype=dtype)

    for step in range(args.probe_steps):
        if not simulation_app.is_running():
            break

        x_hat = make_x_hat(robot, velocity_frame="world")
        foot_pos = robot.data.body_pos_w[:, foot_indices, :]
        base_pos = x_hat[:, 0:3]
        base_rpy = x_hat[:, 3:6]
        base_vel = x_hat[:, 6:9]
        base_ang_vel = x_hat[:, 9:12]

        out = compute_support_region_ref(
            foot_pos_w=foot_pos,
            base_pos_w=base_pos,
            base_rpy_w=base_rpy,
            stance_mask=stance_mask,
            prev_base_ref=prev_base_ref,
            cfg=cfg,
        )
        base_ref = out.base_ref.detach().clone()

        if args.target_mode == "manual":
            base_ref[:, 0] = base_pos[:, 0] + args.manual_dx
            base_ref[:, 1] = base_pos[:, 1] + args.manual_dy
            base_ref[:, 2] = args.height_ref

        prev_base_ref = base_ref.detach().clone()

        foot_forces, desired_wrench = distribute_wrench_to_feet(
            base_pos=base_pos,
            base_ref=base_ref,
            base_vel=base_vel,
            base_rpy=base_rpy,
            base_ang_vel=base_ang_vel,
            foot_pos=foot_pos,
            stance_mask=stance_mask,
            mass=mass,
        )

        jv_feet = get_jacobian_feet(robot, foot_indices)
        tau_cmd = jt_force_to_tau(jv_feet, foot_forces)
        tau_cmd = torch.clamp(tau_cmd, -args.max_total_tau, args.max_total_tau)

        action = tau_cmd
        env.step(action)

        if step % max(args.print_every, 1) == 0:
            print_debug(step, x_hat, base_ref, out, foot_forces, desired_wrench, tau_cmd, action, robot)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
