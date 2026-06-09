# isaaclab_carto/scripts/run_spot_b8q2_upright_base_ref_probe.py
#
# B8-q2: upright-first base reference probe.
#
# Why this exists:
#   B8-q failed because the robot already fell backward during the zero-effort
#   settle phase before the controller actually started. Once the base pitch is
#   around -1 rad, the support region and force distribution are no longer a
#   meaningful upright locomotion test.
#
# B8-q2 changes the diagnostic:
#   1. Do NOT do a long zero-effort settle.
#   2. Capture the reset joint pose as q_nom.
#   3. From the first control step, apply:
#        posture torque + vertical support + gentle horizontal base-ref force
#   4. First verify upright hold.
#   5. Then verify small manual/base_ref shift.
#
# This is still not final MPC/WBC.
# It is an authority/stability bridge before wrapping the cleaner MPC/WBC layer.

import os
import sys
import argparse
from typing import Any

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CARTO/TRACER B8-q2 upright-first base ref probe")

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--warmup_steps", type=int, default=120)
parser.add_argument("--probe_steps", type=int, default=360)
parser.add_argument("--print_every", type=int, default=20)

# Env / actuator
parser.add_argument("--spawn_z", type=float, default=0.60)
parser.add_argument("--pd_scale", type=float, default=0.65)

# Target mode
parser.add_argument("--target_mode", type=str, default="manual", choices=["hold", "manual", "support_center"])
parser.add_argument("--manual_dx", type=float, default=0.025)
parser.add_argument("--manual_dy", type=float, default=0.0)
parser.add_argument("--height_ref", type=float, default=0.54)
parser.add_argument("--max_shift_per_step", type=float, default=0.001)

# Support region ref config
parser.add_argument("--alpha", type=float, default=0.20)
parser.add_argument("--margin", type=float, default=0.030)

# Base force gains. Keep gentle first.
parser.add_argument("--kp_xy", type=float, default=40.0)
parser.add_argument("--kd_xy", type=float, default=18.0)
parser.add_argument("--kp_z", type=float, default=260.0)
parser.add_argument("--kd_z", type=float, default=45.0)

# Optional attitude moment. Disabled by default because B8-q overreacted when fallen.
parser.add_argument("--use_attitude_moment", action="store_true")
parser.add_argument("--kp_roll", type=float, default=40.0)
parser.add_argument("--kd_roll", type=float, default=8.0)
parser.add_argument("--kp_pitch", type=float, default=40.0)
parser.add_argument("--kd_pitch", type=float, default=8.0)

# Joint posture stabilizer
parser.add_argument("--kp_posture", type=float, default=8.0)
parser.add_argument("--kd_posture", type=float, default=0.8)
parser.add_argument("--max_posture_tau", type=float, default=8.0)

# Force limits
parser.add_argument("--mass_override", type=float, default=32.0)
parser.add_argument("--mu", type=float, default=0.7)
parser.add_argument("--min_fz", type=float, default=10.0)
parser.add_argument("--max_fz", type=float, default=130.0)

# Torque limits/scales
parser.add_argument("--tau_force_scale", type=float, default=0.35)
parser.add_argument("--tau_posture_scale", type=float, default=1.0)
parser.add_argument("--tau_sign", type=float, default=1.0, choices=[1.0, -1.0])
parser.add_argument("--max_total_tau", type=float, default=14.0)

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
        masses = robot.root_physx_view.get_masses()
        return float(masses.sum().detach().cpu())
    except Exception:
        return args.mass_override


def get_jacobian_feet(robot, foot_indices):
    jac = robot.root_physx_view.get_jacobians()
    if jac.shape[-1] == robot.num_joints + 6:
        jac = jac[..., 6:]
    return jac[:, foot_indices, 0:3, :]


def make_base_ref(step, x_hat, foot_pos, stance_mask, prev_base_ref, cfg):
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

    # Warmup: hold current xy and target height.
    if step < args.warmup_steps or args.target_mode == "hold":
        base_ref[:, 0:2] = base_pos[:, 0:2]
        base_ref[:, 2] = args.height_ref
        base_ref[:, 3:5] = 0.0
        return base_ref, out

    if args.target_mode == "manual":
        # Use a rate-limited manual offset relative to the base position at this step.
        desired = base_pos[:, 0:2].clone()
        desired[:, 0] += args.manual_dx
        desired[:, 1] += args.manual_dy

        if prev_base_ref is None:
            prev_xy = base_pos[:, 0:2]
        else:
            prev_xy = prev_base_ref[:, 0:2]
        delta = desired - prev_xy
        norm = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1.0e-9)
        scale = torch.clamp(args.max_shift_per_step / norm, max=1.0)
        base_ref[:, 0:2] = prev_xy + scale * delta
        base_ref[:, 2] = args.height_ref
        base_ref[:, 3:5] = 0.0

    return base_ref, out


def compute_foot_forces(x_hat, base_ref, foot_pos, stance_mask, mass):
    device = x_hat.device
    dtype = x_hat.dtype
    n = x_hat.shape[0]
    g = 9.81

    base_pos = x_hat[:, 0:3]
    base_rpy = x_hat[:, 3:6]
    base_vel = x_hat[:, 6:9]
    base_ang = x_hat[:, 9:12]

    pos_err = base_ref[:, 0:3] - base_pos
    rpy_err = base_ref[:, 3:6] - base_rpy

    F_des = torch.zeros((n, 3), device=device, dtype=dtype)
    M_des = torch.zeros((n, 3), device=device, dtype=dtype)

    F_des[:, 0] = args.kp_xy * pos_err[:, 0] - args.kd_xy * base_vel[:, 0]
    F_des[:, 1] = args.kp_xy * pos_err[:, 1] - args.kd_xy * base_vel[:, 1]
    F_des[:, 2] = mass * g + args.kp_z * pos_err[:, 2] - args.kd_z * base_vel[:, 2]

    if args.use_attitude_moment:
        M_des[:, 0] = args.kp_roll * rpy_err[:, 0] - args.kd_roll * base_ang[:, 0]
        M_des[:, 1] = args.kp_pitch * rpy_err[:, 1] - args.kd_pitch * base_ang[:, 1]
    else:
        M_des[:, 0:2] = 0.0
    M_des[:, 2] = 0.0

    foot_forces = torch.zeros((n, 4, 3), device=device, dtype=dtype)

    for e in range(n):
        stance_ids = torch.where(stance_mask[e] > 0.5)[0]
        ns = max(int(stance_ids.numel()), 1)

        # Equal vertical support for first robust diagnostic.
        fz = torch.ones((ns,), device=device, dtype=dtype) * (F_des[e, 2] / ns)

        # Optional roll/pitch moment via vertical force correction.
        if args.use_attitude_moment and ns >= 3:
            r = foot_pos[e, stance_ids, :] - base_pos[e].unsqueeze(0)
            A = torch.stack([torch.ones((ns,), device=device, dtype=dtype), r[:, 1], -r[:, 0]], dim=0)
            b = torch.stack([F_des[e, 2], M_des[e, 0], M_des[e, 1]], dim=0).unsqueeze(1)
            try:
                fz = (A.T @ torch.linalg.solve(A @ A.T + 1e-4 * torch.eye(3, device=device, dtype=dtype), b)).squeeze(1)
            except Exception:
                pass

        fz = torch.clamp(fz, args.min_fz, args.max_fz)

        fx_i = F_des[e, 0] / ns
        fy_i = F_des[e, 1] / ns
        fxy_norm = torch.sqrt(fx_i * fx_i + fy_i * fy_i).clamp_min(1.0e-9)
        max_fxy = args.mu * fz
        scale = torch.clamp(max_fxy / fxy_norm, max=1.0)

        foot_forces[e, stance_ids, 0] = fx_i * scale
        foot_forces[e, stance_ids, 1] = fy_i * scale
        foot_forces[e, stance_ids, 2] = fz

    wrench = torch.cat([F_des, M_des], dim=1)
    return foot_forces, wrench


def jt_force_to_tau(jv_feet, foot_forces):
    return torch.einsum("nfij,nfi->nj", jv_feet, foot_forces)


def posture_tau(robot, q_nom):
    q = robot.data.joint_pos
    qd = robot.data.joint_vel
    tau = args.kp_posture * (q_nom - q) - args.kd_posture * qd
    return torch.clamp(tau, -args.max_posture_tau, args.max_posture_tau)


def print_debug(step, x_hat, base_ref, out, foot_forces, wrench, tau_force, tau_post, action, robot):
    print("\n" + "=" * 132)
    print(f"[B8-q2 UPRIGHT BASE REF PROBE] step={step}")
    print("=" * 132)
    print("mode:", "warmup" if step < args.warmup_steps else args.target_mode)
    print("[support-region]")
    print("support_center_xy:", out.support_center_xy[0].detach().cpu().numpy())
    print("current_xy:", out.current_xy[0].detach().cpu().numpy())
    print("target_xy:", out.target_xy[0].detach().cpu().numpy())
    print("margin_to_edge:", float(out.margin_to_edge[0].detach().cpu()))
    print("swing_allowed:", bool(out.swing_allowed[0].detach().cpu()))

    print("\n[base]")
    print("base pos xyz:", x_hat[0, 0:3].detach().cpu().numpy())
    print("base rpy:", x_hat[0, 3:6].detach().cpu().numpy())
    print("base vel:", x_hat[0, 6:9].detach().cpu().numpy())
    print("base ref:", base_ref[0].detach().cpu().numpy())
    print("pos err:", (base_ref[0, 0:3] - x_hat[0, 0:3]).detach().cpu().numpy())
    print("rpy err:", (base_ref[0, 3:6] - x_hat[0, 3:6]).detach().cpu().numpy())

    print("\n[force]")
    print("desired wrench Fx,Fy,Fz,Mx,My,Mz:", wrench[0].detach().cpu().numpy())
    print("foot forces LF/RF/LH/RH:")
    print(foot_forces[0].detach().cpu().numpy())

    print("\n[tau]")
    print("tau_force:", tau_force[0].detach().cpu().numpy())
    print("tau_posture:", tau_post[0].detach().cpu().numpy())
    print("action:", action[0].detach().cpu().numpy())
    print("tau_force max_abs:", float(tau_force.abs().max().detach().cpu()))
    print("tau_posture max_abs:", float(tau_post.abs().max().detach().cpu()))
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
    mass = get_mass(robot)

    q_nom = robot.data.joint_pos.detach().clone()
    print("[INFO] Captured q_nom from reset:", q_nom[0].detach().cpu().numpy())

    cfg = SupportRegionRefConfig(
        alpha=args.alpha,
        margin=args.margin,
        max_shift_per_step=args.max_shift_per_step,
        height_ref=args.height_ref,
    )

    stance_mask = torch.ones((args.num_envs, 4), device=device, dtype=dtype)
    prev_base_ref = None

    print("\n" + "=" * 132)
    print("[INFO] Starting B8-q2 upright-first base ref probe")
    print("mass:", mass, "cfg:", cfg)
    print("target_mode:", args.target_mode)
    print("warmup_steps:", args.warmup_steps)
    print("=" * 132)

    total_steps = args.warmup_steps + args.probe_steps

    for step in range(total_steps):
        if not simulation_app.is_running():
            break

        x_hat = make_x_hat(robot, velocity_frame="world")
        foot_pos = robot.data.body_pos_w[:, foot_indices, :]

        base_ref, out = make_base_ref(step, x_hat, foot_pos, stance_mask, prev_base_ref, cfg)
        prev_base_ref = base_ref.detach().clone()

        foot_forces, wrench = compute_foot_forces(x_hat, base_ref, foot_pos, stance_mask, mass)
        jv_feet = get_jacobian_feet(robot, foot_indices)
        tau_force = args.tau_sign * args.tau_force_scale * jt_force_to_tau(jv_feet, foot_forces)
        tau_post = args.tau_posture_scale * posture_tau(robot, q_nom)

        action = torch.clamp(tau_force + tau_post, -args.max_total_tau, args.max_total_tau)

        env.step(action)

        if step % max(args.print_every, 1) == 0:
            print_debug(step, x_hat, base_ref, out, foot_forces, wrench, tau_force, tau_post, action, robot)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
