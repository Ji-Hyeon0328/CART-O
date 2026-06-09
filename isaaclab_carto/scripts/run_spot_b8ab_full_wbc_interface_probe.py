# isaaclab_carto/scripts/run_spot_b8ab_full_wbc_interface_probe.py
#
# B8-ab: Full floating-base WBC interface probe.
#
# Why:
#   The slides/design describe a full-body WBC:
#
#     M(q) qdd + h(q,qd) = S^T tau + Jc^T f
#
#   with task objectives:
#     base acceleration tracking
#     swing foot acceleration tracking
#     MPC GRF tracking
#     torque penalties / smoothness
#
#   B8-y/B8-z/B8-aa confirmed the software interface and contact schedule,
#   but only used bridge/WBC-lite approximations. The next firm step is to
#   inspect the exact Isaac tensors needed for a full floating-base WBC QP.
#
# This script does NOT solve the final WBC yet.
# It checks:
#   - full mass matrix shape
#   - bias/gravity/coriolis availability
#   - full foot Jacobian shape and base/joint column split
#   - base DOF ordering assumptions
#   - selection matrix shape
#   - floating-base dynamics row dimensions
#   - stance/swing contact indexing
#
# It also keeps the robot upright using the stable B8-y bridge-style torque
# so the probe runs in a meaningful pose.

import os
import sys
import argparse
from typing import Any, Optional

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CARTO/TRACER B8-ab full WBC interface probe")

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--print_every", type=int, default=20)
parser.add_argument("--test_leg", type=str, default="RF", choices=["LF", "RF", "LH", "RH"])

# Env / actuator
parser.add_argument("--spawn_z", type=float, default=0.60)
parser.add_argument("--pd_scale", type=float, default=0.65)

# Simple stable standing bridge
parser.add_argument("--height_ref", type=float, default=0.665)
parser.add_argument("--kp_xy", type=float, default=20.0)
parser.add_argument("--kd_xy", type=float, default=14.0)
parser.add_argument("--kp_z", type=float, default=180.0)
parser.add_argument("--kd_z", type=float, default=35.0)
parser.add_argument("--tau_sign", type=float, default=-1.0)
parser.add_argument("--tau_force_scale", type=float, default=0.22)
parser.add_argument("--kp_posture", type=float, default=8.0)
parser.add_argument("--kd_posture", type=float, default=0.8)
parser.add_argument("--max_posture_tau", type=float, default=8.0)
parser.add_argument("--max_tau", type=float, default=18.0)

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


def try_call(name, fn):
    try:
        out = fn()
        print(f"[OK] {name}: type={type(out)}, shape={getattr(out, 'shape', None)}")
        return out
    except Exception as exc:
        print(f"[MISS] {name}: {type(exc).__name__}: {exc}")
        return None


def get_full_mass_matrix(robot):
    return try_call("root_physx_view.get_generalized_mass_matrices()", lambda: robot.root_physx_view.get_generalized_mass_matrices())


def get_bias_like_terms(robot):
    terms = {}
    terms["gravity"] = try_call("root_physx_view.get_generalized_gravity_forces()", lambda: robot.root_physx_view.get_generalized_gravity_forces())
    terms["coriolis"] = try_call("root_physx_view.get_coriolis_and_centrifugal_forces()", lambda: robot.root_physx_view.get_coriolis_and_centrifugal_forces())
    terms["bias"] = try_call("root_physx_view.get_generalized_bias_forces()", lambda: robot.root_physx_view.get_generalized_bias_forces())
    terms["dof_forces"] = try_call("robot.data.applied_torque", lambda: robot.data.applied_torque)
    return terms


def get_full_jacobian(robot):
    return try_call("root_physx_view.get_jacobians()", lambda: robot.root_physx_view.get_jacobians())


def get_full_velocity(robot):
    # For full WBC ordering we need base twist + joint velocity.
    # We print both candidates; final WBC will choose one consistent with Jacobian columns.
    root_lin = getattr(robot.data, "root_lin_vel_w", None)
    root_ang = getattr(robot.data, "root_ang_vel_w", None)
    if root_lin is not None and root_ang is not None:
        qd_full_world = torch.cat([root_lin, root_ang, robot.data.joint_vel], dim=1)
    else:
        qd_full_world = None

    root_lin_b = getattr(robot.data, "root_lin_vel_b", None)
    root_ang_b = getattr(robot.data, "root_ang_vel_b", None)
    if root_lin_b is not None and root_ang_b is not None:
        qd_full_body = torch.cat([root_lin_b, root_ang_b, robot.data.joint_vel], dim=1)
    else:
        qd_full_body = None

    return qd_full_world, qd_full_body


def simple_standing_tau(robot, foot_indices, q_nom):
    """Stable standing torque based on the sign convention found earlier.

    This is only for keeping the robot upright during tensor probing.
    """
    x_hat = make_x_hat(robot, velocity_frame="world")
    foot_pos = robot.data.body_pos_w[:, foot_indices, :]
    jac = robot.root_physx_view.get_jacobians()
    if jac.shape[-1] == robot.num_joints + 6:
        jv = jac[:, foot_indices, 0:3, 6:]
    else:
        jv = jac[:, foot_indices, 0:3, :]

    n = robot.data.joint_pos.shape[0]
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype

    mass = 31.6
    try:
        mass = float(robot.root_physx_view.get_masses().sum().detach().cpu())
    except Exception:
        pass

    pos_err = torch.zeros((n, 3), device=device, dtype=dtype)
    pos_err[:, 2] = args.height_ref - x_hat[:, 2]

    F_des = torch.zeros((n, 3), device=device, dtype=dtype)
    F_des[:, 0] = -args.kd_xy * x_hat[:, 6]
    F_des[:, 1] = -args.kd_xy * x_hat[:, 7]
    F_des[:, 2] = mass * 9.81 + args.kp_z * pos_err[:, 2] - args.kd_z * x_hat[:, 8]

    f = torch.zeros((n, 4, 3), device=device, dtype=dtype)
    f[:, :, 0] = F_des[:, 0:1] / 4.0
    f[:, :, 1] = F_des[:, 1:2] / 4.0
    f[:, :, 2] = torch.clamp(F_des[:, 2:3] / 4.0, 5.0, 180.0)

    tau_force = torch.einsum("nfij,nfi->nj", jv, f)
    tau_force = args.tau_sign * args.tau_force_scale * tau_force

    q = robot.data.joint_pos
    qd = robot.data.joint_vel
    tau_post = args.kp_posture * (q_nom - q) - args.kd_posture * qd
    tau_post = torch.clamp(tau_post, -args.max_posture_tau, args.max_posture_tau)

    tau = torch.clamp(tau_force + tau_post, -args.max_tau, args.max_tau)
    return tau


def inspect_full_wbc_shapes(robot, foot_indices, step):
    print("\n" + "=" * 140)
    print(f"[B8-ab FULL WBC INTERFACE PROBE] step={step}")
    print("=" * 140)

    print("\n[robot metadata]")
    print("num_envs:", robot.data.joint_pos.shape[0])
    print("num_joints:", robot.num_joints)
    print("joint_names:", robot.joint_names)
    print("body_names:", robot.body_names)
    print("foot_indices:", foot_indices, "foot_names:", FOOT_NAMES)

    x_hat = make_x_hat(robot, velocity_frame="world")
    print("\n[base state]")
    print("base pos xyz:", x_hat[0, 0:3].detach().cpu().numpy())
    print("base rpy:", x_hat[0, 3:6].detach().cpu().numpy())
    print("base lin vel:", x_hat[0, 6:9].detach().cpu().numpy())
    print("base ang vel:", x_hat[0, 9:12].detach().cpu().numpy())

    M = get_full_mass_matrix(robot)
    jac = get_full_jacobian(robot)
    terms = get_bias_like_terms(robot)
    qd_w, qd_b = get_full_velocity(robot)

    print("\n[full velocity candidates]")
    print("qd_full_world shape:", None if qd_w is None else qd_w.shape)
    print("qd_full_body shape:", None if qd_b is None else qd_b.shape)
    if qd_w is not None:
        print("qd_full_world env0 first 6:", qd_w[0, :6].detach().cpu().numpy())
    if qd_b is not None:
        print("qd_full_body env0 first 6:", qd_b[0, :6].detach().cpu().numpy())

    print("\n[WBC expected dimensions]")
    nq_full = 6 + robot.num_joints
    ntau = robot.num_joints
    nf_all = 4 * 3
    print("nq_full = 6 + num_joints =", nq_full)
    print("tau dim:", ntau)
    print("contact force dim:", nf_all)
    print("QP v0 variables expected: qdd_full + tau + f =", nq_full + ntau + nf_all)

    if M is not None:
        print("\n[mass matrix check]")
        print("M shape:", M.shape)
        print("M env0 top-left 6x6:")
        print(M[0, :6, :6].detach().cpu().numpy() if M.dim() == 3 else M[:6, :6].detach().cpu().numpy())
        print("M env0 joint block diag sample:")
        Me = M[0] if M.dim() == 3 else M
        print(torch.diagonal(Me)[0:min(18, Me.shape[-1])].detach().cpu().numpy())

    if jac is not None:
        print("\n[jacobian check]")
        print("jac shape:", jac.shape)
        print("Assumption A: last dimension includes 6 floating base columns + 12 joint columns if shape[-1] == 18.")
        if jac.shape[-1] == robot.num_joints + 6:
            Jfeet_full = jac[:, foot_indices, 0:3, :]
            Jfeet_base = Jfeet_full[..., :6]
            Jfeet_joint = Jfeet_full[..., 6:]
            print("Jfeet_full shape:", Jfeet_full.shape)
            print("Jfeet_base shape:", Jfeet_base.shape)
            print("Jfeet_joint shape:", Jfeet_joint.shape)
            print("RF/LF selected foot J_full env0:")
            leg = LEG_TO_ID[args.test_leg]
            print(Jfeet_full[0, leg].detach().cpu().numpy())
        else:
            Jfeet_joint = jac[:, foot_indices, 0:3, :]
            print("Jacobian appears joint-only. Jfeet_joint shape:", Jfeet_joint.shape)

    print("\n[selection matrix S]")
    S = torch.zeros((robot.num_joints, 6 + robot.num_joints))
    S[:, 6:] = torch.eye(robot.num_joints)
    print("S shape:", tuple(S.shape))
    print("S maps qdd_full/tau relation as S^T tau in full dynamics.")

    print("\n[contact schedule example]")
    stance_mask = torch.ones((robot.data.joint_pos.shape[0], 4), device=robot.data.joint_pos.device)
    stance_mask[:, LEG_TO_ID[args.test_leg]] = 0.0
    print("test_leg:", args.test_leg)
    print("stance_mask LF/RF/LH/RH:", stance_mask[0].detach().cpu().numpy())
    print("In full WBC: contact force rows for swing leg should be constrained to zero.")

    print("\n[full WBC QP v0 intended form]")
    print("Decision: x = [qdd_full(18), tau(12), f(12)]")
    print("Equality dynamics: M qdd + h = S^T tau + Jc^T f")
    print("Stance task: J_stance qdd + Jdot_qdot ~= 0")
    print("Swing task: J_swing qdd + Jdot_qdot ~= xdd_swing_ref")
    print("MPC force tracking: f_stance ~= f_mpc, f_swing = 0")
    print("Regularization: posture/nullspace, torque magnitude, torque rate")
    print("=" * 140 + "\n")


def main():
    env_cfg = CartoEffortEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    patch_flat_safe_env(env_cfg)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()
    robot = env.scene["robot"]
    print_robot_debug_info(robot)

    foot_indices = get_foot_indices(robot)
    q_nom = robot.data.joint_pos.detach().clone()

    for step in range(args.steps):
        if not simulation_app.is_running():
            break

        tau = simple_standing_tau(robot, foot_indices, q_nom)
        env.step(tau)

        if step % max(args.print_every, 1) == 0:
            inspect_full_wbc_shapes(robot, foot_indices, step)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
