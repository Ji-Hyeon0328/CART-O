# isaaclab_carto/scripts/run_spot_b8ah_stable_warmup_decoupled_preshift.py
#
# B8-au: fall-safe gated release + RF single-step probe.
#
# Problem found in B8-af:
#   With --require_margin, swing_enabled became False, and the script
#   recomputed both contact mask and base_ref using all-stance [1,1,1,1].
#   Therefore no real pre-shift toward the future 3-leg support region occurred.
#
# Fix:
#   Use two masks:
#
#     ref_mask:
#       future support mask, e.g. RF swing -> [1,0,1,1]
#       used for support-region base_ref pre-shift
#
#     contact_mask:
#       actual force/contact mask used by MPC/WBC
#       remains [1,1,1,1] until the future-support margin is safe
#
#   Once the trial/future support gate passes:
#       contact_mask -> [1,0,1,1]
#
# This script is a diagnostic:
#   Keep the stable B8-ah no-assist controller.
#   Diagnose whether swing target z is truly separated from measured foot z.
#   Diagnose whether no-clearance is target-generation, contact-sticking, or actuator-interface related.

import os
import sys
import argparse
from typing import Any

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CARTO/TRACER B8-au fall-safe gated release + RF single-step probe")

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--warmup_steps", type=int, default=120)
parser.add_argument("--shift_steps", type=int, default=700)
parser.add_argument("--lift_steps", type=int, default=160)
parser.add_argument("--hold_steps", type=int, default=120)
parser.add_argument("--lower_steps", type=int, default=160)
parser.add_argument("--settle_steps", type=int, default=220)
parser.add_argument("--print_every", type=int, default=20)

# B8-as: contact-calibrated warmup.
parser.add_argument("--enable_contact_calibration", action="store_true")
parser.add_argument("--calib_hold_initial_targets", action="store_true")
parser.add_argument("--calib_hold_all_joints", action="store_true")
parser.add_argument("--calib_velocity_zero", action="store_true")
parser.add_argument("--calib_target_max_delta", type=float, default=0.45)
parser.add_argument("--calib_contact_tol", type=float, default=0.018)
parser.add_argument("--calib_min_ready_legs", type=int, default=4)
parser.add_argument("--calib_max_abs_roll", type=float, default=0.12)
parser.add_argument("--calib_max_abs_pitch", type=float, default=0.12)
parser.add_argument("--calib_require_ready_for_shift", action="store_true")

# B8-at: bypass conservative support-margin gate once contact calibration is ready.
parser.add_argument("--release_on_contact_ready", action="store_true")
parser.add_argument("--contact_ready_release_after_phase_step", type=int, default=20)
parser.add_argument("--contact_ready_release_max_abs_roll", type=float, default=0.18)
parser.add_argument("--contact_ready_release_max_abs_pitch", type=float, default=0.18)

# B8-au: fall-safe release conditions. Unlike B8-at, this does not release
# when the support margin is negative or when the base has already sagged/rolled.
parser.add_argument("--fallsafe_release", action="store_true")
parser.add_argument("--fallsafe_min_margin", type=float, default=0.005)
parser.add_argument("--fallsafe_min_base_z", type=float, default=0.575)
parser.add_argument("--fallsafe_max_abs_roll", type=float, default=0.060)
parser.add_argument("--fallsafe_max_abs_pitch", type=float, default=0.080)
parser.add_argument("--fallsafe_require_current_contact_ready", action="store_true")
parser.add_argument("--disable_unsafe_ready_forced", action="store_true",
                    help="If set, ignore B8-at ready_forced and use only fallsafe_ready_forced.")

parser.add_argument("--test_leg", type=str, default="RF", choices=["LF", "RF", "LH", "RH"])

# Env / actuator
parser.add_argument("--spawn_z", type=float, default=0.60)
parser.add_argument("--pd_scale", type=float, default=0.35)

# Reference
parser.add_argument("--height_ref", type=float, default=0.625)
parser.add_argument("--warmup_height_ref", type=float, default=0.600)
parser.add_argument("--freeze_xy_during_warmup", action="store_true")
parser.add_argument("--freeze_xy_until_shift", type=int, default=80)
parser.add_argument("--alpha", type=float, default=0.04)
parser.add_argument("--margin", type=float, default=0.055)
parser.add_argument("--max_shift_per_step", type=float, default=0.00012)

# Future-support gate
parser.add_argument("--future_margin_gate", type=float, default=0.040)
parser.add_argument("--future_margin_release", type=float, default=0.025)
parser.add_argument("--min_shift_steps_before_release", type=int, default=260)
parser.add_argument("--force_release_after_shift", action="store_true")

# MPC GRF
parser.add_argument("--kp_xy", type=float, default=20.0)
parser.add_argument("--kd_xy", type=float, default=14.0)
parser.add_argument("--kp_z", type=float, default=180.0)
parser.add_argument("--kd_z", type=float, default=35.0)
parser.add_argument("--mass_override", type=float, default=32.0)
parser.add_argument("--mu", type=float, default=0.70)
parser.add_argument("--min_fz", type=float, default=8.0)
parser.add_argument("--max_fz", type=float, default=180.0)

# Full WBC QP v1 weights
parser.add_argument("--w_dyn", type=float, default=25.0)
parser.add_argument("--w_base_acc", type=float, default=3.0)
parser.add_argument("--w_stance_acc", type=float, default=35.0)
parser.add_argument("--w_swing_acc", type=float, default=25.0)
parser.add_argument("--w_force_track", type=float, default=1.0)
parser.add_argument("--w_swing_force_zero", type=float, default=80.0)
parser.add_argument("--w_tau_posture", type=float, default=0.08)
parser.add_argument("--w_tau_reg", type=float, default=0.03)
parser.add_argument("--w_qdd_reg", type=float, default=0.03)

# Swing task. Default disabled by zero clearance.
parser.add_argument("--swing_clearance", type=float, default=0.000)
parser.add_argument("--kp_swing_z", type=float, default=80.0)
parser.add_argument("--kd_swing_z", type=float, default=12.0)
parser.add_argument("--max_swing_acc", type=float, default=5.0)

# B8-ak swing-target diagnostics
parser.add_argument("--anchor_swing_on_release", action="store_true")
parser.add_argument("--target_z_mode", type=str, default="ramp", choices=["ramp", "step", "hold"])
parser.add_argument("--diagnostic_clearance", type=float, default=None,
                    help="If set, use this value instead of --swing_clearance for target z diagnostics.")
parser.add_argument("--print_all_feet_z", action="store_true")

# B8-am crouched nominal stance diagnostic.
# Applies selected-joint implicit targets to HY/KN joints only, with a slow ramp.
# This avoids full-vector target overwrite while making a walking-ready bent-knee posture.
parser.add_argument("--enable_crouch_target", action="store_true")
parser.add_argument("--crouch_start_step", type=int, default=80)
parser.add_argument("--crouch_ramp_steps", type=int, default=360)
parser.add_argument("--crouch_hold_before_shift", type=int, default=80)
parser.add_argument("--crouch_base", type=str, default="initial", choices=["initial", "current", "custom"])
parser.add_argument("--crouch_hy_offset", type=float, default=0.0)
parser.add_argument("--crouch_kn_offset", type=float, default=0.0)
parser.add_argument("--crouch_hy_abs", type=float, default=0.25)
parser.add_argument("--crouch_kn_abs", type=float, default=-0.75)
parser.add_argument("--crouch_max_delta", type=float, default=0.75)
parser.add_argument("--use_crouch_q_nom", action="store_true")

# B8-an: Jacobian IK swing adapter for the selected leg.
# Uses the current foot Jacobian to map foot target error -> selected joint target delta.
# This is a robot-adapter layer: WBC/MPC still handles stance support and force masking.
parser.add_argument("--enable_jacobian_ik_swing", action="store_true")
parser.add_argument("--ik_start_profile", type=float, default=0.05)
parser.add_argument("--ik_gain", type=float, default=0.65)
parser.add_argument("--ik_damping", type=float, default=0.025)
parser.add_argument("--ik_max_joint_delta", type=float, default=0.18)
parser.add_argument("--ik_include_xy", action="store_true")
parser.add_argument("--ik_use_hx", action="store_true")
parser.add_argument("--ik_target_scale_z", type=float, default=1.0)
parser.add_argument("--ik_apply_velocity_zero", action="store_true")

# B8-aq: single-step landing target.
# The swing target moves RF forward in world +x, then ramps z back down during lower.
parser.add_argument("--swing_forward_step", type=float, default=0.055)
parser.add_argument("--swing_lateral_step", type=float, default=0.0)
parser.add_argument("--landing_extra_down", type=float, default=0.000,
                    help="Optional lower target below anchor during final lower phase. Usually keep 0.")

# B8-ap: controlled forced release gate.
# Used to test release + IK activation from the previously stable upright stance.
parser.add_argument("--controlled_force_release", action="store_true")
parser.add_argument("--force_release_after_phase_step", type=int, default=180)
parser.add_argument("--force_release_max_abs_roll", type=float, default=0.16)
parser.add_argument("--force_release_max_abs_pitch", type=float, default=0.16)
parser.add_argument("--force_release_require_margin", type=float, default=-999.0)
parser.add_argument("--disable_margin_gate_for_release", action="store_true")

# Torque
parser.add_argument("--max_tau", type=float, default=24.0)
parser.add_argument("--tau_output_sign", type=float, default=1.0, choices=[1.0, -1.0])
parser.add_argument("--tau_cmd_scale", type=float, default=1.0)

# Target alignment
parser.add_argument("--align_targets", action="store_true")
parser.add_argument("--target_mode", type=str, default="nominal", choices=["current", "nominal"])

# Optional very small swing assist. Default off.
parser.add_argument("--swing_target_assist", action="store_true")
parser.add_argument("--swing_hy_bias", type=float, default=0.0)
parser.add_argument("--swing_kn_bias", type=float, default=0.0)
parser.add_argument("--target_max_delta", type=float, default=0.12)

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
from isaaclab_carto.lowlevel.tracer_mpc_wbc_bridge import MpcWbcBridgeConfig, distribute_grf_ls  # noqa: E402
from isaaclab_carto.lowlevel.tracer_full_wbc_qp_v1 import FullWbcQpV1Config, solve_full_wbc_qp_v1  # noqa: E402


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


def smooth01(s):
    s = max(0.0, min(1.0, s))
    return float(0.5 - 0.5 * torch.cos(torch.tensor(torch.pi * s)).item())


def patch_flat_safe_env(env_cfg: Any):
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
    except Exception as exc:
        print(f"[WARN] terrain patch failed: {exc}")

    try:
        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, args.spawn_z)
    except Exception:
        pass

    try:
        for actuator in env_cfg.scene.robot.actuators.values():
            actuator.stiffness = float(actuator.stiffness) * args.pd_scale
            actuator.damping = float(actuator.damping) * args.pd_scale
        print(f"[INFO] Scaled implicit actuator stiffness/damping by {args.pd_scale}")
    except Exception as exc:
        print(f"[WARN] actuator scale failed: {exc}")


def get_foot_indices(robot):
    name_to_idx = {name: i for i, name in enumerate(robot.body_names)}
    return [name_to_idx[name] for name in FOOT_NAMES]


def get_mass(robot):
    try:
        return float(robot.root_physx_view.get_masses().sum().detach().cpu())
    except Exception:
        return args.mass_override


def get_phase(step):
    w, s, l, h, lo, st = (
        args.warmup_steps,
        args.shift_steps,
        args.lift_steps,
        args.hold_steps,
        args.lower_steps,
        args.settle_steps,
    )
    if step < w:
        return "warmup", 0.0, step
    t = step - w
    if t < s:
        return "shift", 0.0, t
    t -= s
    if t < l:
        return "lift", smooth01(t / max(l, 1)), t
    t -= l
    if t < h:
        return "hold_lift", smooth01(t / max(h, 1)), t
    t -= h
    if t < lo:
        return "lower", smooth01(1.0 - t / max(lo, 1)), t
    t -= lo
    if t < st:
        return "settle", 0.0, t
    return "done", 0.0, t


def all_stance_mask(num_envs, device, dtype):
    return torch.ones((num_envs, 4), device=device, dtype=dtype)


def future_swing_mask(num_envs, device, dtype):
    stance = torch.ones((num_envs, 4), device=device, dtype=dtype)
    stance[:, LEG_TO_ID[args.test_leg]] = 0.0
    return stance


def make_base_ref(x_hat, foot_pos, ref_mask, prev_base_ref, cfg, phase, phase_step):
    out = compute_support_region_ref(
        foot_pos_w=foot_pos,
        base_pos_w=x_hat[:, 0:3],
        base_rpy_w=x_hat[:, 3:6],
        stance_mask=ref_mask,
        prev_base_ref=prev_base_ref,
        cfg=cfg,
    )
    base_ref = out.base_ref.detach().clone()

    # B8-ah fix:
    # In B8-ag, warmup used support-region xy targets instead of current xy.
    # That produced a large horizontal command before any gait transition and
    # caused dog-sitting. During warmup, keep xy at current and use a lower
    # settle height. At the beginning of shift, optionally keep xy frozen for a
    # short time so the robot does not receive an immediate lateral/fore-aft jerk.
    if phase == "warmup" or (phase == "shift" and phase_step < args.freeze_xy_until_shift):
        base_ref[:, 0:2] = x_hat[:, 0:2]
    elif args.freeze_xy_during_warmup and prev_base_ref is not None:
        pass

    if phase == "warmup":
        base_ref[:, 2] = args.warmup_height_ref
    else:
        base_ref[:, 2] = args.height_ref

    base_ref[:, 3:5] = 0.0
    return base_ref, out


def gate_swing(phase, phase_step, trial_out, x_hat):
    if phase not in ["lift", "hold_lift", "lower"]:
        return False

    roll = float(x_hat[0, 3].detach().cpu())
    pitch = float(x_hat[0, 4].detach().cpu())
    roll_ok_default = abs(roll) < 0.20
    pitch_ok_default = abs(pitch) < 0.25

    trial_margin = float(trial_out.margin_to_edge[0].detach().cpu())
    trial_allowed = bool(trial_out.swing_allowed[0].detach().cpu())

    normal_ok = roll_ok_default and pitch_ok_default and trial_allowed and (trial_margin >= args.future_margin_release)

    old_forced = False
    if args.force_release_after_shift and phase in ["lift", "hold_lift", "lower"]:
        old_forced = roll_ok_default and pitch_ok_default and trial_allowed

    controlled_forced = False
    if args.controlled_force_release and phase in ["lift", "hold_lift", "lower"]:
        controlled_forced = (
            phase_step >= args.force_release_after_phase_step
            and abs(roll) < args.force_release_max_abs_roll
            and abs(pitch) < args.force_release_max_abs_pitch
            and trial_margin >= args.force_release_require_margin
        )

    if args.disable_margin_gate_for_release:
        return bool(old_forced or controlled_forced)
    return bool(normal_ok or old_forced or controlled_forced)


def gate_debug_flags(phase, phase_step, trial_out, x_hat):
    if phase not in ["lift", "hold_lift", "lower"]:
        return False, False

    roll = float(x_hat[0, 3].detach().cpu())
    pitch = float(x_hat[0, 4].detach().cpu())
    trial_margin = float(trial_out.margin_to_edge[0].detach().cpu())
    trial_allowed = bool(trial_out.swing_allowed[0].detach().cpu())

    old_forced = False
    if args.force_release_after_shift:
        old_forced = (abs(roll) < 0.20) and (abs(pitch) < 0.25) and trial_allowed

    controlled_forced = False
    if args.controlled_force_release:
        controlled_forced = (
            phase_step >= args.force_release_after_phase_step
            and abs(roll) < args.force_release_max_abs_roll
            and abs(pitch) < args.force_release_max_abs_pitch
            and trial_margin >= args.force_release_require_margin
        )
    return bool(old_forced), bool(controlled_forced)


def build_crouch_target(q_initial, q_now):
    """Build a symmetric crouched nominal posture for HY/KN joints.

    Spot/Isaac joint signs observed so far:
      stretched standing: HY near 0, KN near -0.25
      bent initial stance: HY positive, KN more negative

    Therefore default crouch pushes HY positive and KN negative.
    """
    if args.crouch_base == "current":
        q_target = q_now.detach().clone()
    elif args.crouch_base == "custom":
        q_target = q_now.detach().clone()
        for leg in range(4):
            q_target[:, HY[leg]] = args.crouch_hy_abs
            q_target[:, KN[leg]] = args.crouch_kn_abs
    else:
        q_target = q_initial.detach().clone()

    for leg in range(4):
        q_target[:, HY[leg]] = q_target[:, HY[leg]] + args.crouch_hy_offset
        q_target[:, KN[leg]] = q_target[:, KN[leg]] + args.crouch_kn_offset

    # Limit relative to current state so the actuator target does not jump too far.
    delta = torch.clamp(q_target - q_now, -args.crouch_max_delta, args.crouch_max_delta)
    return q_now + delta


def apply_crouch_selected_joint_targets(robot, q_initial, step):
    """Apply slow selected-joint HY/KN targets to all legs.

    This is intentionally not global align_targets:
      - only HY/KN joint_ids are targeted
      - all legs are targeted symmetrically
      - ramp is slow
      - intended to find a walking-ready bent-knee standing posture
    """
    q_now = robot.data.joint_pos.detach()
    q_delta_full = torch.zeros_like(q_now)

    if not args.enable_crouch_target:
        return None, q_delta_full, 0.0

    if step < args.crouch_start_step:
        return None, q_delta_full, 0.0

    ramp = float(max(0.0, min(1.0, (step - args.crouch_start_step) / max(1, args.crouch_ramp_steps))))
    q_goal = build_crouch_target(q_initial, q_now)

    joint_ids = []
    for leg in range(4):
        joint_ids.extend([int(HY[leg]), int(KN[leg])])

    q_selected_now = q_now[:, joint_ids]
    q_selected_goal = q_goal[:, joint_ids]
    q_selected = (1.0 - ramp) * q_selected_now + ramp * q_selected_goal

    joint_ids_tensor = torch.tensor(joint_ids, device=q_now.device, dtype=torch.long)
    robot.set_joint_position_target(q_selected, joint_ids=joint_ids_tensor)
    robot.set_joint_velocity_target(torch.zeros_like(q_selected), joint_ids=joint_ids_tensor)

    q_delta_full[:, joint_ids] = q_selected - q_selected_now
    q_target_full = q_now.clone()
    q_target_full[:, joint_ids] = q_selected
    return q_target_full, q_delta_full, ramp






def contact_ready_release_gate(phase, phase_step, x_hat, contact_ready_seen):
    if not args.release_on_contact_ready:
        return False
    if phase not in ["lift", "hold_lift", "lower"]:
        return False
    if not contact_ready_seen:
        return False
    roll = float(x_hat[0, 3].detach().cpu())
    pitch = float(x_hat[0, 4].detach().cpu())
    return bool(
        phase_step >= args.contact_ready_release_after_phase_step
        and abs(roll) <= args.contact_ready_release_max_abs_roll
        and abs(pitch) <= args.contact_ready_release_max_abs_pitch
    )



def fallsafe_ready_release_gate(phase, phase_step, x_hat, trial_out, calib_status, contact_ready_seen):
    if not args.fallsafe_release:
        return False
    if phase not in ["lift", "hold_lift", "lower"]:
        return False
    if not contact_ready_seen:
        return False
    if phase_step < args.contact_ready_release_after_phase_step:
        return False

    roll = float(x_hat[0, 3].detach().cpu())
    pitch = float(x_hat[0, 4].detach().cpu())
    base_z = float(x_hat[0, 2].detach().cpu())
    margin = float(trial_out.margin_to_edge[0].detach().cpu())

    if args.fallsafe_require_current_contact_ready and not bool(calib_status["ready"]):
        return False
    if abs(roll) > args.fallsafe_max_abs_roll:
        return False
    if abs(pitch) > args.fallsafe_max_abs_pitch:
        return False
    if base_z < args.fallsafe_min_base_z:
        return False
    if margin < args.fallsafe_min_margin:
        return False
    return True


def contact_calibration_status(x_hat, foot_pos):
    z = foot_pos[0, :, 2]
    z_mean = torch.mean(z)
    z_rel = torch.abs(z - z_mean)
    ready_legs = int(torch.sum(z_rel <= args.calib_contact_tol).detach().cpu())

    roll = float(x_hat[0, 3].detach().cpu())
    pitch = float(x_hat[0, 4].detach().cpu())
    rpy_ready = (abs(roll) <= args.calib_max_abs_roll) and (abs(pitch) <= args.calib_max_abs_pitch)
    feet_ready = ready_legs >= args.calib_min_ready_legs
    ready = bool(feet_ready and rpy_ready)

    return {
        "ready": ready,
        "ready_legs": ready_legs,
        "z_mean": float(z_mean.detach().cpu()),
        "z_rel": z_rel.detach().clone(),
        "rpy_ready": rpy_ready,
        "feet_ready": feet_ready,
    }


def apply_contact_calibration_targets(robot, q_initial):
    if not (args.enable_contact_calibration and args.calib_hold_initial_targets):
        return None

    q_now = robot.data.joint_pos.detach()
    if args.calib_hold_all_joints:
        joint_ids = list(range(q_now.shape[1]))
    else:
        joint_ids = []
        for leg in range(4):
            joint_ids.extend([int(HY[leg]), int(KN[leg])])

    joint_ids_tensor = torch.tensor(joint_ids, device=q_now.device, dtype=torch.long)
    q_des0 = q_initial[:, joint_ids].detach().clone()
    q_now_sel = q_now[:, joint_ids]
    q_des = q_now_sel + torch.clamp(q_des0 - q_now_sel, -args.calib_target_max_delta, args.calib_target_max_delta)

    robot.set_joint_position_target(q_des, joint_ids=joint_ids_tensor)
    if args.calib_velocity_zero:
        robot.set_joint_velocity_target(torch.zeros_like(q_des), joint_ids=joint_ids_tensor)

    q_full = q_now.clone()
    q_full[:, joint_ids] = q_des
    return q_full



def damped_least_squares_delta(J, err, damping):
    """Batch damped least squares: dq = J^T (J J^T + λ² I)^-1 err.

    J: [B, r, m]
    err: [B, r]
    returns dq: [B, m]
    """
    B, r, m = J.shape
    JJt = torch.bmm(J, J.transpose(1, 2))
    eye = torch.eye(r, device=J.device, dtype=J.dtype).unsqueeze(0).expand(B, r, r)
    rhs = err.unsqueeze(-1)
    sol = torch.linalg.solve(JJt + (damping ** 2) * eye, rhs)
    dq = torch.bmm(J.transpose(1, 2), sol).squeeze(-1)
    return dq


def apply_jacobian_ik_swing_target(robot, Jfeet_full, foot_pos, swing_target, phase, profile, swing_enabled):
    """Apply selected-joint position target from a local Jacobian IK step.

    This is intentionally an adapter layer:
      - WBC/MPC still computes support forces and swing reference.
      - During swing only, the selected leg receives a q_des from DLS IK.
      - Stance legs are not touched by this helper.
    """
    q_now = robot.data.joint_pos.detach()
    q_target_full = None
    q_delta_full = torch.zeros_like(q_now)
    ik_info = {
        "active": False,
        "joint_ids": [],
        "foot_err": torch.zeros((q_now.shape[0], 3), device=q_now.device, dtype=q_now.dtype),
        "dq_cmd": torch.zeros((q_now.shape[0], 3), device=q_now.device, dtype=q_now.dtype),
    }

    if not args.enable_jacobian_ik_swing:
        return None, q_delta_full, ik_info
    if not swing_enabled:
        return None, q_delta_full, ik_info
    if phase not in ["lift", "hold_lift", "lower"]:
        return None, q_delta_full, ik_info
    if profile < args.ik_start_profile:
        return None, q_delta_full, ik_info

    leg = LEG_TO_ID[args.test_leg]
    foot_err_3 = swing_target[:, leg, :] - foot_pos[:, leg, :]
    foot_err_3 = foot_err_3.clone()
    foot_err_3[:, 2] = args.ik_target_scale_z * foot_err_3[:, 2]

    if args.ik_use_hx:
        joint_ids = [int(HX[leg]), int(HY[leg]), int(KN[leg])]
    else:
        joint_ids = [int(HY[leg]), int(KN[leg])]

    full_cols = [6 + jid for jid in joint_ids]
    Jleg_full = Jfeet_full[:, leg, :, full_cols]  # [B, 3, m]

    if args.ik_include_xy:
        J_use = Jleg_full
        err_use = foot_err_3
    else:
        J_use = Jleg_full[:, 2:3, :]
        err_use = foot_err_3[:, 2:3]

    dq = damped_least_squares_delta(J_use, err_use, args.ik_damping)
    dq = args.ik_gain * dq
    dq = torch.clamp(dq, -args.ik_max_joint_delta, args.ik_max_joint_delta)

    q_selected = q_now[:, joint_ids] + dq
    joint_ids_tensor = torch.tensor(joint_ids, device=q_now.device, dtype=torch.long)
    robot.set_joint_position_target(q_selected, joint_ids=joint_ids_tensor)
    if args.ik_apply_velocity_zero:
        robot.set_joint_velocity_target(torch.zeros_like(q_selected), joint_ids=joint_ids_tensor)

    q_delta_full[:, joint_ids] = q_selected - q_now[:, joint_ids]
    q_target_full = q_now.clone()
    q_target_full[:, joint_ids] = q_selected

    ik_info["active"] = True
    ik_info["joint_ids"] = joint_ids
    ik_info["foot_err"] = foot_err_3
    # pad dq to length 3 for readable logging [hx/hy/kn], even if hx not used
    dq_pad = torch.zeros((q_now.shape[0], 3), device=q_now.device, dtype=q_now.dtype)
    if args.ik_use_hx:
        dq_pad[:, :] = dq
    else:
        dq_pad[:, 1:] = dq
    ik_info["dq_cmd"] = dq_pad
    return q_target_full, q_delta_full, ik_info



def make_swing_target(phase, profile, foot_pos, foot_swing0, swing_enabled):
    """Single-step swing target.

    Target policy:
      lift       : raise z and move halfway forward
      hold_lift  : keep z high and move from halfway to full forward
      lower      : keep full forward x/y and ramp z back down
      settle     : target follows measured foot; contact restores to all stance
    """
    target = foot_pos.detach().clone()
    if foot_swing0 is None:
        return target

    leg = LEG_TO_ID[args.test_leg]
    target[:, leg, :] = foot_swing0[:, leg, :]

    clearance = args.swing_clearance if args.diagnostic_clearance is None else args.diagnostic_clearance

    if phase in ["lift", "hold_lift", "lower"] and swing_enabled:
        if phase == "lift":
            xy_profile = 0.50 * profile
            z_profile = profile
        elif phase == "hold_lift":
            xy_profile = 0.50 + 0.50 * profile
            z_profile = 1.0
        else:  # lower: profile is 1 -> 0
            xy_profile = 1.0
            z_profile = profile

        target[:, leg, 0] = foot_swing0[:, leg, 0] + args.swing_forward_step * xy_profile
        target[:, leg, 1] = foot_swing0[:, leg, 1] + args.swing_lateral_step * xy_profile

        if args.target_z_mode == "hold":
            z_offset = 0.0
        elif args.target_z_mode == "step":
            # Step mode is intentionally not used for landing tests; lower still ramps down.
            z_offset = clearance if phase != "lower" else clearance * z_profile
        else:
            z_offset = clearance * z_profile

        if phase == "lower":
            z_offset = z_offset - args.landing_extra_down * (1.0 - z_profile)

        target[:, leg, 2] = foot_swing0[:, leg, 2] + z_offset

    return target


def get_foot_vel(robot, foot_indices):
    try:
        return robot.data.body_lin_vel_w[:, foot_indices, :]
    except Exception:
        return torch.zeros_like(robot.data.body_pos_w[:, foot_indices, :])


def get_gravity(robot):
    try:
        return robot.root_physx_view.get_generalized_gravity_forces()
    except Exception:
        return None


def get_coriolis(robot):
    try:
        return robot.root_physx_view.get_coriolis_and_centrifugal_forces()
    except Exception:
        return None


def apply_implicit_target_alignment(robot, q_nom, phase, profile, swing_enabled):
    if not args.align_targets:
        return None

    q = robot.data.joint_pos.detach()
    if args.target_mode == "nominal":
        q_target = q_nom.detach().clone()
    else:
        q_target = q.detach().clone()

    leg = LEG_TO_ID[args.test_leg]
    if args.swing_target_assist and swing_enabled and phase in ["lift", "hold_lift", "lower"]:
        q_target[:, HY[leg]] = q_target[:, HY[leg]] + args.swing_hy_bias * profile
        q_target[:, KN[leg]] = q_target[:, KN[leg]] + args.swing_kn_bias * profile

    delta = torch.clamp(q_target - q, -args.target_max_delta, args.target_max_delta)
    q_target = q + delta

    try:
        robot.set_joint_position_target(q_target)
    except Exception as exc:
        print(f"[WARN] set_joint_position_target failed: {exc}")
        return None
    return q_target


def print_debug(step, phase, profile, phase_step, x_hat, base_ref, trial_out, ref_mask, contact_mask,
                f_mpc, foot_pos, foot_swing0, swing_target, tau_cmd, qpd, robot, swing_enabled, q_target,
                swing_anchor_step, combined_delta_full, crouch_target, crouch_ramp, ik_target, ik_delta_full, ik_info,
                old_forced, controlled_forced, ready_forced, unsafe_ready_forced, fallsafe_ready_forced,
                calib_status, contact_ready_seen):
    leg = LEG_TO_ID[args.test_leg]
    idx = [HX[leg], HY[leg], KN[leg]]

    q = robot.data.joint_pos
    qd = robot.data.joint_vel
    applied = robot.data.applied_torque
    residual = applied - tau_cmd
    ratio = applied.abs() / tau_cmd.abs().clamp_min(1.0)

    foot_delta_swing = foot_pos - foot_swing0 if foot_swing0 is not None else torch.zeros_like(foot_pos)
    target_err = swing_target - foot_pos

    print("\n" + "=" * 150)
    print(f"[B8-au FALLSAFE-GATED-RELEASE] step={step}")
    print("=" * 150)
    print("phase:", phase, "phase_step:", phase_step, "test_leg:", args.test_leg, "profile:", profile)
    print("height_ref:", args.height_ref, "pd_scale:", args.pd_scale, "align_targets:", args.align_targets, "target_mode:", args.target_mode)
    print("anchor_swing_on_release:", args.anchor_swing_on_release, "swing_anchor_step:", swing_anchor_step,
          "target_z_mode:", args.target_z_mode, "diagnostic_clearance:", args.diagnostic_clearance)
    print("enable_crouch_target:", args.enable_crouch_target, "crouch_ramp:", crouch_ramp,
          "crouch_base:", args.crouch_base, "hy_offset:", args.crouch_hy_offset,
          "kn_offset:", args.crouch_kn_offset)
    print("enable_jacobian_ik_swing:", args.enable_jacobian_ik_swing,
          "ik_active:", ik_info["active"],
          "ik_gain:", args.ik_gain,
          "ik_max_joint_delta:", args.ik_max_joint_delta)
    print("old_forced:", old_forced, "controlled_forced:", controlled_forced,
          "ready_forced:", ready_forced,
          "unsafe_ready_forced:", unsafe_ready_forced,
          "fallsafe_ready_forced:", fallsafe_ready_forced,
          "disable_margin_gate_for_release:", args.disable_margin_gate_for_release,
          "force_release_require_margin:", args.force_release_require_margin)
    print("contact_calib_ready:", calib_status["ready"],
          "contact_ready_seen:", contact_ready_seen,
          "ready_legs:", calib_status["ready_legs"],
          "feet_ready:", calib_status["feet_ready"],
          "rpy_ready:", calib_status["rpy_ready"],
          "foot_z_rel_to_mean:", calib_status["z_rel"].detach().cpu().numpy())
    print("swing_enabled:", swing_enabled)
    print("ref_mask LF/RF/LH/RH:", ref_mask[0].detach().cpu().numpy())
    print("contact_mask LF/RF/LH/RH:", contact_mask[0].detach().cpu().numpy())
    print("trial_margin_to_edge:", float(trial_out.margin_to_edge[0].detach().cpu()))
    print("trial_swing_allowed:", bool(trial_out.swing_allowed[0].detach().cpu()))
    print("future_margin_release:", args.future_margin_release)

    print("\n[base]")
    print("base pos xyz:", x_hat[0, 0:3].detach().cpu().numpy())
    print("base rpy:", x_hat[0, 3:6].detach().cpu().numpy())
    print("base ref:", base_ref[0].detach().cpu().numpy())

    print("\n[MPC/WBC]")
    print("f_mpc:")
    print(f_mpc[0].detach().cpu().numpy())
    print("f_qp:")
    print(qpd["f_qp"][0].detach().cpu().numpy())
    print("swing_leg_f_mpc:", f_mpc[0, leg].detach().cpu().numpy())
    print("swing_leg_f_qp:", qpd["f_qp"][0, leg].detach().cpu().numpy())
    print("swing_acc_des selected:", qpd["swing_acc_des"][0, leg].detach().cpu().numpy())
    print("qdd_full selected leg [hx,hy,kn]:", qpd["qdd_full"][0, [6 + leg, 6 + leg + 4, 6 + leg + 8]].detach().cpu().numpy())
    print("h_full base:", qpd["h_full"][0, 0:6].detach().cpu().numpy())
    print("residual_norm:", float(qpd["residual_norm"][0].detach().cpu()))

    print("\n[target / adapter selected leg]")
    if q_target is not None:
        print("q_target_or_adapter_target:", q_target[0, idx].detach().cpu().numpy())
        print("target_minus_q:", (q_target - q)[0, idx].detach().cpu().numpy())
    else:
        print("q_target_or_adapter_target: None")
    if crouch_target is None:
        print("crouch_target: None")
    else:
        print("crouch_target_selected:", crouch_target[0, idx].detach().cpu().numpy())
        print("crouch_target_minus_q_selected:", (crouch_target[0, idx] - q[0, idx]).detach().cpu().numpy())
    if ik_target is None:
        print("ik_target: None")
    else:
        print("ik_target_selected:", ik_target[0, idx].detach().cpu().numpy())
        print("ik_target_minus_q_selected:", (ik_target[0, idx] - q[0, idx]).detach().cpu().numpy())
    print("combined_delta_full_selected:", combined_delta_full[0, idx].detach().cpu().numpy())
    print("ik_delta_full_selected:", ik_delta_full[0, idx].detach().cpu().numpy())
    print("ik_foot_err_xyz:", ik_info["foot_err"][0].detach().cpu().numpy())
    print("ik_dq_cmd_hxhykn:", ik_info["dq_cmd"][0].detach().cpu().numpy())
    print("all HY q LF/RF/LH/RH:", q[0, HY].detach().cpu().numpy())
    print("all KN q LF/RF/LH/RH:", q[0, KN].detach().cpu().numpy())

    print("\n[actuator selected leg hx/hy/kn]")
    print("joint q:", q[0, idx].detach().cpu().numpy())
    print("joint qd:", qd[0, idx].detach().cpu().numpy())
    print("tau_cmd:", tau_cmd[0, idx].detach().cpu().numpy())
    print("applied_torque:", applied[0, idx].detach().cpu().numpy())
    print("applied_minus_cmd:", residual[0, idx].detach().cpu().numpy())
    print("abs(applied)/max(abs(cmd),1):", ratio[0, idx].detach().cpu().numpy())
    print("all tau_cmd max_abs:", float(tau_cmd.abs().max().detach().cpu()))
    print("all applied_torque max_abs:", float(applied.abs().max().detach().cpu()))
    print("all residual max_abs:", float(residual.abs().max().detach().cpu()))

    print("\n[swing foot]")
    print("swing target:", swing_target[0, leg].detach().cpu().numpy())
    print("foot pos:", foot_pos[0, leg].detach().cpu().numpy())
    print("foot target error:", target_err[0, leg].detach().cpu().numpy())
    print("test foot_delta_from_swing_start:", foot_delta_swing[0, leg].detach().cpu().numpy())
    if foot_swing0 is not None:
        print("foot_swing0 selected:", foot_swing0[0, leg].detach().cpu().numpy())
        print("target_z_minus_anchor_z:", float((swing_target[0, leg, 2] - foot_swing0[0, leg, 2]).detach().cpu()))
        print("foot_z_minus_anchor_z:", float((foot_pos[0, leg, 2] - foot_swing0[0, leg, 2]).detach().cpu()))
    print("target_x_minus_anchor_x:", float((swing_target[0, leg, 0] - foot_swing0[0, leg, 0]).detach().cpu()) if foot_swing0 is not None else 0.0)
    print("foot_x_minus_anchor_x:", float(foot_delta_swing[0, leg, 0].detach().cpu()))
    print("target_z_minus_foot_z:", float(target_err[0, leg, 2].detach().cpu()))
    print("foot_z_world:", float(foot_pos[0, leg, 2].detach().cpu()))
    print("target_z_world:", float(swing_target[0, leg, 2].detach().cpu()))
    print("clearance_z_from_swing_start:", float(foot_delta_swing[0, leg, 2].detach().cpu()))
    if args.print_all_feet_z:
        print("all feet z LF/RF/LH/RH:", foot_pos[0, :, 2].detach().cpu().numpy())
        print("all target z LF/RF/LH/RH:", swing_target[0, :, 2].detach().cpu().numpy())
    print("=" * 150 + "\n")


def main():
    env_cfg = CartoEffortEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    patch_flat_safe_env(env_cfg)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()
    robot = env.scene["robot"]
    print_robot_debug_info(robot)

    device, dtype = robot.data.joint_pos.device, robot.data.joint_pos.dtype
    foot_indices = get_foot_indices(robot)
    mass = get_mass(robot)

    ref_cfg = SupportRegionRefConfig(alpha=args.alpha, margin=args.margin, max_shift_per_step=args.max_shift_per_step, height_ref=args.height_ref)
    mpc_cfg = MpcWbcBridgeConfig(kp_xy=args.kp_xy, kd_xy=args.kd_xy, kp_z=args.kp_z, kd_z=args.kd_z, mass=mass, mu=args.mu, min_fz=args.min_fz, max_fz=args.max_fz)
    wbc_cfg = FullWbcQpV1Config(
        mass=mass,
        w_dyn=args.w_dyn,
        w_base_acc=args.w_base_acc,
        w_stance_acc=args.w_stance_acc,
        w_swing_acc=args.w_swing_acc,
        w_force_track=args.w_force_track,
        w_swing_force_zero=args.w_swing_force_zero,
        w_tau_posture=args.w_tau_posture,
        w_tau_reg=args.w_tau_reg,
        w_qdd_reg=args.w_qdd_reg,
        kp_swing_z=args.kp_swing_z,
        kd_swing_z=args.kd_swing_z,
        max_swing_acc=args.max_swing_acc,
        mu=args.mu,
        max_fz=args.max_fz,
        max_tau=args.max_tau,
        tau_output_sign=args.tau_output_sign,
    )

    q_initial = robot.data.joint_pos.detach().clone()
    q_nom = q_initial.clone()
    if args.enable_crouch_target and args.use_crouch_q_nom:
        q_nom = build_crouch_target(q_initial, q_initial).detach().clone()
    prev_base_ref = None
    foot_swing0 = None
    swing_anchor_step = -1
    current_step_key = None
    contact_ready_seen = False
    calib_hold_target = None
    swing_start_step = args.warmup_steps + args.shift_steps
    total_steps = args.warmup_steps + args.shift_steps + args.lift_steps + args.hold_steps + args.lower_steps + args.settle_steps

    print("\n" + "=" * 150)
    print("[INFO] Starting B8-au fall-safe gated release + RF single-step probe")
    print("test_leg:", args.test_leg, "mass:", mass, "height_ref:", args.height_ref, "pd_scale:", args.pd_scale)
    print("enable_crouch_target:", args.enable_crouch_target,
          "crouch_base:", args.crouch_base,
          "hy_offset:", args.crouch_hy_offset,
          "kn_offset:", args.crouch_kn_offset,
          "use_crouch_q_nom:", args.use_crouch_q_nom)
    print("enable_jacobian_ik_swing:", args.enable_jacobian_ik_swing,
          "ik_gain:", args.ik_gain,
          "ik_damping:", args.ik_damping,
          "ik_max_joint_delta:", args.ik_max_joint_delta,
          "ik_include_xy:", args.ik_include_xy,
          "ik_use_hx:", args.ik_use_hx)
    print("controlled_force_release:", args.controlled_force_release,
          "force_release_after_phase_step:", args.force_release_after_phase_step,
          "max_roll:", args.force_release_max_abs_roll,
          "max_pitch:", args.force_release_max_abs_pitch,
          "disable_margin_gate_for_release:", args.disable_margin_gate_for_release,
          "force_release_require_margin:", args.force_release_require_margin)
    print("single_step_target:",
          "forward:", args.swing_forward_step,
          "lateral:", args.swing_lateral_step,
          "landing_extra_down:", args.landing_extra_down,
          "settle_steps:", args.settle_steps)
    print("contact_calibration:",
          "enabled:", args.enable_contact_calibration,
          "hold_initial_targets:", args.calib_hold_initial_targets,
          "hold_all_joints:", args.calib_hold_all_joints,
          "contact_tol:", args.calib_contact_tol,
          "require_ready_for_shift:", args.calib_require_ready_for_shift)
    print("contact_ready_release:",
          "release_on_contact_ready:", args.release_on_contact_ready,
          "after_phase_step:", args.contact_ready_release_after_phase_step,
          "max_roll:", args.contact_ready_release_max_abs_roll,
          "max_pitch:", args.contact_ready_release_max_abs_pitch)
    print("fallsafe_release:",
          "enabled:", args.fallsafe_release,
          "min_margin:", args.fallsafe_min_margin,
          "min_base_z:", args.fallsafe_min_base_z,
          "max_roll:", args.fallsafe_max_abs_roll,
          "max_pitch:", args.fallsafe_max_abs_pitch,
          "require_current_contact_ready:", args.fallsafe_require_current_contact_ready,
          "disable_unsafe_ready_forced:", args.disable_unsafe_ready_forced)
    print("future_margin_release:", args.future_margin_release, "min_shift_steps_before_release:", args.min_shift_steps_before_release)
    print("wbc_cfg:", wbc_cfg)
    print("=" * 150)

    for step in range(total_steps):
        if not simulation_app.is_running():
            break

        x_hat = make_x_hat(robot, velocity_frame="world")
        foot_pos = robot.data.body_pos_w[:, foot_indices, :]
        foot_vel = get_foot_vel(robot, foot_indices)
        M = robot.root_physx_view.get_generalized_mass_matrices()
        Jfeet_full = robot.root_physx_view.get_jacobians()[:, foot_indices, 0:3, :]
        gravity, coriolis = get_gravity(robot), get_coriolis(robot)

        phase, profile, phase_step = get_phase(step)
        calib_status = contact_calibration_status(x_hat, foot_pos)

        if args.enable_contact_calibration:
            if calib_status["ready"]:
                contact_ready_seen = True
            if args.calib_require_ready_for_shift and (not contact_ready_seen) and phase != "warmup":
                # Keep all-stance warmup behavior until four-foot geometry and base attitude look sane.
                phase, profile, phase_step = "warmup", 0.0, step

        ref_mask = all_stance_mask(args.num_envs, device, dtype) if phase in ["warmup", "settle"] else future_swing_mask(args.num_envs, device, dtype)
        base_ref, trial_out = make_base_ref(x_hat, foot_pos, ref_mask, prev_base_ref, ref_cfg, phase, phase_step)
        prev_base_ref = base_ref.detach().clone()

        swing_enabled = gate_swing(phase, phase_step, trial_out, x_hat)
        old_forced, controlled_forced = gate_debug_flags(phase, phase_step, trial_out, x_hat)

        unsafe_ready_forced = contact_ready_release_gate(phase, phase_step, x_hat, contact_ready_seen)
        fallsafe_ready_forced = fallsafe_ready_release_gate(
            phase, phase_step, x_hat, trial_out, calib_status, contact_ready_seen
        )

        ready_forced = fallsafe_ready_forced if args.disable_unsafe_ready_forced else unsafe_ready_forced
        if ready_forced:
            swing_enabled = True

        if phase == "shift":
            # pre-shift only; keep all contacts
            swing_enabled = False
            ready_forced = False
            unsafe_ready_forced = False
            fallsafe_ready_forced = False

        contact_mask = future_swing_mask(args.num_envs, device, dtype) if swing_enabled else all_stance_mask(args.num_envs, device, dtype)

        # B8-ak: choose when to anchor the swing-foot reference.
        # Default reproduces B8-ah: anchor at nominal lift start.
        # With --anchor_swing_on_release, anchor only when the gate actually releases the leg.
        if foot_swing0 is None:
            if args.anchor_swing_on_release:
                if swing_enabled:
                    foot_swing0 = foot_pos.detach().clone()
                    swing_anchor_step = step
            elif step >= swing_start_step:
                foot_swing0 = foot_pos.detach().clone()
                swing_anchor_step = step

        f_mpc, _ = distribute_grf_ls(
            base_pos_w=x_hat[:, 0:3],
            base_rpy_w=x_hat[:, 3:6],
            base_lin_vel_w=x_hat[:, 6:9],
            base_ang_vel_w=x_hat[:, 9:12],
            base_ref=base_ref,
            foot_pos_w=foot_pos,
            stance_mask=contact_mask,
            cfg=mpc_cfg,
        )

        swing_target = make_swing_target(phase, profile, foot_pos, foot_swing0, swing_enabled)

        tau_cmd, qpd = solve_full_wbc_qp_v1(
            M_full=M,
            Jfeet_full=Jfeet_full,
            f_mpc=f_mpc,
            q=robot.data.joint_pos,
            qd=robot.data.joint_vel,
            q_nom=q_nom,
            x_hat=x_hat,
            base_ref=base_ref,
            foot_pos_w=foot_pos,
            foot_vel_w=foot_vel,
            swing_target_pos_w=swing_target,
            stance_mask=contact_mask,
            gravity_forces=gravity,
            coriolis_forces=coriolis,
            cfg=wbc_cfg,
        )

        tau_cmd = args.tau_cmd_scale * tau_cmd
        crouch_target, crouch_delta_full, crouch_ramp = apply_crouch_selected_joint_targets(robot, q_initial, step)

        ik_target, ik_delta_full, ik_info = apply_jacobian_ik_swing_target(
            robot, Jfeet_full, foot_pos, swing_target, phase, profile, swing_enabled
        )

        q_target = apply_implicit_target_alignment(robot, q_nom, phase, profile, swing_enabled)

        calib_hold_target = None
        if phase == "warmup" and not swing_enabled:
            calib_hold_target = apply_contact_calibration_targets(robot, q_initial)

        # For logging: if global alignment is off, report the adapter target actually used.
        if q_target is None:
            q_target = ik_target if ik_target is not None else (crouch_target if crouch_target is not None else calib_hold_target)

        combined_delta_full = crouch_delta_full + ik_delta_full

        env.step(tau_cmd)

        if step % max(args.print_every, 1) == 0:
            print_debug(
                step, phase, profile, phase_step, x_hat, base_ref, trial_out, ref_mask, contact_mask,
                f_mpc, foot_pos, foot_swing0, swing_target, tau_cmd, qpd, robot, swing_enabled, q_target,
                swing_anchor_step, combined_delta_full, crouch_target, crouch_ramp, ik_target, ik_delta_full, ik_info,
                old_forced, controlled_forced, ready_forced, unsafe_ready_forced, fallsafe_ready_forced,
                calib_status, contact_ready_seen
            )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
