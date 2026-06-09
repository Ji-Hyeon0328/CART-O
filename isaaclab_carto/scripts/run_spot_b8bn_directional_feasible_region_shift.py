# isaaclab_carto/scripts/run_spot_b8ah_stable_warmup_decoupled_preshift.py
#
# B8-bn: directional feasible-region-lite trunk shift.
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
from pathlib import Path
import argparse
from typing import Any

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CARTO/TRACER B8-bn directional feasible-region-lite trunk shift")

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--warmup_steps", type=int, default=120)
parser.add_argument("--shift_steps", type=int, default=700)
parser.add_argument("--lift_steps", type=int, default=160)
parser.add_argument("--hold_steps", type=int, default=120)
parser.add_argument("--lower_steps", type=int, default=160)
parser.add_argument("--settle_steps", type=int, default=220)
parser.add_argument("--print_every", type=int, default=20)
parser.add_argument("--log_file", type=str, default="",
                    help="If provided, tee all stdout/stderr to this file.")
parser.add_argument("--log_flush_every_print", action="store_true",
                    help="Flush the log after every Python print.")
# B8-az: explicit visible swing assist in joint-target adapter.
# This is diagnostic/robot-adapter only: WBC/MPC/contact masking remain unchanged.
parser.add_argument("--enable_visible_swing_assist", action="store_true")
parser.add_argument("--visible_assist_front_hy_delta", type=float, default=0.08)
parser.add_argument("--visible_assist_front_kn_delta", type=float, default=-0.18)
parser.add_argument("--visible_assist_hind_hy_delta", type=float, default=0.06)
parser.add_argument("--visible_assist_hind_kn_delta", type=float, default=-0.14)
parser.add_argument("--visible_assist_min_profile", type=float, default=0.15)
parser.add_argument("--visible_assist_max_delta_norm", type=float, default=0.35)
# B8-ba: bound overshoot and do not start the next step until the previous swing foot is really down.
parser.add_argument("--enable_swing_overshoot_guard", action="store_true")
parser.add_argument("--swing_overshoot_margin", type=float, default=0.012,
                    help="Disable visible assist if foot_z exceeds target_z by this margin.")
parser.add_argument("--enable_real_touchdown_gate", action="store_true")
parser.add_argument("--touchdown_foot_z_tol", type=float, default=0.012,
                    help="A foot is considered down when its z is within max(stance_mean_z)+tol.")
parser.add_argument("--touchdown_hold_steps", type=int, default=40,
                    help="Extra all-stance hold steps inserted until the previous swing foot is down.")
parser.add_argument("--reset_joint_target_on_touchdown", action="store_true")
parser.add_argument("--touchdown_reset_velocity_zero", action="store_true")
# B8-bc: insert all-stance base recenter before step_idx>=1.
parser.add_argument("--enable_base_recenter_between_steps", action="store_true")
parser.add_argument("--recenter_steps", type=int, default=260)
parser.add_argument("--recenter_freeze_xy", action="store_true",
                    help="During recenter, set base_ref xy to current xy.")
parser.add_argument("--recenter_reset_prev_base_ref", action="store_true",
                    help="Reset prev_base_ref when entering recenter or a new step.")
parser.add_argument("--recenter_require_safe_for_next", action="store_true",
                    help="If set, second-step swing gate requires recenter safe flag.")
parser.add_argument("--recenter_max_abs_roll", type=float, default=0.08)
parser.add_argument("--recenter_max_abs_pitch", type=float, default=0.08)
parser.add_argument("--recenter_min_base_z", type=float, default=0.64)

parser.add_argument("--test_leg", type=str, default="RF", choices=["LF", "RF", "LH", "RH"])
parser.add_argument("--enable_two_step_sequence", action="store_true",
                    help="If set, run RF -> settle -> LH. If not set, behave like B8-aq single RF step.")
parser.add_argument("--step_order", type=str, default="RF,LH",
                    help="Comma-separated step order used only when --enable_two_step_sequence is set.")
parser.add_argument("--auto_select_second_leg", action="store_true",
                    help="At step_idx=1, choose the next swing leg with the best current support margin.")
parser.add_argument("--second_leg_candidates", type=str, default="LF,LH,RH",
                    help="Candidates for --auto_select_second_leg. Default excludes RF.")
parser.add_argument("--reset_base_ref_on_step_change", action="store_true",
                    help="Reset prev_base_ref when switching from one step to the next.")
parser.add_argument("--restore_contact_in_late_lower", action="store_true",
                    help="During the late part of lower phase, restore all-stance contact instead of keeping the swing leg in the air.")
parser.add_argument("--touchdown_profile_threshold", type=float, default=0.25,
                    help="If lower profile <= threshold, restore all stance.")
parser.add_argument("--print_touchdown_gate", action="store_true")

# Env / actuator
parser.add_argument("--spawn_z", type=float, default=0.60)
parser.add_argument("--pd_scale", type=float, default=0.35)
parser.add_argument("--disable_env_terminations", action="store_true",
                    help="Disable time_out and base-height termination for diagnostic runs.")
parser.add_argument("--episode_length_s_override", type=float, default=1000000.0,
                    help="Large episode length when terminations are disabled.")
parser.add_argument("--base_height_threshold_override", type=float, default=-100.0,
                    help="Very low threshold to prevent base-height termination.")
parser.add_argument("--detect_env_reset", action="store_true",
                    help="Detect unexpected env reset/respawn and clear controller internal state.")
parser.add_argument("--reset_detection_distance", type=float, default=0.20)
parser.add_argument("--reset_detection_spawn_tol", type=float, default=0.08)
# B8-be: do not leave shift until the selected swing leg is actually feasible.
parser.add_argument("--enable_strict_support_shift_gate", action="store_true")
parser.add_argument("--shift_gate_margin", type=float, default=0.060,
                    help="Required future-support margin before lift can start.")
parser.add_argument("--shift_gate_min_steps", type=int, default=220)
parser.add_argument("--shift_gate_max_steps", type=int, default=900)
parser.add_argument("--shift_gate_max_abs_roll", type=float, default=0.080)
parser.add_argument("--shift_gate_max_abs_pitch", type=float, default=0.080)
parser.add_argument("--shift_gate_min_base_z", type=float, default=0.640)
parser.add_argument("--shift_gate_max_base_ref_xy_err", type=float, default=0.045)
parser.add_argument("--shift_gate_hold_safe_steps", type=int, default=20,
                    help="Require N consecutive safe shift samples before lift.")
parser.add_argument("--shift_gate_use_selected_leg_mask", action="store_true",
                    help="Build the future support mask from active_leg, not args.test_leg.")

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
# B8-bf: explicit foothold target instead of tiny/place-back swing.
parser.add_argument("--enable_explicit_foothold_target", action="store_true")
parser.add_argument("--foothold_forward_step", type=float, default=0.055)
parser.add_argument("--foothold_lateral_step", type=float, default=0.0)
parser.add_argument("--foothold_use_base_yaw", action="store_true",
                    help="Interpret foothold forward/lateral in the current base-yaw frame.")
parser.add_argument("--foothold_target_z_from_anchor", action="store_true",
                    help="Keep landing z equal to swing anchor z.")
parser.add_argument("--foothold_hold_during_lower", action="store_true",
                    help="Keep xy target fixed at foothold during lower phase.")
parser.add_argument("--foothold_accept_on_late_touchdown", action="store_true",
                    help="When late touchdown begins, reset selected joint target to current and accept actual foot as stance.")
# B8-bg: touchdown is valid only when the foot is load-bearing and near its foothold.
parser.add_argument("--enable_load_bearing_touchdown_gate", action="store_true")
parser.add_argument("--touchdown_min_fz", type=float, default=45.0)
parser.add_argument("--touchdown_max_foot_speed", type=float, default=0.18)
parser.add_argument("--touchdown_max_foothold_xy_err", type=float, default=0.045)
parser.add_argument("--touchdown_load_hold_steps", type=int, default=60)
parser.add_argument("--touchdown_max_abs_roll", type=float, default=0.12)
parser.add_argument("--touchdown_max_abs_pitch", type=float, default=0.12)
parser.add_argument("--touchdown_min_base_z", type=float, default=0.62)
parser.add_argument("--keep_foothold_until_load_valid", action="store_true",
                    help="Do not clear foothold target at late touchdown. Keep it until load-bearing validation passes.")
# B8-bh: after touchdown, push the selected leg back to its pre-swing stance posture.
parser.add_argument("--enable_touchdown_stance_recovery", action="store_true")
parser.add_argument("--touchdown_recovery_steps", type=int, default=160)
parser.add_argument("--touchdown_recovery_min_alpha", type=float, default=0.15)
parser.add_argument("--touchdown_recovery_max_delta", type=float, default=0.35)
parser.add_argument("--touchdown_recovery_include_hx", action="store_true")
parser.add_argument("--touchdown_recovery_hy_offset", type=float, default=0.0)
parser.add_argument("--touchdown_recovery_kn_offset", type=float, default=0.0)
parser.add_argument("--touchdown_recovery_zero_velocity", action="store_true")
parser.add_argument("--record_safe_stance_max_abs_roll", type=float, default=0.08)
parser.add_argument("--record_safe_stance_max_abs_pitch", type=float, default=0.08)
parser.add_argument("--record_safe_stance_min_base_z", type=float, default=0.62)
# B8-bi: freeze step-key update while touchdown recovery is active.
parser.add_argument("--freeze_step_key_during_touchdown_recovery", action="store_true")
parser.add_argument("--force_touchdown_recovery_active", action="store_true",
                    help="Force stance recovery target while previous_swing_leg is held by touchdown gate.")
parser.add_argument("--print_touchdown_state_machine", action="store_true")
# B8-bj: freeze trunk/base reference during swing/lower/touchdown/recovery.
parser.add_argument("--freeze_base_ref_during_swing", action="store_true")
parser.add_argument("--freeze_base_ref_during_touchdown_recovery", action="store_true")
parser.add_argument("--freeze_base_ref_keep_z", action="store_true",
                    help="Freeze only x/y/yaw, but keep current height_ref and zero roll/pitch targets.")

# B8-bj: recover touchdown stance by locking foot to explicit foothold, not old pre-swing q.
parser.add_argument("--enable_foothold_lock_recovery", action="store_true")
parser.add_argument("--foothold_lock_gain", type=float, default=0.55)
parser.add_argument("--foothold_lock_damping", type=float, default=0.04)
parser.add_argument("--foothold_lock_max_joint_delta", type=float, default=0.12)
parser.add_argument("--foothold_lock_target_scale_xy", type=float, default=1.0)
parser.add_argument("--foothold_lock_target_scale_z", type=float, default=0.35)
parser.add_argument("--foothold_lock_zero_velocity", action="store_true")
# B8-bk: do not restore contact just because lower phase is late.
# Keep the swing controller alive until the toe is actually close to the explicit foothold.
parser.add_argument("--defer_contact_restore_until_foothold_valid", action="store_true")
parser.add_argument("--late_touchdown_keep_swing_controller", action="store_true")
parser.add_argument("--late_touchdown_force_foothold_lock", action="store_true")
parser.add_argument("--late_touchdown_xy_tol", type=float, default=0.030)
parser.add_argument("--late_touchdown_z_tol", type=float, default=0.018)
parser.add_argument("--late_touchdown_max_foot_speed", type=float, default=0.35)
# B8-bl: previous_swing_leg must come from an actual contact-free swing,
# not from nominal step_key/schedule transition.
parser.add_argument("--enable_real_swing_commit_state", action="store_true")
parser.add_argument("--real_swing_min_active_steps", type=int, default=20)
parser.add_argument("--real_swing_commit_only_after_lower", action="store_true")
parser.add_argument("--block_recenter_until_previous_valid", action="store_true",
                    help="If a previous swing is not load-bearing valid, do not enter next recenter/shift.")
parser.add_argument("--second_shift_requires_margin", type=float, default=0.050,
                    help="Minimum future-support margin before second leg may detach.")
# B8-bm: Abdalla-style feasible-region-lite CoM/trunk shift.
# Full Abdalla et al. feasible region considers balance, torque, and kinematic constraints.
# This lite version first moves the CoM/base projection toward the incenter/centroid
# of the future support polygon before detach.
parser.add_argument("--enable_feasible_region_lite_shift", action="store_true")
parser.add_argument("--fr_lite_mode", type=str, default="incenter", choices=["incenter", "centroid"])
parser.add_argument("--fr_lite_apply_in_shift", action="store_true")
parser.add_argument("--fr_lite_apply_in_recenter", action="store_true")
parser.add_argument("--fr_lite_start_after_steps", type=int, default=0)
parser.add_argument("--fr_lite_max_shift_per_step", type=float, default=0.0012)
parser.add_argument("--fr_lite_blend", type=float, default=1.0,
                    help="Blend from current support-region base_ref xy to feasible-region-lite target xy.")
parser.add_argument("--fr_lite_forward_bias", type=float, default=0.010,
                    help="Small base-yaw forward bias added to the support-region safe point.")
parser.add_argument("--fr_lite_lateral_bias_away_from_swing", type=float, default=0.020,
                    help="Bias away from the planned swing leg side in base/body y direction.")
parser.add_argument("--fr_lite_min_margin_for_detach", type=float, default=0.060,
                    help="Detach only when trial margin reaches this threshold if feasible-region-lite is enabled.")
# B8-bn: directional feasible-region-lite.
# Instead of using the incenter as the target directly, build a desired forward
# body-progress point and blend/project it toward the future support safe point.
parser.add_argument("--enable_directional_fr_lite", action="store_true")
parser.add_argument("--fr_directional_forward_distance", type=float, default=0.050)
parser.add_argument("--fr_directional_weight", type=float, default=0.70,
                    help="0=incenter only, 1=desired forward point. The result is still margin-gated.")
parser.add_argument("--fr_no_backward_along_yaw", action="store_true",
                    help="Prevent the feasible-region-lite target from moving backward along current yaw.")
parser.add_argument("--fr_min_forward_delta", type=float, default=0.000,
                    help="Minimum nonnegative forward progress when --fr_no_backward_along_yaw is set.")
parser.add_argument("--force_no_swing_in_shift_phase", action="store_true",
                    help="Hard state-machine guard: shift phase can move trunk only, never detach a foot.")

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
ID_TO_LEG = {v: k for k, v in LEG_TO_ID.items()}
HX = [0, 1, 2, 3]
HY = [4, 5, 6, 7]
KN = [8, 9, 10, 11]



class TeeStream:
    def __init__(self, *streams, flush_every_write=False):
        self.streams = streams
        self.flush_every_write = flush_every_write

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            if self.flush_every_write:
                stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def setup_log_file():
    if not args.log_file:
        return None
    log_path = Path(args.log_file).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.__stdout__, f, flush_every_write=args.log_flush_every_print)
    sys.stderr = TeeStream(sys.__stderr__, f, flush_every_write=True)
    print(f"[INFO] Logging stdout/stderr to: {log_path.resolve()}")
    return f


@configclass
class CartoEffortActionsCfg:
    joint_effort = JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0)


@configclass
class CartoEffortEnvCfg(CartoEnvCfg):
    actions: CartoEffortActionsCfg = CartoEffortActionsCfg()


def smooth01(s):
    s = max(0.0, min(1.0, s))
    return float(0.5 - 0.5 * torch.cos(torch.tensor(torch.pi * s)).item())


def make_world_step_from_base_yaw(x_hat, forward, lateral):
    yaw = x_hat[:, 5]
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    dx = forward * c - lateral * s
    dy = forward * s + lateral * c
    return dx, dy


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

    # B8-bd: diagnostic runs should not be polluted by automatic respawn.
    # Disable/relax the termination terms that trigger env reset.
    if args.disable_env_terminations:
        try:
            env_cfg.episode_length_s = float(args.episode_length_s_override)
            print(f"[INFO] Overrode episode_length_s = {env_cfg.episode_length_s}")
        except Exception as exc:
            print(f"[WARN] episode_length_s override failed: {exc}")
        try:
            env_cfg.terminations.time_out = None
            print("[INFO] Disabled time_out termination")
        except Exception as exc:
            print(f"[WARN] disabling time_out failed: {exc}")
        try:
            env_cfg.terminations.base_height_termination = None
            print("[INFO] Disabled base_height_termination")
        except Exception as exc:
            print(f"[WARN] disabling base_height_termination failed: {exc}")
            try:
                env_cfg.terminations.base_height_termination.params["threshold"] = float(args.base_height_threshold_override)
                print(f"[INFO] Relaxed base_height_termination threshold = {args.base_height_threshold_override}")
            except Exception as exc2:
                print(f"[WARN] relaxing base_height_termination threshold failed: {exc2}")

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


def detect_unexpected_reset(prev_root_pos, curr_root_pos, step):
    if not args.detect_env_reset or prev_root_pos is None or step < 5:
        return False

    jump = torch.linalg.norm(curr_root_pos[0, 0:3] - prev_root_pos[0, 0:3]).detach()
    spawn = torch.tensor([0.0, 0.0, args.spawn_z], device=curr_root_pos.device, dtype=curr_root_pos.dtype)
    dist_to_spawn = torch.linalg.norm(curr_root_pos[0, 0:3] - spawn).detach()
    prev_dist_to_spawn = torch.linalg.norm(prev_root_pos[0, 0:3] - spawn).detach()

    jumped = bool((jump > args.reset_detection_distance).detach().cpu())
    respawned_near_origin = bool(
        (dist_to_spawn < args.reset_detection_spawn_tol and prev_dist_to_spawn > args.reset_detection_distance).detach().cpu()
    )
    return jumped or respawned_near_origin


def parse_step_order():
    order = [x.strip().upper() for x in args.step_order.split(",") if x.strip()]
    if not order:
        order = [args.test_leg]
    for leg in order:
        if leg not in LEG_TO_ID:
            raise ValueError(f"Invalid leg in --step_order: {leg}")
    return order


def one_step_len():
    return args.shift_steps + args.lift_steps + args.hold_steps + args.lower_steps + args.settle_steps


def sequence_total_len(num_steps):
    if num_steps <= 0:
        return 0
    if not args.enable_base_recenter_between_steps:
        return one_step_len() * num_steps
    # no recenter before the first step; recenter before each later step
    return one_step_len() + (num_steps - 1) * (args.recenter_steps + one_step_len())


def get_phase(step):
    """B8-av phase function.

    Default mode reproduces B8-aq single-step timing.

    If --enable_two_step_sequence is used, each leg has:
        shift -> lift -> hold_lift -> lower -> settle
    with the active leg selected from --step_order.
    """
    if step < args.warmup_steps:
        active_leg = args.test_leg
        return "warmup", 0.0, step, active_leg, 0

    if not args.enable_two_step_sequence:
        t = step - args.warmup_steps
        active_leg = args.test_leg
        if t < args.shift_steps:
            return "shift", 0.0, t, active_leg, 0
        t -= args.shift_steps
        if t < args.lift_steps:
            return "lift", smooth01(t / max(args.lift_steps, 1)), t, active_leg, 0
        t -= args.lift_steps
        if t < args.hold_steps:
            return "hold_lift", 1.0, t, active_leg, 0
        t -= args.hold_steps
        if t < args.lower_steps:
            return "lower", smooth01(1.0 - t / max(args.lower_steps, 1)), t, active_leg, 0
        t -= args.lower_steps
        if t < args.settle_steps:
            return "settle", 0.0, t, active_leg, 0
        return "done", 0.0, t, active_leg, 0

    order = parse_step_order()
    t_global = step - args.warmup_steps

    if not args.enable_base_recenter_between_steps:
        per = one_step_len()
        step_idx = int(t_global // max(per, 1))
        t = int(t_global % max(per, 1))
    else:
        first_len = one_step_len()
        if t_global < first_len:
            step_idx = 0
            t = int(t_global)
        else:
            t2 = int(t_global - first_len)
            block = args.recenter_steps + one_step_len()
            step_idx = 1 + int(t2 // max(block, 1))
            t_in_block = int(t2 % max(block, 1))
            if t_in_block < args.recenter_steps:
                active_leg = order[min(step_idx, len(order)-1)]
                return "recenter", 0.0, t_in_block, active_leg, step_idx
            t = t_in_block - args.recenter_steps

    active_leg = order[min(step_idx, len(order)-1)]

    if t < args.shift_steps:
        return "shift", 0.0, t, active_leg, step_idx
    t -= args.shift_steps
    if t < args.lift_steps:
        return "lift", smooth01(t / max(args.lift_steps, 1)), t, active_leg, step_idx
    t -= args.lift_steps
    if t < args.hold_steps:
        return "hold_lift", 1.0, t, active_leg, step_idx
    t -= args.hold_steps
    if t < args.lower_steps:
        return "lower", smooth01(1.0 - t / max(args.lower_steps, 1)), t, active_leg, step_idx
    t -= args.lower_steps
    if t < args.settle_steps:
        return "settle", 0.0, t, active_leg, step_idx
    return "done", 0.0, t, active_leg, step_idx


def all_stance_mask(num_envs, device, dtype):
    return torch.ones((num_envs, 4), device=device, dtype=dtype)


def future_swing_mask(num_envs, device, dtype):
    stance = torch.ones((num_envs, 4), device=device, dtype=dtype)
    stance[:, LEG_TO_ID[args.test_leg]] = 0.0
    return stance


def future_swing_mask_for_leg(num_envs, device, dtype, leg_name):
    stance = torch.ones((num_envs, 4), device=device, dtype=dtype)
    stance[:, LEG_TO_ID[str(leg_name)]] = 0.0
    return stance


def swing_mask_for_leg(leg_name, num_envs, device, dtype):
    stance = torch.ones((num_envs, 4), device=device, dtype=dtype)
    stance[:, LEG_TO_ID[leg_name]] = 0.0
    return stance


def parse_second_leg_candidates():
    out = [x.strip().upper() for x in args.second_leg_candidates.split(",") if x.strip()]
    out = [x for x in out if x in LEG_TO_ID]
    return out if out else ["LF", "LH", "RH"]


def evaluate_candidate_margins(x_hat, foot_pos, cfg, device, dtype):
    """Evaluate support-region margin for each candidate swing leg.

    This is a diagnostic selection rule: choose the next leg whose future 3-leg
    support polygon currently has the largest positive/least-negative margin.
    """
    margins = {}
    for leg in parse_second_leg_candidates():
        mask = swing_mask_for_leg(leg, args.num_envs, device, dtype)
        out = compute_support_region_ref(
            foot_pos_w=foot_pos,
            base_pos_w=x_hat[:, 0:3],
            base_rpy_w=x_hat[:, 3:6],
            stance_mask=mask,
            prev_base_ref=None,
            cfg=cfg,
        )
        margins[leg] = float(out.margin_to_edge[0].detach().cpu())
    return margins


def select_best_second_leg(x_hat, foot_pos, cfg, device, dtype):
    margins = evaluate_candidate_margins(x_hat, foot_pos, cfg, device, dtype)
    best_leg = max(margins, key=lambda k: margins[k])
    return best_leg, margins


def _base_yaw_unit_vectors(x_hat):
    yaw = x_hat[:, 5]
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    fwd = torch.stack([c, s], dim=-1)
    left = torch.stack([-s, c], dim=-1)
    return fwd, left


def _safe_point_of_support_polygon_lite(foot_pos, stance_mask, x_hat, active_leg=None):
    """Abdalla-inspired feasible-region-lite safe point.

    This is not the full improved feasible region from Abdalla et al.
    It is a cheap proxy:
      - use the future stance polygon
      - choose its incenter/centroid as a maximum-margin-like CoM projection
      - add small forward and away-from-swing biases
    """
    B = foot_pos.shape[0]
    xy = foot_pos[:, :, 0:2]
    out = x_hat[:, 0:2].detach().clone()
    info = {
        "active": False,
        "mode": args.fr_lite_mode,
        "stance_count": 0,
        "raw_safe_xy": out.detach().clone(),
        "biased_safe_xy": out.detach().clone(),
        "bias_xy": torch.zeros_like(out),
    }

    mask = stance_mask > 0.5
    # This diagnostic usually uses B=1, but keep batch logic simple.
    for b in range(B):
        ids = torch.nonzero(mask[b], as_tuple=False).flatten()
        if ids.numel() < 2:
            continue
        pts = xy[b, ids, :]
        if args.fr_lite_mode == "centroid" or ids.numel() != 3:
            safe = pts.mean(dim=0)
        else:
            A, C, D = pts[0], pts[1], pts[2]
            # Incenter weights are opposite side lengths.
            a = torch.linalg.norm(C - D)
            c = torch.linalg.norm(A - D)
            d = torch.linalg.norm(A - C)
            denom = (a + c + d).clamp_min(1.0e-6)
            safe = (a * A + c * C + d * D) / denom

        fwd, left = _base_yaw_unit_vectors(x_hat[b:b+1])
        bias = float(args.fr_lite_forward_bias) * fwd[0]

        # Move away from the planned swing side.
        # left legs have positive y, right legs negative y in body convention.
        if active_leg is not None and active_leg in [0, 2]:       # LF/LH swing -> bias right
            bias = bias - float(args.fr_lite_lateral_bias_away_from_swing) * left[0]
        elif active_leg is not None and active_leg in [1, 3]:     # RF/RH swing -> bias left
            bias = bias + float(args.fr_lite_lateral_bias_away_from_swing) * left[0]

        out[b] = safe + bias
        info["active"] = True
        info["stance_count"] = int(ids.numel())
        info["raw_safe_xy"][b] = safe
        info["biased_safe_xy"][b] = out[b]
        info["bias_xy"][b] = bias

    return out, info


def _step_toward_xy(current_xy, target_xy, max_step):
    delta = target_xy - current_xy
    norm = torch.linalg.norm(delta, dim=1, keepdim=True).clamp_min(1.0e-9)
    scale = torch.clamp(float(max_step) / norm, max=1.0)
    return current_xy + delta * scale


def make_base_ref(x_hat, foot_pos, ref_mask, prev_base_ref, cfg, phase, phase_step, active_leg=None):
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
    if phase == "warmup" or phase == "recenter" or (phase == "shift" and phase_step < args.freeze_xy_until_shift):
        if phase != "recenter" or args.recenter_freeze_xy:
            base_ref[:, 0:2] = x_hat[:, 0:2]
    elif args.freeze_xy_during_warmup and prev_base_ref is not None:
        pass

    fr_lite_info = {
        "active": False,
        "mode": args.fr_lite_mode if hasattr(args, "fr_lite_mode") else "none",
        "stance_count": 0,
        "raw_safe_xy": x_hat[:, 0:2].detach().clone(),
        "biased_safe_xy": x_hat[:, 0:2].detach().clone(),
        "bias_xy": torch.zeros_like(x_hat[:, 0:2]),
        "target_xy": x_hat[:, 0:2].detach().clone(),
        "directional_desired_xy": x_hat[:, 0:2].detach().clone(),
        "directional_delta_xy": torch.zeros_like(x_hat[:, 0:2]),
        "applied_delta_xy": torch.zeros_like(x_hat[:, 0:2]),
    }

    fr_phase_ok = (
        (phase == "shift" and args.fr_lite_apply_in_shift)
        or (phase == "recenter" and args.fr_lite_apply_in_recenter)
    )
    if (
        args.enable_feasible_region_lite_shift
        and fr_phase_ok
        and phase_step >= args.fr_lite_start_after_steps
    ):
        safe_xy, fr_lite_info = _safe_point_of_support_polygon_lite(foot_pos, ref_mask, x_hat, active_leg=active_leg)

        target_xy = safe_xy
        if args.enable_directional_fr_lite:
            fwd, _left = _base_yaw_unit_vectors(x_hat)
            desired_forward_xy = x_hat[:, 0:2] + float(args.fr_directional_forward_distance) * fwd

            # Blend toward commanded forward progress, but still anchor the point at the safe support-region point.
            w_dir = float(args.fr_directional_weight)
            target_xy = (1.0 - w_dir) * safe_xy + w_dir * desired_forward_xy

            if args.fr_no_backward_along_yaw:
                cur_xy = x_hat[:, 0:2]
                delta = target_xy - cur_xy
                fwd_prog = torch.sum(delta * fwd, dim=1, keepdim=True)
                min_prog = torch.full_like(fwd_prog, float(args.fr_min_forward_delta))
                correction = torch.clamp(min_prog - fwd_prog, min=0.0)
                target_xy = target_xy + correction * fwd

            fr_lite_info["directional_desired_xy"] = desired_forward_xy.detach().clone()
            fr_lite_info["directional_delta_xy"] = (target_xy - safe_xy).detach().clone()

        blended_xy = (1.0 - float(args.fr_lite_blend)) * base_ref[:, 0:2] + float(args.fr_lite_blend) * target_xy
        start_xy = x_hat[:, 0:2] if prev_base_ref is None else prev_base_ref[:, 0:2]
        next_xy = _step_toward_xy(start_xy, blended_xy, args.fr_lite_max_shift_per_step)
        fr_lite_info["target_xy"] = blended_xy.detach().clone()
        fr_lite_info["applied_delta_xy"] = (next_xy - x_hat[:, 0:2]).detach().clone()
        base_ref[:, 0:2] = next_xy

    if phase == "warmup":
        base_ref[:, 2] = args.warmup_height_ref
    else:
        base_ref[:, 2] = args.height_ref

    base_ref[:, 3:5] = 0.0
    out.fr_lite_info = fr_lite_info
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



def visible_swing_assist_delta(leg, profile, q_now, foot_pos=None, swing_target=None):
    """Small explicit leg-folding target delta for visible clearance.

    This is not part of the WBC formulation; it is a Spot/implicit-actuator adapter
    used only to test whether a selected swing leg can visibly leave the ground.
    """
    delta = torch.zeros((q_now.shape[0], 3), device=q_now.device, dtype=q_now.dtype)
    if not args.enable_visible_swing_assist:
        return delta
    if args.enable_swing_overshoot_guard and foot_pos is not None and swing_target is not None:
        foot_z = float(foot_pos[0, leg, 2].detach().cpu())
        target_z = float(swing_target[0, leg, 2].detach().cpu())
        if foot_z > target_z + args.swing_overshoot_margin:
            return delta

    if leg in [LEG_TO_ID["LF"], LEG_TO_ID["RF"]]:
        hy_delta = args.visible_assist_front_hy_delta
        kn_delta = args.visible_assist_front_kn_delta
    else:
        hy_delta = args.visible_assist_hind_hy_delta
        kn_delta = args.visible_assist_hind_kn_delta

    p = max(float(profile), float(args.visible_assist_min_profile))
    p = max(0.0, min(1.0, p))

    # [hx, hy, kn]. Keep hx untouched; let IK handle lateral/fore-aft correction.
    delta[:, 1] = float(hy_delta) * p
    delta[:, 2] = float(kn_delta) * p

    # Bound the assist so it does not dominate the stabilizing implicit targets.
    n = torch.linalg.norm(delta, dim=1, keepdim=True).clamp_min(1.0e-6)
    scale = torch.clamp(float(args.visible_assist_max_delta_norm) / n, max=1.0)
    return delta * scale



def is_selected_foot_down(foot_pos, leg, tol):
    z_all = foot_pos[0, :, 2].detach()
    stance_ids = [i for i in range(4) if i != int(leg)]
    stance_ref_z = torch.max(z_all[stance_ids])
    selected_z = z_all[int(leg)]
    return bool((selected_z <= stance_ref_z + float(tol)).detach().cpu())


def reset_selected_joint_targets_to_current(robot, leg, zero_velocity=True):
    joint_ids = [int(HX[leg]), int(HY[leg]), int(KN[leg])]
    q_now = robot.data.joint_pos.detach()
    q_sel = q_now[:, joint_ids]
    joint_ids_tensor = torch.tensor(joint_ids, device=q_now.device, dtype=torch.long)
    robot.set_joint_position_target(q_sel, joint_ids=joint_ids_tensor)
    if zero_velocity:
        robot.set_joint_velocity_target(torch.zeros_like(q_sel), joint_ids=joint_ids_tensor)


def is_safe_all_stance_for_record(x_hat):
    return (
        abs(float(x_hat[0, 3].detach().cpu())) <= args.record_safe_stance_max_abs_roll
        and abs(float(x_hat[0, 4].detach().cpu())) <= args.record_safe_stance_max_abs_pitch
        and float(x_hat[0, 2].detach().cpu()) >= args.record_safe_stance_min_base_z
    )


def apply_touchdown_stance_recovery_target(robot, stance_q_ref, leg, active, hold_count):
    """Restore selected leg's implicit PD target to its pre-swing stance posture.

    This avoids the bad pattern:
      touchdown -> reset target to current folded pose -> implicit PD holds folded leg.
    """
    q_now = robot.data.joint_pos.detach()
    q_delta_full = torch.zeros_like(q_now)
    info = {
        "active": False,
        "leg": None,
        "alpha": 0.0,
        "joint_ids": [],
        "target": None,
        "target_minus_q": torch.zeros((q_now.shape[0], 3), device=q_now.device, dtype=q_now.dtype),
    }

    if not args.enable_touchdown_stance_recovery:
        return None, q_delta_full, info
    if not active or stance_q_ref is None or leg is None:
        return None, q_delta_full, info

    if args.touchdown_recovery_include_hx:
        joint_ids = [int(HX[leg]), int(HY[leg]), int(KN[leg])]
    else:
        joint_ids = [int(HY[leg]), int(KN[leg])]

    q_goal_full = stance_q_ref.detach().clone()
    q_goal_full[:, HY[leg]] = q_goal_full[:, HY[leg]] + float(args.touchdown_recovery_hy_offset)
    q_goal_full[:, KN[leg]] = q_goal_full[:, KN[leg]] + float(args.touchdown_recovery_kn_offset)

    q_goal = q_goal_full[:, joint_ids]
    q_sel_now = q_now[:, joint_ids]

    ramp = min(1.0, max(float(args.touchdown_recovery_min_alpha),
                        float(hold_count) / max(1.0, float(args.touchdown_recovery_steps))))
    q_sel = (1.0 - ramp) * q_sel_now + ramp * q_goal
    delta = torch.clamp(q_sel - q_sel_now,
                        -float(args.touchdown_recovery_max_delta),
                        float(args.touchdown_recovery_max_delta))
    q_sel = q_sel_now + delta

    joint_ids_tensor = torch.tensor(joint_ids, device=q_now.device, dtype=torch.long)
    robot.set_joint_position_target(q_sel, joint_ids=joint_ids_tensor)
    if args.touchdown_recovery_zero_velocity:
        robot.set_joint_velocity_target(torch.zeros_like(q_sel), joint_ids=joint_ids_tensor)

    q_delta_full[:, joint_ids] = q_sel - q_sel_now
    q_target_full = q_now.clone()
    q_target_full[:, joint_ids] = q_sel

    # Pack into 3 values for hx/hy/kn reporting.
    target_minus_q_3 = torch.zeros((q_now.shape[0], 3), device=q_now.device, dtype=q_now.dtype)
    if args.touchdown_recovery_include_hx:
        target_minus_q_3[:, :] = q_delta_full[:, [HX[leg], HY[leg], KN[leg]]]
    else:
        target_minus_q_3[:, 1:] = q_delta_full[:, [HY[leg], KN[leg]]]

    info.update({
        "active": True,
        "leg": int(leg),
        "alpha": float(ramp),
        "joint_ids": joint_ids,
        "target": q_target_full,
        "target_minus_q": target_minus_q_3,
    })
    return q_target_full, q_delta_full, info


def check_late_touchdown_foothold_ready(foot_pos, foot_vel, leg, foothold_target_w, foot_swing0):
    """B8-bk: contact is restored only when toe is close to the target foothold."""
    has_foothold = foothold_target_w is not None
    if leg is None or not has_foothold:
        return False, {
            "has_foothold": bool(has_foothold),
            "xy_err": 999.0,
            "z_err": 999.0,
            "speed": 999.0,
            "xy_ok": False,
            "z_ok": False,
            "speed_ok": False,
        }

    xy_err = float(torch.linalg.norm(foot_pos[0, leg, 0:2] - foothold_target_w[0, 0:2]).detach().cpu())

    if foot_swing0 is not None:
        landing_z = foot_swing0[0, leg, 2]
    else:
        landing_z = foothold_target_w[0, 2]
    z_err = float(torch.abs(foot_pos[0, leg, 2] - landing_z).detach().cpu())
    speed = float(torch.linalg.norm(foot_vel[0, leg, :]).detach().cpu())

    xy_ok = xy_err <= float(args.late_touchdown_xy_tol)
    z_ok = z_err <= float(args.late_touchdown_z_tol)
    speed_ok = speed <= float(args.late_touchdown_max_foot_speed)

    return bool(xy_ok and z_ok and speed_ok), {
        "has_foothold": True,
        "xy_err": xy_err,
        "z_err": z_err,
        "speed": speed,
        "xy_ok": bool(xy_ok),
        "z_ok": bool(z_ok),
        "speed_ok": bool(speed_ok),
    }


def apply_foothold_lock_recovery_target(robot, Jfeet_full, foot_pos, leg, foothold_target_w, active):
    """During touchdown recovery, lock previous swing foot to its new foothold."""
    q_now = robot.data.joint_pos.detach()
    q_delta_full = torch.zeros_like(q_now)
    info = {
        "active": False,
        "leg": None,
        "joint_ids": [],
        "foot_err": torch.zeros((q_now.shape[0], 3), device=q_now.device, dtype=q_now.dtype),
        "dq_cmd": torch.zeros((q_now.shape[0], 3), device=q_now.device, dtype=q_now.dtype),
        "target_minus_q": torch.zeros((q_now.shape[0], 3), device=q_now.device, dtype=q_now.dtype),
        "target": None,
    }

    if not args.enable_foothold_lock_recovery:
        return None, q_delta_full, info
    if not active or leg is None or foothold_target_w is None:
        return None, q_delta_full, info

    joint_ids = [int(HX[leg]), int(HY[leg]), int(KN[leg])]
    full_cols = [6 + jid for jid in joint_ids]

    foot_err = foothold_target_w[:, 0:3] - foot_pos[:, leg, :]
    foot_err = foot_err.clone()
    foot_err[:, 0:2] = float(args.foothold_lock_target_scale_xy) * foot_err[:, 0:2]
    foot_err[:, 2] = float(args.foothold_lock_target_scale_z) * foot_err[:, 2]

    Jleg_full = Jfeet_full[:, leg, :, full_cols]
    dq = damped_least_squares_delta(Jleg_full, foot_err, args.foothold_lock_damping)
    dq = float(args.foothold_lock_gain) * dq
    dq = torch.clamp(dq, -float(args.foothold_lock_max_joint_delta), float(args.foothold_lock_max_joint_delta))

    q_sel = q_now[:, joint_ids] + dq
    joint_ids_tensor = torch.tensor(joint_ids, device=q_now.device, dtype=torch.long)
    robot.set_joint_position_target(q_sel, joint_ids=joint_ids_tensor)
    if args.foothold_lock_zero_velocity:
        robot.set_joint_velocity_target(torch.zeros_like(q_sel), joint_ids=joint_ids_tensor)

    q_delta_full[:, joint_ids] = q_sel - q_now[:, joint_ids]
    q_target_full = q_now.clone()
    q_target_full[:, joint_ids] = q_sel

    info.update({
        "active": True,
        "leg": int(leg),
        "joint_ids": joint_ids,
        "foot_err": foot_err,
        "dq_cmd": dq,
        "target_minus_q": q_delta_full[:, [HX[leg], HY[leg], KN[leg]]],
        "target": q_target_full,
    })
    return q_target_full, q_delta_full, info


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
        "visible_assist": torch.zeros((q_now.shape[0], 3), device=q_now.device, dtype=q_now.dtype),
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

    assist_delta_hxhykn = visible_swing_assist_delta(leg, profile, q_now, foot_pos=foot_pos, swing_target=swing_target)
    if args.ik_use_hx:
        assist_selected = assist_delta_hxhykn[:, :]
    else:
        assist_selected = assist_delta_hxhykn[:, 1:]

    q_selected = q_now[:, joint_ids] + dq + assist_selected
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
    ik_info["visible_assist"] = assist_delta_hxhykn
    return q_target_full, q_delta_full, ik_info



def make_swing_target(phase, profile, foot_pos, foot_swing0, swing_enabled, foothold_target_w=None):
    """Swing target with optional explicit foothold.

    B8-bf:
      - If explicit foothold is available, xy target moves to foothold_target_w.
      - This prevents RF from simply lifting and landing near the original point.
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
        else:
            xy_profile = 1.0
            z_profile = profile

        if foothold_target_w is not None and args.enable_explicit_foothold_target:
            if phase == "lower" and args.foothold_hold_during_lower:
                xy_profile = 1.0
            target[:, leg, 0] = foot_swing0[:, leg, 0] + (foothold_target_w[:, 0] - foot_swing0[:, leg, 0]) * xy_profile
            target[:, leg, 1] = foot_swing0[:, leg, 1] + (foothold_target_w[:, 1] - foot_swing0[:, leg, 1]) * xy_profile
        else:
            target[:, leg, 0] = foot_swing0[:, leg, 0] + args.swing_forward_step * xy_profile
            target[:, leg, 1] = foot_swing0[:, leg, 1] + args.swing_lateral_step * xy_profile

        if args.target_z_mode == "hold":
            z_offset = 0.0
        elif args.target_z_mode == "step":
            z_offset = clearance if phase != "lower" else clearance * z_profile
        else:
            z_offset = clearance * z_profile

        if phase == "lower":
            z_offset = z_offset - args.landing_extra_down * (1.0 - z_profile)

        landing_z = foot_swing0[:, leg, 2]
        if foothold_target_w is not None and args.enable_explicit_foothold_target and not args.foothold_target_z_from_anchor:
            landing_z = foothold_target_w[:, 2]
        target[:, leg, 2] = landing_z + z_offset

    return target


def build_explicit_foothold_target(x_hat, foot_swing0, leg):
    target = foot_swing0[:, leg, :].detach().clone()
    if args.foothold_use_base_yaw:
        dx, dy = make_world_step_from_base_yaw(x_hat, args.foothold_forward_step, args.foothold_lateral_step)
    else:
        dx = torch.full_like(target[:, 0], float(args.foothold_forward_step))
        dy = torch.full_like(target[:, 1], float(args.foothold_lateral_step))
    target[:, 0] = target[:, 0] + dx
    target[:, 1] = target[:, 1] + dy
    if args.foothold_target_z_from_anchor:
        target[:, 2] = foot_swing0[:, leg, 2]
    return target


def check_load_bearing_touchdown(x_hat, foot_pos, foot_vel, last_f_qp, leg, foothold_target_w):
    """Return load-bearing touchdown diagnostics for the previous swing leg."""
    foot_down = is_selected_foot_down(foot_pos, leg, args.touchdown_foot_z_tol)
    speed = float(torch.linalg.norm(foot_vel[0, leg, :]).detach().cpu())
    fz = 0.0
    if last_f_qp is not None:
        fz = float(last_f_qp[0, leg, 2].detach().cpu())

    xy_err = 0.0
    has_foothold = foothold_target_w is not None
    if has_foothold:
        xy_err = float(torch.linalg.norm(foot_pos[0, leg, 0:2] - foothold_target_w[0, 0:2]).detach().cpu())

    roll = abs(float(x_hat[0, 3].detach().cpu()))
    pitch = abs(float(x_hat[0, 4].detach().cpu()))
    base_z = float(x_hat[0, 2].detach().cpu())

    valid = (
        foot_down
        and speed <= args.touchdown_max_foot_speed
        and fz >= args.touchdown_min_fz
        and (not has_foothold or xy_err <= args.touchdown_max_foothold_xy_err)
        and roll <= args.touchdown_max_abs_roll
        and pitch <= args.touchdown_max_abs_pitch
        and base_z >= args.touchdown_min_base_z
    )
    return {
        "valid": bool(valid),
        "foot_down": bool(foot_down),
        "speed": speed,
        "fz": fz,
        "xy_err": xy_err,
        "has_foothold": bool(has_foothold),
        "roll_abs": roll,
        "pitch_abs": pitch,
        "base_z": base_z,
    }


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
                old_forced, controlled_forced, step_idx, selected_second_leg=None, selected_second_leg_margins=None,
                late_touchdown=False, touchdown_gate_active=False, touchdown_gate_down=True, previous_swing_leg=None,
                recenter_active=False, recenter_safe_now=True, recenter_safe_seen=False,
                reset_detected_count=0, reset_detected_last=False,
                shift_gate_active=False, shift_gate_safe_now=True, shift_gate_passed=True,
                shift_gate_safe_count=0, shift_gate_base_ref_xy_err=0.0,
                foothold_target_w=None, foothold_target_leg=None,
                previous_swing_foothold_target_w=None, touchdown_load_diag=None,
                touchdown_load_hold_count=0, touchdown_load_valid=True,
                previous_swing_stance_q_ref=None, touchdown_recovery_info=None,
                touchdown_recovery_frozen_key=None, touchdown_recovery_freeze_count=0,
                freeze_step_key_for_touchdown=False,
                frozen_base_ref_active=False, frozen_base_ref=None,
                foothold_lock_info=None,
                pre_touchdown_lock_info=None,
                late_touchdown_candidate=False, late_touchdown_hold_swing=False,
                late_touchdown_ready=True, late_touchdown_diag=None,
                real_swing_leg=None, real_swing_active_steps=0, real_swing_seen_lower=False,
                previous_load_bearing_valid=True):
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
    print(f"[B8-bn DIRECTIONAL-FEASIBLE-REGION-SHIFT] step={step}")
    print("=" * 150)
    print("phase:", phase, "phase_step:", phase_step, "test_leg:", args.test_leg, "step_idx:", step_idx, "profile:", profile,
          "late_touchdown:", late_touchdown,
          "touchdown_gate_active:", touchdown_gate_active,
          "touchdown_gate_down:", touchdown_gate_down,
          "previous_swing_leg:", previous_swing_leg,
          "recenter_active:", recenter_active,
          "recenter_safe_now:", recenter_safe_now,
          "recenter_safe_seen:", recenter_safe_seen,
          "reset_detected_count:", reset_detected_count,
          "reset_detected_last:", reset_detected_last,
          "shift_gate_active:", shift_gate_active,
          "shift_gate_safe_now:", shift_gate_safe_now,
          "shift_gate_passed:", shift_gate_passed,
          "shift_gate_safe_count:", shift_gate_safe_count,
          "shift_gate_base_ref_xy_err:", shift_gate_base_ref_xy_err)
    if args.auto_select_second_leg and selected_second_leg_margins is not None:
        print("auto_second_selected:", selected_second_leg,
              "auto_second_candidate_margins:", selected_second_leg_margins)
    print("height_ref:", args.height_ref, "pd_scale:", args.pd_scale, "align_targets:", args.align_targets, "target_mode:", args.target_mode)
    print("anchor_swing_on_release:", args.anchor_swing_on_release, "swing_anchor_step:", swing_anchor_step,
          "target_z_mode:", args.target_z_mode, "diagnostic_clearance:", args.diagnostic_clearance)
    print("explicit_foothold:", args.enable_explicit_foothold_target,
          "foothold_target_leg:", foothold_target_leg,
          "foothold_target_w:", None if foothold_target_w is None else foothold_target_w[0].detach().cpu().numpy())
    print("load_bearing_touchdown:",
          "valid:", touchdown_load_valid,
          "hold_count:", touchdown_load_hold_count,
          "diag:", touchdown_load_diag,
          "previous_foothold_target_w:", None if previous_swing_foothold_target_w is None else previous_swing_foothold_target_w[0].detach().cpu().numpy())
    print("touchdown_stance_recovery:",
          None if touchdown_recovery_info is None else {
              "active": touchdown_recovery_info.get("active"),
              "leg": touchdown_recovery_info.get("leg"),
              "alpha": touchdown_recovery_info.get("alpha"),
              "joint_ids": touchdown_recovery_info.get("joint_ids"),
              "target_minus_q": touchdown_recovery_info.get("target_minus_q")[0].detach().cpu().numpy() if touchdown_recovery_info.get("target_minus_q") is not None else None,
          })
    if previous_swing_stance_q_ref is not None:
        _leg_for_ref = previous_swing_leg if previous_swing_leg is not None else LEG_TO_ID[args.test_leg]
        print("previous_stance_q_ref_selected hx/hy/kn:",
              previous_swing_stance_q_ref[0, [HX[_leg_for_ref], HY[_leg_for_ref], KN[_leg_for_ref]]].detach().cpu().numpy())
    print("touchdown_state_machine:",
          "freeze_step_key_for_touchdown:", freeze_step_key_for_touchdown,
          "frozen_key:", touchdown_recovery_frozen_key,
          "freeze_count:", touchdown_recovery_freeze_count,
          "current_test_leg:", args.test_leg)
    print("base_freeze:",
          "active:", frozen_base_ref_active,
          "frozen_base_ref:", None if frozen_base_ref is None else frozen_base_ref[0].detach().cpu().numpy())
    print("foothold_lock_recovery:",
          None if foothold_lock_info is None else {
              "active": foothold_lock_info.get("active"),
              "leg": foothold_lock_info.get("leg"),
              "joint_ids": foothold_lock_info.get("joint_ids"),
              "foot_err": foothold_lock_info.get("foot_err")[0].detach().cpu().numpy() if foothold_lock_info.get("foot_err") is not None else None,
              "dq_cmd": foothold_lock_info.get("dq_cmd")[0].detach().cpu().numpy() if foothold_lock_info.get("dq_cmd") is not None else None,
              "target_minus_q": foothold_lock_info.get("target_minus_q")[0].detach().cpu().numpy() if foothold_lock_info.get("target_minus_q") is not None else None,
          })
    print("delayed_touchdown:",
          "candidate:", late_touchdown_candidate,
          "hold_swing:", late_touchdown_hold_swing,
          "ready:", late_touchdown_ready,
          "diag:", late_touchdown_diag)
    print("pre_touchdown_foothold_lock:",
          None if pre_touchdown_lock_info is None else {
              "active": pre_touchdown_lock_info.get("active"),
              "leg": pre_touchdown_lock_info.get("leg"),
              "joint_ids": pre_touchdown_lock_info.get("joint_ids"),
              "foot_err": pre_touchdown_lock_info.get("foot_err")[0].detach().cpu().numpy() if pre_touchdown_lock_info.get("foot_err") is not None else None,
              "dq_cmd": pre_touchdown_lock_info.get("dq_cmd")[0].detach().cpu().numpy() if pre_touchdown_lock_info.get("dq_cmd") is not None else None,
              "target_minus_q": pre_touchdown_lock_info.get("target_minus_q")[0].detach().cpu().numpy() if pre_touchdown_lock_info.get("target_minus_q") is not None else None,
          })
    print("real_swing_commit_state:",
          "real_swing_leg:", real_swing_leg,
          "real_swing_active_steps:", real_swing_active_steps,
          "real_swing_seen_lower:", real_swing_seen_lower,
          "previous_load_bearing_valid:", previous_load_bearing_valid)
    print("enable_crouch_target:", args.enable_crouch_target, "crouch_ramp:", crouch_ramp,
          "crouch_base:", args.crouch_base, "hy_offset:", args.crouch_hy_offset,
          "kn_offset:", args.crouch_kn_offset)
    print("enable_jacobian_ik_swing:", args.enable_jacobian_ik_swing,
          "ik_active:", ik_info["active"],
          "ik_gain:", args.ik_gain,
          "ik_max_joint_delta:", args.ik_max_joint_delta)
    print("old_forced:", old_forced, "controlled_forced:", controlled_forced,
          "disable_margin_gate_for_release:", args.disable_margin_gate_for_release,
          "force_release_require_margin:", args.force_release_require_margin)
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
    fr_info = getattr(trial_out, "fr_lite_info", None)
    if fr_info is not None:
        print("feasible_region_lite:",
              {
                  "active": fr_info.get("active"),
                  "mode": fr_info.get("mode"),
                  "stance_count": fr_info.get("stance_count"),
                  "raw_safe_xy": fr_info.get("raw_safe_xy")[0].detach().cpu().numpy() if fr_info.get("raw_safe_xy") is not None else None,
                  "biased_safe_xy": fr_info.get("biased_safe_xy")[0].detach().cpu().numpy() if fr_info.get("biased_safe_xy") is not None else None,
                  "target_xy": fr_info.get("target_xy")[0].detach().cpu().numpy() if fr_info.get("target_xy") is not None else None,
                  "directional_desired_xy": fr_info.get("directional_desired_xy")[0].detach().cpu().numpy() if fr_info.get("directional_desired_xy") is not None else None,
                  "directional_delta_xy": fr_info.get("directional_delta_xy")[0].detach().cpu().numpy() if fr_info.get("directional_delta_xy") is not None else None,
                  "applied_delta_xy": fr_info.get("applied_delta_xy")[0].detach().cpu().numpy() if fr_info.get("applied_delta_xy") is not None else None,
              })

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
    if "visible_assist" in ik_info:
        print("visible_assist_hxhykn:", ik_info["visible_assist"][0].detach().cpu().numpy())
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
    if foothold_target_w is not None and foothold_target_leg == leg:
        print("foothold_target_minus_anchor_xyz:", (foothold_target_w[0] - foot_swing0[0, leg]).detach().cpu().numpy() if foot_swing0 is not None else None)
        print("foot_minus_foothold_xyz:", (foot_pos[0, leg] - foothold_target_w[0]).detach().cpu().numpy())
    print("target_z_minus_foot_z:", float(target_err[0, leg, 2].detach().cpu()))
    print("foot_z_world:", float(foot_pos[0, leg, 2].detach().cpu()))
    print("target_z_world:", float(swing_target[0, leg, 2].detach().cpu()))
    print("clearance_z_from_swing_start:", float(foot_delta_swing[0, leg, 2].detach().cpu()))
    if args.print_all_feet_z:
        print("all feet z LF/RF/LH/RH:", foot_pos[0, :, 2].detach().cpu().numpy())
        print("all target z LF/RF/LH/RH:", swing_target[0, :, 2].detach().cpu().numpy())
    print("=" * 150 + "\n")


def main():
    log_handle = setup_log_file()
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
    foothold_target_w = None
    foothold_target_leg = None
    previous_swing_foothold_target_w = None
    current_swing_stance_q_ref = None
    previous_swing_stance_q_ref = None
    last_safe_stance_q = None
    touchdown_load_hold_count = 0
    touchdown_load_diag = None
    touchdown_recovery_active = False
    touchdown_recovery_target = None
    touchdown_recovery_delta_full = None
    touchdown_recovery_info = None
    touchdown_recovery_frozen_key = None
    touchdown_recovery_freeze_count = 0
    frozen_base_ref = None
    frozen_base_ref_active = False
    foothold_lock_target = None
    foothold_lock_delta_full = None
    foothold_lock_info = None
    pre_touchdown_lock_target = None
    pre_touchdown_lock_delta_full = None
    pre_touchdown_lock_info = None
    late_touchdown_candidate = False
    late_touchdown_hold_swing = False
    late_touchdown_ready = False
    late_touchdown_diag = None
    real_swing_leg = None
    real_swing_active_steps = 0
    real_swing_seen_lower = False
    real_swing_foothold_target_w = None
    real_swing_stance_q_ref = None
    previous_load_bearing_valid = True
    last_f_qp = None
    swing_anchor_step = -1
    current_step_key = None
    selected_second_leg = None
    selected_second_leg_margins = None
    previous_swing_leg = None
    touchdown_gate_hold_count = 0
    recenter_safe_seen = False
    recenter_active = False
    recenter_safe_now = True
    reset_detected_count = 0
    reset_detected_last = False
    shift_gate_safe_count = 0
    lift_unlocked_by_shift_gate = {}
    swing_start_step = args.warmup_steps + args.shift_steps
    total_steps = args.warmup_steps + args.shift_steps + args.lift_steps + args.hold_steps + args.lower_steps + args.settle_steps
    if args.enable_two_step_sequence:
        total_steps = args.warmup_steps + sequence_total_len(len(parse_step_order()))

    print("\n" + "=" * 150)
    print("[INFO] Starting B8-bn directional feasible-region-lite trunk shift")
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
    print("visible_swing_assist:",
          "enabled:", args.enable_visible_swing_assist,
          "front_hy_delta:", args.visible_assist_front_hy_delta,
          "front_kn_delta:", args.visible_assist_front_kn_delta,
          "hind_hy_delta:", args.visible_assist_hind_hy_delta,
          "hind_kn_delta:", args.visible_assist_hind_kn_delta,
          "min_profile:", args.visible_assist_min_profile,
          "max_delta_norm:", args.visible_assist_max_delta_norm)
    print("touchdown_gate:",
          "overshoot_guard:", args.enable_swing_overshoot_guard,
          "overshoot_margin:", args.swing_overshoot_margin,
          "real_touchdown_gate:", args.enable_real_touchdown_gate,
          "touchdown_foot_z_tol:", args.touchdown_foot_z_tol,
          "touchdown_hold_steps:", args.touchdown_hold_steps,
          "reset_joint_target_on_touchdown:", args.reset_joint_target_on_touchdown)
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
    print("sequence_mode:",
          "enable_two_step_sequence:", args.enable_two_step_sequence,
          "step_order:", parse_step_order() if args.enable_two_step_sequence else [args.test_leg],
          "auto_select_second_leg:", args.auto_select_second_leg,
          "second_leg_candidates:", parse_second_leg_candidates(),
          "reset_base_ref_on_step_change:", args.reset_base_ref_on_step_change,
          "restore_contact_in_late_lower:", args.restore_contact_in_late_lower,
          "touchdown_profile_threshold:", args.touchdown_profile_threshold,
          "enable_base_recenter_between_steps:", args.enable_base_recenter_between_steps,
          "recenter_steps:", args.recenter_steps,
          "recenter_freeze_xy:", args.recenter_freeze_xy,
          "recenter_require_safe_for_next:", args.recenter_require_safe_for_next,
          "one_step_len:", one_step_len(),
          "sequence_total_len:", sequence_total_len(len(parse_step_order())) if args.enable_two_step_sequence else one_step_len(),
          "total_steps:", total_steps)
    print("future_margin_release:", args.future_margin_release, "min_shift_steps_before_release:", args.min_shift_steps_before_release)
    print("reset_safety:",
          "disable_env_terminations:", args.disable_env_terminations,
          "episode_length_s_override:", args.episode_length_s_override,
          "base_height_threshold_override:", args.base_height_threshold_override,
          "detect_env_reset:", args.detect_env_reset,
          "reset_detection_distance:", args.reset_detection_distance,
          "reset_detection_spawn_tol:", args.reset_detection_spawn_tol)
    print("strict_shift_gate:",
          "enabled:", args.enable_strict_support_shift_gate,
          "margin:", args.shift_gate_margin,
          "min_steps:", args.shift_gate_min_steps,
          "max_steps:", args.shift_gate_max_steps,
          "max_roll:", args.shift_gate_max_abs_roll,
          "max_pitch:", args.shift_gate_max_abs_pitch,
          "min_base_z:", args.shift_gate_min_base_z,
          "max_base_ref_xy_err:", args.shift_gate_max_base_ref_xy_err,
          "hold_safe_steps:", args.shift_gate_hold_safe_steps,
          "use_selected_leg_mask:", args.shift_gate_use_selected_leg_mask)
    print("explicit_foothold:",
          "enabled:", args.enable_explicit_foothold_target,
          "forward_step:", args.foothold_forward_step,
          "lateral_step:", args.foothold_lateral_step,
          "use_base_yaw:", args.foothold_use_base_yaw,
          "target_z_from_anchor:", args.foothold_target_z_from_anchor,
          "hold_during_lower:", args.foothold_hold_during_lower,
          "accept_on_late_touchdown:", args.foothold_accept_on_late_touchdown)
    print("load_bearing_touchdown:",
          "enabled:", args.enable_load_bearing_touchdown_gate,
          "min_fz:", args.touchdown_min_fz,
          "max_foot_speed:", args.touchdown_max_foot_speed,
          "max_foothold_xy_err:", args.touchdown_max_foothold_xy_err,
          "hold_steps:", args.touchdown_load_hold_steps,
          "max_roll:", args.touchdown_max_abs_roll,
          "max_pitch:", args.touchdown_max_abs_pitch,
          "min_base_z:", args.touchdown_min_base_z,
          "keep_foothold_until_valid:", args.keep_foothold_until_load_valid)
    print("touchdown_stance_recovery:",
          "enabled:", args.enable_touchdown_stance_recovery,
          "steps:", args.touchdown_recovery_steps,
          "min_alpha:", args.touchdown_recovery_min_alpha,
          "max_delta:", args.touchdown_recovery_max_delta,
          "include_hx:", args.touchdown_recovery_include_hx,
          "hy_offset:", args.touchdown_recovery_hy_offset,
          "kn_offset:", args.touchdown_recovery_kn_offset,
          "zero_velocity:", args.touchdown_recovery_zero_velocity)
    print("touchdown_state_machine:",
          "freeze_step_key_during_touchdown_recovery:", args.freeze_step_key_during_touchdown_recovery,
          "force_touchdown_recovery_active:", args.force_touchdown_recovery_active,
          "print_touchdown_state_machine:", args.print_touchdown_state_machine)
    print("base_freeze:",
          "during_swing:", args.freeze_base_ref_during_swing,
          "during_touchdown_recovery:", args.freeze_base_ref_during_touchdown_recovery,
          "keep_z:", args.freeze_base_ref_keep_z)
    print("foothold_lock_recovery:",
          "enabled:", args.enable_foothold_lock_recovery,
          "gain:", args.foothold_lock_gain,
          "damping:", args.foothold_lock_damping,
          "max_joint_delta:", args.foothold_lock_max_joint_delta,
          "target_scale_xy:", args.foothold_lock_target_scale_xy,
          "target_scale_z:", args.foothold_lock_target_scale_z,
          "zero_velocity:", args.foothold_lock_zero_velocity)
    print("delayed_touchdown:",
          "defer_contact_restore_until_foothold_valid:", args.defer_contact_restore_until_foothold_valid,
          "keep_swing_controller:", args.late_touchdown_keep_swing_controller,
          "force_foothold_lock:", args.late_touchdown_force_foothold_lock,
          "xy_tol:", args.late_touchdown_xy_tol,
          "z_tol:", args.late_touchdown_z_tol,
          "max_foot_speed:", args.late_touchdown_max_foot_speed)
    print("real_swing_commit_state:",
          "enabled:", args.enable_real_swing_commit_state,
          "min_active_steps:", args.real_swing_min_active_steps,
          "commit_only_after_lower:", args.real_swing_commit_only_after_lower,
          "block_recenter_until_previous_valid:", args.block_recenter_until_previous_valid,
          "second_shift_requires_margin:", args.second_shift_requires_margin)
    print("feasible_region_lite_shift:",
          "enabled:", args.enable_feasible_region_lite_shift,
          "mode:", args.fr_lite_mode,
          "apply_shift:", args.fr_lite_apply_in_shift,
          "apply_recenter:", args.fr_lite_apply_in_recenter,
          "start_after_steps:", args.fr_lite_start_after_steps,
          "max_shift_per_step:", args.fr_lite_max_shift_per_step,
          "blend:", args.fr_lite_blend,
          "forward_bias:", args.fr_lite_forward_bias,
          "lateral_bias_away_from_swing:", args.fr_lite_lateral_bias_away_from_swing,
          "min_margin_for_detach:", args.fr_lite_min_margin_for_detach)
    print("directional_feasible_region_lite:",
          "enabled:", args.enable_directional_fr_lite,
          "forward_distance:", args.fr_directional_forward_distance,
          "directional_weight:", args.fr_directional_weight,
          "no_backward_along_yaw:", args.fr_no_backward_along_yaw,
          "min_forward_delta:", args.fr_min_forward_delta,
          "force_no_swing_in_shift_phase:", args.force_no_swing_in_shift_phase)
    print("wbc_cfg:", wbc_cfg)
    print("=" * 150)

    for step in range(total_steps):
        if not simulation_app.is_running():
            break

        root_pos_before_step = robot.data.root_pos_w.detach().clone()
        x_hat = make_x_hat(robot, velocity_frame="world")
        foot_pos = robot.data.body_pos_w[:, foot_indices, :]
        foot_vel = get_foot_vel(robot, foot_indices)
        M = robot.root_physx_view.get_generalized_mass_matrices()
        Jfeet_full = robot.root_physx_view.get_jacobians()[:, foot_indices, 0:3, :]
        gravity, coriolis = get_gravity(robot), get_coriolis(robot)

        phase, profile, phase_step, active_leg, step_idx = get_phase(step)

        recenter_active = (phase == "recenter")
        recenter_safe_now = (
            abs(float(x_hat[0, 3].detach().cpu())) <= args.recenter_max_abs_roll
            and abs(float(x_hat[0, 4].detach().cpu())) <= args.recenter_max_abs_pitch
            and float(x_hat[0, 2].detach().cpu()) >= args.recenter_min_base_z
        )
        if recenter_active:
            if args.recenter_reset_prev_base_ref and phase_step == 0:
                prev_base_ref = None
            if recenter_safe_now:
                recenter_safe_seen = True

        # Record a safe all-stance posture to use later as touchdown stance recovery target.
        if (phase in ["warmup", "shift", "settle", "recenter"]) and is_safe_all_stance_for_record(x_hat):
            # Avoid recording while a leg is actively swinging; this should be a support posture.
            # During shift, contact_mask is still all stance before release, so it is useful.
            last_safe_stance_q = robot.data.joint_pos.detach().clone()

        # B8-bg: before starting the next step, require the previous swing foot to be
        # physically down AND load-bearing near its foothold.
        touchdown_gate_active = False
        touchdown_gate_down = True
        touchdown_load_valid = True
        if args.enable_real_touchdown_gate and args.enable_two_step_sequence and step_idx >= 1:
            if previous_swing_leg is not None:
                touchdown_gate_down = is_selected_foot_down(foot_pos, previous_swing_leg, args.touchdown_foot_z_tol)
                touchdown_load_diag = check_load_bearing_touchdown(
                    x_hat, foot_pos, foot_vel, last_f_qp, previous_swing_leg, previous_swing_foothold_target_w
                )
                touchdown_load_valid = bool(touchdown_load_diag["valid"]) if args.enable_load_bearing_touchdown_gate else True

                if touchdown_load_valid:
                    touchdown_load_hold_count += 1
                else:
                    touchdown_load_hold_count = 0

                need_hold = touchdown_gate_hold_count < args.touchdown_hold_steps
                need_load_hold = args.enable_load_bearing_touchdown_gate and touchdown_load_hold_count < args.touchdown_load_hold_steps

                if (not touchdown_gate_down) or (not touchdown_load_valid) or need_hold or need_load_hold:
                    phase = "settle"
                    profile = 0.0
                    phase_step = max(touchdown_gate_hold_count, touchdown_load_hold_count)
                    active_leg = ID_TO_LEG.get(previous_swing_leg, args.test_leg)
                    touchdown_gate_active = True
                    touchdown_gate_hold_count += 1

                    # Keep the previous foothold target visible and persistent until validation passes.
                    if args.keep_foothold_until_load_valid and previous_swing_foothold_target_w is not None:
                        foothold_target_w = previous_swing_foothold_target_w
                        foothold_target_leg = previous_swing_leg

                    if touchdown_gate_down and args.reset_joint_target_on_touchdown and not args.enable_touchdown_stance_recovery:
                        reset_selected_joint_targets_to_current(
                            robot, previous_swing_leg, zero_velocity=args.touchdown_reset_velocity_zero
                        )
                else:
                    previous_swing_leg = None
                    previous_swing_foothold_target_w = None
                    previous_swing_stance_q_ref = None
                    touchdown_load_diag = None
                    touchdown_recovery_frozen_key = None
                    touchdown_recovery_freeze_count = 0
                    frozen_base_ref = None
                    frozen_base_ref_active = False
                    previous_load_bearing_valid = True
                    real_swing_leg = None
                    real_swing_active_steps = 0
                    real_swing_seen_lower = False
                    real_swing_foothold_target_w = None
                    real_swing_stance_q_ref = None

        if args.auto_select_second_leg and args.enable_two_step_sequence and step_idx == 1 and not touchdown_gate_active and not recenter_active:
            if selected_second_leg is None:
                selected_second_leg, selected_second_leg_margins = select_best_second_leg(
                    x_hat, foot_pos, ref_cfg, device, dtype
                )
                print("[AUTO-SECOND-LEG] selected:", selected_second_leg,
                      "candidate_margins:", selected_second_leg_margins)
            active_leg = selected_second_leg

        if args.test_leg != active_leg:
            args.test_leg = active_leg

        step_key = (step_idx, active_leg)

        # B8-bl:
        # Do NOT create previous_swing_leg from current_step_key/nominal schedule.
        # Only an actual contact-free swing is allowed to become previous_swing_leg.
        freeze_step_key_for_touchdown = (
            args.freeze_step_key_during_touchdown_recovery
            and previous_swing_leg is not None
            and step_idx >= 1
        )
        if freeze_step_key_for_touchdown:
            touchdown_recovery_freeze_count += 1
            if touchdown_recovery_frozen_key is None:
                touchdown_recovery_frozen_key = current_step_key
            if args.print_touchdown_state_machine and step % max(args.print_every, 1) == 0:
                print("[TOUCHDOWN-SM] freeze step-key update:",
                      "step:", step,
                      "nominal_step_key:", step_key,
                      "current_step_key:", current_step_key,
                      "previous_swing_leg:", previous_swing_leg,
                      "real_swing_leg:", real_swing_leg,
                      "freeze_count:", touchdown_recovery_freeze_count)
        elif step_key != current_step_key:
            # Nominal schedule changed. Clear swing anchor for the new planned leg,
            # but do not mark it as previous_swing_leg yet.
            if step_idx >= 1 and not recenter_active:
                pass
            foot_swing0 = None
            foothold_target_w = None
            foothold_target_leg = None
            current_swing_stance_q_ref = None
            frozen_base_ref = None
            frozen_base_ref_active = False
            swing_anchor_step = -1
            shift_gate_safe_count = 0
            real_swing_leg = None
            real_swing_active_steps = 0
            real_swing_seen_lower = False
            real_swing_foothold_target_w = None
            real_swing_stance_q_ref = None
            if args.reset_base_ref_on_step_change or args.recenter_reset_prev_base_ref:
                prev_base_ref = None
            current_step_key = step_key

        late_touchdown_candidate = (
            args.restore_contact_in_late_lower
            and phase == "lower"
            and profile <= args.touchdown_profile_threshold
        )

        late_touchdown_ready = True
        late_touchdown_diag = None
        if late_touchdown_candidate and args.defer_contact_restore_until_foothold_valid:
            _td_leg = LEG_TO_ID[args.test_leg]
            late_touchdown_ready, late_touchdown_diag = check_late_touchdown_foothold_ready(
                foot_pos, foot_vel, _td_leg, foothold_target_w, foot_swing0
            )

        late_touchdown = bool(late_touchdown_candidate and late_touchdown_ready)
        late_touchdown_hold_swing = bool(late_touchdown_candidate and not late_touchdown)

        if late_touchdown and args.foothold_accept_on_late_touchdown:
            # Do not reset to current folded pose when stance recovery is enabled.
            if not args.enable_touchdown_stance_recovery:
                reset_selected_joint_targets_to_current(robot, LEG_TO_ID[args.test_leg], zero_velocity=args.touchdown_reset_velocity_zero)
            if not args.keep_foothold_until_load_valid:
                foothold_target_w = None
                foothold_target_leg = None

        # B8-be: use selected-leg future support mask for the active leg.
        def _future_mask_for_active_leg():
            if args.shift_gate_use_selected_leg_mask:
                return future_swing_mask_for_leg(args.num_envs, device, dtype, active_leg)
            return future_swing_mask(args.num_envs, device, dtype)

        ref_mask = all_stance_mask(args.num_envs, device, dtype) if phase in ["warmup", "settle", "recenter", "done"] or late_touchdown else _future_mask_for_active_leg()
        base_ref, trial_out = make_base_ref(x_hat, foot_pos, ref_mask, prev_base_ref, ref_cfg, phase, phase_step, active_leg=active_leg)
        base_ref_raw = base_ref.detach().clone()

        # B8-bj: separate trunk shift from swing/recovery.
        freeze_base_now = (
            (args.freeze_base_ref_during_swing and phase in ["lift", "hold_lift", "lower"] and (swing_enabled or late_touchdown_hold_swing))
            or (args.freeze_base_ref_during_touchdown_recovery and (touchdown_gate_active or freeze_step_key_for_touchdown or late_touchdown_hold_swing))
        )
        if freeze_base_now:
            if frozen_base_ref is None:
                frozen_base_ref = base_ref_raw.detach().clone()
            base_ref = frozen_base_ref.detach().clone()
            if args.freeze_base_ref_keep_z:
                base_ref[:, 2] = args.height_ref
                base_ref[:, 3] = 0.0
                base_ref[:, 4] = 0.0
                base_ref[:, 5] = 0.0
            frozen_base_ref_active = True
        else:
            frozen_base_ref = None
            frozen_base_ref_active = False

        # Strict event gate: lift/hold/lower cannot begin until the previous shift was actually safe.
        shift_gate_active = False
        shift_gate_passed = True
        shift_gate_safe_now = True
        shift_gate_base_ref_xy_err = float(torch.linalg.norm((base_ref - x_hat[:, 0:6])[:, 0:2], dim=1)[0].detach().cpu())
        shift_gate_key = (int(step_idx), str(active_leg))

        if args.enable_strict_support_shift_gate and phase not in ["warmup", "settle", "recenter", "done"]:
            roll = abs(float(x_hat[0, 3].detach().cpu()))
            pitch = abs(float(x_hat[0, 4].detach().cpu()))
            base_z = float(x_hat[0, 2].detach().cpu())
            trial_margin = float(trial_out.margin_to_edge[0].detach().cpu())
            trial_allowed = bool(trial_out.swing_allowed[0].detach().cpu())

            shift_gate_safe_now = (
                trial_allowed
                and trial_margin >= args.shift_gate_margin
                and roll <= args.shift_gate_max_abs_roll
                and pitch <= args.shift_gate_max_abs_pitch
                and base_z >= args.shift_gate_min_base_z
                and shift_gate_base_ref_xy_err <= args.shift_gate_max_base_ref_xy_err
            )

            if phase == "shift":
                if shift_gate_safe_now and phase_step >= args.shift_gate_min_steps:
                    shift_gate_safe_count += 1
                else:
                    shift_gate_safe_count = 0
                if shift_gate_safe_count >= args.shift_gate_hold_safe_steps:
                    lift_unlocked_by_shift_gate[shift_gate_key] = True
            elif phase in ["lift", "hold_lift", "lower"]:
                shift_gate_passed = bool(lift_unlocked_by_shift_gate.get(shift_gate_key, False))
                if not shift_gate_passed and phase_step < args.shift_gate_max_steps:
                    # Hold in support-shift mode with selected-leg future ref_mask, but keep all contacts.
                    phase = "shift"
                    profile = 0.0
                    shift_gate_active = True
                    late_touchdown = False
                    ref_mask = _future_mask_for_active_leg()
                    base_ref, trial_out = make_base_ref(x_hat, foot_pos, ref_mask, prev_base_ref, ref_cfg, phase, args.shift_gate_min_steps)
                    shift_gate_base_ref_xy_err = float(torch.linalg.norm((base_ref - x_hat[:, 0:6])[:, 0:2], dim=1)[0].detach().cpu())
                elif not shift_gate_passed:
                    print(f"[WARN][SHIFT-GATE] max wait exceeded at step={step}, step_idx={step_idx}, leg={active_leg}; allowing phase={phase}")

        prev_base_ref = base_ref.detach().clone()

        swing_enabled = gate_swing(phase, phase_step, trial_out, x_hat)
        old_forced, controlled_forced = gate_debug_flags(phase, phase_step, trial_out, x_hat)
        if args.enable_feasible_region_lite_shift and phase in ["lift", "hold_lift", "lower"]:
            if float(trial_out.margin_to_edge[0].detach().cpu()) < float(args.fr_lite_min_margin_for_detach):
                swing_enabled = False
        if phase in ["shift", "settle", "recenter", "warmup", "done"] or late_touchdown or (args.recenter_require_safe_for_next and step_idx >= 1 and not recenter_safe_seen):
            # pre-shift / settle / recenter / warmup keep all contacts.
            # B8-bk: late_touchdown only becomes True after foothold-ready check passes.
            swing_enabled = False

        if late_touchdown_hold_swing and args.late_touchdown_keep_swing_controller:
            # Keep selected leg in swing/contact-free mode while lower phase waits for foothold commit.
            swing_enabled = True

        # B8-bl: second leg may not detach until it has a real feasible-support shift.
        if step_idx >= 1 and phase in ["lift", "hold_lift", "lower"]:
            _margin_now = float(trial_out.margin_to_edge[0].detach().cpu())
            _needed_margin = float(args.fr_lite_min_margin_for_detach) if args.enable_feasible_region_lite_shift else float(args.second_shift_requires_margin)
            if _margin_now < _needed_margin:
                swing_enabled = False

        # If previous RF touchdown is still not load-bearing valid, block next recenter/shift detachment.
        if args.block_recenter_until_previous_valid and step_idx >= 1 and not previous_load_bearing_valid:
            swing_enabled = False

        # B8-bn: state-machine cleanup. Shift is only for trunk/CoM motion.
        # Foot detachment must wait until lift/hold_lift/lower.
        if args.force_no_swing_in_shift_phase and phase == "shift":
            swing_enabled = False

        contact_mask = _future_mask_for_active_leg() if swing_enabled else all_stance_mask(args.num_envs, device, dtype)

        # B8-bl: record only real contact-free swing activity.
        if args.enable_real_swing_commit_state and swing_enabled:
            _sleg = LEG_TO_ID[args.test_leg]
            if real_swing_leg is None:
                real_swing_leg = _sleg
                real_swing_active_steps = 0
                real_swing_seen_lower = False
                real_swing_foothold_target_w = None
                real_swing_stance_q_ref = None
            if real_swing_leg == _sleg:
                real_swing_active_steps += 1
                if phase == "lower":
                    real_swing_seen_lower = True
                if foothold_target_w is not None:
                    real_swing_foothold_target_w = foothold_target_w.detach().clone()
                if current_swing_stance_q_ref is not None:
                    real_swing_stance_q_ref = current_swing_stance_q_ref.detach().clone()

        # Commit the previous swing only when the real swing actually happened and touchdown is accepted.
        if (
            args.enable_real_swing_commit_state
            and late_touchdown
            and real_swing_leg is not None
            and real_swing_active_steps >= args.real_swing_min_active_steps
            and ((not args.real_swing_commit_only_after_lower) or real_swing_seen_lower)
        ):
            previous_swing_leg = int(real_swing_leg)
            previous_swing_foothold_target_w = (
                real_swing_foothold_target_w.detach().clone()
                if real_swing_foothold_target_w is not None
                else (foothold_target_w.detach().clone() if foothold_target_w is not None else None)
            )
            previous_swing_stance_q_ref = (
                real_swing_stance_q_ref.detach().clone()
                if real_swing_stance_q_ref is not None
                else (current_swing_stance_q_ref.detach().clone() if current_swing_stance_q_ref is not None else None)
            )
            touchdown_gate_hold_count = 0
            touchdown_load_hold_count = 0
            touchdown_load_diag = None
            touchdown_recovery_frozen_key = current_step_key
            touchdown_recovery_freeze_count = 0
            previous_load_bearing_valid = False

        # B8-ak: choose when to anchor the swing-foot reference.
        # Default reproduces B8-ah: anchor at nominal lift start.
        # With --anchor_swing_on_release, anchor only when the gate actually releases the leg.
        if foot_swing0 is None:
            if args.anchor_swing_on_release:
                if swing_enabled:
                    foot_swing0 = foot_pos.detach().clone()
                    swing_anchor_step = step
                    current_swing_stance_q_ref = last_safe_stance_q.detach().clone() if last_safe_stance_q is not None else robot.data.joint_pos.detach().clone()
                    if args.enable_explicit_foothold_target:
                        foothold_target_leg = LEG_TO_ID[args.test_leg]
                        foothold_target_w = build_explicit_foothold_target(x_hat, foot_swing0, foothold_target_leg)
            elif step >= swing_start_step:
                foot_swing0 = foot_pos.detach().clone()
                swing_anchor_step = step
                current_swing_stance_q_ref = last_safe_stance_q.detach().clone() if last_safe_stance_q is not None else robot.data.joint_pos.detach().clone()
                if args.enable_explicit_foothold_target:
                    foothold_target_leg = LEG_TO_ID[args.test_leg]
                    foothold_target_w = build_explicit_foothold_target(x_hat, foot_swing0, foothold_target_leg)

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

        swing_target = make_swing_target(phase, profile, foot_pos, foot_swing0, swing_enabled, foothold_target_w)

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

        pre_touchdown_lock_target, pre_touchdown_lock_delta_full, pre_touchdown_lock_info = apply_foothold_lock_recovery_target(
            robot, Jfeet_full, foot_pos, LEG_TO_ID[args.test_leg], foothold_target_w,
            bool(args.late_touchdown_force_foothold_lock and late_touchdown_hold_swing)
        )

        touchdown_recovery_active = bool(
            args.enable_touchdown_stance_recovery
            and previous_swing_leg is not None
            and (
                touchdown_gate_active
                or args.force_touchdown_recovery_active
                or freeze_step_key_for_touchdown
            )
        )
        touchdown_recovery_target, touchdown_recovery_delta_full, touchdown_recovery_info = apply_touchdown_stance_recovery_target(
            robot, previous_swing_stance_q_ref, previous_swing_leg,
            touchdown_recovery_active and not args.enable_foothold_lock_recovery,
            touchdown_load_hold_count
        )

        foothold_lock_target, foothold_lock_delta_full, foothold_lock_info = apply_foothold_lock_recovery_target(
            robot, Jfeet_full, foot_pos, previous_swing_leg, previous_swing_foothold_target_w, touchdown_recovery_active
        )

        q_target = apply_implicit_target_alignment(robot, q_nom, phase, profile, swing_enabled)
        # For logging: report the adapter target actually used.
        if q_target is None:
            q_target = (
                pre_touchdown_lock_target
                if pre_touchdown_lock_target is not None
                else (
                    foothold_lock_target
                    if foothold_lock_target is not None
                    else (
                        touchdown_recovery_target
                        if touchdown_recovery_target is not None
                        else (ik_target if ik_target is not None else crouch_target)
                    )
                )
            )

        combined_delta_full = (
            crouch_delta_full
            + ik_delta_full
            + (pre_touchdown_lock_delta_full if pre_touchdown_lock_delta_full is not None else torch.zeros_like(crouch_delta_full))
            + (touchdown_recovery_delta_full if touchdown_recovery_delta_full is not None else torch.zeros_like(crouch_delta_full))
            + (foothold_lock_delta_full if foothold_lock_delta_full is not None else torch.zeros_like(crouch_delta_full))
        )
        last_f_qp = qpd["f_qp"].detach().clone()

        env.step(tau_cmd)

        reset_detected_last = False
        if args.detect_env_reset:
            root_pos_after_step = robot.data.root_pos_w.detach().clone()
            if detect_unexpected_reset(root_pos_before_step, root_pos_after_step, step):
                reset_detected_count += 1
                reset_detected_last = True
                print("\n" + "!" * 120)
                print(f"[RESET-DETECTED] step={step} count={reset_detected_count}")
                print("root_before:", root_pos_before_step[0].detach().cpu().numpy())
                print("root_after: ", root_pos_after_step[0].detach().cpu().numpy())
                print("[RESET-DETECTED] Clearing controller internal state to match respawned simulator state.")
                print("!" * 120 + "\n")

                # Clear controller internal states so a respawned robot is not commanded
                # as if it were still in the previous step/phase.
                prev_base_ref = None
                foot_swing0 = None
                foothold_target_w = None
                foothold_target_leg = None
                previous_swing_foothold_target_w = None
                current_swing_stance_q_ref = None
                previous_swing_stance_q_ref = None
                last_safe_stance_q = None
                swing_anchor_step = -1
                current_step_key = None
                selected_second_leg = None
                selected_second_leg_margins = None
                previous_swing_leg = None
                touchdown_gate_hold_count = 0
                touchdown_load_hold_count = 0
                touchdown_load_diag = None
                touchdown_recovery_frozen_key = None
                touchdown_recovery_freeze_count = 0
                frozen_base_ref = None
                frozen_base_ref_active = False
                late_touchdown_candidate = False
                late_touchdown_hold_swing = False
                late_touchdown_ready = False
                late_touchdown_diag = None
                real_swing_leg = None
                real_swing_active_steps = 0
                real_swing_seen_lower = False
                real_swing_foothold_target_w = None
                real_swing_stance_q_ref = None
                previous_load_bearing_valid = True
                recenter_safe_seen = False
                shift_gate_safe_count = 0
                lift_unlocked_by_shift_gate = {}
                q_initial = robot.data.joint_pos.detach().clone()
                q_nom = q_initial.clone()
                if args.enable_crouch_target and args.use_crouch_q_nom:
                    q_nom = build_crouch_target(q_initial, q_initial).detach().clone()

        if step % max(args.print_every, 1) == 0:
            print_debug(
                step, phase, profile, phase_step, x_hat, base_ref, trial_out, ref_mask, contact_mask,
                f_mpc, foot_pos, foot_swing0, swing_target, tau_cmd, qpd, robot, swing_enabled, q_target,
                swing_anchor_step, combined_delta_full, crouch_target, crouch_ramp, ik_target, ik_delta_full, ik_info,
                old_forced, controlled_forced, step_idx, selected_second_leg, selected_second_leg_margins,
                late_touchdown, touchdown_gate_active, touchdown_gate_down, previous_swing_leg,
                recenter_active, recenter_safe_now, recenter_safe_seen,
                reset_detected_count, reset_detected_last,
                shift_gate_active, shift_gate_safe_now, shift_gate_passed,
                shift_gate_safe_count, shift_gate_base_ref_xy_err,
                foothold_target_w, foothold_target_leg,
                previous_swing_foothold_target_w, touchdown_load_diag,
                touchdown_load_hold_count, touchdown_load_valid,
                previous_swing_stance_q_ref, touchdown_recovery_info,
                touchdown_recovery_frozen_key, touchdown_recovery_freeze_count,
                freeze_step_key_for_touchdown,
                frozen_base_ref_active, frozen_base_ref,
                foothold_lock_info,
                pre_touchdown_lock_info,
                late_touchdown_candidate, late_touchdown_hold_swing,
                late_touchdown_ready, late_touchdown_diag,
                real_swing_leg, real_swing_active_steps, real_swing_seen_lower,
                previous_load_bearing_valid
            )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
