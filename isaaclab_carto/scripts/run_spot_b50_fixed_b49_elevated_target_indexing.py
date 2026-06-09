# isaaclab_carto/scripts/run_spot_b8ah_stable_warmup_decoupled_preshift.py
#
# B39: Latched candidate-specific support-region recenter before hind-leg liftoff.
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
import copy
from typing import Any

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CARTO/TRACER B39 latched support recenter")

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

# B9-a: canonical gait schedule. S_t is the single source of truth for contact/controller switching.
parser.add_argument("--enable_canonical_gait_schedule", action="store_true")
parser.add_argument("--gait_period_steps", type=int, default=560,
                    help="Gait period T in control steps.")
parser.add_argument("--gait_duty", type=float, default=0.72,
                    help="Duty factor xi. stance if phi < xi, swing otherwise.")
parser.add_argument("--gait_phase_offsets", type=str, default="LF:0.50,RF:0.00,LH:0.75,RH:0.25",
                    help="Leg phase offsets phi_i. Format LF:0.5,RF:0.0,LH:0.75,RH:0.25")
parser.add_argument("--canonical_gait_swing_height", type=float, default=0.030)
parser.add_argument("--canonical_gait_forward_step", type=float, default=0.045)
parser.add_argument("--canonical_gait_lateral_step", type=float, default=0.0)
parser.add_argument("--canonical_gait_use_base_yaw", action="store_true")
parser.add_argument("--canonical_gait_disable_event_overrides", action="store_true",
                    help="Disable B8 event overrides and let S_t drive contact/controller switching.")
parser.add_argument("--canonical_gait_print_edges", action="store_true")
parser.add_argument("--canonical_gait_start_step", type=int, default=320,
                    help="Before this control step, force all legs to stance and do not advance gait phase.")
parser.add_argument("--canonical_gait_force_all_stance_before_start", action="store_true")
parser.add_argument("--canonical_gait_use_relative_time", action="store_true",
                    help="Use (step - start_step)/T instead of step/T for phase.")
parser.add_argument("--canonical_debug_use_active_swing_leg", action="store_true",
                    help="Temporarily print selected-leg diagnostics for the first scheduled swing leg.")
# B9-c: schedule execution layer. Requested S_t from phase is filtered into executed S_t.
parser.add_argument("--canonical_single_swing_gate", action="store_true",
                    help="Allow at most one swing leg at a time.")
parser.add_argument("--canonical_min_all_stance_gap_steps", type=int, default=80,
                    help="Required all-stance control steps between different swing legs.")
parser.add_argument("--canonical_liftoff_min_margin", type=float, default=0.085,
                    help="Minimum future support margin before allowing a requested liftoff.")
parser.add_argument("--canonical_liftoff_max_abs_roll", type=float, default=0.055)
parser.add_argument("--canonical_liftoff_max_abs_pitch", type=float, default=0.060)
parser.add_argument("--canonical_liftoff_min_base_z", type=float, default=0.645)
parser.add_argument("--canonical_first_swing_leg", type=str, default="LH", choices=["LF", "RF", "LH", "RH"],
                    help="Preferred first swing leg when multiple/early requests exist.")
parser.add_argument("--canonical_crawl_priority", type=str, default="LH,RF,RH,LF",
                    help="Priority order for resolving simultaneous requested swing legs.")
parser.add_argument("--canonical_print_execution_gate", action="store_true")
parser.add_argument("--canonical_enable_feasible_trunk_shift", action="store_true",
                    help="Move base_ref toward a feasibility-inspired safe point before/while executing swing.")
parser.add_argument("--canonical_shift_before_liftoff", action="store_true",
                    help="If requested liftoff is gated, keep all-stance and shift trunk toward future support safe point.")
parser.add_argument("--canonical_shift_during_swing", action="store_true")
parser.add_argument("--canonical_shift_max_step", type=float, default=0.0010)
parser.add_argument("--canonical_shift_blend", type=float, default=1.0)
parser.add_argument("--canonical_shift_forward_bias", type=float, default=0.002)
parser.add_argument("--canonical_shift_lateral_bias", type=float, default=0.018)
parser.add_argument("--canonical_shift_margin_stop", type=float, default=0.115,
                    help="When future support margin exceeds this value, liftoff can proceed without further all-stance shift hold.")
# B9-d: track-aware feasible shift. Prevent base_ref from running away from the actual base.
parser.add_argument("--canonical_shift_track_actual_base", action="store_true",
                    help="Use actual base xy as shift start and freeze target update if base_ref gets too far.")
parser.add_argument("--canonical_shift_max_base_ref_xy_err", type=float, default=0.055)
parser.add_argument("--canonical_shift_hold_when_ref_error", action="store_true",
                    help="If base_ref-actual-base xy error is too large, hold the previous base_ref instead of moving farther.")
parser.add_argument("--canonical_liftoff_allow_after_shift_steps", type=int, default=180,
                    help="If margin is close but not over threshold, allow liftoff after this many tracked shift steps.")
parser.add_argument("--canonical_liftoff_margin_relaxed", type=float, default=0.075,
                    help="Relaxed future margin threshold after enough tracked shift steps.")
parser.add_argument("--canonical_print_shift_tracking", action="store_true")
# B9-e: once the gate opens, execute a full local swing trajectory independent of requested sigma.
parser.add_argument("--canonical_enable_latched_executed_swing", action="store_true")
parser.add_argument("--canonical_executed_swing_duration_steps", type=int, default=180)
parser.add_argument("--canonical_executed_swing_min_duration_steps", type=int, default=120)
parser.add_argument("--canonical_executed_swing_force_gap_after_touchdown", type=int, default=120)
parser.add_argument("--canonical_print_latched_swing", action="store_true")
# B9-g: after executed swing ends, keep the landed foothold pinned long enough to become load-bearing.
parser.add_argument("--canonical_enable_touchdown_foothold_commit", action="store_true")
parser.add_argument("--canonical_commit_steps", type=int, default=220)
parser.add_argument("--canonical_commit_ik_gain", type=float, default=0.95)
parser.add_argument("--canonical_commit_ik_damping", type=float, default=0.025)
parser.add_argument("--canonical_commit_max_joint_delta", type=float, default=0.18)
parser.add_argument("--canonical_commit_target_scale_xy", type=float, default=1.0)
parser.add_argument("--canonical_commit_target_scale_z", type=float, default=0.6)
parser.add_argument("--canonical_commit_zero_velocity", action="store_true")
parser.add_argument("--canonical_commit_print", action="store_true")
# B9-h: event-driven crawl queue. Do not let analytical phase request the next leg while current swing/commit is active.
parser.add_argument("--canonical_enable_event_crawl_queue", action="store_true")
parser.add_argument("--canonical_event_queue", type=str, default="RF,LF,RH,LH")
parser.add_argument("--canonical_event_shift_min_future_margin", type=float, default=0.0,
                    help="Do not move trunk for the next leg if its future support margin is below this value.")
parser.add_argument("--canonical_event_commit_anchor_to_target", action="store_true",
                    help="When a swing ends, write the committed landing target into canonical_foot_anchor_w.")
parser.add_argument("--canonical_event_print", action="store_true")
# B9-i: capture the actual touchdown configuration instead of dragging a stance foot to a desired Cartesian target.
parser.add_argument("--canonical_enable_touchdown_capture", action="store_true")
parser.add_argument("--canonical_capture_steps", type=int, default=260)
parser.add_argument("--canonical_capture_update_anchor_actual", action="store_true")
parser.add_argument("--canonical_capture_zero_velocity", action="store_true")
parser.add_argument("--canonical_capture_print", action="store_true")
parser.add_argument("--canonical_freeze_base_ref_during_swing_commit", action="store_true")
parser.add_argument("--canonical_freeze_base_ref_keep_current_xy", action="store_true")
parser.add_argument("--canonical_freeze_base_ref_height", type=float, default=-1.0)
parser.add_argument("--canonical_event_require_stable_before_next", action="store_true")
parser.add_argument("--canonical_event_stable_roll", type=float, default=0.08)
parser.add_argument("--canonical_event_stable_pitch", type=float, default=0.08)
parser.add_argument("--canonical_event_stable_min_z", type=float, default=0.62)
parser.add_argument("--canonical_event_stable_steps", type=int, default=80)
# B9-k: stronger support-region target planner, replacing the earlier lite heuristic.
parser.add_argument("--canonical_enable_support_region_target_planner", action="store_true")
parser.add_argument("--canonical_fr_target_min_margin", type=float, default=0.020)
parser.add_argument("--canonical_fr_liftoff_min_geom_margin", type=float, default=0.008)
parser.add_argument("--canonical_fr_liftoff_target_err_tol", type=float, default=0.030)
parser.add_argument("--canonical_fr_liftoff_min_shift_steps", type=int, default=220)
parser.add_argument("--canonical_fr_hind_forward_bias", type=float, default=0.045)
parser.add_argument("--canonical_fr_front_forward_bias", type=float, default=-0.005)
parser.add_argument("--canonical_fr_hind_lateral_bias", type=float, default=0.018)
parser.add_argument("--canonical_fr_front_lateral_bias", type=float, default=0.012)
parser.add_argument("--canonical_fr_print", action="store_true")
# B9-l: use the closest feasible projection instead of the support-triangle incenter.
parser.add_argument("--canonical_fr_use_projected_target", action="store_true")
parser.add_argument("--canonical_fr_projection_grid", type=int, default=23)
parser.add_argument("--canonical_fr_projection_margin_weight", type=float, default=0.30)
parser.add_argument("--canonical_fr_projection_max_step_from_current", type=float, default=0.090)
parser.add_argument("--canonical_fr_projection_inner_push", type=float, default=0.012)
# B10: leg-agnostic feasibility state-machine diagnostics.
parser.add_argument("--canonical_enable_b10_fsm_diagnostics", action="store_true")
parser.add_argument("--b10_relocation_target_err_tol", type=float, default=0.035)
parser.add_argument("--b10_relocation_geom_margin_tol", type=float, default=0.002)
parser.add_argument("--b10_relocation_min_shift_steps", type=int, default=180)
parser.add_argument("--b10_allow_liftoff_on_b10_gate", action="store_true")
parser.add_argument("--b10_print", action="store_true")
# B11: during relocation-before-liftoff, use a stronger WBC base task.
# No front/hind branching: it activates for whichever candidate is selected.
parser.add_argument("--enable_b11_wbc_relocation_task", action="store_true")
parser.add_argument("--b11_reloc_kp_base_xy", type=float, default=45.0)
parser.add_argument("--b11_reloc_kd_base_xy", type=float, default=12.0)
parser.add_argument("--b11_reloc_w_base_acc", type=float, default=45.0)
parser.add_argument("--b11_reloc_w_stance_acc", type=float, default=120.0)
parser.add_argument("--b11_reloc_max_base_acc_lin", type=float, default=1.20)
parser.add_argument("--b11_reloc_max_tau", type=float, default=24.0)
parser.add_argument("--b11_reloc_max_base_ref_xy_err", type=float, default=0.095)
parser.add_argument("--b11_reloc_print", action="store_true")
# B12: align stance-leg posture/torque target with relocation base_ref.
# This is leg-agnostic: for candidate i, all legs except i receive a small stance IK-compatible target.
parser.add_argument("--enable_b12_pd_target_alignment", action="store_true")
parser.add_argument("--b12_align_gain", type=float, default=0.35)
parser.add_argument("--b12_align_damping", type=float, default=0.050)
parser.add_argument("--b12_align_max_joint_delta", type=float, default=0.035)
parser.add_argument("--b12_align_max_foot_xy_cmd", type=float, default=0.020)
parser.add_argument("--b12_align_include_hx", action="store_true")
parser.add_argument("--b12_align_sign", type=float, default=-1.0)
parser.add_argument("--b12_align_ramp_steps", type=int, default=180)
parser.add_argument("--b12_align_min_base_ref_xy_err", type=float, default=0.004)
parser.add_argument("--b12_align_max_base_ref_xy_err", type=float, default=0.095)
parser.add_argument("--b12_align_torque_kp", type=float, default=14.0)
parser.add_argument("--b12_align_torque_kd", type=float, default=2.0)
parser.add_argument("--b12_align_max_tau", type=float, default=2.5)
parser.add_argument("--b12_align_print", action="store_true")
# B13: choose the next swing leg by current support feasibility instead of a fixed queue slot.
# Keeps the first leg deterministic by default, then adapts the next candidate.
parser.add_argument("--enable_b13_adaptive_next_leg", action="store_true")
parser.add_argument("--b13_adaptive_start_index", type=int, default=1,
                    help="Queue index at which adaptive selection starts. Default 1 keeps the first RF step fixed.")
parser.add_argument("--b13_candidate_legs", type=str, default="LF,LH,RH",
                    help="Candidate legs for adaptive selection after the first step.")
parser.add_argument("--b13_exclude_last_completed", action="store_true")
parser.add_argument("--b13_score_margin_weight", type=float, default=1.0)
parser.add_argument("--b13_score_geom_weight", type=float, default=0.35)
parser.add_argument("--b13_score_target_err_weight", type=float, default=0.45)
parser.add_argument("--b13_print", action="store_true")
# B14: do NOT mutate the persistent queue. Select a temporary current candidate only.
parser.add_argument("--enable_b14_no_queue_mutation", action="store_true")
parser.add_argument("--b14_completed_cooldown_steps", type=int, default=700)
parser.add_argument("--b14_require_positive_best_margin", action="store_true",
                    help="If enabled, keep all-stance when every adaptive candidate has negative margin.")
parser.add_argument("--b14_print", action="store_true")
# B15: if all candidate legs remain infeasible, temporarily stop candidate-specific shifting
# and recenter the trunk toward an all-feet support-center target.
parser.add_argument("--enable_b15_recenter_fallback", action="store_true")
parser.add_argument("--b15_recenter_trigger_steps", type=int, default=260)
parser.add_argument("--b15_recenter_duration_steps", type=int, default=420)
parser.add_argument("--b15_recenter_future_margin_threshold", type=float, default=-0.005)
parser.add_argument("--b15_recenter_geom_margin_threshold", type=float, default=-0.005)
parser.add_argument("--b15_recenter_max_step", type=float, default=0.0012)
parser.add_argument("--b15_recenter_target_blend_current", type=float, default=0.25,
                    help="0 uses foot centroid; larger keeps target closer to current base.")
parser.add_argument("--b15_recenter_wbc_kp_xy", type=float, default=38.0)
parser.add_argument("--b15_recenter_wbc_kd_xy", type=float, default=10.0)
parser.add_argument("--b15_recenter_w_base_acc", type=float, default=38.0)
parser.add_argument("--b15_recenter_w_stance_acc", type=float, default=130.0)
parser.add_argument("--b15_recenter_print", action="store_true")
# B16: recenter-specific all-stance PD target alignment.
# Unlike B12, this does not exclude a candidate leg and uses recenter_count based ramp.
parser.add_argument("--enable_b16_recenter_all_stance_pd_alignment", action="store_true")
parser.add_argument("--b16_recenter_align_gain", type=float, default=0.30)
parser.add_argument("--b16_recenter_align_damping", type=float, default=0.060)
parser.add_argument("--b16_recenter_align_max_joint_delta", type=float, default=0.025)
parser.add_argument("--b16_recenter_align_max_foot_xy_cmd", type=float, default=0.014)
parser.add_argument("--b16_recenter_align_include_hx", action="store_true")
parser.add_argument("--b16_recenter_align_sign", type=float, default=-1.0)
parser.add_argument("--b16_recenter_align_ramp_steps", type=int, default=160)
parser.add_argument("--b16_recenter_align_torque_kp", type=float, default=10.0)
parser.add_argument("--b16_recenter_align_torque_kd", type=float, default=1.5)
parser.add_argument("--b16_recenter_align_max_tau", type=float, default=1.8)
parser.add_argument("--b16_recenter_align_print", action="store_true")
# B17: TRACER Decoder/Mapper-level support-geometry-aware foothold selection.
# This does not change MPC/WBC. It only changes the swing foot reference.
parser.add_argument("--enable_b17_support_aware_foothold_mapper", action="store_true")
parser.add_argument("--b17_foothold_forward_candidates", type=str, default="0.000,0.010,0.020")
parser.add_argument("--b17_foothold_lateral_candidates", type=str, default="-0.010,0.000,0.010,0.020",
                    help="Positive means outward from body side, negative means inward.")
parser.add_argument("--b17_foothold_max_extra_xy", type=float, default=0.025)
parser.add_argument("--b17_foothold_score_margin_weight", type=float, default=1.0)
parser.add_argument("--b17_foothold_score_geom_weight", type=float, default=0.35)
parser.add_argument("--b17_foothold_score_target_err_weight", type=float, default=0.45)
parser.add_argument("--b17_foothold_step_penalty", type=float, default=0.20)
parser.add_argument("--b17_foothold_next_candidates", type=str, default="LH,RH,LF,RF",
                    help="Candidate next swing legs used to score the touchdown geometry.")
parser.add_argument("--b17_foothold_apply_only_to", type=str, default="RF,LF",
                    help="Comma-separated swing legs whose touchdown target can be modified.")
parser.add_argument("--b17_foothold_print", action="store_true")
# B48: Clemente-style post-landing transition-feasibility foothold planner.
# It samples touchdown candidates for the *current* swing leg, then evaluates the
# post-landing support/feasible region for the *next* swing leg. Abdalla-lite
# support-region metrics are used as the inner feasibility checker.
parser.add_argument("--enable_b48_post_landing_feasibility_foothold_planner", action="store_true")
parser.add_argument("--b48_apply_only_to", type=str, default="RF",
                    help="Comma-separated current swing legs whose touchdown can be selected by B48. Start with RF for RF->LH diagnosis.")
parser.add_argument("--b48_next_leg_mode", type=str, default="fixed", choices=["fixed", "score_candidates"],
                    help="fixed: evaluate one expected next leg. score_candidates: choose the best next leg among candidates.")
parser.add_argument("--b48_fixed_next_leg", type=str, default="LH", choices=["LF", "RF", "LH", "RH"],
                    help="Expected next swing leg used for post-landing feasibility scoring in fixed mode.")
parser.add_argument("--b48_next_candidates", type=str, default="LH,RH,LF",
                    help="Next-leg candidates used when --b48_next_leg_mode score_candidates.")
parser.add_argument("--b48_forward_candidates", type=str, default="0.000,0.020,0.040,0.060,0.080",
                    help="Extra forward offsets from the nominal touchdown, in meters.")
parser.add_argument("--b48_lateral_candidates", type=str, default="-0.020,-0.010,0.000,0.010,0.020",
                    help="Extra lateral offsets from nominal touchdown. Positive means outward from body side.")
parser.add_argument("--b48_max_extra_xy", type=float, default=0.085)
parser.add_argument("--b48_score_margin_weight", type=float, default=3.0)
parser.add_argument("--b48_score_geom_weight", type=float, default=1.0)
parser.add_argument("--b48_score_target_err_weight", type=float, default=0.80)
parser.add_argument("--b48_score_step_penalty", type=float, default=0.25)
parser.add_argument("--b48_score_progress_weight", type=float, default=0.50)
parser.add_argument("--b48_min_post_future_margin", type=float, default=-999.0,
                    help="Optional hard filter for post-landing next-leg future margin. Default keeps best candidate even if all are negative.")
parser.add_argument("--b48_disable_b19_when_active", action="store_true",
                    help="If B48 selected a foothold, prevent B19 forced bias from overwriting it.")
parser.add_argument("--b48_print", action="store_true")

# B49: Use the elevated swing reference (not the ground-level committed touchdown)
# as the tracking target for swing-execution assists.  B18 committed foothold must
# remain a touchdown target at ground height, but B20/B21/B37 should see the
# time-varying swing target z so the foot actually lifts during swing.
parser.add_argument("--enable_b49_elevated_swing_tracking_reference", action="store_true")
parser.add_argument("--b49_apply_to_b20", action="store_true",
                    help="Use elevated swing z for B20 direct swing IK target.")
parser.add_argument("--b49_apply_to_b21", action="store_true",
                    help="Use elevated swing z for B21 Cartesian swing torque target.")
parser.add_argument("--b49_apply_to_b37", action="store_true",
                    help="Make B37/B45-style target picking prefer B23 target_after over committed touchdown.")
parser.add_argument("--b49_min_sigma", type=float, default=0.03)
parser.add_argument("--b49_max_sigma", type=float, default=0.98)
parser.add_argument("--b49_min_clearance", type=float, default=0.035,
                    help="Minimum desired swing clearance above the liftoff anchor, multiplied by sin(pi*sigma).")
parser.add_argument("--b49_z_boost", type=float, default=0.0,
                    help="Additional z offset added to the elevated swing reference.")
parser.add_argument("--b49_print", action="store_true")
# B18: commit B17-selected touchdown targets and keep them through swing/capture.
parser.add_argument("--enable_b18_committed_foothold_pinning", action="store_true")
parser.add_argument("--b18_commit_min_sigma", type=float, default=0.05)
parser.add_argument("--b18_pin_during_swing", action="store_true")
parser.add_argument("--b18_pin_during_capture", action="store_true")
parser.add_argument("--b18_pin_extra_after_capture_steps", type=int, default=180)
parser.add_argument("--b18_pin_target_scale_xy", type=float, default=1.0)
parser.add_argument("--b18_pin_target_scale_z", type=float, default=0.25)
parser.add_argument("--b18_pin_gain", type=float, default=0.70)
parser.add_argument("--b18_pin_damping", type=float, default=0.045)
parser.add_argument("--b18_pin_max_joint_delta", type=float, default=0.11)
parser.add_argument("--b18_pin_include_hx", action="store_true")
parser.add_argument("--b18_pin_torque_kp", type=float, default=10.0)
parser.add_argument("--b18_pin_torque_kd", type=float, default=1.4)
parser.add_argument("--b18_pin_max_tau", type=float, default=1.8)
parser.add_argument("--b18_post_capture_geometry_print", action="store_true")
parser.add_argument("--b18_print", action="store_true")
parser.add_argument("--b18_print_reuse", action="store_true", help="Print committed foothold reuse at print_every cadence.")
# B19: force a small exploratory foothold bias before B17 scoring is trusted.
parser.add_argument("--enable_b19_forced_exploratory_foothold_bias", action="store_true")
parser.add_argument("--b19_force_bias_legs", type=str, default="RF,LF")
parser.add_argument("--b19_force_forward_bias", type=float, default=0.020)
parser.add_argument("--b19_force_outward_bias", type=float, default=0.010)
parser.add_argument("--b19_force_max_extra_xy", type=float, default=0.030)
parser.add_argument("--b19_force_min_sigma", type=float, default=0.02)
parser.add_argument("--b19_force_override_b17_selection", action="store_true")
parser.add_argument("--b19_force_print", action="store_true")
# B20: directly drive the swing leg to the committed foothold before touchdown.
parser.add_argument("--enable_b20_committed_foothold_direct_swing_ik", action="store_true")
parser.add_argument("--b20_direct_ik_start_sigma", type=float, default=0.08)
parser.add_argument("--b20_direct_ik_end_sigma", type=float, default=0.98)
parser.add_argument("--b20_direct_ik_xy_scale", type=float, default=1.0)
parser.add_argument("--b20_direct_ik_z_scale", type=float, default=1.0)
parser.add_argument("--b20_direct_ik_gain", type=float, default=0.95)
parser.add_argument("--b20_direct_ik_damping", type=float, default=0.030)
parser.add_argument("--b20_direct_ik_max_joint_delta", type=float, default=0.18)
parser.add_argument("--b20_direct_ik_include_hx", action="store_true")
parser.add_argument("--b20_direct_ik_torque_kp", type=float, default=12.0)
parser.add_argument("--b20_direct_ik_torque_kd", type=float, default=1.6)
parser.add_argument("--b20_direct_ik_max_tau", type=float, default=2.2)
parser.add_argument("--b20_direct_ik_print", action="store_true")
# B21: add direct Cartesian torque assist for committed swing foothold tracking.
parser.add_argument("--enable_b21_cartesian_committed_swing_torque", action="store_true")
parser.add_argument("--b21_cart_kp_xy", type=float, default=85.0)
parser.add_argument("--b21_cart_kp_z", type=float, default=35.0)
parser.add_argument("--b21_cart_kd_xy", type=float, default=7.0)
parser.add_argument("--b21_cart_kd_z", type=float, default=4.0)
parser.add_argument("--b21_cart_max_tau", type=float, default=7.0)
parser.add_argument("--b21_cart_force_sign", type=float, default=1.0)
parser.add_argument("--b21_cart_apply_during_swing", action="store_true")
parser.add_argument("--b21_cart_apply_during_capture", action="store_true")
parser.add_argument("--b21_cart_apply_post_capture_extra", action="store_true")
parser.add_argument("--b21_cart_start_sigma", type=float, default=0.03)
parser.add_argument("--b21_cart_end_sigma", type=float, default=1.00)
parser.add_argument("--b21_cart_print", action="store_true")
# B22: actuator/WBC authority calibration.
# This does not try to improve gait directly. It measures and changes who has authority:
# WBC torque, assist torques, or the implicit actuator PD.
parser.add_argument("--enable_b22_wbc_authority_calibration", action="store_true")
parser.add_argument("--b22_authority_mode", type=str, default="normal",
                    choices=["normal", "zero_action", "wbc_only", "assist_only", "wbc_plus_assist_scaled"])
parser.add_argument("--b22_wbc_scale", type=float, default=1.0)
parser.add_argument("--b22_assist_scale", type=float, default=1.0)
parser.add_argument("--b22_ramp_start_step", type=int, default=120)
parser.add_argument("--b22_ramp_steps", type=int, default=600)
parser.add_argument("--b22_use_ramp", action="store_true")
parser.add_argument("--b22_print", action="store_true")
parser.add_argument("--b22_print_selected_leg", type=str, default="RF")
# B23: make the committed foothold the main swing foot reference seen by WBC.
parser.add_argument("--enable_b23_wbc_main_committed_footref", action="store_true")
parser.add_argument("--b23_apply_during_swing", action="store_true")
parser.add_argument("--b23_apply_during_capture", action="store_true")
parser.add_argument("--b23_apply_post_capture_extra", action="store_true")
parser.add_argument("--b23_start_sigma", type=float, default=0.03)
parser.add_argument("--b23_end_sigma", type=float, default=1.00)
parser.add_argument("--b23_xy_mode", type=str, default="direct", choices=["direct", "sigma_blend"])
parser.add_argument("--b23_preserve_swing_z", action="store_true")
parser.add_argument("--b23_commit_anchor_on_capture_start", action="store_true")
parser.add_argument("--b23_print", action="store_true")
# B24: increase WBC swing tracking authority.
parser.add_argument("--enable_b24_wbc_swing_authority_boost", action="store_true")
parser.add_argument("--b24_w_swing_acc", type=float, default=90.0)
parser.add_argument("--b24_w_stance_acc", type=float, default=35.0)
parser.add_argument("--b24_w_base_acc", type=float, default=7.0)
parser.add_argument("--b24_kp_swing_xy", type=float, default=60.0)
parser.add_argument("--b24_kd_swing_xy", type=float, default=8.0)
parser.add_argument("--b24_kp_swing_z", type=float, default=110.0)
parser.add_argument("--b24_kd_swing_z", type=float, default=14.0)
parser.add_argument("--b24_max_swing_acc", type=float, default=25.0)
parser.add_argument("--b24_w_swing_force_zero", type=float, default=130.0)
parser.add_argument("--b24_w_tau_reg", type=float, default=0.035)
parser.add_argument("--b24_w_tau_posture", type=float, default=0.08)
parser.add_argument("--b24_print", action="store_true")
# B25: explicitly feed committed-swing IK target to Isaac implicit PD joint target.
# Previous B20/B18 targets affected WBC q_nom and torque assist, but canonical mode nulled q_target,
# so the simulator's implicit PD target could remain near its default stance.
parser.add_argument("--enable_b25_canonical_swing_pd_bridge", action="store_true")
parser.add_argument("--b25_use_b20_direct_ik_target", action="store_true")
parser.add_argument("--b25_use_b18_pin_target", action="store_true")
parser.add_argument("--b25_zero_target_velocity", action="store_true")
parser.add_argument("--b25_max_joint_delta", type=float, default=0.22)
parser.add_argument("--b25_print", action="store_true")
# B26: safer PD bridge. B25 can ratchet because the IK target is recomputed as current_q + delta every step.
parser.add_argument("--enable_b26_latched_safe_swing_pd_bridge", action="store_true")
parser.add_argument("--b26_use_b20_direct_ik_target", action="store_true")
parser.add_argument("--b26_use_b18_pin_target", action="store_true")
parser.add_argument("--b26_zero_target_velocity", action="store_true")
parser.add_argument("--b26_max_joint_delta", type=float, default=0.035)
parser.add_argument("--b26_max_abs_from_latch", type=float, default=0.070)
parser.add_argument("--b26_retarget_interval", type=int, default=999999)
parser.add_argument("--b26_reset_when_no_target", action="store_true")
parser.add_argument("--b26_print", action="store_true")
# B27: signed version of the latched bridge.
# B26 was safe, but the RF foot moved opposite the committed forward foothold.
# This probes whether the IK/PD bridge direction is inverted in joint space.
parser.add_argument("--enable_b27_signed_latched_swing_pd_bridge", action="store_true")
parser.add_argument("--b27_use_b20_direct_ik_target", action="store_true")
parser.add_argument("--b27_zero_target_velocity", action="store_true")
parser.add_argument("--b27_hx_sign", type=float, default=-1.0)
parser.add_argument("--b27_hy_sign", type=float, default=-1.0)
parser.add_argument("--b27_kn_sign", type=float, default=1.0)
parser.add_argument("--b27_max_joint_delta", type=float, default=0.030)
parser.add_argument("--b27_max_abs_from_latch", type=float, default=0.060)
parser.add_argument("--b27_retarget_interval", type=int, default=999999)
parser.add_argument("--b27_reset_when_no_target", action="store_true")
parser.add_argument("--b27_print", action="store_true")
# B28: hold the best signed latched target through capture/post-capture.
parser.add_argument("--enable_b28_hold_latched_swing_pd_bridge", action="store_true")
parser.add_argument("--b28_use_b20_direct_ik_target", action="store_true")
parser.add_argument("--b28_zero_target_velocity", action="store_true")
parser.add_argument("--b28_hx_sign", type=float, default=-1.0)
parser.add_argument("--b28_hy_sign", type=float, default=-1.0)
parser.add_argument("--b28_kn_sign", type=float, default=1.0)
parser.add_argument("--b28_max_joint_delta", type=float, default=0.030)
parser.add_argument("--b28_max_abs_from_latch", type=float, default=0.060)
parser.add_argument("--b28_hold_without_target_steps", type=int, default=520)
parser.add_argument("--b28_retarget_interval", type=int, default=999999)
parser.add_argument("--b28_reset_when_hold_expires", action="store_true")
parser.add_argument("--b28_print", action="store_true")
# B29: task-space foot-error bridge.
# B28 showed joint-space latch strength trades x against y. B29 directly computes
# q target correction from committed_foot - current_foot using the foot Jacobian.
parser.add_argument("--enable_b29_task_space_foot_error_bridge", action="store_true")
parser.add_argument("--b29_use_committed_target", action="store_true")
parser.add_argument("--b29_apply_during_swing", action="store_true")
parser.add_argument("--b29_apply_during_capture", action="store_true")
parser.add_argument("--b29_apply_post_capture_extra", action="store_true")
parser.add_argument("--b29_kp_xy", type=float, default=0.35)
parser.add_argument("--b29_kp_z", type=float, default=0.04)
parser.add_argument("--b29_damping", type=float, default=0.04)
parser.add_argument("--b29_max_foot_err_xy", type=float, default=0.040)
parser.add_argument("--b29_max_foot_err_z", type=float, default=0.010)
parser.add_argument("--b29_max_joint_delta", type=float, default=0.025)
parser.add_argument("--b29_max_abs_from_latch", type=float, default=0.090)
parser.add_argument("--b29_latch_goal", action="store_true")
parser.add_argument("--b29_hold_without_target_steps", type=int, default=420)
parser.add_argument("--b29_zero_target_velocity", action="store_true")
parser.add_argument("--b29_print", action="store_true")
# B30: same task-space bridge as B29, but uses the already-computed Jfeet_full tensor.
# B29 failed because it tried to fetch Jacobians from root_physx_view, while the script
# already has the correct Jfeet_full used by B18/B20/B21.
parser.add_argument("--enable_b30_jfeet_task_space_bridge", action="store_true")
parser.add_argument("--b30_use_committed_target", action="store_true")
parser.add_argument("--b30_apply_during_swing", action="store_true")
parser.add_argument("--b30_apply_during_capture", action="store_true")
parser.add_argument("--b30_apply_post_capture_extra", action="store_true")
parser.add_argument("--b30_kp_xy", type=float, default=0.35)
parser.add_argument("--b30_kp_z", type=float, default=0.04)
parser.add_argument("--b30_damping", type=float, default=0.04)
parser.add_argument("--b30_max_foot_err_xy", type=float, default=0.040)
parser.add_argument("--b30_max_foot_err_z", type=float, default=0.010)
parser.add_argument("--b30_max_joint_delta", type=float, default=0.025)
parser.add_argument("--b30_max_abs_from_latch", type=float, default=0.090)
parser.add_argument("--b30_latch_goal", action="store_true")
parser.add_argument("--b30_hold_without_target_steps", type=int, default=420)
parser.add_argument("--b30_zero_target_velocity", action="store_true")
parser.add_argument("--b30_print", action="store_true")
# B31: passive Jacobian response probe.
# It records B30's J*dq prediction and compares it against the actual foot displacement
# after a short delay. This diagnoses sign/frame/column-order mismatch without adding
# a separate perturbation controller.
parser.add_argument("--enable_b31_jacobian_response_probe", action="store_true")
parser.add_argument("--b31_delay_steps", type=int, default=6)
parser.add_argument("--b31_min_gap_steps", type=int, default=40)
parser.add_argument("--b31_min_delta_norm", type=float, default=1.0e-5)
parser.add_argument("--b31_print", action="store_true")
# B32: active direct joint-target response probe.
# This temporarily overrides one swing leg's implicit PD joint target by a small
# single-axis delta and measures actual foot displacement after a short delay.
# It diagnoses the empirical actuator-target -> foot-response map, independent of Jfeet_full.
parser.add_argument("--enable_b32_direct_joint_response_probe", action="store_true")
parser.add_argument("--b32_probe_leg", type=str, default="RF", choices=["LF", "RF", "LH", "RH"])
parser.add_argument("--b32_start_step", type=int, default=500)
parser.add_argument("--b32_axes", type=str, default="hx,hy,kn")
parser.add_argument("--b32_signs", type=str, default="plus,minus")
parser.add_argument("--b32_delta", type=float, default=0.025)
parser.add_argument("--b32_hold_steps", type=int, default=8)
parser.add_argument("--b32_gap_steps", type=int, default=35)
parser.add_argument("--b32_max_probes", type=int, default=6)
parser.add_argument("--b32_zero_target_velocity", action="store_true")
parser.add_argument("--b32_require_swing_or_capture", action="store_true")
parser.add_argument("--b32_disable_b30_during_probe", action="store_true")
parser.add_argument("--b32_print", action="store_true")

# B42: LH task-space authority / empirical Jacobian probe.
# This is a convenience wrapper around B32 direct joint response probe, with
# naming and defaults matched to the current LH swing-tracking diagnosis.
parser.add_argument("--enable_b42_lh_taskspace_authority_probe", action="store_true")
parser.add_argument("--b42_probe_leg", type=str, default="LH", choices=["LF", "RF", "LH", "RH"])
parser.add_argument("--b42_start_step", type=int, default=1080)
parser.add_argument("--b42_axes", type=str, default="hx,hy,kn")
parser.add_argument("--b42_signs", type=str, default="plus,minus")
parser.add_argument("--b42_delta", type=float, default=0.025)
parser.add_argument("--b42_hold_steps", type=int, default=8)
parser.add_argument("--b42_gap_steps", type=int, default=25)
parser.add_argument("--b42_max_probes", type=int, default=6)
parser.add_argument("--b42_zero_target_velocity", action="store_true")
parser.add_argument("--b42_require_swing_or_capture", action="store_true")
parser.add_argument("--b42_disable_b30_during_probe", action="store_true")
parser.add_argument("--b42_print", action="store_true")

# B43: empirical lateral-y bridge for hind swing.
# Uses B42-measured authority: foot-y is strongly affected by HY and moderately by HX.
# Error convention: e_y = target_y - foot_y. For LH/RH defaults:
#   dq_hx = sign_hx * k_hx_y * e_y, dq_hy = sign_hy * k_hy_y * e_y
# Default signs make e_y < 0 produce hx+ and hy-, which should reduce LH foot_y.
parser.add_argument("--enable_b43_hind_empirical_lateral_y_bridge", action="store_true")
parser.add_argument("--b43_legs", type=str, default="LH,RH")
parser.add_argument("--b43_use_committed_target", action="store_true")
parser.add_argument("--b43_apply_during_swing", action="store_true")
parser.add_argument("--b43_apply_during_capture", action="store_true")
parser.add_argument("--b43_apply_post_capture_extra", action="store_true")
parser.add_argument("--b43_k_hx_per_y", type=float, default=0.10)
parser.add_argument("--b43_k_hy_per_y", type=float, default=0.12)
parser.add_argument("--b43_deadband_y", type=float, default=0.004)
parser.add_argument("--b43_max_y_err", type=float, default=0.060)
parser.add_argument("--b43_max_hx_delta", type=float, default=0.012)
parser.add_argument("--b43_max_hy_delta", type=float, default=0.010)
parser.add_argument("--b43_min_sigma", type=float, default=0.03)
parser.add_argument("--b43_max_sigma", type=float, default=1.00)
parser.add_argument("--b43_hx_signs", type=str, default="LH:-1,RH:-1")
parser.add_argument("--b43_hy_signs", type=str, default="LH:1,RH:1")
parser.add_argument("--b43_zero_target_velocity", action="store_true")
parser.add_argument("--b43_print", action="store_true")

# B44: latched hind lateral-y bridge + authority probe.
# Difference from B43:
#   B43 wrote q_cmd = q_now + small dq each step.
#   B44 latches q at the beginning of hind swing/capture/post phase and keeps
#   a bounded q_goal = q_latch + delta(e_y), then moves toward it with a
#   per-step bound. This tests whether the LH/RH y correction needs persistent
#   joint-space authority rather than a tiny instantaneous offset.
parser.add_argument("--enable_b44_latched_hind_y_bridge", action="store_true")
parser.add_argument("--b44_legs", type=str, default="LH,RH")
parser.add_argument("--b44_use_committed_target", action="store_true")
parser.add_argument("--b44_apply_during_swing", action="store_true")
parser.add_argument("--b44_apply_during_capture", action="store_true")
parser.add_argument("--b44_apply_post_capture_extra", action="store_true")
parser.add_argument("--b44_k_hx_per_y", type=float, default=0.35)
parser.add_argument("--b44_k_hy_per_y", type=float, default=0.22)
parser.add_argument("--b44_deadband_y", type=float, default=0.004)
parser.add_argument("--b44_max_y_err", type=float, default=0.100)
parser.add_argument("--b44_max_abs_hx_from_latch", type=float, default=0.050)
parser.add_argument("--b44_max_abs_hy_from_latch", type=float, default=0.035)
parser.add_argument("--b44_max_step_hx_delta", type=float, default=0.018)
parser.add_argument("--b44_max_step_hy_delta", type=float, default=0.012)
parser.add_argument("--b44_min_sigma", type=float, default=0.03)
parser.add_argument("--b44_max_sigma", type=float, default=1.00)
parser.add_argument("--b44_hx_signs", type=str, default="LH:-1,RH:-1")
parser.add_argument("--b44_hy_signs", type=str, default="LH:1,RH:1")
parser.add_argument("--b44_zero_target_velocity", action="store_true")
parser.add_argument("--b44_reset_latch_on_phase_change", action="store_true")
parser.add_argument("--b44_print", action="store_true")


# B45: directional current-state hind lateral-y push bridge.
# Motivation from B44:
#   B44's q_latch -> q_goal restore logic can reverse the actual per-step
#   direction once q_now has already drifted past q_goal. B45 therefore does
#   NOT chase q_goal. It applies a direction-only per-step push computed from
#   e_y around q_now, with a phase-latch guard that prevents unlimited runaway
#   but never reverses the commanded correction just to return to latch.
parser.add_argument("--enable_b45_directional_hind_y_push_bridge", action="store_true")
parser.add_argument("--b45_legs", type=str, default="LH,RH")
parser.add_argument("--b45_use_committed_target", action="store_true")
parser.add_argument("--b45_apply_during_swing", action="store_true")
parser.add_argument("--b45_apply_during_capture", action="store_true")
parser.add_argument("--b45_apply_post_capture_extra", action="store_true")
parser.add_argument("--b45_k_hx_per_y", type=float, default=0.30)
parser.add_argument("--b45_k_hy_per_y", type=float, default=0.08)
parser.add_argument("--b45_enable_hy", action="store_true",
                    help="Enable small HY directional assist. Default is HX-only because B44 showed HY drift/coupling can dominate.")
parser.add_argument("--b45_deadband_y", type=float, default=0.004)
parser.add_argument("--b45_max_y_err", type=float, default=0.100)
parser.add_argument("--b45_max_step_hx_delta", type=float, default=0.020)
parser.add_argument("--b45_max_step_hy_delta", type=float, default=0.006)
parser.add_argument("--b45_max_abs_hx_from_latch", type=float, default=0.120)
parser.add_argument("--b45_max_abs_hy_from_latch", type=float, default=0.080)
parser.add_argument("--b45_min_sigma", type=float, default=0.03)
parser.add_argument("--b45_max_sigma", type=float, default=1.00)
parser.add_argument("--b45_hx_signs", type=str, default="LH:-1,RH:-1")
parser.add_argument("--b45_hy_signs", type=str, default="LH:1,RH:1")
parser.add_argument("--b45_zero_target_velocity", action="store_true")
parser.add_argument("--b45_reset_latch_on_phase_change", action="store_true")
parser.add_argument("--b45_print", action="store_true")
# B33: baseline-cancelled response probe.
# For each axis/sign, first measure natural foot drift with no extra perturbation,
# then immediately measure perturbation response, and report:
# corrected_delta = perturb_delta - baseline_delta.
parser.add_argument("--enable_b33_baseline_cancelled_response_probe", action="store_true")
parser.add_argument("--b33_probe_leg", type=str, default="RF", choices=["LF", "RF", "LH", "RH"])
parser.add_argument("--b33_start_step", type=int, default=520)
parser.add_argument("--b33_axes", type=str, default="hx,hy,kn")
parser.add_argument("--b33_signs", type=str, default="plus,minus")
parser.add_argument("--b33_delta", type=float, default=0.025)
parser.add_argument("--b33_baseline_steps", type=int, default=8)
parser.add_argument("--b33_perturb_steps", type=int, default=8)
parser.add_argument("--b33_gap_steps", type=int, default=25)
parser.add_argument("--b33_max_pairs", type=int, default=6)
parser.add_argument("--b33_zero_target_velocity", action="store_true")
parser.add_argument("--b33_require_swing_or_capture", action="store_true")
parser.add_argument("--b33_print", action="store_true")
# B34: empirical RF x-directional correction bridge.
# B33 showed RF hy is the dominant empirical x actuator:
#   hy+ -> foot x negative, hy- -> foot x positive.
# So if committed_x - foot_x is positive, drive hy negative.
parser.add_argument("--enable_b34_empirical_rf_x_bridge", action="store_true")
parser.add_argument("--b34_leg", type=str, default="RF", choices=["LF", "RF", "LH", "RH"])
parser.add_argument("--b34_use_committed_target", action="store_true")
parser.add_argument("--b34_apply_during_swing", action="store_true")
parser.add_argument("--b34_apply_during_capture", action="store_true")
parser.add_argument("--b34_apply_post_capture_extra", action="store_true")
parser.add_argument("--b34_k_hy_per_x", type=float, default=4.0)
parser.add_argument("--b34_deadband_x", type=float, default=0.004)
parser.add_argument("--b34_max_x_err", type=float, default=0.045)
parser.add_argument("--b34_max_joint_delta", type=float, default=0.025)
parser.add_argument("--b34_max_abs_from_latch", type=float, default=0.080)
parser.add_argument("--b34_hold_without_target_steps", type=int, default=420)
parser.add_argument("--b34_zero_target_velocity", action="store_true")
parser.add_argument("--b34_include_small_hx", action="store_true")
parser.add_argument("--b34_k_hx_per_y", type=float, default=0.5)
parser.add_argument("--b34_max_hx_delta", type=float, default=0.010)
parser.add_argument("--b34_print", action="store_true")
# B35: pure additive empirical HY assist.
# Same empirical rule as B34, but without q_latch/q_goal hold contamination.
# It only writes HY for the selected leg:
#   q_cmd = q_now; q_cmd[hy] = q_now[hy] + dq_hy
parser.add_argument("--enable_b35_pure_empirical_hy_assist", action="store_true")
parser.add_argument("--b35_leg", type=str, default="RF", choices=["LF", "RF", "LH", "RH"])
parser.add_argument("--b35_use_committed_target", action="store_true")
parser.add_argument("--b35_apply_during_swing", action="store_true")
parser.add_argument("--b35_apply_during_capture", action="store_true")
parser.add_argument("--b35_apply_post_capture_extra", action="store_true")
parser.add_argument("--b35_k_hy_per_x", type=float, default=4.0)
parser.add_argument("--b35_deadband_x", type=float, default=0.004)
parser.add_argument("--b35_max_x_err", type=float, default=0.045)
parser.add_argument("--b35_max_hy_delta", type=float, default=0.025)
parser.add_argument("--b35_min_sigma", type=float, default=0.03)
parser.add_argument("--b35_max_sigma", type=float, default=1.00)
parser.add_argument("--b35_zero_target_velocity", action="store_true")
parser.add_argument("--b35_no_hold_without_target", action="store_true")
parser.add_argument("--b35_print", action="store_true")
# B36: pure additive empirical HY+KN assist.
# Clean additive version: no latch, no hold. Adds explicit bounded KN term.
parser.add_argument("--enable_b36_pure_empirical_hy_kn_assist", action="store_true")
parser.add_argument("--b36_leg", type=str, default="RF", choices=["LF", "RF", "LH", "RH"])
parser.add_argument("--b36_use_committed_target", action="store_true")
parser.add_argument("--b36_apply_during_swing", action="store_true")
parser.add_argument("--b36_apply_during_capture", action="store_true")
parser.add_argument("--b36_apply_post_capture_extra", action="store_true")
parser.add_argument("--b36_k_hy_per_x", type=float, default=4.0)
parser.add_argument("--b36_k_kn_per_x", type=float, default=0.45)
parser.add_argument("--b36_kn_sign", type=float, default=1.0)
parser.add_argument("--b36_deadband_x", type=float, default=0.004)
parser.add_argument("--b36_max_x_err", type=float, default=0.045)
parser.add_argument("--b36_max_hy_delta", type=float, default=0.025)
parser.add_argument("--b36_max_kn_delta", type=float, default=0.012)
parser.add_argument("--b36_min_sigma", type=float, default=0.03)
parser.add_argument("--b36_max_sigma", type=float, default=1.00)
parser.add_argument("--b36_zero_target_velocity", action="store_true")
parser.add_argument("--b36_print", action="store_true")
# B37: all-leg empirical HY+KN assist.
# Purpose: verify that the same low-level bridge mechanism can move all legs, not only RF.
# It applies the clean additive HY+KN structure to any committed/active leg in the allow-list.
parser.add_argument("--enable_b37_all_leg_empirical_hy_kn_assist", action="store_true")
parser.add_argument("--b37_legs", type=str, default="RF,LH,LF,RH")
parser.add_argument("--b37_use_committed_target", action="store_true")
parser.add_argument("--b37_apply_during_swing", action="store_true")
parser.add_argument("--b37_apply_during_capture", action="store_true")
parser.add_argument("--b37_apply_post_capture_extra", action="store_true")
parser.add_argument("--b37_k_hy_per_x", type=float, default=4.0)
parser.add_argument("--b37_k_kn_per_x", type=float, default=0.8)
parser.add_argument("--b37_deadband_x", type=float, default=0.004)
parser.add_argument("--b37_max_x_err", type=float, default=0.045)
parser.add_argument("--b37_max_hy_delta", type=float, default=0.025)
parser.add_argument("--b37_max_kn_delta", type=float, default=0.018)
parser.add_argument("--b37_min_sigma", type=float, default=0.03)
parser.add_argument("--b37_max_sigma", type=float, default=1.00)
parser.add_argument("--b37_hy_signs", type=str, default="RF:-1,LH:-1,LF:-1,RH:-1")
parser.add_argument("--b37_kn_signs", type=str, default="RF:1,LH:1,LF:1,RH:1")
parser.add_argument("--b37_zero_target_velocity", action="store_true")
parser.add_argument("--b37_print", action="store_true")
# B38: candidate-specific support-region recenter.
# Purpose: when the next swing candidate (especially LH/RH) is blocked by support
# geometry, move the base reference directly toward the feasible support-triangle
# target for that candidate instead of waiting at the full-support centroid.
parser.add_argument("--enable_b38_candidate_support_recenter", action="store_true")
parser.add_argument("--b38_candidate_legs", type=str, default="LH,RH")
parser.add_argument("--b38_force_shift_even_if_future_bad", action="store_true")
parser.add_argument("--b38_use_actual_base_start", action="store_true")
parser.add_argument("--b38_max_step", type=float, default=0.0012)
parser.add_argument("--b38_target_err_tol", type=float, default=0.045)
parser.add_argument("--b38_geom_margin_tol", type=float, default=-0.020)
parser.add_argument("--b38_min_shift_steps", type=int, default=260)
parser.add_argument("--b38_allow_liftoff_on_b38_gate", action="store_true")
parser.add_argument("--b38_print", action="store_true")
# B39: latched candidate-specific support-region recenter.
# Difference from B38: the commanded base_ref_xy is accumulated/latch-stepped toward
# the candidate support target instead of restarting from the actual base every step.
parser.add_argument("--enable_b39_latched_support_recenter", action="store_true")
parser.add_argument("--b39_candidate_legs", type=str, default="LH,RH")
parser.add_argument("--b39_force_shift_even_if_future_bad", action="store_true")
parser.add_argument("--b39_max_step", type=float, default=0.0018)
parser.add_argument("--b39_target_err_tol", type=float, default=0.045)
parser.add_argument("--b39_geom_margin_tol", type=float, default=-0.020)
parser.add_argument("--b39_min_shift_steps", type=int, default=320)
parser.add_argument("--b39_allow_liftoff_on_b39_gate", action="store_true")
parser.add_argument("--b39_print", action="store_true")

# B40: clamp committed foothold displacement for hind legs before B23/B37 tracking.
# This prevents support-aware foothold generation from asking LH/RH to move laterally by 20+ cm.
parser.add_argument("--enable_b40_hind_target_clamp", action="store_true")
parser.add_argument("--b40_clamp_legs", type=str, default="LH,RH")
parser.add_argument("--b40_max_forward_from_anchor", type=float, default=0.055)
parser.add_argument("--b40_max_lateral_from_anchor", type=float, default=0.045)
parser.add_argument("--b40_max_xy_from_anchor", type=float, default=0.070)
parser.add_argument("--b40_clamp_z", action="store_true")
parser.add_argument("--b40_max_z_from_anchor", type=float, default=0.020)
parser.add_argument("--b40_print", action="store_true")

# B41: apply hind target clamp as early as the committed foothold is created,
# so B23/B20/B21/B37 all see the same bounded target during active swing.
parser.add_argument("--enable_b41_swing_phase_hind_target_clamp", action="store_true")
parser.add_argument("--b41_clamp_legs", type=str, default="LH,RH")
parser.add_argument("--b41_max_forward_from_anchor", type=float, default=0.055)
parser.add_argument("--b41_max_lateral_from_anchor", type=float, default=0.045)
parser.add_argument("--b41_max_xy_from_anchor", type=float, default=0.070)
parser.add_argument("--b41_clamp_z", action="store_true")
parser.add_argument("--b41_max_z_from_anchor", type=float, default=0.020)
parser.add_argument("--b41_print", action="store_true")
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
parser.add_argument("--ik_use_canonical_active_leg", action="store_true",
                    help="Use the currently executed/latched canonical swing leg instead of --test_leg for IK swing target.")

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
# B8-bo: once a real swing starts, keep it latched until touchdown commit.
# Margin gates are pre-detach only; they must not cancel an already airborne foot.
parser.add_argument("--enable_latched_swing_until_touchdown", action="store_true")
parser.add_argument("--latched_swing_min_phase", type=str, default="lift", choices=["lift", "hold_lift", "lower"])
parser.add_argument("--latched_swing_keep_base_freeze", action="store_true")
parser.add_argument("--latched_swing_force_contact_open", action="store_true")
parser.add_argument("--latched_swing_force_ik", action="store_true")
parser.add_argument("--latched_swing_release_on_late_touchdown", action="store_true")
parser.add_argument("--post_touchdown_lock_steps", type=int, default=80,
                    help="Keep foothold lock/trunk freeze after touchdown to prevent old-stance PD pullback.")
# B8-bp: after touchdown is accepted, do not allow the same leg/step to relatch.
parser.add_argument("--enable_touchdown_committed_state", action="store_true")
parser.add_argument("--touchdown_committed_blocks_same_step_relatch", action="store_true")
parser.add_argument("--touchdown_committed_force_all_stance", action="store_true")
parser.add_argument("--touchdown_committed_keep_base_freeze", action="store_true")
parser.add_argument("--touchdown_committed_until_next_step", action="store_true",
                    help="Keep touchdown committed state until the nominal step key changes.")
# B8-bq: committed foothold should stay pinned, not just all-stance.
parser.add_argument("--enable_committed_foothold_pinning", action="store_true")
parser.add_argument("--committed_pin_until_next_step", action="store_true")
parser.add_argument("--committed_pin_ignore_late_touchdown_logic", action="store_true")
parser.add_argument("--committed_pin_gain", type=float, default=0.85)
parser.add_argument("--committed_pin_damping", type=float, default=0.035)
parser.add_argument("--committed_pin_max_joint_delta", type=float, default=0.16)
parser.add_argument("--committed_pin_target_scale_xy", type=float, default=1.0)
parser.add_argument("--committed_pin_target_scale_z", type=float, default=0.20)
parser.add_argument("--committed_pin_zero_velocity", action="store_true")
parser.add_argument("--committed_pin_extra_steps", type=int, default=160,
                    help="Fallback pin duration if not using committed_pin_until_next_step.")
parser.add_argument("--touchdown_xy_tol_x", type=float, default=-1.0,
                    help="Optional axis-wise touchdown x tolerance. Negative disables.")
parser.add_argument("--touchdown_xy_tol_y", type=float, default=-1.0,
                    help="Optional axis-wise touchdown y tolerance. Negative disables.")

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
parser.add_argument("--kp_swing_xy", type=float, default=0.0)
parser.add_argument("--kd_swing_xy", type=float, default=0.0)
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

# B42 is implemented as a named diagnostic layer on top of the existing B32
# direct joint response probe. Keeping B32 as the execution backend avoids
# duplicating low-level joint target code, while B42 gives the run/log a clear
# purpose: identify which LH joint axis actually moves foot y during swing.
if getattr(args, "enable_b42_lh_taskspace_authority_probe", False):
    args.enable_b32_direct_joint_response_probe = True
    args.b32_probe_leg = str(args.b42_probe_leg).upper()
    args.b32_start_step = int(args.b42_start_step)
    args.b32_axes = str(args.b42_axes)
    args.b32_signs = str(args.b42_signs)
    args.b32_delta = float(args.b42_delta)
    args.b32_hold_steps = int(args.b42_hold_steps)
    args.b32_gap_steps = int(args.b42_gap_steps)
    args.b32_max_probes = int(args.b42_max_probes)
    args.b32_zero_target_velocity = bool(args.b42_zero_target_velocity)
    args.b32_require_swing_or_capture = bool(args.b42_require_swing_or_capture)
    args.b32_disable_b30_during_probe = bool(args.b42_disable_b30_during_probe)
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


def quat_to_yaw(q):
    """Return yaw from quaternion.

    Isaac Lab root_quat_w is usually wxyz. If the tensor appears xyzw-like,
    this still keeps the diagnostic robust enough for flat-yaw walking tests.
    """
    if q.shape[-1] != 4:
        return torch.zeros(q.shape[:-1], device=q.device, dtype=q.dtype)
    # Assume w, x, y, z.
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def parse_phase_offsets(s):
    out = {"LF": 0.50, "RF": 0.00, "LH": 0.75, "RH": 0.25}
    if s is None or str(s).strip() == "":
        return out
    for part in str(s).split(","):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip().upper()
        if k in out:
            out[k] = float(v) % 1.0
    return out


def parse_canonical_event_queue():
    out = []
    for x in str(args.canonical_event_queue).split(","):
        x = x.strip().upper()
        if x in LEG_TO_ID and LEG_TO_ID[x] not in out:
            out.append(LEG_TO_ID[x])
    if not out:
        out = [LEG_TO_ID[x] for x in ["RF", "LF", "RH", "LH"]]
    return out


def event_queue_requested_mask(queue_ids, queue_index, num_envs, device, dtype):
    req = torch.ones((num_envs, 4), device=device, dtype=dtype)
    sigma = torch.zeros((num_envs, 4), device=device, dtype=dtype)
    if queue_ids:
        leg = int(queue_ids[int(queue_index) % len(queue_ids)])
        req[:, leg] = 0.0
    return req, sigma



def compute_canonical_gait_schedule(step, num_envs, device, dtype):
    """Canonical schedule:
       phi_i,k|t = (phi_i + k*dt/T) mod 1
       s_i,k|t = 1 if phi_i,k|t < xi_i else 0.

    B9-b: before canonical_gait_start_step, force all-stance.
    """
    offsets = parse_phase_offsets(args.gait_phase_offsets)
    T = max(1, int(args.gait_period_steps))
    xi = max(0.05, min(0.95, float(args.gait_duty)))

    if args.canonical_gait_force_all_stance_before_start and step < int(args.canonical_gait_start_step):
        phi_t = torch.zeros((num_envs, 4), device=device, dtype=dtype)
        s_t = torch.ones((num_envs, 4), device=device, dtype=dtype)
        sigma_t = torch.zeros((num_envs, 4), device=device, dtype=dtype)
        return phi_t, s_t, sigma_t

    step_for_phase = step
    if args.canonical_gait_use_relative_time:
        step_for_phase = max(0, step - int(args.canonical_gait_start_step))
    phase_scalar = (float(step_for_phase) / float(T)) % 1.0
    phi_vals, s_vals, sigma_vals = [], [], []
    for leg_name in ["LF", "RF", "LH", "RH"]:
        phi = (offsets[leg_name] + phase_scalar) % 1.0
        stance = 1.0 if phi < xi else 0.0
        sigma = 0.0
        if stance < 0.5:
            sigma = (phi - xi) / max(1.0e-6, 1.0 - xi)
            sigma = max(0.0, min(1.0, sigma))
        phi_vals.append(phi)
        s_vals.append(stance)
        sigma_vals.append(sigma)
    phi_t = torch.tensor(phi_vals, device=device, dtype=dtype).unsqueeze(0).repeat(num_envs, 1)
    s_t = torch.tensor(s_vals, device=device, dtype=dtype).unsqueeze(0).repeat(num_envs, 1)
    sigma_t = torch.tensor(sigma_vals, device=device, dtype=dtype).unsqueeze(0).repeat(num_envs, 1)
    return phi_t, s_t, sigma_t


def canonical_swing_ref_from_schedule(foot_anchor_w, base_quat_w, s_t, sigma_t):
    """Foot reference from S_t. Stance legs hold anchor; swing legs move anchor -> foothold."""
    target = foot_anchor_w.clone()
    for li, leg_name in enumerate(["LF", "RF", "LH", "RH"]):
        if float((1.0 - s_t[0, li]).detach().cpu()) < 0.5:
            continue
        sig = sigma_t[:, li].clamp(0.0, 1.0).view(-1, 1)

        fwd = torch.zeros_like(target[:, li, :])
        if args.canonical_gait_use_base_yaw:
            yaw = quat_to_yaw(base_quat_w)
            fwd[:, 0] = torch.cos(yaw)
            fwd[:, 1] = torch.sin(yaw)
        else:
            fwd[:, 0] = 1.0

        lat = torch.zeros_like(fwd)
        lat[:, 0] = -fwd[:, 1]
        lat[:, 1] = fwd[:, 0]

        final = foot_anchor_w[:, li, :] + float(args.canonical_gait_forward_step) * fwd + float(args.canonical_gait_lateral_step) * lat
        target[:, li, 0:2] = (1.0 - sig) * foot_anchor_w[:, li, 0:2] + sig * final[:, 0:2]
        target[:, li, 2] = foot_anchor_w[:, li, 2] + float(args.canonical_gait_swing_height) * torch.sin(torch.pi * sig[:, 0])
    return target


def _canonical_priority_list():
    out = []
    for x in str(args.canonical_crawl_priority).split(","):
        x = x.strip().upper()
        if x in LEG_TO_ID and x not in out:
            out.append(x)
    # Ensure all legs exist.
    for x in ["LH", "RF", "RH", "LF"]:
        if x not in out:
            out.append(x)
    return out


def _choose_single_swing_leg(requested_s_t, sigma_t, current_leg=None):
    """Resolve requested S_t to a single swing leg id."""
    req_ids = [int(x.detach().cpu()) for x in torch.nonzero(requested_s_t[0] < 0.5, as_tuple=False).flatten()]
    if len(req_ids) == 0:
        return None
    if current_leg is not None and current_leg in req_ids:
        return int(current_leg)
    # Prefer explicit first leg if it is currently requested.
    first_id = LEG_TO_ID.get(str(args.canonical_first_swing_leg).upper(), None)
    if first_id is not None and first_id in req_ids:
        return int(first_id)
    priority = _canonical_priority_list()
    for name in priority:
        li = LEG_TO_ID[name]
        if li in req_ids:
            return int(li)
    # Fallback: largest swing progress.
    best = max(req_ids, key=lambda li: float(sigma_t[0, li].detach().cpu()))
    return int(best)


def _canonical_future_margin(x_hat, foot_pos, stance_mask, cfg):
    out = compute_support_region_ref(
        foot_pos_w=foot_pos,
        base_pos_w=x_hat[:, 0:3],
        base_rpy_w=x_hat[:, 3:6],
        stance_mask=stance_mask,
        prev_base_ref=None,
        cfg=cfg,
    )
    return float(out.margin_to_edge[0].detach().cpu()), bool(out.swing_allowed[0].detach().cpu()), out


def _b9k_order_triangle_points(pts):
    center = pts.mean(dim=0)
    ang = torch.atan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = torch.argsort(ang)
    return pts[order]


def _b9k_triangle_signed_margins(point_xy, tri_xy):
    """Return signed distances to the oriented triangle edges.

    Positive means inside all edges. Works for a single point and 3 vertices.
    """
    tri = _b9k_order_triangle_points(tri_xy)
    p = point_xy
    # triangle orientation
    e01 = tri[1] - tri[0]
    e02 = tri[2] - tri[0]
    orient = e01[0] * e02[1] - e01[1] * e02[0]
    sign = 1.0 if float(orient.detach().cpu()) >= 0.0 else -1.0
    ds = []
    for i in range(3):
        a = tri[i]
        b = tri[(i + 1) % 3]
        e = b - a
        v = p - a
        cross = e[0] * v[1] - e[1] * v[0]
        d = sign * cross / torch.linalg.norm(e).clamp_min(1.0e-9)
        ds.append(d)
    return torch.stack(ds), tri


def _b9k_triangle_incenter(tri_xy):
    tri = _b9k_order_triangle_points(tri_xy)
    A, B, C = tri[0], tri[1], tri[2]
    a = torch.linalg.norm(B - C)  # opposite A
    b = torch.linalg.norm(A - C)  # opposite B
    c = torch.linalg.norm(A - B)  # opposite C
    denom = (a + b + c).clamp_min(1.0e-9)
    return (a * A + b * B + c * C) / denom, tri


def _b9l_closest_projected_feasible_point(current_xy, safe_xy, tri_xy):
    """Approximate the closest feasible point in a shrunken support triangle.

    This replaces the aggressive B9-k incenter target. We sample the triangle
    and choose a point that is close to the current base projection, while having
    sufficient positive signed edge margin. This is closer to a feasibility
    projection than "go all the way to the incenter".
    """
    tri = _b9k_order_triangle_points(tri_xy)
    min_margin = float(args.canonical_fr_target_min_margin)
    grid_n = max(7, int(args.canonical_fr_projection_grid))
    best = None
    best_score = None
    best_margin = None
    best_raw_dist = None

    # Barycentric grid over the triangle.
    for i in range(grid_n + 1):
        a = i / float(grid_n)
        for j in range(grid_n + 1 - i):
            b = j / float(grid_n)
            c = 1.0 - a - b
            p = a * tri[0] + b * tri[1] + c * tri[2]
            margins, _ = _b9k_triangle_signed_margins(p, tri)
            m = torch.min(margins)
            raw_dist = torch.linalg.norm(p - current_xy)
            # Prefer points satisfying target margin, but keep a fallback.
            violation = torch.clamp(torch.as_tensor(min_margin, device=p.device, dtype=p.dtype) - m, min=0.0)
            score = raw_dist + float(args.canonical_fr_projection_margin_weight) * violation
            if best is None or float(score.detach().cpu()) < float(best_score.detach().cpu()):
                best = p
                best_score = score
                best_margin = m
                best_raw_dist = raw_dist

    if best is None:
        best = safe_xy
        best_margin = torch.min(_b9k_triangle_signed_margins(best, tri)[0])
        best_raw_dist = torch.linalg.norm(best - current_xy)

    # If the best sampled point is still too far, limit the step from current.
    max_step = float(args.canonical_fr_projection_max_step_from_current)
    delta = best - current_xy
    d = torch.linalg.norm(delta)
    if max_step > 0.0 and float(d.detach().cpu()) > max_step:
        best = current_xy + delta / d.clamp_min(1.0e-9) * max_step

    # Small push toward incenter to increase numerical margin, but keep it local.
    inner_push = float(args.canonical_fr_projection_inner_push)
    if inner_push > 0.0:
        to_safe = safe_xy - best
        dn = torch.linalg.norm(to_safe)
        if float(dn.detach().cpu()) > 1.0e-9:
            best = best + to_safe / dn.clamp_min(1.0e-9) * inner_push

    best_margin = torch.min(_b9k_triangle_signed_margins(best, tri)[0])
    best_raw_dist = torch.linalg.norm(best - current_xy)
    return best, best_margin, best_raw_dist


def _b9k_candidate_support_target(foot_pos, x_hat, candidate):
    """Compute a stronger per-leg support-region target.

    The previous lite planner mostly chose a safe point plus a generic lateral bias.
    B9-k explicitly builds the future 3-leg support triangle, chooses its incenter
    as a max-min-margin point, then applies leg-specific forward/lateral bias while
    keeping the final target inside a minimum-margin shrunken triangle.
    """
    B = foot_pos.shape[0]
    out = x_hat[:, 0:2].detach().clone()
    current_margin = torch.zeros((B,), device=foot_pos.device, dtype=foot_pos.dtype)
    target_margin = torch.zeros((B,), device=foot_pos.device, dtype=foot_pos.dtype)
    target_err = torch.zeros((B,), device=foot_pos.device, dtype=foot_pos.dtype)
    raw_safe = out.detach().clone()
    bias_xy = torch.zeros_like(out)
    target_raw = out.detach().clone()
    vertices = torch.zeros((B, 3, 2), device=foot_pos.device, dtype=foot_pos.dtype)

    for b in range(B):
        stance_ids = [i for i in range(4) if i != int(candidate)]
        pts = foot_pos[b, stance_ids, 0:2].detach()
        safe, tri = _b9k_triangle_incenter(pts)
        vertices[b] = tri
        raw_safe[b] = safe

        fwd, left = _base_yaw_unit_vectors(x_hat[b:b+1])
        fwd = fwd[0]
        left = left[0]

        # Left swing -> shift right. Right swing -> shift left.
        if int(candidate) in [0, 2]:
            lat_dir = -left
        else:
            lat_dir = left

        if int(candidate) in [2, 3]:  # hind swing needs forward trunk relocation
            bias = float(args.canonical_fr_hind_forward_bias) * fwd + float(args.canonical_fr_hind_lateral_bias) * lat_dir
        else:
            bias = float(args.canonical_fr_front_forward_bias) * fwd + float(args.canonical_fr_front_lateral_bias) * lat_dir

        if args.canonical_fr_use_projected_target:
            # B9-l: closest feasible projection target.
            # This avoids commanding a ~0.38 m jump to the incenter when a local
            # feasible correction is enough.
            target, proj_margin, proj_dist = _b9l_closest_projected_feasible_point(x_hat[b, 0:2], safe, tri)
            target_raw[b] = target
        else:
            # B9-k: biased incenter, then retract toward incenter until safely inside.
            target = safe + bias
            target_raw[b] = target
            min_margin = float(args.canonical_fr_target_min_margin)
            for _ in range(12):
                m, _ = _b9k_triangle_signed_margins(target, tri)
                if float(torch.min(m).detach().cpu()) >= min_margin:
                    break
                target = 0.65 * target + 0.35 * safe

        out[b] = target
        bias_xy[b] = target - safe
        mcur, _ = _b9k_triangle_signed_margins(x_hat[b, 0:2], tri)
        mtgt, _ = _b9k_triangle_signed_margins(target, tri)
        current_margin[b] = torch.min(mcur)
        target_margin[b] = torch.min(mtgt)
        target_err[b] = torch.linalg.norm(target - x_hat[b, 0:2])

    info = {
        "active": True,
        "candidate_leg": int(candidate),
        "support_vertices": vertices.detach().clone(),
        "raw_safe_xy": raw_safe.detach().clone(),
        "target_raw_xy": target_raw.detach().clone(),
        "target_xy": out.detach().clone(),
        "bias_xy": bias_xy.detach().clone(),
        "geom_margin_current": current_margin.detach().clone(),
        "geom_margin_target": target_margin.detach().clone(),
        "target_err": target_err.detach().clone(),
    }
    return out, info


def _b9k_support_gate_ok(x_hat, foot_pos, candidate, shift_hold_count):
    if not args.canonical_enable_support_region_target_planner or candidate is None:
        return False, {}
    target_xy, info = _b9k_candidate_support_target(foot_pos, x_hat, int(candidate))
    geom_margin = float(info["geom_margin_current"][0].detach().cpu())
    target_margin = float(info["geom_margin_target"][0].detach().cpu())
    target_err = float(info["target_err"][0].detach().cpu())
    ok = (
        geom_margin >= float(args.canonical_fr_liftoff_min_geom_margin)
        and target_margin >= float(args.canonical_fr_target_min_margin) * 0.5
        and target_err <= float(args.canonical_fr_liftoff_target_err_tol)
        and int(shift_hold_count) >= int(args.canonical_fr_liftoff_min_shift_steps)
    )
    info["gate_ok"] = bool(ok)
    return bool(ok), info



def _b10_fsm_status_from_infos(candidate, requested_s, executed_s, exec_info, shift_info):
    if candidate is None:
        return {"state": "ALL_STANCE_IDLE", "candidate": None, "candidate_name": None, "ready": False}
    cand = int(candidate)
    try:
        req_is_swing = float(requested_s[0, cand].detach().cpu()) < 0.5
        exe_is_swing = float(executed_s[0, cand].detach().cpu()) < 0.5
    except Exception:
        req_is_swing, exe_is_swing = False, False

    def _as_float_from_tensor_dict(d, key, default):
        try:
            v = d.get(key, None)
            if v is None:
                return default
            if hasattr(v, "detach"):
                return float(v[0].detach().cpu())
            return float(v)
        except Exception:
            return default

    target_err = _as_float_from_tensor_dict(shift_info, "b9k_target_err", None)
    geom = _as_float_from_tensor_dict(shift_info, "b9k_geom_margin_current", None)
    try:
        shift_hold = int(shift_info.get("shift_hold_count", 0))
    except Exception:
        shift_hold = 0

    ready = (
        target_err is not None
        and geom is not None
        and target_err <= float(args.b10_relocation_target_err_tol)
        and geom >= float(args.b10_relocation_geom_margin_tol)
        and shift_hold >= int(args.b10_relocation_min_shift_steps)
    )

    if req_is_swing and not exe_is_swing:
        state = "RELOCATION_BEFORE_LIFTOFF"
    elif req_is_swing and exe_is_swing:
        state = "SWING_EXECUTION"
    else:
        state = "ALL_STANCE_BETWEEN_STEPS"

    return {
        "state": state,
        "candidate": cand,
        "candidate_name": ID_TO_LEG.get(cand, str(cand)),
        "ready": bool(ready),
        "target_err": target_err,
        "geom_margin": geom,
        "shift_hold_count": shift_hold,
        "gate_reason": exec_info.get("gate_reason", None) if isinstance(exec_info, dict) else None,
    }


def _b11_make_relocation_wbc_cfg(base_cfg, shift_info, x_hat):
    """Create an active WBC config for trunk relocation.

    The WBC already maps base_ref tracking into generalized torque via:
      desired base accel -> constrained qdd/contact-force QP -> tau.
    B11 only strengthens that base task during relocation-before-liftoff.
    """
    info = {
        "active": False,
        "reason": "disabled",
        "base_ref_xy_err": None,
        "candidate": None,
    }
    if not args.enable_b11_wbc_relocation_task:
        return base_cfg, info
    if not isinstance(shift_info, dict) or not bool(shift_info.get("active", False)):
        info["reason"] = "shift_inactive"
        return base_cfg, info

    candidate = shift_info.get("candidate_leg", None)
    if candidate is None:
        info["reason"] = "no_candidate"
        return base_cfg, info

    base_ref_xy_err = None
    try:
        base_ref_xy_err = float(shift_info.get("base_ref_xy_err")[0].detach().cpu())
    except Exception:
        base_ref_xy_err = None

    # Avoid asking the WBC to chase a very far reference. If the reference gets too
    # far, the QP can demand aggressive horizontal forces and the robot may fall.
    if base_ref_xy_err is not None and base_ref_xy_err > float(args.b11_reloc_max_base_ref_xy_err):
        info["reason"] = "base_ref_xy_err_too_large"
        info["base_ref_xy_err"] = base_ref_xy_err
        info["candidate"] = int(candidate)
        return base_cfg, info

    cfg = copy.copy(base_cfg)
    cfg.kp_base_xy = float(args.b11_reloc_kp_base_xy)
    cfg.kd_base_xy = float(args.b11_reloc_kd_base_xy)
    cfg.w_base_acc = float(args.b11_reloc_w_base_acc)
    cfg.w_stance_acc = float(args.b11_reloc_w_stance_acc)
    cfg.max_base_acc_lin = float(args.b11_reloc_max_base_acc_lin)
    cfg.max_tau = float(args.b11_reloc_max_tau)

    info["active"] = True
    info["reason"] = "relocation_wbc"
    info["base_ref_xy_err"] = base_ref_xy_err
    info["candidate"] = int(candidate)
    return cfg, info


def _b12_get_leg_joint_ids(leg):
    """Return joint ids for LF/RF/LH/RH without relying on a global LEG_JOINT_IDS name."""
    try:
        return LEG_JOINT_IDS[leg]
    except NameError:
        pass
    try:
        return JOINT_IDS_BY_LEG[leg]
    except NameError:
        pass
    # Parsed Spot mapping from startup:
    # LF=[fl_hx, fl_hy, fl_kn], RF=[fr_hx, fr_hy, fr_kn],
    # LH=[hl_hx, hl_hy, hl_kn], RH=[hr_hx, hr_hy, hr_kn]
    fallback = {
        0: [0, 4, 8],
        1: [1, 5, 9],
        2: [2, 6, 10],
        3: [3, 7, 11],
    }
    return fallback[int(leg)]


def _b12_compute_pd_alignment_target(robot, Jfeet_full, shift_info, x_hat):
    """Compute relocation-compatible stance joint target.

    During all-stance relocation, the base should move while stance feet remain
    fixed in the world. If base_ref wants the trunk to move by +dx, the stance
    feet need a small opposite Cartesian command in body-relative kinematics.
    This helper maps that command to joint-space deltas with DLS IK.

    It returns q_align_target and a diagnostic info dictionary. The target is used
    both as WBC posture reference and as a small additive torque assist.
    """
    q_cur = robot.data.joint_pos
    q_align = q_cur.clone()
    info = {
        "active": False,
        "reason": "disabled",
        "candidate": None,
        "base_ref_xy_err": None,
        "foot_cmd_xy": None,
        "max_delta": 0.0,
        "ramp": 0.0,
    }
    if not args.enable_b12_pd_target_alignment:
        return q_align, info
    if not isinstance(shift_info, dict) or not bool(shift_info.get("active", False)):
        info["reason"] = "shift_inactive"
        return q_align, info

    cand = shift_info.get("candidate_leg", None)
    if cand is None:
        info["reason"] = "no_candidate"
        return q_align, info
    cand = int(cand)
    info["candidate"] = cand

    try:
        base_ref_xy_err = float(shift_info.get("base_ref_xy_err")[0].detach().cpu())
    except Exception:
        base_ref_xy_err = None
    info["base_ref_xy_err"] = base_ref_xy_err

    if base_ref_xy_err is None:
        info["reason"] = "no_base_ref_xy_err"
        return q_align, info
    if base_ref_xy_err < float(args.b12_align_min_base_ref_xy_err):
        info["reason"] = "small_error"
        return q_align, info
    if base_ref_xy_err > float(args.b12_align_max_base_ref_xy_err):
        info["reason"] = "base_ref_xy_err_too_large"
        return q_align, info

    try:
        base_ref = shift_info.get("base_ref", None)
        if base_ref is not None:
            shift_xy = (base_ref[0, 0:2] - x_hat[0, 0:2]).detach()
        else:
            target_xy = shift_info.get("target_xy", None)
            shift_xy = (target_xy[0, 0:2] - x_hat[0, 0:2]).detach()
    except Exception:
        info["reason"] = "no_shift_xy"
        return q_align, info

    max_xy = float(args.b12_align_max_foot_xy_cmd)
    foot_cmd_xy = float(args.b12_align_sign) * shift_xy
    if float(torch.norm(foot_cmd_xy).detach().cpu()) > max_xy:
        foot_cmd_xy = foot_cmd_xy / torch.norm(foot_cmd_xy).clamp_min(1e-6) * max_xy

    try:
        sh = int(shift_info.get("shift_hold_count", 0))
    except Exception:
        sh = 0
    ramp = min(1.0, max(0.0, sh / max(int(args.b12_align_ramp_steps), 1)))
    foot_cmd_xy = foot_cmd_xy * ramp
    info["ramp"] = ramp
    try:
        info["foot_cmd_xy"] = foot_cmd_xy.detach().cpu().numpy()
    except Exception:
        pass

    max_delta_abs = 0.0
    for leg in range(4):
        if leg == cand:
            continue
        joint_ids = _b12_get_leg_joint_ids(leg)
        try:
            J_leg = Jfeet_full[0, leg, :, joint_ids]  # 3x3
            if args.b12_align_include_hx:
                J_task = J_leg[0:2, :]
                e_task = torch.stack([foot_cmd_xy[0], foot_cmd_xy[1]])
                eye = torch.eye(J_task.shape[0], device=J_task.device)
                A = J_task @ J_task.transpose(0, 1) + (float(args.b12_align_damping) ** 2) * eye
                dq = J_task.transpose(0, 1) @ torch.linalg.solve(A, e_task)
                dq = float(args.b12_align_gain) * dq
                dq = torch.clamp(dq, -float(args.b12_align_max_joint_delta), float(args.b12_align_max_joint_delta))
                q_align[0, joint_ids] = q_cur[0, joint_ids] + dq
                max_delta_abs = max(max_delta_abs, float(torch.max(torch.abs(dq)).detach().cpu()))
            else:
                J_task = J_leg[0:2, 1:3]
                e_task = torch.stack([foot_cmd_xy[0], foot_cmd_xy[1]])
                eye = torch.eye(J_task.shape[0], device=J_task.device)
                A = J_task @ J_task.transpose(0, 1) + (float(args.b12_align_damping) ** 2) * eye
                dq2 = J_task.transpose(0, 1) @ torch.linalg.solve(A, e_task)
                dq2 = float(args.b12_align_gain) * dq2
                dq2 = torch.clamp(dq2, -float(args.b12_align_max_joint_delta), float(args.b12_align_max_joint_delta))
                ids2 = joint_ids[1:3]
                q_align[0, ids2] = q_cur[0, ids2] + dq2
                max_delta_abs = max(max_delta_abs, float(torch.max(torch.abs(dq2)).detach().cpu()))
        except Exception:
            continue

    info["active"] = True
    info["reason"] = "pd_target_aligned"
    info["max_delta"] = max_delta_abs
    return q_align, info


def _b12_alignment_torque_assist(robot, q_align_target, align_info):
    """Small PD torque assist that follows the B12 aligned target.

    This is deliberately capped. It does not replace WBC. It compensates for the
    implicit actuator/posture layer resisting relocation, so the WBC and PD target
    act in the same direction.
    """
    tau = torch.zeros_like(robot.data.joint_pos)
    if not isinstance(align_info, dict) or not bool(align_info.get("active", False)):
        return tau
    cand = align_info.get("candidate", None)
    if cand is None:
        return tau
    cand = int(cand)

    q = robot.data.joint_pos
    qd = robot.data.joint_vel
    for leg in range(4):
        if leg == cand:
            continue
        ids = _b12_get_leg_joint_ids(leg)
        tau_leg = float(args.b12_align_torque_kp) * (q_align_target[0, ids] - q[0, ids]) - float(args.b12_align_torque_kd) * qd[0, ids]
        tau_leg = torch.clamp(tau_leg, -float(args.b12_align_max_tau), float(args.b12_align_max_tau))
        tau[0, ids] = tau_leg
    return tau


def apply_canonical_execution_gate(
    requested_s_t,
    sigma_t,
    prev_exec_s_t,
    current_swing_leg,
    all_stance_gap_count,
    x_hat,
    foot_pos,
    cfg,
    device,
    dtype,
    shift_hold_count=0,
):
    """B9-c execution layer: requested S_t -> executed S_t.

    It keeps the analytical gait schedule, but adds the practical execution layer:
      - at most one swing leg
      - all-stance gap between swing legs
      - pre-liftoff support-margin and trunk-state safety gate
    """
    info = {
        "requested_s": requested_s_t.detach().clone(),
        "executed_s": requested_s_t.detach().clone(),
        "candidate_leg": None,
        "current_swing_leg": current_swing_leg,
        "gate_reason": "none",
        "future_margin": 0.0,
        "future_allowed": True,
        "gap_count": int(all_stance_gap_count),
        "shift_hold_count": int(shift_hold_count),
        "roll": float(x_hat[0, 3].detach().cpu()),
        "pitch": float(x_hat[0, 4].detach().cpu()),
        "base_z": float(x_hat[0, 2].detach().cpu()),
        "future_mask": torch.ones_like(requested_s_t),
    }

    executed = requested_s_t.detach().clone()

    # Resolve one candidate swing.
    candidate = _choose_single_swing_leg(requested_s_t, sigma_t, current_leg=current_swing_leg)
    info["candidate_leg"] = candidate

    if args.canonical_single_swing_gate:
        executed[:] = 1.0
        if candidate is not None:
            executed[:, candidate] = 0.0

    # Determine whether this is a new liftoff from previous executed all-stance/other leg.
    prev_swing_ids = []
    if prev_exec_s_t is not None:
        prev_swing_ids = [int(x.detach().cpu()) for x in torch.nonzero(prev_exec_s_t[0] < 0.5, as_tuple=False).flatten()]
    prev_leg = prev_swing_ids[0] if len(prev_swing_ids) > 0 else None
    requested_has_swing = candidate is not None
    new_liftoff = requested_has_swing and (prev_leg is None or int(prev_leg) != int(candidate))

    # If a swing ended, start/increment all-stance gap.
    if prev_leg is not None and not requested_has_swing:
        all_stance_gap_count = max(int(all_stance_gap_count), int(args.canonical_min_all_stance_gap_steps))

    # Enforce all-stance gap only for new liftoff.
    if new_liftoff and int(all_stance_gap_count) > 0:
        # Even during enforced all-stance gap, compute the candidate future margin.
        # B9-h uses this to decide whether trunk shift is allowed for the next queued leg.
        if candidate is not None:
            future_mask = torch.ones_like(requested_s_t)
            future_mask[:, candidate] = 0.0
            margin, allowed, _out = _canonical_future_margin(x_hat, foot_pos, future_mask, cfg)
            info["future_margin"] = float(margin)
            info["future_allowed"] = bool(allowed)
            info["future_mask"] = future_mask.detach().clone()
        executed[:] = 1.0
        info["gate_reason"] = "all_stance_gap"
        all_stance_gap_count -= 1
        current_swing_leg = None
    elif requested_has_swing:
        # Check future support feasibility for this candidate.
        future_mask = torch.ones_like(requested_s_t)
        future_mask[:, candidate] = 0.0
        margin, allowed, _out = _canonical_future_margin(x_hat, foot_pos, future_mask, cfg)
        info["future_margin"] = float(margin)
        info["future_allowed"] = bool(allowed)
        info["future_mask"] = future_mask.detach().clone()

        roll_ok = abs(info["roll"]) <= float(args.canonical_liftoff_max_abs_roll)
        pitch_ok = abs(info["pitch"]) <= float(args.canonical_liftoff_max_abs_pitch)
        z_ok = info["base_z"] >= float(args.canonical_liftoff_min_base_z)
        margin_ok = margin >= float(args.canonical_liftoff_min_margin)
        relaxed_margin_ok = (
            margin >= float(args.canonical_liftoff_margin_relaxed)
            and int(shift_hold_count) >= int(args.canonical_liftoff_allow_after_shift_steps)
        )
        info["relaxed_margin_ok"] = bool(relaxed_margin_ok)

        # B9-k: stronger feasibility gate based on explicit support-triangle geometry.
        # If the base is close to the computed target and has positive signed distance
        # to all support-triangle edges, allow liftoff even when the older support-region
        # black-box reports future_allowed=False.
        b9k_gate_ok, b9k_info = _b9k_support_gate_ok(x_hat, foot_pos, candidate, shift_hold_count)
        if b9k_gate_ok:
            allowed = True
            margin_ok = True
            relaxed_margin_ok = True
            info["future_allowed"] = True
            info["relaxed_margin_ok"] = True
            info["b9k_support_gate_ok"] = True
            info["b9k_geom_margin_current"] = float(b9k_info["geom_margin_current"][0].detach().cpu())
            info["b9k_geom_margin_target"] = float(b9k_info["geom_margin_target"][0].detach().cpu())
            info["b9k_target_err"] = float(b9k_info["target_err"][0].detach().cpu())
            if args.canonical_fr_print:
                print("[B37 SUPPORT GATE PASS]",
                      "candidate=", ID_TO_LEG[int(candidate)],
                      "geom_margin=", info["b9k_geom_margin_current"],
                      "target_margin=", info["b9k_geom_margin_target"],
                      "target_err=", info["b9k_target_err"],
                      "shift_hold=", int(shift_hold_count))
        else:
            info["b9k_support_gate_ok"] = False
            if b9k_info:
                info["b9k_geom_margin_current"] = float(b9k_info["geom_margin_current"][0].detach().cpu())
                info["b9k_geom_margin_target"] = float(b9k_info["geom_margin_target"][0].detach().cpu())
                info["b9k_target_err"] = float(b9k_info["target_err"][0].detach().cpu())

        if args.b10_allow_liftoff_on_b10_gate and b9k_info:
            try:
                b10_target_err = float(b9k_info["target_err"][0].detach().cpu())
                b10_geom = float(b9k_info["geom_margin_current"][0].detach().cpu())
                b10_ready = (
                    b10_target_err <= float(args.b10_relocation_target_err_tol)
                    and b10_geom >= float(args.b10_relocation_geom_margin_tol)
                    and int(shift_hold_count) >= int(args.b10_relocation_min_shift_steps)
                )
                info["b10_ready"] = bool(b10_ready)
                info["b10_target_err"] = b10_target_err
                info["b10_geom_margin"] = b10_geom
                if b10_ready:
                    allowed = True
                    margin_ok = True
                    relaxed_margin_ok = True
                    info["future_allowed"] = True
                    info["relaxed_margin_ok"] = True
                    if args.b10_print:
                        print("[B25 FSM GATE PASS]",
                              "candidate=", ID_TO_LEG[int(candidate)],
                              "target_err=", b10_target_err,
                              "geom=", b10_geom,
                              "shift_hold=", int(shift_hold_count))
            except Exception as e:
                info["b10_ready"] = False
                info["b10_gate_error"] = str(e)

        if args.enable_b38_candidate_support_recenter and args.b38_allow_liftoff_on_b38_gate and _b38_leg_allowed(candidate) and b9k_info:
            try:
                b38_target_err = float(b9k_info["target_err"][0].detach().cpu())
                b38_geom = float(b9k_info["geom_margin_current"][0].detach().cpu())
                b38_ready = (
                    b38_target_err <= float(args.b38_target_err_tol)
                    and b38_geom >= float(args.b38_geom_margin_tol)
                    and int(shift_hold_count) >= int(args.b38_min_shift_steps)
                )
                info["b38_ready"] = bool(b38_ready)
                info["b38_target_err"] = b38_target_err
                info["b38_geom_margin"] = b38_geom
                if b38_ready:
                    allowed = True
                    margin_ok = True
                    relaxed_margin_ok = True
                    info["future_allowed"] = True
                    info["relaxed_margin_ok"] = True
                    if args.b38_print:
                        print("[B38 SUPPORT RECENTER GATE PASS]",
                              "candidate=", ID_TO_LEG[int(candidate)],
                              "target_err=", b38_target_err,
                              "geom=", b38_geom,
                              "shift_hold=", int(shift_hold_count))
            except Exception as e:
                info["b38_ready"] = False
                info["b38_gate_error"] = str(e)

        if args.enable_b39_latched_support_recenter and args.b39_allow_liftoff_on_b39_gate and _b39_leg_allowed(candidate) and b9k_info:
            try:
                b39_target_err = float(b9k_info["target_err"][0].detach().cpu())
                b39_geom = float(b9k_info["geom_margin_current"][0].detach().cpu())
                b39_ready = (
                    b39_target_err <= float(args.b39_target_err_tol)
                    and b39_geom >= float(args.b39_geom_margin_tol)
                    and int(shift_hold_count) >= int(args.b39_min_shift_steps)
                )
                info["b39_ready"] = bool(b39_ready)
                info["b39_target_err"] = b39_target_err
                info["b39_geom_margin"] = b39_geom
                if b39_ready:
                    allowed = True
                    margin_ok = True
                    relaxed_margin_ok = True
                    info["future_allowed"] = True
                    info["relaxed_margin_ok"] = True
                    if args.b39_print:
                        print("[B39 LATCHED SUPPORT RECENTER GATE PASS]",
                              "candidate=", ID_TO_LEG[int(candidate)],
                              "target_err=", b39_target_err,
                              "geom=", b39_geom,
                              "shift_hold=", int(shift_hold_count))
            except Exception as e:
                info["b39_ready"] = False
                info["b39_gate_error"] = str(e)

        if new_liftoff and (not allowed or ((not margin_ok) and (not relaxed_margin_ok)) or not roll_ok or not pitch_ok or not z_ok):
            executed[:] = 1.0
            current_swing_leg = None
            if not allowed:
                info["gate_reason"] = "future_not_allowed"
            elif (not margin_ok) and (not relaxed_margin_ok):
                info["gate_reason"] = "margin"
            elif not roll_ok:
                info["gate_reason"] = "roll"
            elif not pitch_ok:
                info["gate_reason"] = "pitch"
            elif not z_ok:
                info["gate_reason"] = "base_z"
        else:
            executed[:] = 1.0
            executed[:, candidate] = 0.0
            current_swing_leg = int(candidate)
            info["gate_reason"] = "pass"
    else:
        executed[:] = 1.0
        current_swing_leg = None
        if int(all_stance_gap_count) > 0:
            all_stance_gap_count -= 1
            info["gate_reason"] = "gap_countdown"
        else:
            info["gate_reason"] = "all_stance"

    info["executed_s"] = executed.detach().clone()
    info["current_swing_leg"] = current_swing_leg
    info["gap_count"] = int(all_stance_gap_count)
    info["shift_hold_count"] = int(shift_hold_count)
    return executed, current_swing_leg, int(all_stance_gap_count), info


def apply_canonical_feasible_trunk_shift(base_ref, x_hat, foot_pos, exec_info, prev_base_ref, last_shift_target_xy=None):
    """Move base_ref toward an Abdalla-inspired feasible safe point.

    This is a lite proxy of the feasibility idea:
      use the future support polygon for the candidate swing leg,
      choose its incenter/centroid safe point,
      bias away from the swing side,
      step the trunk target gradually.
    """
    if not args.canonical_enable_feasible_trunk_shift:
        return base_ref, {"active": False}

    candidate = exec_info.get("candidate_leg", None)
    if candidate is None:
        return base_ref, {"active": False, "reason": "no_candidate"}

    # B9-h safety: do not shift toward a candidate whose future support is already bad.
    # B38 exception: for candidate-specific support recenter we intentionally move toward
    # the candidate's feasible support-triangle target even when the current future margin
    # is bad; otherwise LH/RH never gets a chance to become feasible.
    b38_active_for_candidate = bool(
        args.enable_b38_candidate_support_recenter
        and _b38_leg_allowed(candidate)
        and args.b38_force_shift_even_if_future_bad
    )
    b39_active_for_candidate = bool(
        args.enable_b39_latched_support_recenter
        and _b39_leg_allowed(candidate)
        and args.b39_force_shift_even_if_future_bad
    )
    if args.canonical_enable_event_crawl_queue:
        fm = float(exec_info.get("future_margin", 0.0))
        if fm < float(args.canonical_event_shift_min_future_margin) and not (b38_active_for_candidate or b39_active_for_candidate):
            return base_ref, {"active": False, "reason": "future_margin_too_low", "future_margin": fm, "candidate_leg": int(candidate)}

    executed_s = exec_info.get("executed_s")
    requested_s = exec_info.get("requested_s")
    is_executing_swing = bool((executed_s[0] < 0.5).any().detach().cpu()) if executed_s is not None else False
    is_requested_swing = bool((requested_s[0] < 0.5).any().detach().cpu()) if requested_s is not None else False
    shift_before = args.canonical_shift_before_liftoff and is_requested_swing and not is_executing_swing
    shift_during = args.canonical_shift_during_swing and is_executing_swing
    if not (shift_before or shift_during):
        return base_ref, {"active": False, "reason": "not_in_shift_window"}

    future_mask = torch.ones_like(requested_s)
    future_mask[:, int(candidate)] = 0.0

    b9k_info = {}
    if args.canonical_enable_support_region_target_planner:
        target_xy, b9k_info = _b9k_candidate_support_target(foot_pos, x_hat, int(candidate))
        safe_xy = b9k_info["raw_safe_xy"]
        bias = b9k_info["bias_xy"]
        fr_info = b9k_info
    else:
        safe_xy, fr_info = _safe_point_of_support_polygon_lite(foot_pos, future_mask, x_hat, active_leg=int(candidate))

        # B9-c specific smaller biases, independent of B8 fr_lite args.
        fwd, left = _base_yaw_unit_vectors(x_hat)
        bias = float(args.canonical_shift_forward_bias) * fwd
        if int(candidate) in [0, 2]:       # left leg swing -> shift right
            bias = bias - float(args.canonical_shift_lateral_bias) * left
        elif int(candidate) in [1, 3]:     # right leg swing -> shift left
            bias = bias + float(args.canonical_shift_lateral_bias) * left

        target_xy = safe_xy + bias
    blended_xy = (1.0 - float(args.canonical_shift_blend)) * base_ref[:, 0:2] + float(args.canonical_shift_blend) * target_xy

    base_ref_xy_err = torch.linalg.norm(base_ref[:, 0:2] - x_hat[:, 0:2], dim=1)
    err_too_large = bool((base_ref_xy_err[0] > float(args.canonical_shift_max_base_ref_xy_err)).detach().cpu())

    # B39 is deliberately latched/accumulated: use the previous commanded shift
    # as the next start point, not the measured base.  B38 optionally restarted
    # from x_hat each frame, which made target_err stay nearly constant.
    if b39_active_for_candidate:
        if last_shift_target_xy is not None:
            start_xy = last_shift_target_xy.detach().clone()
        elif prev_base_ref is not None:
            start_xy = prev_base_ref[:, 0:2].detach().clone()
        else:
            start_xy = base_ref[:, 0:2].detach().clone()
    elif args.canonical_shift_track_actual_base or (b38_active_for_candidate and args.b38_use_actual_base_start):
        start_xy = x_hat[:, 0:2]
    else:
        start_xy = base_ref[:, 0:2] if prev_base_ref is None else prev_base_ref[:, 0:2]

    hold_due_to_error = bool(args.canonical_shift_hold_when_ref_error and err_too_large and last_shift_target_xy is not None)
    active_max_step = float(args.b39_max_step) if b39_active_for_candidate else (float(args.b38_max_step) if b38_active_for_candidate else float(args.canonical_shift_max_step))
    if hold_due_to_error:
        next_xy = last_shift_target_xy.detach().clone()
    else:
        next_xy = _step_toward_xy(start_xy, blended_xy, active_max_step)

    out = base_ref.detach().clone()
    out[:, 0:2] = next_xy
    info = {
        "active": True,
        "candidate_leg": int(candidate),
        "shift_before": bool(shift_before),
        "shift_during": bool(shift_during),
        "raw_safe_xy": safe_xy.detach().clone(),
        "target_xy": target_xy.detach().clone(),
        "next_xy": next_xy.detach().clone(),
        "applied_delta_xy": (next_xy - base_ref[:, 0:2]).detach().clone(),
        "b38_candidate_support_recenter": bool(b38_active_for_candidate),
        "b39_latched_support_recenter": bool(b39_active_for_candidate),
        "b38_max_step": active_max_step,
        "b39_max_step": active_max_step,
        "base_ref_xy_err": base_ref_xy_err.detach().clone(),
        "err_too_large": bool(err_too_large),
        "hold_due_to_error": bool(hold_due_to_error),
        "future_margin": exec_info.get("future_margin", 0.0),
        "gate_reason": exec_info.get("gate_reason", "none"),
        "b9k_geom_margin_current": (
            b9k_info.get("geom_margin_current", torch.zeros((1,), device=foot_pos.device, dtype=foot_pos.dtype)).detach().clone()
            if isinstance(b9k_info, dict) else torch.zeros((1,), device=foot_pos.device, dtype=foot_pos.dtype)
        ),
        "b9k_geom_margin_target": (
            b9k_info.get("geom_margin_target", torch.zeros((1,), device=foot_pos.device, dtype=foot_pos.dtype)).detach().clone()
            if isinstance(b9k_info, dict) else torch.zeros((1,), device=foot_pos.device, dtype=foot_pos.dtype)
        ),
        "b9k_target_err": (
            b9k_info.get("target_err", torch.zeros((1,), device=foot_pos.device, dtype=foot_pos.dtype)).detach().clone()
            if isinstance(b9k_info, dict) else torch.zeros((1,), device=foot_pos.device, dtype=foot_pos.dtype)
        ),
    }
    return out, info


def swing_mask_for_leg(leg_name, num_envs, device, dtype):
    stance = torch.ones((num_envs, 4), device=device, dtype=dtype)
    stance[:, LEG_TO_ID[leg_name]] = 0.0
    return stance


def _b13_parse_candidate_legs():
    out = []
    for x in str(args.b13_candidate_legs).split(","):
        x = x.strip().upper()
        if x in LEG_TO_ID and x not in out:
            out.append(x)
    return out if out else ["LF", "LH", "RH"]


def _b13_score_candidate_leg(x_hat, foot_pos, leg_id, cfg, device, dtype):
    """Score a candidate swing leg using current support feasibility.

    Higher is better. The score combines:
      - current future-support margin from compute_support_region_ref
      - explicit support-triangle current signed margin from B9-k
      - distance to the projected feasible target from B9-l

    This avoids hard-coding RF->LH when LH is geometrically unfavorable.
    """
    mask = torch.ones((args.num_envs, 4), device=device, dtype=dtype)
    mask[:, int(leg_id)] = 0.0
    margin, allowed, _out = _canonical_future_margin(x_hat, foot_pos, mask, cfg)

    geom = 0.0
    target_err = 0.0
    b9k_gate = False
    try:
        b9k_gate, b9k_info = _b9k_support_gate_ok(x_hat, foot_pos, int(leg_id), shift_hold_count=10**9)
        if b9k_info:
            geom = float(b9k_info["geom_margin_current"][0].detach().cpu())
            target_err = float(b9k_info["target_err"][0].detach().cpu())
    except Exception:
        b9k_gate = False

    score = (
        float(args.b13_score_margin_weight) * float(margin)
        + float(args.b13_score_geom_weight) * float(geom)
        - float(args.b13_score_target_err_weight) * float(target_err)
    )
    return {
        "leg": ID_TO_LEG[int(leg_id)],
        "leg_id": int(leg_id),
        "score": float(score),
        "future_margin": float(margin),
        "future_allowed": bool(allowed),
        "geom_margin": float(geom),
        "target_err": float(target_err),
        "b9k_gate": bool(b9k_gate),
    }


def _b13_select_adaptive_next_leg(x_hat, foot_pos, cfg, device, dtype, queue_index, last_completed_leg=None, completed_step_by_leg=None, step_now=None):
    if not args.enable_b13_adaptive_next_leg:
        return None, []
    if int(queue_index) < int(args.b13_adaptive_start_index):
        return None, []

    candidates = []
    for name in _b13_parse_candidate_legs():
        li = LEG_TO_ID[name]
        if args.b13_exclude_last_completed and last_completed_leg is not None and int(li) == int(last_completed_leg):
            continue
        if args.enable_b14_no_queue_mutation and completed_step_by_leg is not None and step_now is not None:
            last_step = int(completed_step_by_leg.get(int(li), -10**9))
            if int(step_now) - last_step < int(args.b14_completed_cooldown_steps):
                continue
        candidates.append(int(li))

    if not candidates:
        return None, []

    infos = []
    for li in candidates:
        infos.append(_b13_score_candidate_leg(x_hat, foot_pos, li, cfg, device, dtype))

    best = max(infos, key=lambda d: d["score"])
    if args.enable_b14_no_queue_mutation and args.b14_require_positive_best_margin:
        if float(best.get("future_margin", -1e9)) < 0.0 and float(best.get("geom_margin", -1e9)) < 0.0:
            return None, infos
    return int(best["leg_id"]), infos


def _b15_candidates_all_infeasible(scores):
    if not args.enable_b15_recenter_fallback:
        return False
    if not scores:
        return False
    best_future = max(float(s.get("future_margin", -1e9)) for s in scores)
    best_geom = max(float(s.get("geom_margin", -1e9)) for s in scores)
    return (
        best_future < float(args.b15_recenter_future_margin_threshold)
        and best_geom < float(args.b15_recenter_geom_margin_threshold)
    )


def _b15_apply_all_stance_recenter(base_ref, x_hat, foot_pos, prev_target_xy=None):
    """All-stance recenter fallback.

    When every adaptive candidate is infeasible, candidate-specific support target
    can keep pushing the trunk toward an unreachable corner. This fallback keeps
    all feet in stance and moves the base reference toward the centroid of the
    current four foot contacts, blended with current base xy for conservatism.
    """
    info = {
        "active": False,
        "target_xy": None,
        "next_xy": None,
        "base_ref_xy_err": None,
        "raw_centroid_xy": None,
    }
    if not args.enable_b15_recenter_fallback:
        return base_ref, info

    raw_centroid = torch.mean(foot_pos[:, :, 0:2], dim=1)
    target_xy = (
        (1.0 - float(args.b15_recenter_target_blend_current)) * raw_centroid
        + float(args.b15_recenter_target_blend_current) * x_hat[:, 0:2]
    )

    current_xy = base_ref[:, 0:2]
    if prev_target_xy is not None:
        current_xy = prev_target_xy

    delta = target_xy - current_xy
    norm = torch.norm(delta, dim=1, keepdim=True).clamp_min(1e-6)
    max_step = float(args.b15_recenter_max_step)
    scale = torch.clamp(max_step / norm, max=1.0)
    next_xy = current_xy + delta * scale

    out = base_ref.detach().clone()
    out[:, 0:2] = next_xy
    out[:, 2] = float(args.height_ref)
    out[:, 3:6] = 0.0

    err = torch.norm(out[:, 0:2] - x_hat[:, 0:2], dim=1)
    info.update({
        "active": True,
        "target_xy": target_xy.detach().clone(),
        "next_xy": next_xy.detach().clone(),
        "base_ref_xy_err": err.detach().clone(),
        "raw_centroid_xy": raw_centroid.detach().clone(),
    })
    return out, info


def _b16_compute_recenter_all_stance_target(robot, Jfeet_full, base_ref, x_hat, recenter_elapsed_steps):
    """All-stance target alignment used only during B15 recenter.

    B12 alignment was candidate-based and its ramp could be zero in recenter mode.
    This helper explicitly uses all four stance legs and a recenter elapsed counter.
    """
    q_cur = robot.data.joint_pos
    q_align = q_cur.clone()
    info = {
        "active": False,
        "reason": "disabled",
        "base_ref_xy_err": 0.0,
        "foot_cmd_xy": None,
        "max_delta": 0.0,
        "ramp": 0.0,
    }
    if not args.enable_b16_recenter_all_stance_pd_alignment:
        return q_align, info

    shift_xy = (base_ref[0, 0:2] - x_hat[0, 0:2]).detach()
    err = float(torch.norm(shift_xy).detach().cpu())
    info["base_ref_xy_err"] = err
    if err < 1e-5:
        info["reason"] = "small_error"
        return q_align, info

    max_xy = float(args.b16_recenter_align_max_foot_xy_cmd)
    foot_cmd_xy = float(args.b16_recenter_align_sign) * shift_xy
    if float(torch.norm(foot_cmd_xy).detach().cpu()) > max_xy:
        foot_cmd_xy = foot_cmd_xy / torch.norm(foot_cmd_xy).clamp_min(1e-6) * max_xy

    ramp = min(1.0, max(0.0, float(recenter_elapsed_steps) / max(int(args.b16_recenter_align_ramp_steps), 1)))
    foot_cmd_xy = foot_cmd_xy * ramp
    info["ramp"] = ramp
    try:
        info["foot_cmd_xy"] = foot_cmd_xy.detach().cpu().numpy()
    except Exception:
        pass

    max_delta_abs = 0.0
    for leg in range(4):
        joint_ids = _b12_get_leg_joint_ids(leg)
        try:
            J_leg = Jfeet_full[0, leg, :, joint_ids]
            if args.b16_recenter_align_include_hx:
                J_task = J_leg[0:2, :]
                e_task = torch.stack([foot_cmd_xy[0], foot_cmd_xy[1]])
                eye = torch.eye(J_task.shape[0], device=J_task.device)
                A = J_task @ J_task.transpose(0, 1) + (float(args.b16_recenter_align_damping) ** 2) * eye
                dq = J_task.transpose(0, 1) @ torch.linalg.solve(A, e_task)
                dq = float(args.b16_recenter_align_gain) * dq
                dq = torch.clamp(dq, -float(args.b16_recenter_align_max_joint_delta), float(args.b16_recenter_align_max_joint_delta))
                q_align[0, joint_ids] = q_cur[0, joint_ids] + dq
                max_delta_abs = max(max_delta_abs, float(torch.max(torch.abs(dq)).detach().cpu()))
            else:
                J_task = J_leg[0:2, 1:3]
                e_task = torch.stack([foot_cmd_xy[0], foot_cmd_xy[1]])
                eye = torch.eye(J_task.shape[0], device=J_task.device)
                A = J_task @ J_task.transpose(0, 1) + (float(args.b16_recenter_align_damping) ** 2) * eye
                dq2 = J_task.transpose(0, 1) @ torch.linalg.solve(A, e_task)
                dq2 = float(args.b16_recenter_align_gain) * dq2
                dq2 = torch.clamp(dq2, -float(args.b16_recenter_align_max_joint_delta), float(args.b16_recenter_align_max_joint_delta))
                ids2 = joint_ids[1:3]
                q_align[0, ids2] = q_cur[0, ids2] + dq2
                max_delta_abs = max(max_delta_abs, float(torch.max(torch.abs(dq2)).detach().cpu()))
        except Exception:
            continue

    info["active"] = True
    info["reason"] = "b16_recenter_all_stance_aligned"
    info["max_delta"] = max_delta_abs
    return q_align, info


def _b16_recenter_alignment_torque_assist(robot, q_align_target, align_info):
    tau = torch.zeros_like(robot.data.joint_pos)
    if not isinstance(align_info, dict) or not bool(align_info.get("active", False)):
        return tau
    q = robot.data.joint_pos
    qd = robot.data.joint_vel
    for leg in range(4):
        ids = _b12_get_leg_joint_ids(leg)
        tau_leg = (
            float(args.b16_recenter_align_torque_kp) * (q_align_target[0, ids] - q[0, ids])
            - float(args.b16_recenter_align_torque_kd) * qd[0, ids]
        )
        tau_leg = torch.clamp(tau_leg, -float(args.b16_recenter_align_max_tau), float(args.b16_recenter_align_max_tau))
        tau[0, ids] = tau_leg
    return tau


def _b17_parse_float_list(s):
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if not x:
            continue
        try:
            out.append(float(x))
        except Exception:
            pass
    return out if out else [0.0]


def _b17_parse_leg_list(s):
    out = []
    for x in str(s).split(","):
        x = x.strip().upper()
        if x in LEG_TO_ID and x not in out:
            out.append(x)
    return out


def _b17_nominal_final_foothold(foot_anchor_w, base_quat_w, leg):
    fwd = torch.zeros_like(foot_anchor_w[:, leg, :])
    if args.canonical_gait_use_base_yaw:
        yaw = quat_to_yaw(base_quat_w)
        fwd[:, 0] = torch.cos(yaw)
        fwd[:, 1] = torch.sin(yaw)
    else:
        fwd[:, 0] = 1.0

    left = torch.zeros_like(fwd)
    left[:, 0] = -fwd[:, 1]
    left[:, 1] = fwd[:, 0]

    nominal = foot_anchor_w[:, leg, :].detach().clone()
    nominal[:, 0:2] = (
        foot_anchor_w[:, leg, 0:2]
        + float(args.canonical_gait_forward_step) * fwd[:, 0:2]
        + float(args.canonical_gait_lateral_step) * left[:, 0:2]
    )
    return nominal, fwd, left


def _b17_score_foothold_geometry(x_hat, foot_pos, swing_leg, candidate_final_w, cfg, device, dtype):
    """Score a proposed touchdown by evaluating future support feasibility.

    This is a Decoder/Mapper operation: it only changes the foot reference, then
    uses the existing support-region metrics as a critic. MPC/WBC are unchanged.
    """
    temp_foot_pos = foot_pos.detach().clone()
    temp_foot_pos[:, int(swing_leg), :] = candidate_final_w.detach()

    next_names = _b17_parse_leg_list(args.b17_foothold_next_candidates)
    if not next_names:
        next_names = ["LH", "RH", "LF", "RF"]

    infos = []
    for name in next_names:
        li = LEG_TO_ID[name]
        if int(li) == int(swing_leg):
            continue
        try:
            info = _b13_score_candidate_leg(x_hat, temp_foot_pos, li, cfg, device, dtype)
            score = (
                float(args.b17_foothold_score_margin_weight) * float(info.get("future_margin", 0.0))
                + float(args.b17_foothold_score_geom_weight) * float(info.get("geom_margin", 0.0))
                - float(args.b17_foothold_score_target_err_weight) * float(info.get("target_err", 0.0))
            )
            info = dict(info)
            info["b17_score"] = float(score)
            infos.append(info)
        except Exception:
            continue

    if not infos:
        return -1.0e9, [], None
    best_next = max(infos, key=lambda d: d["b17_score"])
    return float(best_next["b17_score"]), infos, best_next



def _b48_next_leg_names_for_scoring(swing_leg):
    mode = str(getattr(args, "b48_next_leg_mode", "fixed")).strip().lower()
    if mode == "fixed":
        names = [str(getattr(args, "b48_fixed_next_leg", "LH")).strip().upper()]
    else:
        names = _b17_parse_leg_list(getattr(args, "b48_next_candidates", "LH,RH,LF"))
    out = []
    for name in names:
        if name in LEG_TO_ID and int(LEG_TO_ID[name]) != int(swing_leg) and name not in out:
            out.append(name)
    return out


def _b48_score_post_landing_candidate(x_hat, foot_pos, swing_leg, candidate_final_w, fwd_xy, cfg, device, dtype):
    """Clemente outer loop + Abdalla-lite inner checker.

    A touchdown candidate is scored by pretending the current swing foot has
    landed at candidate_final_w, then checking whether the next swing leg can be
    lifted using the existing support-region / feasible-region metrics.
    """
    temp_foot_pos = foot_pos.detach().clone()
    temp_foot_pos[:, int(swing_leg), :] = candidate_final_w.detach()

    next_names = _b48_next_leg_names_for_scoring(swing_leg)
    infos = []
    for name in next_names:
        li = LEG_TO_ID[name]
        try:
            leg_info = _b13_score_candidate_leg(x_hat, temp_foot_pos, int(li), cfg, device, dtype)
        except Exception as e:
            leg_info = {"leg": name, "leg_id": int(li), "score": -1e9, "error": repr(e),
                        "future_margin": -1e9, "geom_margin": -1e9, "target_err": 1e9}
        leg_info = dict(leg_info)
        infos.append(leg_info)

    if not infos:
        return -1.0e9, [], None

    # Fixed mode returns the fixed next-leg score; candidate mode chooses best.
    best_next = max(infos, key=lambda d: float(d.get("score", -1e9)))
    return float(best_next.get("score", -1e9)), infos, best_next


def _b48_apply_post_landing_feasibility_foothold_planner(
    canonical_foot_target_w,
    canonical_foot_anchor_w,
    base_quat_w,
    canonical_s_t,
    canonical_sigma_t,
    x_hat,
    foot_pos,
    cfg,
    device,
    dtype,
    step,
):
    info = {
        "active": False,
        "reason": "disabled",
        "swing_leg": None,
        "selected_delta": None,
        "selected_next": None,
        "best_score": None,
        "scores": [],
    }
    if not bool(getattr(args, "enable_b48_post_landing_feasibility_foothold_planner", False)):
        return canonical_foot_target_w, info
    if canonical_foot_anchor_w is None or canonical_s_t is None or canonical_sigma_t is None:
        info["reason"] = "missing_canonical_state"
        return canonical_foot_target_w, info

    active = [int(i) for i in torch.nonzero(canonical_s_t[0] < 0.5, as_tuple=False).flatten()]
    if len(active) != 1:
        info["reason"] = "not_single_active_swing"
        return canonical_foot_target_w, info

    swing_leg = int(active[0])
    swing_name = ID_TO_LEG[int(swing_leg)]
    apply_names = _b17_parse_leg_list(getattr(args, "b48_apply_only_to", "RF"))
    if apply_names and swing_name not in apply_names:
        info["reason"] = "leg_not_enabled"
        info["swing_leg"] = swing_name
        return canonical_foot_target_w, info

    sig = canonical_sigma_t[:, swing_leg].clamp(0.0, 1.0).view(-1, 1)
    nominal_final, fwd, left = _b17_nominal_final_foothold(canonical_foot_anchor_w, base_quat_w, swing_leg)
    side_sign = 1.0 if swing_leg in [0, 2] else -1.0
    forward_vals = _b17_parse_float_list(getattr(args, "b48_forward_candidates", "0.0,0.02,0.04"))
    lateral_vals = _b17_parse_float_list(getattr(args, "b48_lateral_candidates", "-0.01,0.0,0.01"))

    best = None
    scored = []
    for df in forward_vals:
        for dl in lateral_vals:
            extra = float(df) * fwd[:, 0:2] + side_sign * float(dl) * left[:, 0:2]
            extra_norm_t = torch.linalg.norm(extra, dim=1, keepdim=True).clamp_min(1.0e-9)
            max_extra = float(getattr(args, "b48_max_extra_xy", 0.085))
            if max_extra > 0.0:
                scale = torch.clamp(torch.as_tensor(max_extra, device=extra.device, dtype=extra.dtype) / extra_norm_t, max=1.0)
                extra = extra * scale
            extra_norm = float(torch.linalg.norm(extra[0]).detach().cpu())

            cand_final = nominal_final.detach().clone()
            cand_final[:, 0:2] = nominal_final[:, 0:2] + extra
            transition_raw, next_infos, best_next = _b48_score_post_landing_candidate(
                x_hat, foot_pos, swing_leg, cand_final, fwd[:, 0:2], cfg, device, dtype
            )
            if best_next is None:
                continue
            post_margin = float(best_next.get("future_margin", -1e9))
            if post_margin < float(getattr(args, "b48_min_post_future_margin", -999.0)):
                hard_filtered = True
            else:
                hard_filtered = False

            progress = float(torch.sum((cand_final[0, 0:2] - canonical_foot_anchor_w[0, swing_leg, 0:2]) * fwd[0, 0:2]).detach().cpu())
            score = (
                float(getattr(args, "b48_score_margin_weight", 3.0)) * float(best_next.get("future_margin", 0.0))
                + float(getattr(args, "b48_score_geom_weight", 1.0)) * float(best_next.get("geom_margin", 0.0))
                - float(getattr(args, "b48_score_target_err_weight", 0.8)) * float(best_next.get("target_err", 0.0))
                + float(getattr(args, "b48_score_progress_weight", 0.5)) * progress
                - float(getattr(args, "b48_score_step_penalty", 0.25)) * extra_norm
            )
            if hard_filtered:
                score -= 1.0e6

            row = {
                "df": float(df),
                "dl_outward": float(dl),
                "extra_norm": float(extra_norm),
                "progress": float(progress),
                "score": float(score),
                "post_margin": float(post_margin),
                "best_next": best_next,
                "all_next": next_infos,
                "hard_filtered": bool(hard_filtered),
            }
            scored.append(row)
            if best is None or float(score) > float(best["score"]):
                best = {
                    "score": float(score),
                    "candidate_final": cand_final.detach().clone(),
                    "df": float(df),
                    "dl_outward": float(dl),
                    "extra_norm": float(extra_norm),
                    "progress": float(progress),
                    "best_next": best_next,
                    "all_next": next_infos,
                    "hard_filtered": bool(hard_filtered),
                }

    if best is None:
        info["reason"] = "no_candidate"
        return canonical_foot_target_w, info

    out = canonical_foot_target_w.detach().clone()
    out[:, swing_leg, 0:2] = (
        (1.0 - sig) * canonical_foot_anchor_w[:, swing_leg, 0:2]
        + sig * best["candidate_final"][:, 0:2]
    )
    info.update({
        "active": True,
        "reason": "b48_post_landing_feasibility_foothold",
        "swing_leg": swing_name,
        "selected_delta": (best["candidate_final"][0, 0:2] - nominal_final[0, 0:2]).detach().cpu().numpy(),
        "selected_df": best["df"],
        "selected_dl_outward": best["dl_outward"],
        "selected_next": best["best_next"],
        "best_score": best["score"],
        "progress": best["progress"],
        "scores": scored,
        "nominal_final": nominal_final[0].detach().cpu().numpy(),
        "selected_final": best["candidate_final"][0].detach().cpu().numpy(),
    })

    if bool(getattr(args, "b48_print", False)) and (step % max(args.print_every, 1) == 0):
        compact = []
        for r in scored:
            bn = r.get("best_next") or {}
            compact.append({
                "df": round(r["df"], 3),
                "dl": round(r["dl_outward"], 3),
                "score": round(r["score"], 5),
                "prog": round(r["progress"], 5),
                "next": bn.get("leg"),
                "fm": round(float(bn.get("future_margin", 0.0)), 5) if bn else None,
                "geom": round(float(bn.get("geom_margin", 0.0)), 5) if bn else None,
                "terr": round(float(bn.get("target_err", 0.0)), 5) if bn else None,
                "filtered": bool(r.get("hard_filtered", False)),
            })
        print("[B48 POST-LANDING FEAS FOOTHOLD]",
              "step=", step,
              "swing=", swing_name,
              "next_mode=", getattr(args, "b48_next_leg_mode", "fixed"),
              "selected_delta=", info["selected_delta"],
              "selected_next=", info["selected_next"],
              "best_score=", info["best_score"],
              "selected_final=", info["selected_final"],
              "scores=", compact)
    return out, info


def _b17_apply_support_aware_foothold_mapper(
    canonical_foot_target_w,
    canonical_foot_anchor_w,
    base_quat_w,
    canonical_s_t,
    canonical_sigma_t,
    x_hat,
    foot_pos,
    cfg,
    device,
    dtype,
    step,
):
    info = {
        "active": False,
        "reason": "disabled",
        "swing_leg": None,
        "selected_delta": None,
        "selected_next": None,
        "best_score": None,
        "scores": [],
    }
    if not args.enable_b17_support_aware_foothold_mapper:
        return canonical_foot_target_w, info
    if canonical_foot_anchor_w is None or canonical_s_t is None or canonical_sigma_t is None:
        info["reason"] = "missing_canonical_state"
        return canonical_foot_target_w, info

    apply_names = _b17_parse_leg_list(args.b17_foothold_apply_only_to)
    active = [int(i) for i in torch.nonzero(canonical_s_t[0] < 0.5, as_tuple=False).flatten()]
    if len(active) != 1:
        info["reason"] = "not_single_active_swing"
        return canonical_foot_target_w, info

    swing_leg = int(active[0])
    if apply_names and ID_TO_LEG[int(swing_leg)] not in apply_names:
        info["reason"] = "leg_not_enabled"
        info["swing_leg"] = ID_TO_LEG[int(swing_leg)]
        return canonical_foot_target_w, info

    sig = canonical_sigma_t[:, swing_leg].clamp(0.0, 1.0).view(-1, 1)
    nominal_final, fwd, left = _b17_nominal_final_foothold(canonical_foot_anchor_w, base_quat_w, swing_leg)

    # Positive lateral candidate means outward from the body side.
    # LF/LH are left-side legs -> outward is +left. RF/RH -> outward is -left.
    side_sign = 1.0 if swing_leg in [0, 2] else -1.0

    forward_vals = _b17_parse_float_list(args.b17_foothold_forward_candidates)
    lateral_vals = _b17_parse_float_list(args.b17_foothold_lateral_candidates)

    best = None
    scored = []
    for df in forward_vals:
        for dl in lateral_vals:
            extra = float(df) * fwd[:, 0:2] + (side_sign * float(dl)) * left[:, 0:2]
            extra_norm = float(torch.linalg.norm(extra[0]).detach().cpu())
            if extra_norm > float(args.b17_foothold_max_extra_xy):
                extra = extra / torch.linalg.norm(extra, dim=1, keepdim=True).clamp_min(1e-6) * float(args.b17_foothold_max_extra_xy)
                extra_norm = float(args.b17_foothold_max_extra_xy)

            cand_final = nominal_final.detach().clone()
            cand_final[:, 0:2] = nominal_final[:, 0:2] + extra
            score_raw, next_infos, best_next = _b17_score_foothold_geometry(
                x_hat, foot_pos, swing_leg, cand_final, cfg, device, dtype
            )
            score = float(score_raw) - float(args.b17_foothold_step_penalty) * extra_norm
            row = {
                "df": float(df),
                "dl_outward": float(dl),
                "extra_norm": float(extra_norm),
                "score": float(score),
                "score_raw": float(score_raw),
                "best_next": best_next,
            }
            scored.append(row)
            if best is None or score > best["score"]:
                best = {
                    "score": float(score),
                    "candidate_final": cand_final.detach().clone(),
                    "df": float(df),
                    "dl_outward": float(dl),
                    "extra_norm": float(extra_norm),
                    "best_next": best_next,
                    "all_next": next_infos,
                }

    if best is None:
        info["reason"] = "no_candidate"
        return canonical_foot_target_w, info

    out = canonical_foot_target_w.detach().clone()
    out[:, swing_leg, 0:2] = (
        (1.0 - sig) * canonical_foot_anchor_w[:, swing_leg, 0:2]
        + sig * best["candidate_final"][:, 0:2]
    )
    # Keep z profile from existing canonical target.
    info.update({
        "active": True,
        "reason": "support_geometry_aware_foothold",
        "swing_leg": ID_TO_LEG[int(swing_leg)],
        "selected_delta": (best["candidate_final"][0, 0:2] - nominal_final[0, 0:2]).detach().cpu().numpy(),
        "selected_df": best["df"],
        "selected_dl_outward": best["dl_outward"],
        "selected_next": best["best_next"],
        "best_score": best["score"],
        "scores": scored,
        "nominal_final": nominal_final[0].detach().cpu().numpy(),
        "selected_final": best["candidate_final"][0].detach().cpu().numpy(),
    })

    if args.b17_foothold_print and (step % max(args.print_every, 1) == 0):
        compact = []
        for r in scored:
            bn = r.get("best_next") or {}
            compact.append({
                "df": r["df"],
                "dl": r["dl_outward"],
                "score": round(r["score"], 5),
                "next": bn.get("leg"),
                "fm": round(float(bn.get("future_margin", 0.0)), 5) if bn else None,
                "terr": round(float(bn.get("target_err", 0.0)), 5) if bn else None,
            })
        print("[B25 FOOTHOLD MAPPER]",
              "step=", step,
              "swing=", ID_TO_LEG[int(swing_leg)],
              "selected_delta=", info["selected_delta"],
              "selected_next=", info["selected_next"],
              "best_score=", info["best_score"],
              "scores=", compact)
    return out, info



def _b40_parse_leg_list(s):
    out = []
    for item in str(s).split(','):
        name = item.strip().upper()
        if name in LEG_TO_ID:
            out.append(name)
    return out


def _b40_clamp_committed_target_to_anchor(leg, target, anchor, step=-1, source=""):
    """Clamp a committed foothold target relative to the swing anchor.

    The failure mode in B39 was a hind-leg committed target with a very large
    lateral jump (about 22 cm).  B40 keeps the same high-level target pipeline,
    but adds a safety filter before the target is consumed by B23/B37/B20/B21.
    """
    if not bool(getattr(args, "enable_b40_hind_target_clamp", False)):
        return target, {"active": False, "reason": "disabled"}
    leg_name = ID_TO_LEG.get(int(leg), str(leg))
    allowed = _b40_parse_leg_list(getattr(args, "b40_clamp_legs", "LH,RH"))
    if allowed and leg_name not in allowed:
        return target, {"active": False, "reason": "leg_not_enabled", "leg": leg_name}
    if target is None or anchor is None:
        return target, {"active": False, "reason": "missing_target_or_anchor", "leg": leg_name}

    t = target.detach().clone()
    a = anchor.detach().clone()
    delta = t - a
    before = delta.detach().clone()

    # Clamp in world x/y. For the current flat/no-yaw diagnostic, this is exactly
    # forward/lateral. If yawed trials are used later, replace this with base-yaw frame.
    dx = float(args.b40_max_forward_from_anchor)
    dy = float(args.b40_max_lateral_from_anchor)
    t[0] = a[0] + torch.clamp(delta[0], min=-dx, max=dx)
    t[1] = a[1] + torch.clamp(delta[1], min=-dy, max=dy)

    # Optional total xy radius clamp after axis-wise clamp.
    max_xy = float(args.b40_max_xy_from_anchor)
    dxy = t[0:2] - a[0:2]
    nxy = torch.linalg.norm(dxy).clamp_min(1e-8)
    if max_xy > 0.0 and float(nxy.detach().cpu()) > max_xy:
        t[0:2] = a[0:2] + dxy / nxy * max_xy

    if bool(args.b40_clamp_z):
        dz = float(args.b40_max_z_from_anchor)
        t[2] = a[2] + torch.clamp(delta[2], min=-dz, max=dz)

    after = (t - a).detach().clone()
    changed = bool(torch.linalg.norm(after - before).detach().cpu() > 1e-8)
    info = {
        "active": changed,
        "reason": "clamped" if changed else "within_limits",
        "leg": leg_name,
        "source": source,
        "target_before": target.detach().cpu().numpy(),
        "target_after": t.detach().cpu().numpy(),
        "anchor": a.detach().cpu().numpy(),
        "delta_before": before.detach().cpu().numpy(),
        "delta_after": after.detach().cpu().numpy(),
    }
    if changed and bool(args.b40_print):
        print("[B40 HIND TARGET CLAMP]",
              "step=", step,
              "source=", source,
              "leg=", leg_name,
              "delta_before=", before.detach().cpu().numpy(),
              "delta_after=", after.detach().cpu().numpy(),
              "target_before=", target.detach().cpu().numpy(),
              "target_after=", t.detach().cpu().numpy())
    return t, info




def _b41_parse_leg_list(s):
    out = []
    for item in str(s).split(','):
        name = item.strip().upper()
        if name in LEG_TO_ID:
            out.append(name)
    return out


def _b41_clamp_committed_target_to_anchor(leg, target, anchor, step=-1, source=""):
    """B41 clamp: same safety idea as B40, but used at commit time.

    The important difference from B40 is timing. B40 proved that the clamp value
    is useful, but it was applied late in capture. B41 clamps the committed target
    immediately when it is created, using the true swing anchor, so all downstream
    controllers consume the bounded target during active swing.
    """
    if not bool(getattr(args, "enable_b41_swing_phase_hind_target_clamp", False)):
        return target, {"active": False, "reason": "disabled"}
    leg_name = ID_TO_LEG.get(int(leg), str(leg))
    allowed = _b41_parse_leg_list(getattr(args, "b41_clamp_legs", "LH,RH"))
    if allowed and leg_name not in allowed:
        return target, {"active": False, "reason": "leg_not_enabled", "leg": leg_name}
    if target is None or anchor is None:
        return target, {"active": False, "reason": "missing_target_or_anchor", "leg": leg_name}

    t = target.detach().clone()
    a = anchor.detach().clone()
    delta = t - a
    before = delta.detach().clone()

    dx = float(args.b41_max_forward_from_anchor)
    dy = float(args.b41_max_lateral_from_anchor)
    t[0] = a[0] + torch.clamp(delta[0], min=-dx, max=dx)
    t[1] = a[1] + torch.clamp(delta[1], min=-dy, max=dy)

    max_xy = float(args.b41_max_xy_from_anchor)
    dxy = t[0:2] - a[0:2]
    nxy = torch.linalg.norm(dxy).clamp_min(1e-8)
    if max_xy > 0.0 and float(nxy.detach().cpu()) > max_xy:
        t[0:2] = a[0:2] + dxy / nxy * max_xy

    if bool(args.b41_clamp_z):
        dz = float(args.b41_max_z_from_anchor)
        t[2] = a[2] + torch.clamp(delta[2], min=-dz, max=dz)

    after = (t - a).detach().clone()
    changed = bool(torch.linalg.norm(after - before).detach().cpu() > 1e-8)
    info = {
        "active": changed,
        "reason": "clamped" if changed else "within_limits",
        "leg": leg_name,
        "source": source,
        "target_before": target.detach().cpu().numpy(),
        "target_after": t.detach().cpu().numpy(),
        "anchor": a.detach().cpu().numpy(),
        "delta_before": before.detach().cpu().numpy(),
        "delta_after": after.detach().cpu().numpy(),
    }
    if bool(args.b41_print):
        print("[B41 SWING-PHASE HIND TARGET CLAMP]",
              "step=", step,
              "source=", source,
              "leg=", leg_name,
              "reason=", info["reason"],
              "delta_before=", before.detach().cpu().numpy(),
              "delta_after=", after.detach().cpu().numpy(),
              "target_before=", target.detach().cpu().numpy(),
              "target_after=", t.detach().cpu().numpy())
    return t, info

def _b18_commit_or_reuse_foothold_target(canonical_foot_target_w, canonical_foot_anchor_w, b17_info, canonical_s_t, canonical_sigma_t,
                                         committed_targets, committed_valid, committed_step, step):
    info = {"active": False, "reason": "disabled", "leg": None, "committed_target": None, "selected_delta": None}
    if not args.enable_b18_committed_foothold_pinning:
        return canonical_foot_target_w, committed_targets, committed_valid, committed_step, info

    active = [int(i) for i in torch.nonzero(canonical_s_t[0] < 0.5, as_tuple=False).flatten()]
    if len(active) != 1:
        info["reason"] = "not_single_swing"
        return canonical_foot_target_w, committed_targets, committed_valid, committed_step, info

    leg = int(active[0])
    sigma = float(canonical_sigma_t[0, leg].detach().cpu())
    info["leg"] = ID_TO_LEG.get(leg, str(leg))

    if (not bool(committed_valid[leg])) and isinstance(b17_info, dict) and b17_info.get("active", False) and sigma >= float(args.b18_commit_min_sigma):
        try:
            selected_final = torch.as_tensor(b17_info["selected_final"], device=canonical_foot_target_w.device, dtype=canonical_foot_target_w.dtype).view(3)
            committed_targets[leg, :] = selected_final
            info["reason"] = "new_commit_from_b17"
        except Exception:
            committed_targets[leg, :] = canonical_foot_target_w[0, leg, :].detach()
            info["reason"] = "new_commit_from_current_target"
        # B41 safety filter: clamp immediately at commit time using the true swing anchor.
        # This is the key fix over B40, which clamped too late in capture.
        if bool(getattr(args, "enable_b41_swing_phase_hind_target_clamp", False)) and canonical_foot_anchor_w is not None:
            try:
                anchor_for_clamp = canonical_foot_anchor_w[0, leg, :].detach().clone()
                clamped_t, b41_info = _b41_clamp_committed_target_to_anchor(leg, committed_targets[leg, :], anchor_for_clamp, step=step, source="b18_new_commit_pre_swing")
                committed_targets[leg, :] = clamped_t
                info["b41_clamp"] = b41_info
            except Exception as _e:
                info["b41_clamp_error"] = str(_e)
        # Keep B40 fallback available, but B41 should normally supersede it.
        elif bool(getattr(args, "enable_b40_hind_target_clamp", False)) and canonical_foot_target_w is not None:
            try:
                anchor_for_clamp = canonical_foot_target_w[0, leg, :].detach().clone()
                clamped_t, b40_info = _b40_clamp_committed_target_to_anchor(leg, committed_targets[leg, :], anchor_for_clamp, step=step, source="b18_new_commit")
                committed_targets[leg, :] = clamped_t
                info["b40_clamp"] = b40_info
            except Exception as _e:
                info["b40_clamp_error"] = str(_e)
        committed_valid[leg] = True
        committed_step[leg] = int(step)
        info["active"] = True

    if bool(committed_valid[leg]):
        out = canonical_foot_target_w.detach().clone()
        sig = canonical_sigma_t[:, leg].clamp(0.0, 1.0).view(-1, 1)
        out[:, leg, 0:2] = (1.0 - sig) * out[:, leg, 0:2] + sig * committed_targets[leg, 0:2].view(1, 2)
        out[:, leg, 2] = (1.0 - float(args.b18_pin_target_scale_z)) * out[:, leg, 2] + float(args.b18_pin_target_scale_z) * committed_targets[leg, 2]
        # A true "active commit" should mean a new commit was created this step.
        # Reusing an existing committed target is useful state, but should not spam commit logs.
        _is_new_commit = info["reason"].startswith("new_commit")
        info.update({
            "active": bool(_is_new_commit),
            "reuse": bool(not _is_new_commit),
            "reason": info["reason"] if info["reason"] != "disabled" else "using_committed_target",
            "committed_target": committed_targets[leg, :].detach().cpu().numpy(),
            "selected_delta": b17_info.get("selected_delta") if isinstance(b17_info, dict) else None,
        })
        return out, committed_targets, committed_valid, committed_step, info

    info["reason"] = "not_committed_yet"
    return canonical_foot_target_w, committed_targets, committed_valid, committed_step, info


def _b18_foothold_pin_target(robot, Jfeet_full, foot_pos, committed_targets, committed_valid, pin_leg):
    q_cur = robot.data.joint_pos
    q_pin = q_cur.clone()
    info = {"active": False, "leg": None, "foot_err": None, "dq_cmd": None, "max_delta": 0.0}
    if not args.enable_b18_committed_foothold_pinning or pin_leg is None:
        return q_pin, info
    leg = int(pin_leg)
    if not (0 <= leg < 4) or not bool(committed_valid[leg]):
        return q_pin, info
    try:
        joint_ids = _b12_get_leg_joint_ids(leg)
        err = committed_targets[leg, :] - foot_pos[0, leg, :]
        err_scaled = err.detach().clone()
        err_scaled[0:2] *= float(args.b18_pin_target_scale_xy)
        err_scaled[2] *= float(args.b18_pin_target_scale_z)
        J_leg = Jfeet_full[0, leg, :, joint_ids]
        if args.b18_pin_include_hx:
            J_task = J_leg
            ids = joint_ids
        else:
            J_task = J_leg[:, 1:3]
            ids = joint_ids[1:3]
        eye = torch.eye(J_task.shape[0], device=J_task.device)
        A = J_task @ J_task.transpose(0, 1) + (float(args.b18_pin_damping) ** 2) * eye
        dq = J_task.transpose(0, 1) @ torch.linalg.solve(A, err_scaled)
        dq = float(args.b18_pin_gain) * dq
        dq = torch.clamp(dq, -float(args.b18_pin_max_joint_delta), float(args.b18_pin_max_joint_delta))
        q_pin[0, ids] = q_cur[0, ids] + dq
        info.update({
            "active": True,
            "leg": ID_TO_LEG.get(leg, str(leg)),
            "foot_err": err.detach().cpu().numpy(),
            "dq_cmd": dq.detach().cpu().numpy(),
            "max_delta": float(torch.max(torch.abs(dq)).detach().cpu()),
        })
    except Exception as e:
        info["reason"] = repr(e)
    return q_pin, info


def _b18_foothold_pin_torque_assist(robot, q_pin_target, pin_info):
    tau = torch.zeros_like(robot.data.joint_pos)
    if not isinstance(pin_info, dict) or not bool(pin_info.get("active", False)):
        return tau
    leg_name = pin_info.get("leg")
    if leg_name not in LEG_TO_ID:
        return tau
    leg = LEG_TO_ID[leg_name]
    q = robot.data.joint_pos
    qd = robot.data.joint_vel
    ids = _b12_get_leg_joint_ids(leg)
    tau_leg = float(args.b18_pin_torque_kp) * (q_pin_target[0, ids] - q[0, ids]) - float(args.b18_pin_torque_kd) * qd[0, ids]
    tau_leg = torch.clamp(tau_leg, -float(args.b18_pin_max_tau), float(args.b18_pin_max_tau))
    tau[0, ids] = tau_leg
    return tau


def _b18_print_post_capture_geometry(step, x_hat, foot_pos, cfg, device, dtype, label=""):
    if not args.b18_post_capture_geometry_print:
        return
    rows = []
    for name in _b17_parse_leg_list(args.b17_foothold_next_candidates):
        try:
            info = _b13_score_candidate_leg(x_hat, foot_pos, LEG_TO_ID[name], cfg, device, dtype)
            rows.append({
                "leg": name,
                "future_margin": round(float(info.get("future_margin", 0.0)), 5),
                "geom_margin": round(float(info.get("geom_margin", 0.0)), 5),
                "target_err": round(float(info.get("target_err", 0.0)), 5),
                "allowed": bool(info.get("future_allowed", False)),
            })
        except Exception:
            pass
    print("[B37 POST-CAPTURE GEOMETRY]", "step=", step, "label=", label, "rows=", rows)


def _b19_apply_forced_exploratory_foothold_bias(canonical_foot_target_w, canonical_foot_anchor_w, base_quat_w,
                                                canonical_s_t, canonical_sigma_t, b17_foothold_info, step):
    info = {"active": False, "reason": "disabled", "leg": None, "forced_delta": None, "forced_final": None}
    if not args.enable_b19_forced_exploratory_foothold_bias:
        return canonical_foot_target_w, b17_foothold_info, info
    if canonical_foot_anchor_w is None or canonical_s_t is None or canonical_sigma_t is None:
        info["reason"] = "missing_canonical_state"
        return canonical_foot_target_w, b17_foothold_info, info
    if (bool(getattr(args, "b48_disable_b19_when_active", False))
            and isinstance(b17_foothold_info, dict)
            and b17_foothold_info.get("active", False)
            and str(b17_foothold_info.get("reason", "")).startswith("b48_")):
        info["reason"] = "b48_active_skip_b19_override"
        return canonical_foot_target_w, b17_foothold_info, info

    active = [int(i) for i in torch.nonzero(canonical_s_t[0] < 0.5, as_tuple=False).flatten()]
    if len(active) != 1:
        info["reason"] = "not_single_swing"
        return canonical_foot_target_w, b17_foothold_info, info

    leg = int(active[0])
    leg_name = ID_TO_LEG.get(leg, str(leg))
    allowed = _b17_parse_leg_list(args.b19_force_bias_legs)
    if allowed and leg_name not in allowed:
        info["reason"] = "leg_not_enabled"
        info["leg"] = leg_name
        return canonical_foot_target_w, b17_foothold_info, info

    sigma = canonical_sigma_t[:, leg].clamp(0.0, 1.0).view(-1, 1)
    if float(sigma[0, 0].detach().cpu()) < float(args.b19_force_min_sigma):
        info["reason"] = "sigma_too_small"
        info["leg"] = leg_name
        return canonical_foot_target_w, b17_foothold_info, info

    nominal_final, fwd, left = _b17_nominal_final_foothold(canonical_foot_anchor_w, base_quat_w, leg)
    side_sign = 1.0 if leg in [0, 2] else -1.0
    extra = float(args.b19_force_forward_bias) * fwd[:, 0:2] + side_sign * float(args.b19_force_outward_bias) * left[:, 0:2]

    norm = torch.linalg.norm(extra, dim=1, keepdim=True).clamp_min(1e-6)
    scale = torch.clamp(float(args.b19_force_max_extra_xy) / norm, max=1.0)
    extra = extra * scale

    forced_final = nominal_final.detach().clone()
    forced_final[:, 0:2] = nominal_final[:, 0:2] + extra

    out = canonical_foot_target_w.detach().clone()
    out[:, leg, 0:2] = (1.0 - sigma) * canonical_foot_anchor_w[:, leg, 0:2] + sigma * forced_final[:, 0:2]

    if not isinstance(b17_foothold_info, dict):
        b17_foothold_info = {}
    if args.b19_force_override_b17_selection or not b17_foothold_info.get("active", False):
        b17_foothold_info = dict(b17_foothold_info)
        b17_foothold_info.update({
            "active": True,
            "reason": "b19_forced_exploratory_bias",
            "swing_leg": leg_name,
            "selected_delta": (forced_final[0, 0:2] - nominal_final[0, 0:2]).detach().cpu().numpy(),
            "selected_final": forced_final[0].detach().cpu().numpy(),
            "selected_next": {"leg": "FORCED_BIAS"},
            "best_score": None,
            "nominal_final": nominal_final[0].detach().cpu().numpy(),
        })

    info.update({
        "active": True,
        "reason": "forced_exploratory_bias",
        "leg": leg_name,
        "forced_delta": (forced_final[0, 0:2] - nominal_final[0, 0:2]).detach().cpu().numpy(),
        "forced_final": forced_final[0].detach().cpu().numpy(),
        "sigma": float(sigma[0, 0].detach().cpu()),
    })

    if args.b19_force_print and (step % max(args.print_every, 1) == 0):
        print("[B19 FORCED FOOTHOLD BIAS]",
              "step=", step,
              "leg=", leg_name,
              "sigma=", info["sigma"],
              "forced_delta=", info["forced_delta"],
              "forced_final=", info["forced_final"],
              "override_b17=", args.b19_force_override_b17_selection)
    return out, b17_foothold_info, info



def _b49_make_elevated_tracking_targets(committed_targets, committed_valid,
                                        canonical_foot_target_w, canonical_foot_anchor_w,
                                        canonical_s_t, canonical_sigma_t, step):
    """Return a clone of B18 committed targets with swing-leg z replaced by elevated swing reference.

    B18 committed targets are touchdown references and should stay at ground height for
    post-landing validation.  But B20/B21 swing execution targets need the time-varying
    swing height; otherwise their z-error is almost zero and they do not help lift.
    """
    info = {"active": False, "reason": "disabled", "leg": None, "sigma": None,
            "old_target": None, "new_target": None, "z_ref": None, "z_min": None}
    if not bool(getattr(args, "enable_b49_elevated_swing_tracking_reference", False)):
        return committed_targets, info
    if committed_targets is None or canonical_s_t is None or canonical_sigma_t is None or canonical_foot_target_w is None:
        info["reason"] = "missing_state"
        return committed_targets, info
    out = committed_targets.detach().clone()
    try:
        active = [int(i) for i in torch.nonzero(canonical_s_t[0] < 0.5, as_tuple=False).flatten()]
        if len(active) != 1:
            info["reason"] = "not_single_swing"
            return out, info
        leg = int(active[0])
        sigma = float(canonical_sigma_t[0, leg].detach().cpu())
        info["leg"] = ID_TO_LEG.get(leg, str(leg))
        info["sigma"] = sigma
        if not bool(committed_valid[leg]):
            info["reason"] = "no_committed_target"
            return out, info
        if sigma < float(getattr(args, "b49_min_sigma", 0.03)) or sigma > float(getattr(args, "b49_max_sigma", 0.98)):
            info["reason"] = "sigma_outside_window"
            return out, info

        # B50 fix: keep all target/valid indexing robust.
        # In B49 this block could throw during active swing, which made
        # [B49 ELEVATED...] print active=False reason=error and prevented
        # the elevated z reference from reaching B20/B21.
        valid_entry = committed_valid[leg]
        if torch.is_tensor(valid_entry):
            valid_entry = bool(valid_entry.detach().item())
        else:
            valid_entry = bool(valid_entry)
        if not valid_entry:
            info["reason"] = "no_committed_target"
            return out, info

        z_ref = canonical_foot_target_w[0, leg, 2]
        if canonical_foot_anchor_w is not None:
            z_anchor = canonical_foot_anchor_w[0, leg, 2]
            z_min_t = z_anchor + z_ref.new_tensor(
                float(getattr(args, "b49_min_clearance", 0.035)) * math.sin(math.pi * sigma)
            )
            z_new = torch.maximum(
                z_ref + z_ref.new_tensor(float(getattr(args, "b49_z_boost", 0.0))),
                z_min_t,
            )
        else:
            z_min_t = z_ref
            z_new = z_ref + z_ref.new_tensor(float(getattr(args, "b49_z_boost", 0.0)))

        old_target = out[leg, :].detach().clone()
        out[leg, 2] = z_new
        new_target = out[leg, :].detach().clone()
        info.update({
            "active": True,
            "reason": "elevated_swing_tracking_target",
            "old_target": old_target.detach().cpu().numpy(),
            "new_target": new_target.detach().cpu().numpy(),
            "z_ref": float(z_ref.detach().cpu()),
            "z_min": float(z_min_t.detach().cpu()) if torch.is_tensor(z_min_t) else float(z_min_t),
        })
    except Exception as e:
        info["reason"] = "error"
        info["error"] = repr(e)
    return out, info


def _b20_committed_direct_swing_ik_target(robot, Jfeet_full, foot_pos, canonical_s_t, canonical_sigma_t,
                                          committed_targets, committed_valid):
    q_cur = robot.data.joint_pos
    q_target = q_cur.clone()
    info = {"active": False, "reason": "disabled", "leg": None, "sigma": None,
            "foot_err": None, "dq_cmd": None, "max_delta": 0.0, "committed_target": None}
    if not args.enable_b20_committed_foothold_direct_swing_ik:
        return q_target, info
    if canonical_s_t is None or canonical_sigma_t is None:
        info["reason"] = "missing_canonical_state"
        return q_target, info
    active = [int(i) for i in torch.nonzero(canonical_s_t[0] < 0.5, as_tuple=False).flatten()]
    if len(active) != 1:
        info["reason"] = "not_single_swing"
        return q_target, info

    leg = int(active[0])
    sigma = float(canonical_sigma_t[0, leg].detach().cpu())
    info["leg"] = ID_TO_LEG.get(leg, str(leg))
    info["sigma"] = sigma
    if not bool(committed_valid[leg]):
        info["reason"] = "no_committed_target"
        return q_target, info
    if sigma < float(args.b20_direct_ik_start_sigma) or sigma > float(args.b20_direct_ik_end_sigma):
        info["reason"] = "sigma_outside_window"
        return q_target, info

    try:
        joint_ids = _b12_get_leg_joint_ids(leg)
        err = committed_targets[leg, :] - foot_pos[0, leg, :]
        err_scaled = err.detach().clone()
        err_scaled[0:2] *= float(args.b20_direct_ik_xy_scale)
        err_scaled[2] *= float(args.b20_direct_ik_z_scale)

        J_leg = Jfeet_full[0, leg, :, joint_ids]
        if args.b20_direct_ik_include_hx:
            J_task = J_leg
            ids = joint_ids
        else:
            J_task = J_leg[:, 1:3]
            ids = joint_ids[1:3]

        eye = torch.eye(J_task.shape[0], device=J_task.device)
        A = J_task @ J_task.transpose(0, 1) + (float(args.b20_direct_ik_damping) ** 2) * eye
        dq = J_task.transpose(0, 1) @ torch.linalg.solve(A, err_scaled)
        dq = float(args.b20_direct_ik_gain) * dq
        dq = torch.clamp(dq, -float(args.b20_direct_ik_max_joint_delta), float(args.b20_direct_ik_max_joint_delta))
        q_target[0, ids] = q_cur[0, ids] + dq
        info.update({"active": True, "reason": "direct_committed_swing_ik",
                     "foot_err": err.detach().cpu().numpy(),
                     "dq_cmd": dq.detach().cpu().numpy(),
                     "max_delta": float(torch.max(torch.abs(dq)).detach().cpu()),
                     "committed_target": committed_targets[leg, :].detach().cpu().numpy()})
    except Exception as e:
        info["reason"] = "error"
        info["error"] = repr(e)
    return q_target, info


def _b20_direct_swing_ik_torque_assist(robot, q_target, ik_info):
    tau = torch.zeros_like(robot.data.joint_pos)
    if not isinstance(ik_info, dict) or not bool(ik_info.get("active", False)):
        return tau
    leg_name = ik_info.get("leg")
    if leg_name not in LEG_TO_ID:
        return tau
    ids = _b12_get_leg_joint_ids(LEG_TO_ID[leg_name])
    q = robot.data.joint_pos
    qd = robot.data.joint_vel
    tau_leg = float(args.b20_direct_ik_torque_kp) * (q_target[0, ids] - q[0, ids]) - float(args.b20_direct_ik_torque_kd) * qd[0, ids]
    tau_leg = torch.clamp(tau_leg, -float(args.b20_direct_ik_max_tau), float(args.b20_direct_ik_max_tau))
    tau[0, ids] = tau_leg
    return tau


def _b21_select_committed_tracking_leg(canonical_s_t, canonical_sigma_t,
                                       canonical_capture_leg, canonical_capture_count,
                                       b18_pin_extra_count_by_leg, committed_valid):
    """Select a leg for Cartesian committed foothold tracking.

    Priority:
      1) active executed swing from canonical_s_t
      2) capture leg
      3) post-capture extra pin leg
    """
    info = {"leg": None, "leg_name": None, "reason": "none", "sigma": None}

    if args.b21_cart_apply_during_swing and canonical_s_t is not None:
        try:
            active = [int(i) for i in torch.nonzero(canonical_s_t[0] < 0.5, as_tuple=False).flatten()]
            if len(active) == 1:
                leg = int(active[0])
                sigma = float(canonical_sigma_t[0, leg].detach().cpu()) if canonical_sigma_t is not None else 0.5
                if bool(committed_valid[leg]) and sigma >= float(args.b21_cart_start_sigma) and sigma <= float(args.b21_cart_end_sigma):
                    info.update({"leg": leg, "leg_name": ID_TO_LEG.get(leg, str(leg)), "reason": "active_swing", "sigma": sigma})
                    return info
        except Exception:
            pass

    if args.b21_cart_apply_during_capture and canonical_capture_count is not None and int(canonical_capture_count) > 0 and canonical_capture_leg is not None:
        leg = int(canonical_capture_leg)
        if 0 <= leg < 4 and bool(committed_valid[leg]):
            info.update({"leg": leg, "leg_name": ID_TO_LEG.get(leg, str(leg)), "reason": "capture", "sigma": None})
            return info

    if args.b21_cart_apply_post_capture_extra and isinstance(b18_pin_extra_count_by_leg, dict):
        for leg, cnt in b18_pin_extra_count_by_leg.items():
            leg = int(leg)
            if int(cnt) > 0 and 0 <= leg < 4 and bool(committed_valid[leg]):
                info.update({"leg": leg, "leg_name": ID_TO_LEG.get(leg, str(leg)), "reason": "post_capture_extra", "sigma": None})
                return info

    info["reason"] = "no_valid_committed_leg"
    return info


def _b21_cartesian_committed_swing_torque_assist(robot, Jfeet_full, foot_pos, foot_vel,
                                                 canonical_s_t, canonical_sigma_t,
                                                 canonical_capture_leg, canonical_capture_count,
                                                 b18_pin_extra_count_by_leg,
                                                 committed_targets, committed_valid):
    tau = torch.zeros_like(robot.data.joint_pos)
    info = {"active": False, "reason": "disabled", "leg": None, "foot_err": None, "foot_vel": None,
            "f_cmd": None, "tau_leg": None, "max_tau": 0.0, "sigma": None}
    if not args.enable_b21_cartesian_committed_swing_torque:
        return tau, info

    sel = _b21_select_committed_tracking_leg(
        canonical_s_t, canonical_sigma_t,
        canonical_capture_leg, canonical_capture_count,
        b18_pin_extra_count_by_leg, committed_valid
    )
    if sel.get("leg") is None:
        info["reason"] = sel.get("reason")
        return tau, info

    leg = int(sel["leg"])
    try:
        joint_ids = _b12_get_leg_joint_ids(leg)
        err = committed_targets[leg, :] - foot_pos[0, leg, :]
        vel = foot_vel[0, leg, :] if foot_vel.ndim == 3 else foot_vel[leg, :]

        kp = torch.tensor([float(args.b21_cart_kp_xy), float(args.b21_cart_kp_xy), float(args.b21_cart_kp_z)],
                          device=foot_pos.device, dtype=foot_pos.dtype)
        kd = torch.tensor([float(args.b21_cart_kd_xy), float(args.b21_cart_kd_xy), float(args.b21_cart_kd_z)],
                          device=foot_pos.device, dtype=foot_pos.dtype)
        f_cmd = kp * err - kd * vel
        f_cmd = float(args.b21_cart_force_sign) * f_cmd

        J_leg = Jfeet_full[0, leg, :, joint_ids]  # 3 x 3
        tau_leg = J_leg.transpose(0, 1) @ f_cmd
        tau_leg = torch.clamp(tau_leg, -float(args.b21_cart_max_tau), float(args.b21_cart_max_tau))
        tau[0, joint_ids] = tau_leg

        info.update({
            "active": True,
            "reason": sel.get("reason"),
            "leg": ID_TO_LEG.get(leg, str(leg)),
            "sigma": sel.get("sigma"),
            "foot_err": err.detach().cpu().numpy(),
            "foot_vel": vel.detach().cpu().numpy(),
            "f_cmd": f_cmd.detach().cpu().numpy(),
            "tau_leg": tau_leg.detach().cpu().numpy(),
            "max_tau": float(torch.max(torch.abs(tau_leg)).detach().cpu()),
        })
    except Exception as e:
        info["reason"] = "error"
        info["error"] = repr(e)
    return tau, info


def _b22_ramp_alpha(step):
    if not args.b22_use_ramp:
        return 1.0
    if step <= int(args.b22_ramp_start_step):
        return 0.0
    denom = max(int(args.b22_ramp_steps), 1)
    return float(max(0.0, min(1.0, (step - int(args.b22_ramp_start_step)) / denom)))


def _b22_apply_authority_mode(tau_wbc_component, tau_pre_b22, step):
    """Return calibrated tau and an info dictionary.

    tau_wbc_component: WBC/QP torque after args.tau_cmd_scale.
    tau_pre_b22: WBC plus all explicit assists before B22 override.
    The implicit actuator PD, if any, is not in either tensor; it appears later as applied_torque - tau_cmd.
    """
    info = {
        "active": False,
        "mode": args.b22_authority_mode,
        "alpha": _b22_ramp_alpha(step),
        "wbc_norm": 0.0,
        "assist_norm": 0.0,
        "pre_norm": 0.0,
        "out_norm": 0.0,
        "selected_leg": args.b22_print_selected_leg,
    }
    if not args.enable_b22_wbc_authority_calibration:
        return tau_pre_b22, info

    alpha = float(info["alpha"])
    tau_assist_component = tau_pre_b22 - tau_wbc_component

    mode = str(args.b22_authority_mode)
    if mode == "normal":
        tau_out = tau_pre_b22
    elif mode == "zero_action":
        tau_out = torch.zeros_like(tau_pre_b22)
    elif mode == "wbc_only":
        tau_out = alpha * float(args.b22_wbc_scale) * tau_wbc_component
    elif mode == "assist_only":
        tau_out = alpha * float(args.b22_assist_scale) * tau_assist_component
    elif mode == "wbc_plus_assist_scaled":
        tau_out = alpha * (
            float(args.b22_wbc_scale) * tau_wbc_component
            + float(args.b22_assist_scale) * tau_assist_component
        )
    else:
        tau_out = tau_pre_b22

    tau_out = torch.clamp(tau_out, -float(args.max_tau), float(args.max_tau))
    info.update({
        "active": True,
        "wbc_norm": float(torch.linalg.norm(tau_wbc_component).detach().cpu()),
        "assist_norm": float(torch.linalg.norm(tau_assist_component).detach().cpu()),
        "pre_norm": float(torch.linalg.norm(tau_pre_b22).detach().cpu()),
        "out_norm": float(torch.linalg.norm(tau_out).detach().cpu()),
        "wbc_max": float(torch.max(torch.abs(tau_wbc_component)).detach().cpu()),
        "assist_max": float(torch.max(torch.abs(tau_assist_component)).detach().cpu()),
        "pre_max": float(torch.max(torch.abs(tau_pre_b22)).detach().cpu()),
        "out_max": float(torch.max(torch.abs(tau_out)).detach().cpu()),
    })
    return tau_out, info


def _b22_print_authority(step, tau_wbc_component, tau_pre_b22, tau_out, info, robot):
    if not (args.enable_b22_wbc_authority_calibration and args.b22_print and (step % max(args.print_every, 1) == 0)):
        return
    leg_name = str(args.b22_print_selected_leg)
    leg = LEG_TO_ID.get(leg_name, LEG_TO_ID.get(args.test_leg, 1))
    ids = _b12_get_leg_joint_ids(leg)
    tau_assist_component = tau_pre_b22 - tau_wbc_component
    applied_prev = robot.data.applied_torque
    print("[B22 AUTHORITY]",
          "step=", step,
          "mode=", info.get("mode"),
          "alpha=", info.get("alpha"),
          "leg=", ID_TO_LEG.get(leg, leg),
          "norms(wbc,assist,pre,out)=",
          (info.get("wbc_norm"), info.get("assist_norm"), info.get("pre_norm"), info.get("out_norm")),
          "max(wbc,assist,pre,out)=",
          (info.get("wbc_max"), info.get("assist_max"), info.get("pre_max"), info.get("out_max")),
          "tau_wbc_leg=", tau_wbc_component[0, ids].detach().cpu().numpy(),
          "tau_assist_leg=", tau_assist_component[0, ids].detach().cpu().numpy(),
          "tau_pre_leg=", tau_pre_b22[0, ids].detach().cpu().numpy(),
          "tau_out_leg=", tau_out[0, ids].detach().cpu().numpy(),
          "applied_prev_leg=", applied_prev[0, ids].detach().cpu().numpy(),
          "applied_prev_minus_out_leg=", (applied_prev[0, ids] - tau_out[0, ids]).detach().cpu().numpy())


def _b23_select_committed_footref_leg(canonical_s_t, canonical_sigma_t,
                                      canonical_capture_leg, canonical_capture_count,
                                      b18_pin_extra_count_by_leg, committed_valid):
    info = {"leg": None, "leg_name": None, "reason": "none", "sigma": None}
    if not args.enable_b23_wbc_main_committed_footref:
        info["reason"] = "disabled"
        return info

    if args.b23_apply_during_swing and canonical_s_t is not None:
        try:
            active = [int(i) for i in torch.nonzero(canonical_s_t[0] < 0.5, as_tuple=False).flatten()]
            if len(active) == 1:
                leg = int(active[0])
                sigma = float(canonical_sigma_t[0, leg].detach().cpu()) if canonical_sigma_t is not None else 0.5
                if bool(committed_valid[leg]) and sigma >= float(args.b23_start_sigma) and sigma <= float(args.b23_end_sigma):
                    info.update({"leg": leg, "leg_name": ID_TO_LEG.get(leg, str(leg)), "reason": "active_swing", "sigma": sigma})
                    return info
        except Exception:
            pass

    if args.b23_apply_during_capture and canonical_capture_count is not None and int(canonical_capture_count) > 0 and canonical_capture_leg is not None:
        leg = int(canonical_capture_leg)
        if 0 <= leg < 4 and bool(committed_valid[leg]):
            info.update({"leg": leg, "leg_name": ID_TO_LEG.get(leg, str(leg)), "reason": "capture", "sigma": None})
            return info

    if args.b23_apply_post_capture_extra and isinstance(b18_pin_extra_count_by_leg, dict):
        for leg, cnt in b18_pin_extra_count_by_leg.items():
            leg = int(leg)
            if int(cnt) > 0 and 0 <= leg < 4 and bool(committed_valid[leg]):
                info.update({"leg": leg, "leg_name": ID_TO_LEG.get(leg, str(leg)), "reason": "post_capture_extra", "sigma": None})
                return info

    info["reason"] = "no_valid_committed_leg"
    return info


def _b23_apply_wbc_main_committed_footref(canonical_foot_target_w, canonical_foot_anchor_w,
                                          canonical_s_t, canonical_sigma_t,
                                          canonical_capture_leg, canonical_capture_count,
                                          b18_pin_extra_count_by_leg,
                                          committed_targets, committed_valid, step):
    info = {"active": False, "reason": "disabled", "leg": None,
            "target_before": None, "target_after": None, "committed": None,
            "anchor": None, "sigma": None, "xy_mode": None}
    if not args.enable_b23_wbc_main_committed_footref:
        return canonical_foot_target_w, info
    if canonical_foot_target_w is None:
        info["reason"] = "no_canonical_target"
        return canonical_foot_target_w, info

    sel = _b23_select_committed_footref_leg(
        canonical_s_t, canonical_sigma_t,
        canonical_capture_leg, canonical_capture_count,
        b18_pin_extra_count_by_leg, committed_valid
    )
    if sel.get("leg") is None:
        info["reason"] = sel.get("reason")
        return canonical_foot_target_w, info

    leg = int(sel["leg"])
    out = canonical_foot_target_w.detach().clone()
    before = out[0, leg, :].detach().clone()
    committed = committed_targets[leg, :].detach().clone()
    b40_info = {"active": False, "reason": "not_applied"}
    b41_info = {"active": False, "reason": "not_applied"}
    if bool(getattr(args, "enable_b41_swing_phase_hind_target_clamp", False)) and canonical_foot_anchor_w is not None:
        try:
            anchor_for_clamp = canonical_foot_anchor_w[0, leg, :].detach().clone()
            committed, b41_info = _b41_clamp_committed_target_to_anchor(leg, committed, anchor_for_clamp, step=step, source="b23_main_footref_active_swing")
            committed_targets[leg, :] = committed
        except Exception as _e:
            b41_info = {"active": False, "reason": "error", "error": str(_e)}
    elif bool(getattr(args, "enable_b40_hind_target_clamp", False)) and canonical_foot_anchor_w is not None:
        try:
            anchor_for_clamp = canonical_foot_anchor_w[0, leg, :].detach().clone()
            committed, b40_info = _b40_clamp_committed_target_to_anchor(leg, committed, anchor_for_clamp, step=step, source="b23_main_footref")
            committed_targets[leg, :] = committed
        except Exception as _e:
            b40_info = {"active": False, "reason": "error", "error": str(_e)}

    if args.b23_xy_mode == "sigma_blend" and canonical_foot_anchor_w is not None and canonical_sigma_t is not None:
        sig = float(canonical_sigma_t[0, leg].detach().cpu())
        lo = float(args.b23_start_sigma)
        hi = max(float(args.b23_end_sigma), lo + 1e-6)
        alpha = max(0.0, min(1.0, (sig - lo) / (hi - lo)))
        anchor = canonical_foot_anchor_w[0, leg, :].detach().clone()
        out[:, leg, 0:2] = (1.0 - alpha) * canonical_foot_anchor_w[:, leg, 0:2] + alpha * committed[0:2].view(1, 2)
    else:
        anchor = canonical_foot_anchor_w[0, leg, :].detach().clone() if canonical_foot_anchor_w is not None else torch.zeros_like(committed)
        out[:, leg, 0:2] = committed[0:2].view(1, 2)

    if not args.b23_preserve_swing_z:
        out[:, leg, 2] = committed[2]

    after = out[0, leg, :].detach().clone()
    info.update({
        "active": True,
        "reason": sel.get("reason"),
        "leg": ID_TO_LEG.get(leg, str(leg)),
        "sigma": sel.get("sigma"),
        "xy_mode": args.b23_xy_mode,
        "target_before": before.detach().cpu().numpy(),
        "target_after": after.detach().cpu().numpy(),
        "committed": committed.detach().cpu().numpy(),
        "anchor": anchor.detach().cpu().numpy(),
        "target_minus_anchor": (after - anchor).detach().cpu().numpy(),
        "b40_clamp": b40_info,
        "b41_clamp": b41_info,
    })
    return out, info


def _b25_apply_canonical_swing_pd_bridge(robot, b20_target, b20_info, b18_target, b18_info, step):
    """Set Isaac joint position targets for the committed swing leg.

    This is intentionally different from q_nom_active. q_nom_active shapes WBC posture.
    This bridge directly updates the simulator's implicit PD target so PD cooperates
    with WBC instead of holding the old stance.
    """
    info = {"active": False, "reason": "disabled", "source": None, "leg": None,
            "joint_ids": None, "q_now": None, "q_goal_raw": None, "q_goal": None,
            "delta": None}
    if not args.enable_b25_canonical_swing_pd_bridge:
        return info

    src = None
    tgt = None
    leg_name = None

    if args.b25_use_b20_direct_ik_target and isinstance(b20_info, dict) and b20_info.get("active", False) and b20_target is not None:
        src = "b20_direct_ik"
        tgt = b20_target
        leg_name = b20_info.get("leg")
    elif args.b25_use_b18_pin_target and isinstance(b18_info, dict) and b18_info.get("active", False) and b18_target is not None:
        src = "b18_pin"
        tgt = b18_target
        leg_name = b18_info.get("leg")

    if tgt is None or leg_name not in LEG_TO_ID:
        info["reason"] = "no_active_target"
        return info

    leg = LEG_TO_ID[leg_name]
    ids_list = _b12_get_leg_joint_ids(leg)
    q_now_full = robot.data.joint_pos.detach()
    q_goal_full = tgt.detach().clone()

    q_now = q_now_full[:, ids_list]
    q_goal_raw = q_goal_full[:, ids_list]
    delta = torch.clamp(q_goal_raw - q_now, -float(args.b25_max_joint_delta), float(args.b25_max_joint_delta))
    q_goal = q_now + delta

    joint_ids_tensor = torch.tensor(ids_list, device=q_now_full.device, dtype=torch.long)
    try:
        robot.set_joint_position_target(q_goal, joint_ids=joint_ids_tensor)
        if args.b25_zero_target_velocity:
            robot.set_joint_velocity_target(torch.zeros_like(q_goal), joint_ids=joint_ids_tensor)
        info.update({
            "active": True,
            "reason": "pd_target_set",
            "source": src,
            "leg": leg_name,
            "joint_ids": ids_list,
            "q_now": q_now[0].detach().cpu().numpy(),
            "q_goal_raw": q_goal_raw[0].detach().cpu().numpy(),
            "q_goal": q_goal[0].detach().cpu().numpy(),
            "delta": delta[0].detach().cpu().numpy(),
        })
    except Exception as e:
        info["reason"] = "set_target_error"
        info["error"] = repr(e)
    return info


# B26 latched bridge state. One persistent target prevents per-step ratcheting.
_B26_BRIDGE_STATE = {"active": False, "leg_name": None, "source": None,
                     "q_latch": None, "q_goal": None,
                     "step_latch": -1, "last_retarget_step": -1}


def _b26_pick_bridge_target(b20_target, b20_info, b18_target, b18_info):
    src = None; tgt = None; leg_name = None
    if args.b26_use_b20_direct_ik_target and isinstance(b20_info, dict) and b20_info.get("active", False) and b20_target is not None:
        src = "b20_direct_ik"; tgt = b20_target; leg_name = b20_info.get("leg")
    elif args.b26_use_b18_pin_target and isinstance(b18_info, dict) and b18_info.get("active", False) and b18_target is not None:
        src = "b18_pin"; tgt = b18_target; leg_name = b18_info.get("leg")
    return src, tgt, leg_name


def _b26_apply_latched_safe_swing_pd_bridge(robot, b20_target, b20_info, b18_target, b18_info, step):
    info = {"active": False, "reason": "disabled", "source": None, "leg": None,
            "joint_ids": None, "q_now": None, "q_latch": None, "q_goal_raw": None,
            "q_goal": None, "q_cmd": None, "delta_from_now": None, "delta_from_latch": None}
    if not args.enable_b26_latched_safe_swing_pd_bridge:
        return info
    src, tgt, leg_name = _b26_pick_bridge_target(b20_target, b20_info, b18_target, b18_info)
    if tgt is None or leg_name not in LEG_TO_ID:
        if args.b26_reset_when_no_target:
            _B26_BRIDGE_STATE.update({"active": False, "leg_name": None, "source": None,
                                      "q_latch": None, "q_goal": None,
                                      "step_latch": -1, "last_retarget_step": -1})
        info["reason"] = "no_active_target"
        return info
    leg = LEG_TO_ID[leg_name]
    ids_list = _b12_get_leg_joint_ids(leg)
    ids = torch.tensor(ids_list, device=robot.data.joint_pos.device, dtype=torch.long)
    q_now_full = robot.data.joint_pos.detach()
    q_now = q_now_full[:, ids_list]
    q_raw = tgt.detach()[:, ids_list]
    need_new_latch = ((not _B26_BRIDGE_STATE["active"]) or
                      (_B26_BRIDGE_STATE["leg_name"] != leg_name) or
                      (_B26_BRIDGE_STATE["source"] != src) or
                      (step - int(_B26_BRIDGE_STATE.get("last_retarget_step", -999999)) >= int(args.b26_retarget_interval)))
    if need_new_latch:
        q_latch = q_now.detach().clone()
        delta_from_latch = torch.clamp(q_raw - q_latch, -float(args.b26_max_abs_from_latch), float(args.b26_max_abs_from_latch))
        q_goal = q_latch + delta_from_latch
        _B26_BRIDGE_STATE.update({"active": True, "leg_name": leg_name, "source": src,
                                  "q_latch": q_latch.detach().clone(), "q_goal": q_goal.detach().clone(),
                                  "step_latch": int(step), "last_retarget_step": int(step)})
    else:
        q_latch = _B26_BRIDGE_STATE["q_latch"].detach().clone()
        q_goal = _B26_BRIDGE_STATE["q_goal"].detach().clone()
        delta_from_latch = q_goal - q_latch
    delta_from_now = torch.clamp(q_goal - q_now, -float(args.b26_max_joint_delta), float(args.b26_max_joint_delta))
    q_cmd = q_now + delta_from_now
    try:
        robot.set_joint_position_target(q_cmd, joint_ids=ids)
        if args.b26_zero_target_velocity:
            robot.set_joint_velocity_target(torch.zeros_like(q_cmd), joint_ids=ids)
        info.update({"active": True, "reason": "pd_target_set", "source": src, "leg": leg_name,
                     "joint_ids": ids_list, "latched_new": bool(need_new_latch),
                     "q_now": q_now[0].detach().cpu().numpy(),
                     "q_latch": q_latch[0].detach().cpu().numpy(),
                     "q_goal_raw": q_raw[0].detach().cpu().numpy(),
                     "q_goal": q_goal[0].detach().cpu().numpy(),
                     "q_cmd": q_cmd[0].detach().cpu().numpy(),
                     "delta_from_now": delta_from_now[0].detach().cpu().numpy(),
                     "delta_from_latch": delta_from_latch[0].detach().cpu().numpy()})
    except Exception as e:
        info["reason"] = "set_target_error"; info["error"] = repr(e)
    return info


_B27_BRIDGE_STATE = {
    "active": False,
    "leg_name": None,
    "source": None,
    "q_latch": None,
    "q_goal": None,
    "step_latch": -1,
    "last_retarget_step": -1,
}


def _b27_apply_signed_latched_swing_pd_bridge(robot, b20_target, b20_info, step):
    info = {"active": False, "reason": "disabled", "source": None, "leg": None,
            "joint_ids": None, "q_now": None, "q_latch": None, "q_goal_raw": None,
            "q_goal": None, "q_cmd": None, "raw_delta": None, "signed_delta": None,
            "delta_from_now": None}
    if not args.enable_b27_signed_latched_swing_pd_bridge:
        return info
    if not (args.b27_use_b20_direct_ik_target and isinstance(b20_info, dict)
            and b20_info.get("active", False) and b20_target is not None):
        if args.b27_reset_when_no_target:
            _B27_BRIDGE_STATE.update({"active": False, "leg_name": None, "source": None,
                                      "q_latch": None, "q_goal": None,
                                      "step_latch": -1, "last_retarget_step": -1})
        info["reason"] = "no_active_b20_target"
        return info

    leg_name = b20_info.get("leg")
    if leg_name not in LEG_TO_ID:
        info["reason"] = "invalid_leg"
        return info

    leg = LEG_TO_ID[leg_name]
    ids_list = _b12_get_leg_joint_ids(leg)
    ids = torch.tensor(ids_list, device=robot.data.joint_pos.device, dtype=torch.long)
    q_now_full = robot.data.joint_pos.detach()
    q_now = q_now_full[:, ids_list]
    q_raw = b20_target.detach()[:, ids_list]

    need_new_latch = (
        (not _B27_BRIDGE_STATE["active"])
        or (_B27_BRIDGE_STATE["leg_name"] != leg_name)
        or (step - int(_B27_BRIDGE_STATE.get("last_retarget_step", -999999)) >= int(args.b27_retarget_interval))
    )

    sign_vec = torch.tensor([float(args.b27_hx_sign), float(args.b27_hy_sign), float(args.b27_kn_sign)],
                            device=q_now.device, dtype=q_now.dtype).view(1, 3)

    if need_new_latch:
        q_latch = q_now.detach().clone()
        raw_delta = q_raw - q_latch
        signed_delta = raw_delta * sign_vec
        signed_delta = torch.clamp(signed_delta,
                                   -float(args.b27_max_abs_from_latch),
                                   float(args.b27_max_abs_from_latch))
        q_goal = q_latch + signed_delta
        _B27_BRIDGE_STATE.update({
            "active": True,
            "leg_name": leg_name,
            "source": "b20_signed",
            "q_latch": q_latch.detach().clone(),
            "q_goal": q_goal.detach().clone(),
            "step_latch": int(step),
            "last_retarget_step": int(step),
        })
    else:
        q_latch = _B27_BRIDGE_STATE["q_latch"].detach().clone()
        q_goal = _B27_BRIDGE_STATE["q_goal"].detach().clone()
        raw_delta = q_raw - q_latch
        signed_delta = q_goal - q_latch

    delta_from_now = torch.clamp(q_goal - q_now,
                                 -float(args.b27_max_joint_delta),
                                 float(args.b27_max_joint_delta))
    q_cmd = q_now + delta_from_now

    try:
        robot.set_joint_position_target(q_cmd, joint_ids=ids)
        if args.b27_zero_target_velocity:
            robot.set_joint_velocity_target(torch.zeros_like(q_cmd), joint_ids=ids)
        info.update({
            "active": True,
            "reason": "pd_target_set",
            "source": "b20_signed",
            "leg": leg_name,
            "latched_new": bool(need_new_latch),
            "joint_ids": ids_list,
            "q_now": q_now[0].detach().cpu().numpy(),
            "q_latch": q_latch[0].detach().cpu().numpy(),
            "q_goal_raw": q_raw[0].detach().cpu().numpy(),
            "q_goal": q_goal[0].detach().cpu().numpy(),
            "q_cmd": q_cmd[0].detach().cpu().numpy(),
            "raw_delta": raw_delta[0].detach().cpu().numpy(),
            "signed_delta": signed_delta[0].detach().cpu().numpy(),
            "delta_from_now": delta_from_now[0].detach().cpu().numpy(),
        })
    except Exception as e:
        info["reason"] = "set_target_error"
        info["error"] = repr(e)
    return info


_B28_BRIDGE_STATE = {
    "active": False,
    "leg_name": None,
    "source": None,
    "q_latch": None,
    "q_goal": None,
    "step_latch": -1,
    "last_retarget_step": -1,
    "last_active_target_step": -1,
}


def _b28_set_pd_cmd(robot, leg_name, q_goal, step, reason):
    leg = LEG_TO_ID[leg_name]
    ids_list = _b12_get_leg_joint_ids(leg)
    ids = torch.tensor(ids_list, device=robot.data.joint_pos.device, dtype=torch.long)
    q_now = robot.data.joint_pos.detach()[:, ids_list]
    delta_from_now = torch.clamp(q_goal - q_now,
                                 -float(args.b28_max_joint_delta),
                                 float(args.b28_max_joint_delta))
    q_cmd = q_now + delta_from_now
    robot.set_joint_position_target(q_cmd, joint_ids=ids)
    if args.b28_zero_target_velocity:
        robot.set_joint_velocity_target(torch.zeros_like(q_cmd), joint_ids=ids)
    return {
        "active": True,
        "reason": reason,
        "source": _B28_BRIDGE_STATE.get("source"),
        "leg": leg_name,
        "joint_ids": ids_list,
        "q_now": q_now[0].detach().cpu().numpy(),
        "q_latch": _B28_BRIDGE_STATE["q_latch"][0].detach().cpu().numpy() if _B28_BRIDGE_STATE.get("q_latch") is not None else None,
        "q_goal": q_goal[0].detach().cpu().numpy(),
        "q_cmd": q_cmd[0].detach().cpu().numpy(),
        "delta_from_now": delta_from_now[0].detach().cpu().numpy(),
        "hold_age": int(step) - int(_B28_BRIDGE_STATE.get("last_active_target_step", step)),
        "step_latch": int(_B28_BRIDGE_STATE.get("step_latch", -1)),
    }


def _b28_apply_hold_latched_swing_pd_bridge(robot, b20_target, b20_info, step):
    info = {"active": False, "reason": "disabled", "source": None, "leg": None,
            "joint_ids": None, "q_now": None, "q_latch": None, "q_goal_raw": None,
            "q_goal": None, "q_cmd": None, "raw_delta": None, "signed_delta": None,
            "delta_from_now": None, "hold_age": None}
    if not args.enable_b28_hold_latched_swing_pd_bridge:
        return info

    has_b20 = (args.b28_use_b20_direct_ik_target and isinstance(b20_info, dict)
               and b20_info.get("active", False) and b20_target is not None
               and b20_info.get("leg") in LEG_TO_ID)

    if has_b20:
        leg_name = b20_info.get("leg")
        leg = LEG_TO_ID[leg_name]
        ids_list = _b12_get_leg_joint_ids(leg)
        q_now = robot.data.joint_pos.detach()[:, ids_list]
        q_raw = b20_target.detach()[:, ids_list]
        need_new_latch = ((not _B28_BRIDGE_STATE["active"])
                          or (_B28_BRIDGE_STATE["leg_name"] != leg_name)
                          or (step - int(_B28_BRIDGE_STATE.get("last_retarget_step", -999999)) >= int(args.b28_retarget_interval)))
        sign_vec = torch.tensor([float(args.b28_hx_sign), float(args.b28_hy_sign), float(args.b28_kn_sign)],
                                device=q_now.device, dtype=q_now.dtype).view(1, 3)
        if need_new_latch:
            q_latch = q_now.detach().clone()
            raw_delta = q_raw - q_latch
            signed_delta = torch.clamp(raw_delta * sign_vec,
                                       -float(args.b28_max_abs_from_latch),
                                       float(args.b28_max_abs_from_latch))
            q_goal = q_latch + signed_delta
            _B28_BRIDGE_STATE.update({
                "active": True,
                "leg_name": leg_name,
                "source": "b20_signed_hold",
                "q_latch": q_latch.detach().clone(),
                "q_goal": q_goal.detach().clone(),
                "step_latch": int(step),
                "last_retarget_step": int(step),
                "last_active_target_step": int(step),
            })
        else:
            q_latch = _B28_BRIDGE_STATE["q_latch"].detach().clone()
            q_goal = _B28_BRIDGE_STATE["q_goal"].detach().clone()
            raw_delta = q_raw - q_latch
            signed_delta = q_goal - q_latch
            _B28_BRIDGE_STATE["last_active_target_step"] = int(step)
        try:
            out = _b28_set_pd_cmd(robot, leg_name, _B28_BRIDGE_STATE["q_goal"], step, "active_b20_target")
            out.update({"latched_new": bool(need_new_latch),
                        "q_goal_raw": q_raw[0].detach().cpu().numpy(),
                        "raw_delta": raw_delta[0].detach().cpu().numpy(),
                        "signed_delta": signed_delta[0].detach().cpu().numpy()})
            return out
        except Exception as e:
            info["reason"] = "set_target_error"; info["error"] = repr(e); return info

    if _B28_BRIDGE_STATE.get("active", False) and _B28_BRIDGE_STATE.get("q_goal") is not None:
        hold_age = int(step) - int(_B28_BRIDGE_STATE.get("last_active_target_step", -1))
        if hold_age <= int(args.b28_hold_without_target_steps):
            try:
                out = _b28_set_pd_cmd(robot, _B28_BRIDGE_STATE["leg_name"], _B28_BRIDGE_STATE["q_goal"], step, "hold_latched_no_b20")
                out["latched_new"] = False
                return out
            except Exception as e:
                info["reason"] = "hold_set_target_error"; info["error"] = repr(e); return info
        if args.b28_reset_when_hold_expires:
            _B28_BRIDGE_STATE.update({"active": False, "leg_name": None, "source": None,
                                      "q_latch": None, "q_goal": None, "step_latch": -1,
                                      "last_retarget_step": -1, "last_active_target_step": -1})
        info["reason"] = "hold_expired"; info["hold_age"] = hold_age; return info
    info["reason"] = "no_active_b20_target_no_latch"
    return info


_B29_BRIDGE_STATE = {
    "active": False,
    "leg_name": None,
    "q_latch": None,
    "q_goal": None,
    "target_w": None,
    "last_active_step": -1,
    "step_latch": -1,
}


def _b29_get_leg_jacobian_world(robot, leg_id):
    """Return 3x3 translational Jacobian for the selected foot wrt [hx, hy, kn].

    Falls back to None if Isaac's Jacobian tensor is unavailable in the current asset.
    """
    try:
        body_id = int(foot_indices[leg_id])
        joint_ids = _b12_get_leg_joint_ids(leg_id)

        # Isaac Lab variants expose either root_physx_view or body/joint jacobians.
        jac = None
        if hasattr(robot.root_physx_view, "get_jacobians"):
            jac_all = robot.root_physx_view.get_jacobians()
            # Common shape: [num_envs, num_bodies, 6, num_dofs]
            if jac_all is not None and jac_all.ndim == 4:
                jac = jac_all[0, body_id, 0:3, joint_ids]
        elif hasattr(robot.data, "body_jacobian_w"):
            jac_all = robot.data.body_jacobian_w
            if jac_all is not None and jac_all.ndim == 4:
                jac = jac_all[0, body_id, 0:3, joint_ids]

        if jac is None:
            return None
        return jac
    except Exception:
        return None


def _b29_damped_pinv_step(J, e):
    """dq = J^T (J J^T + lambda I)^-1 e"""
    lam = float(args.b29_damping)
    eye = torch.eye(3, device=J.device, dtype=J.dtype)
    A = J @ J.transpose(0, 1) + (lam * lam) * eye
    try:
        return J.transpose(0, 1) @ torch.linalg.solve(A, e)
    except Exception:
        return J.transpose(0, 1) @ e


def _b29_pick_leg_and_target(b23_info, b18_info, active_swing):
    """Choose a leg/committed target for task-space bridge."""
    # Prefer B23's selected committed target because it is exactly the main WBC foot ref.
    if isinstance(b23_info, dict) and b23_info.get("active", False):
        leg_name = b23_info.get("leg")
        target = b23_info.get("committed")
        if leg_name in LEG_TO_ID and target is not None:
            if not torch.is_tensor(target):
                target = torch.as_tensor(target)
            return leg_name, target.reshape(3), "b23_committed"

    # Fallback to B18 pin target info if present.
    if isinstance(b18_info, dict) and b18_info.get("active", False):
        leg_name = b18_info.get("leg")
        target = b18_info.get("target_w", None)
        if target is None:
            target = b18_info.get("committed", None)
        if target is None:
            target = b18_info.get("target", None)
        if leg_name in LEG_TO_ID and target is not None:
            if not torch.is_tensor(target):
                target = torch.as_tensor(target)
            return leg_name, target.reshape(3), "b18_pin_target"

    return None, None, "no_target"


def _b29_phase_allows(leg_name, active_swing, capture_leg, capture_count, post_capture_pin_count):
    in_swing = leg_name in active_swing if leg_name is not None else False
    in_capture = (capture_leg == leg_name and int(capture_count) > 0)
    in_post = (leg_name is not None and int(post_capture_pin_count) > 0)

    allowed = (
        (args.b29_apply_during_swing and in_swing)
        or (args.b29_apply_during_capture and in_capture)
        or (args.b29_apply_post_capture_extra and in_post)
    )
    phase = "swing" if in_swing else ("capture" if in_capture else ("post_capture" if in_post else "not_allowed"))
    return allowed, phase


def _b29_apply_task_space_foot_error_bridge(robot, foot_pos_arg, b23_info, b18_info, active_swing, capture_leg, capture_count, post_capture_pin_count, step):
    info = {"active": False, "reason": "disabled", "leg": None, "source": None, "phase": None,
            "target_w": None, "foot_w": None, "foot_err_raw": None, "foot_err_used": None,
            "J": None, "dq": None, "q_now": None, "q_latch": None, "q_goal": None, "q_cmd": None,
            "delta_from_now": None, "hold_age": None}
    if not args.enable_b29_task_space_foot_error_bridge:
        return info

    leg_name, target_w, source = _b29_pick_leg_and_target(b23_info, b18_info, active_swing)
    has_target = leg_name is not None and target_w is not None
    allowed = False
    phase = "no_target"
    if has_target:
        allowed, phase = _b29_phase_allows(leg_name, active_swing, capture_leg, capture_count, post_capture_pin_count)

    if has_target and allowed:
        leg_id = LEG_TO_ID[leg_name]
        joint_ids = _b12_get_leg_joint_ids(leg_id)
        q_now = robot.data.joint_pos.detach()[:, joint_ids]
        foot_w = foot_pos_arg[0, leg_id, :].detach()
        target_w = target_w.to(device=foot_w.device, dtype=foot_w.dtype)

        foot_err_raw = target_w - foot_w
        foot_err_used = foot_err_raw.clone()
        foot_err_used[0:2] = torch.clamp(foot_err_used[0:2],
                                         -float(args.b29_max_foot_err_xy),
                                         float(args.b29_max_foot_err_xy)) * float(args.b29_kp_xy)
        foot_err_used[2] = torch.clamp(foot_err_used[2],
                                       -float(args.b29_max_foot_err_z),
                                       float(args.b29_max_foot_err_z)) * float(args.b29_kp_z)

        J = _b29_get_leg_jacobian_world(robot, leg_id)
        if J is None:
            info.update({"reason": "no_jacobian", "leg": leg_name, "source": source, "phase": phase,
                         "target_w": target_w.detach().cpu().numpy(),
                         "foot_w": foot_w.detach().cpu().numpy(),
                         "foot_err_raw": foot_err_raw.detach().cpu().numpy()})
            return info

        dq = _b29_damped_pinv_step(J, foot_err_used)
        dq = torch.clamp(dq, -float(args.b29_max_joint_delta), float(args.b29_max_joint_delta)).view(1, 3)

        need_latch = (not _B29_BRIDGE_STATE["active"]) or (_B29_BRIDGE_STATE.get("leg_name") != leg_name)
        if need_latch:
            q_latch = q_now.detach().clone()
            q_goal = q_latch + dq
            _B29_BRIDGE_STATE.update({
                "active": True,
                "leg_name": leg_name,
                "q_latch": q_latch.detach().clone(),
                "q_goal": q_goal.detach().clone(),
                "target_w": target_w.detach().clone(),
                "last_active_step": int(step),
                "step_latch": int(step),
            })
        else:
            q_latch = _B29_BRIDGE_STATE["q_latch"].detach().clone()
            if args.b29_latch_goal:
                # Hold the first goal, but allow a small correction toward current task error.
                q_goal_raw = _B29_BRIDGE_STATE["q_goal"].detach().clone() + 0.25 * dq
            else:
                # Recompute from current q so this behaves like task-space servo, not joint-space latch.
                q_goal_raw = q_now + dq
            delta_from_latch = torch.clamp(q_goal_raw - q_latch,
                                           -float(args.b29_max_abs_from_latch),
                                           float(args.b29_max_abs_from_latch))
            q_goal = q_latch + delta_from_latch
            _B29_BRIDGE_STATE["q_goal"] = q_goal.detach().clone()
            _B29_BRIDGE_STATE["target_w"] = target_w.detach().clone()
            _B29_BRIDGE_STATE["last_active_step"] = int(step)

        delta_from_now = torch.clamp(_B29_BRIDGE_STATE["q_goal"] - q_now,
                                     -float(args.b29_max_joint_delta),
                                     float(args.b29_max_joint_delta))
        q_cmd = q_now + delta_from_now

        try:
            ids = torch.tensor(joint_ids, device=q_now.device, dtype=torch.long)
            robot.set_joint_position_target(q_cmd, joint_ids=ids)
            if args.b29_zero_target_velocity:
                robot.set_joint_velocity_target(torch.zeros_like(q_cmd), joint_ids=ids)
        except Exception as e:
            info["reason"] = "set_target_error"
            info["error"] = repr(e)
            return info

        info.update({
            "active": True,
            "reason": "task_space_target",
            "leg": leg_name,
            "source": source,
            "phase": phase,
            "latched_new": bool(need_latch),
            "target_w": target_w.detach().cpu().numpy(),
            "foot_w": foot_w.detach().cpu().numpy(),
            "foot_err_raw": foot_err_raw.detach().cpu().numpy(),
            "foot_err_used": foot_err_used.detach().cpu().numpy(),
            "J": J.detach().cpu().numpy(),
            "dq": dq[0].detach().cpu().numpy(),
            "q_now": q_now[0].detach().cpu().numpy(),
            "q_latch": _B29_BRIDGE_STATE["q_latch"][0].detach().cpu().numpy(),
            "q_goal": _B29_BRIDGE_STATE["q_goal"][0].detach().cpu().numpy(),
            "q_cmd": q_cmd[0].detach().cpu().numpy(),
            "delta_from_now": delta_from_now[0].detach().cpu().numpy(),
            "hold_age": 0,
        })
        return info

    # No target or not in allowed phase: optionally hold last q_goal through a short gap.
    if _B29_BRIDGE_STATE.get("active", False) and _B29_BRIDGE_STATE.get("q_goal") is not None:
        hold_age = int(step) - int(_B29_BRIDGE_STATE.get("last_active_step", -1))
        if hold_age <= int(args.b29_hold_without_target_steps):
            leg_name = _B29_BRIDGE_STATE.get("leg_name")
            if leg_name in LEG_TO_ID:
                leg_id = LEG_TO_ID[leg_name]
                joint_ids = _b12_get_leg_joint_ids(leg_id)
                q_now = robot.data.joint_pos.detach()[:, joint_ids]
                delta_from_now = torch.clamp(_B29_BRIDGE_STATE["q_goal"] - q_now,
                                             -float(args.b29_max_joint_delta),
                                             float(args.b29_max_joint_delta))
                q_cmd = q_now + delta_from_now
                try:
                    ids = torch.tensor(joint_ids, device=q_now.device, dtype=torch.long)
                    robot.set_joint_position_target(q_cmd, joint_ids=ids)
                    if args.b29_zero_target_velocity:
                        robot.set_joint_velocity_target(torch.zeros_like(q_cmd), joint_ids=ids)
                except Exception as e:
                    info["reason"] = "hold_set_target_error"
                    info["error"] = repr(e)
                    return info
                info.update({
                    "active": True,
                    "reason": "hold_latched_gap",
                    "leg": leg_name,
                    "source": _B29_BRIDGE_STATE.get("source", "held"),
                    "phase": phase,
                    "q_now": q_now[0].detach().cpu().numpy(),
                    "q_latch": _B29_BRIDGE_STATE["q_latch"][0].detach().cpu().numpy(),
                    "q_goal": _B29_BRIDGE_STATE["q_goal"][0].detach().cpu().numpy(),
                    "q_cmd": q_cmd[0].detach().cpu().numpy(),
                    "delta_from_now": delta_from_now[0].detach().cpu().numpy(),
                    "hold_age": hold_age,
                })
                return info

    info.update({"reason": "no_allowed_target", "source": source, "phase": phase})
    return info


_B30_BRIDGE_STATE = {
    "active": False,
    "leg_name": None,
    "q_latch": None,
    "q_goal": None,
    "target_w": None,
    "last_active_step": -1,
    "step_latch": -1,
}


def _b30_pick_leg_and_target(b23_info, b18_info):
    """Choose committed foot target used by the main stack."""
    if isinstance(b23_info, dict) and b23_info.get("active", False):
        leg_name = b23_info.get("leg")
        target = b23_info.get("committed")
        if target is None:
            target = b23_info.get("target_after", None)
        if leg_name in LEG_TO_ID and target is not None:
            if not torch.is_tensor(target):
                target = torch.as_tensor(target)
            return leg_name, target.reshape(3), "b23_committed"

    if isinstance(b18_info, dict) and b18_info.get("active", False):
        leg_name = b18_info.get("leg")
        target = b18_info.get("target_w", None)
        if target is None:
            target = b18_info.get("committed", None)
        if target is None:
            target = b18_info.get("target", None)
        if leg_name in LEG_TO_ID and target is not None:
            if not torch.is_tensor(target):
                target = torch.as_tensor(target)
            return leg_name, target.reshape(3), "b18_target"

    return None, None, "no_target"


def _b30_phase_allows(leg_name, active_swing_names, capture_leg, capture_count, post_capture_pin_count):
    in_swing = leg_name in active_swing_names if leg_name is not None else False
    in_capture = (capture_leg == leg_name and int(capture_count) > 0)
    in_post = (leg_name is not None and int(post_capture_pin_count) > 0)

    allowed = (
        (args.b30_apply_during_swing and in_swing)
        or (args.b30_apply_during_capture and in_capture)
        or (args.b30_apply_post_capture_extra and in_post)
    )
    phase = "swing" if in_swing else ("capture" if in_capture else ("post_capture" if in_post else "not_allowed"))
    return allowed, phase


def _b30_damped_pinv_step(J, e):
    lam = float(args.b30_damping)
    eye = torch.eye(J.shape[0], device=J.device, dtype=J.dtype)
    A = J @ J.transpose(0, 1) + (lam * lam) * eye
    return J.transpose(0, 1) @ torch.linalg.solve(A, e)


def _b30_apply_jfeet_task_space_bridge(robot, Jfeet_full, foot_pos_arg, b23_info, b18_info,
                                       active_swing_names, capture_leg, capture_count,
                                       post_capture_pin_count, step):
    info = {"active": False, "reason": "disabled", "leg": None, "source": None, "phase": None,
            "target_w": None, "foot_w": None, "foot_err_raw": None, "foot_err_used": None,
            "J": None, "dq": None, "q_now": None, "q_latch": None, "q_goal": None,
            "q_cmd": None, "delta_from_now": None, "hold_age": None}
    if not args.enable_b30_jfeet_task_space_bridge:
        return info

    leg_name, target_w, source = _b30_pick_leg_and_target(b23_info, b18_info)
    if leg_name is None or target_w is None:
        info.update({"reason": "no_target", "source": source, "phase": "no_target"})
        return info

    allowed, phase = _b30_phase_allows(leg_name, active_swing_names, capture_leg, capture_count, post_capture_pin_count)
    if not allowed:
        # Keep the most recent target through tiny gaps only.
        if _B30_BRIDGE_STATE.get("active", False) and _B30_BRIDGE_STATE.get("q_goal") is not None:
            hold_age = int(step) - int(_B30_BRIDGE_STATE.get("last_active_step", -1))
            if hold_age <= int(args.b30_hold_without_target_steps):
                leg_hold = _B30_BRIDGE_STATE.get("leg_name")
                if leg_hold in LEG_TO_ID:
                    leg_id = LEG_TO_ID[leg_hold]
                    joint_ids = _b12_get_leg_joint_ids(leg_id)
                    q_now = robot.data.joint_pos.detach()[:, joint_ids]
                    delta_from_now = torch.clamp(_B30_BRIDGE_STATE["q_goal"] - q_now,
                                                 -float(args.b30_max_joint_delta),
                                                 float(args.b30_max_joint_delta))
                    q_cmd = q_now + delta_from_now
                    ids = torch.tensor(joint_ids, device=q_now.device, dtype=torch.long)
                    robot.set_joint_position_target(q_cmd, joint_ids=ids)
                    if args.b30_zero_target_velocity:
                        robot.set_joint_velocity_target(torch.zeros_like(q_cmd), joint_ids=ids)
                    info.update({
                        "active": True, "reason": "hold_latched_gap", "leg": leg_hold,
                        "source": _B30_BRIDGE_STATE.get("source", "held"), "phase": phase,
                        "q_now": q_now[0].detach().cpu().numpy(),
                        "q_latch": _B30_BRIDGE_STATE["q_latch"][0].detach().cpu().numpy(),
                        "q_goal": _B30_BRIDGE_STATE["q_goal"][0].detach().cpu().numpy(),
                        "q_cmd": q_cmd[0].detach().cpu().numpy(),
                        "delta_from_now": delta_from_now[0].detach().cpu().numpy(),
                        "hold_age": hold_age,
                    })
                    return info
        info.update({"reason": "phase_not_allowed", "source": source, "phase": phase, "leg": leg_name})
        return info

    leg_id = LEG_TO_ID[leg_name]
    joint_ids = _b12_get_leg_joint_ids(leg_id)
    try:
        J_leg = Jfeet_full[0, leg_id, :, joint_ids]
    except Exception as e:
        info.update({"reason": "jfeet_access_error", "error": repr(e), "leg": leg_name, "source": source, "phase": phase})
        return info

    q_now = robot.data.joint_pos.detach()[:, joint_ids]
    foot_w = foot_pos_arg[0, leg_id, :].detach()
    target_w = target_w.to(device=foot_w.device, dtype=foot_w.dtype)

    foot_err_raw = target_w - foot_w
    foot_err_used = foot_err_raw.clone()
    foot_err_used[0:2] = torch.clamp(foot_err_used[0:2],
                                     -float(args.b30_max_foot_err_xy),
                                     float(args.b30_max_foot_err_xy)) * float(args.b30_kp_xy)
    foot_err_used[2] = torch.clamp(foot_err_used[2],
                                   -float(args.b30_max_foot_err_z),
                                   float(args.b30_max_foot_err_z)) * float(args.b30_kp_z)

    try:
        dq = _b30_damped_pinv_step(J_leg, foot_err_used)
    except Exception as e:
        info.update({"reason": "dls_error", "error": repr(e), "leg": leg_name, "source": source,
                     "phase": phase, "target_w": target_w.detach().cpu().numpy(),
                     "foot_w": foot_w.detach().cpu().numpy(),
                     "foot_err_raw": foot_err_raw.detach().cpu().numpy(),
                     "J": J_leg.detach().cpu().numpy()})
        return info

    dq = torch.clamp(dq, -float(args.b30_max_joint_delta), float(args.b30_max_joint_delta)).view(1, 3)

    need_latch = (not _B30_BRIDGE_STATE["active"]) or (_B30_BRIDGE_STATE.get("leg_name") != leg_name)
    if need_latch:
        q_latch = q_now.detach().clone()
        q_goal_raw = q_latch + dq
        _B30_BRIDGE_STATE["q_latch"] = q_latch.detach().clone()
        _B30_BRIDGE_STATE["step_latch"] = int(step)
    else:
        q_latch = _B30_BRIDGE_STATE["q_latch"].detach().clone()
        if args.b30_latch_goal:
            q_goal_raw = _B30_BRIDGE_STATE["q_goal"].detach().clone() + 0.25 * dq
        else:
            q_goal_raw = q_now + dq

    delta_from_latch = torch.clamp(q_goal_raw - q_latch,
                                   -float(args.b30_max_abs_from_latch),
                                   float(args.b30_max_abs_from_latch))
    q_goal = q_latch + delta_from_latch
    delta_from_now = torch.clamp(q_goal - q_now,
                                 -float(args.b30_max_joint_delta),
                                 float(args.b30_max_joint_delta))
    q_cmd = q_now + delta_from_now

    ids = torch.tensor(joint_ids, device=q_now.device, dtype=torch.long)
    robot.set_joint_position_target(q_cmd, joint_ids=ids)
    if args.b30_zero_target_velocity:
        robot.set_joint_velocity_target(torch.zeros_like(q_cmd), joint_ids=ids)

    _B30_BRIDGE_STATE.update({
        "active": True,
        "leg_name": leg_name,
        "source": source,
        "q_goal": q_goal.detach().clone(),
        "target_w": target_w.detach().clone(),
        "last_active_step": int(step),
    })

    info.update({
        "active": True,
        "reason": "jfeet_task_space_target",
        "leg": leg_name,
        "source": source,
        "phase": phase,
        "latched_new": bool(need_latch),
        "target_w": target_w.detach().cpu().numpy(),
        "foot_w": foot_w.detach().cpu().numpy(),
        "foot_err_raw": foot_err_raw.detach().cpu().numpy(),
        "foot_err_used": foot_err_used.detach().cpu().numpy(),
        "J": J_leg.detach().cpu().numpy(),
        "dq": dq[0].detach().cpu().numpy(),
        "q_now": q_now[0].detach().cpu().numpy(),
        "q_latch": q_latch[0].detach().cpu().numpy(),
        "q_goal": q_goal[0].detach().cpu().numpy(),
        "q_cmd": q_cmd[0].detach().cpu().numpy(),
        "delta_from_now": delta_from_now[0].detach().cpu().numpy(),
        "hold_age": 0,
    })
    return info


_B31_PROBE_STATE = {
    "pending": None,
    "last_start_step": -1000000000,
    "seq": 0,
}


def _b31_as_tensor(x, device, dtype):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _b31_update_jacobian_response_probe(b30_info, foot_pos_arg, step):
    """Passive sign/frame diagnostic.

    When B30 is active, record:
        pred_delta = Jfeet_full_leg @ delta_from_now

    A few control steps later, compare:
        actual_delta = foot_now - foot_start

    This does not inject any new command. It only observes B30's already-issued
    task-space bridge command.
    """
    info = {
        "event": "disabled",
        "seq": None,
        "leg": None,
        "start_step": None,
        "end_step": None,
        "dt_steps": None,
        "pred_delta": None,
        "actual_delta": None,
        "error": None,
        "pred_norm": None,
        "actual_norm": None,
        "cosine": None,
        "axis_sign_match": None,
        "dq": None,
        "J": None,
        "reason": None,
    }
    if not args.enable_b31_jacobian_response_probe:
        return info

    device = foot_pos_arg.device
    dtype = foot_pos_arg.dtype

    # Complete pending probe first.
    pending = _B31_PROBE_STATE.get("pending", None)
    if pending is not None:
        age = int(step) - int(pending["start_step"])
        if age >= int(args.b31_delay_steps):
            leg_id = int(pending["leg_id"])
            foot_now = foot_pos_arg[0, leg_id, :].detach()
            foot_start = pending["foot_start"].to(device=device, dtype=dtype)
            pred = pending["pred_delta"].to(device=device, dtype=dtype)
            actual = foot_now - foot_start
            err = actual - pred
            pred_norm = torch.linalg.norm(pred).item()
            actual_norm = torch.linalg.norm(actual).item()
            denom = max(pred_norm * actual_norm, 1.0e-9)
            cosine = float(torch.dot(pred.reshape(-1), actual.reshape(-1)).item() / denom)
            sign_match = torch.sign(pred) == torch.sign(actual)
            # Near-zero predicted axes should not be treated as meaningful.
            meaningful = torch.abs(pred) > 1.0e-6
            sign_match_out = []
            for i in range(3):
                if bool(meaningful[i].item()):
                    sign_match_out.append(bool(sign_match[i].item()))
                else:
                    sign_match_out.append(None)

            info.update({
                "event": "complete",
                "seq": pending["seq"],
                "leg": pending["leg_name"],
                "start_step": pending["start_step"],
                "end_step": int(step),
                "dt_steps": age,
                "pred_delta": pred.detach().cpu().numpy(),
                "actual_delta": actual.detach().cpu().numpy(),
                "error": err.detach().cpu().numpy(),
                "pred_norm": pred_norm,
                "actual_norm": actual_norm,
                "cosine": cosine,
                "axis_sign_match": sign_match_out,
                "dq": pending["dq"].detach().cpu().numpy(),
                "J": pending["J"].detach().cpu().numpy(),
                "reason": "completed_probe",
            })
            _B31_PROBE_STATE["pending"] = None
            return info

        info.update({
            "event": "waiting",
            "seq": pending["seq"],
            "leg": pending["leg_name"],
            "start_step": pending["start_step"],
            "end_step": int(step),
            "dt_steps": age,
            "reason": "waiting_for_delay",
        })
        return info

    # Start a new probe only when B30 actually issued a task-space target.
    if not isinstance(b30_info, dict) or not b30_info.get("active", False):
        info.update({"event": "idle", "reason": "b30_inactive"})
        return info
    if b30_info.get("reason") != "jfeet_task_space_target":
        info.update({"event": "idle", "reason": b30_info.get("reason")})
        return info

    if int(step) - int(_B31_PROBE_STATE.get("last_start_step", -1000000000)) < int(args.b31_min_gap_steps):
        info.update({"event": "idle", "reason": "min_gap"})
        return info

    leg_name = b30_info.get("leg", None)
    if leg_name not in LEG_TO_ID:
        info.update({"event": "idle", "reason": "invalid_leg", "leg": leg_name})
        return info

    J = _b31_as_tensor(b30_info.get("J", None), device=device, dtype=dtype)
    dq = _b31_as_tensor(b30_info.get("delta_from_now", None), device=device, dtype=dtype)
    if J is None or dq is None:
        info.update({"event": "idle", "reason": "missing_J_or_dq", "leg": leg_name})
        return info
    J = J.reshape(3, 3)
    dq = dq.reshape(3)

    if torch.linalg.norm(dq).item() < float(args.b31_min_delta_norm):
        info.update({"event": "idle", "reason": "dq_too_small", "leg": leg_name})
        return info

    pred = J @ dq
    if torch.linalg.norm(pred).item() < float(args.b31_min_delta_norm):
        info.update({"event": "idle", "reason": "pred_too_small", "leg": leg_name})
        return info

    leg_id = LEG_TO_ID[leg_name]
    _B31_PROBE_STATE["seq"] = int(_B31_PROBE_STATE.get("seq", 0)) + 1
    seq = _B31_PROBE_STATE["seq"]
    _B31_PROBE_STATE["last_start_step"] = int(step)
    _B31_PROBE_STATE["pending"] = {
        "seq": seq,
        "leg_name": leg_name,
        "leg_id": leg_id,
        "start_step": int(step),
        "foot_start": foot_pos_arg[0, leg_id, :].detach().clone(),
        "pred_delta": pred.detach().clone(),
        "dq": dq.detach().clone(),
        "J": J.detach().clone(),
    }

    info.update({
        "event": "start",
        "seq": seq,
        "leg": leg_name,
        "start_step": int(step),
        "dt_steps": 0,
        "pred_delta": pred.detach().cpu().numpy(),
        "pred_norm": torch.linalg.norm(pred).item(),
        "dq": dq.detach().cpu().numpy(),
        "J": J.detach().cpu().numpy(),
        "reason": "started_probe",
    })
    return info


_B32_PROBE_STATE = {
    "queue": None,
    "active": None,
    "last_end_step": -1000000000,
    "completed": 0,
    "seq": 0,
}


def _b32_parse_probe_queue():
    axes_raw = [x.strip().lower() for x in str(args.b32_axes).split(",") if x.strip()]
    signs_raw = [x.strip().lower() for x in str(args.b32_signs).split(",") if x.strip()]
    valid_axes = {"hx": 0, "hy": 1, "kn": 2}
    valid_signs = {"plus": 1.0, "+": 1.0, "pos": 1.0, "minus": -1.0, "-": -1.0, "neg": -1.0}

    queue = []
    for ax in axes_raw:
        if ax not in valid_axes:
            continue
        for sg in signs_raw:
            if sg not in valid_signs:
                continue
            queue.append({
                "axis": ax,
                "axis_idx": valid_axes[ax],
                "sign_name": "plus" if valid_signs[sg] > 0 else "minus",
                "sign": valid_signs[sg],
            })
    return queue


def _b32_phase_allows_for_leg(probe_leg, active_swing_names, capture_leg, capture_count):
    in_swing = probe_leg in active_swing_names if probe_leg is not None else False
    in_capture = (capture_leg == probe_leg and int(capture_count) > 0)
    if not args.b32_require_swing_or_capture:
        return True, "forced_probe"
    if in_swing:
        return True, "swing"
    if in_capture:
        return True, "capture"
    return False, "not_swing_or_capture"


def _b32_update_direct_joint_response_probe(robot, foot_pos_arg, active_swing_names,
                                            capture_leg, capture_count, step):
    """Actively apply small single-axis joint target offsets and measure foot response.

    This probe bypasses Jfeet_full. It writes a tiny position target offset on one
    joint axis of the selected leg, holds it for a few steps, and compares actual
    foot displacement.

    It is intentionally small and short. It should be run in isolation from B25-B29,
    and optionally with B30 disabled during probe via --b32_disable_b30_during_probe.
    """
    info = {
        "event": "disabled",
        "seq": None,
        "leg": None,
        "axis": None,
        "sign": None,
        "start_step": None,
        "end_step": None,
        "age": None,
        "phase": None,
        "q_start": None,
        "q_cmd": None,
        "dq_cmd": None,
        "foot_start": None,
        "foot_now": None,
        "actual_delta": None,
        "actual_norm": None,
        "delta_per_rad": None,
        "reason": None,
    }
    if not args.enable_b32_direct_joint_response_probe:
        return info

    if _B32_PROBE_STATE["queue"] is None:
        _B32_PROBE_STATE["queue"] = _b32_parse_probe_queue()

    probe_leg = str(args.b32_probe_leg).upper()
    if probe_leg not in LEG_TO_ID:
        info.update({"event": "idle", "reason": "invalid_probe_leg", "leg": probe_leg})
        return info

    # Complete active probe.
    active = _B32_PROBE_STATE.get("active", None)
    if active is not None:
        leg_id = int(active["leg_id"])
        joint_ids = active["joint_ids"]
        q_start = active["q_start"]
        q_cmd = active["q_cmd"]

        # Keep commanding during hold.
        ids = torch.tensor(joint_ids, device=q_cmd.device, dtype=torch.long)
        robot.set_joint_position_target(q_cmd, joint_ids=ids)
        if args.b32_zero_target_velocity:
            robot.set_joint_velocity_target(torch.zeros_like(q_cmd), joint_ids=ids)

        age = int(step) - int(active["start_step"])
        if age >= int(args.b32_hold_steps):
            foot_now = foot_pos_arg[0, leg_id, :].detach()
            foot_start = active["foot_start"].to(device=foot_now.device, dtype=foot_now.dtype)
            actual = foot_now - foot_start
            actual_norm = float(torch.linalg.norm(actual).item())
            delta_mag = float(active["delta_mag"])
            delta_per_rad = actual / max(delta_mag, 1.0e-9)

            info.update({
                "event": "complete",
                "seq": active["seq"],
                "leg": active["leg"],
                "axis": active["axis"],
                "sign": active["sign_name"],
                "start_step": active["start_step"],
                "end_step": int(step),
                "age": age,
                "phase": active["phase"],
                "q_start": q_start[0].detach().cpu().numpy(),
                "q_cmd": q_cmd[0].detach().cpu().numpy(),
                "dq_cmd": active["dq_cmd"].detach().cpu().numpy(),
                "foot_start": foot_start.detach().cpu().numpy(),
                "foot_now": foot_now.detach().cpu().numpy(),
                "actual_delta": actual.detach().cpu().numpy(),
                "actual_norm": actual_norm,
                "delta_per_rad": delta_per_rad.detach().cpu().numpy(),
                "reason": "completed_probe",
            })
            _B32_PROBE_STATE["active"] = None
            _B32_PROBE_STATE["last_end_step"] = int(step)
            _B32_PROBE_STATE["completed"] = int(_B32_PROBE_STATE.get("completed", 0)) + 1
            return info

        info.update({
            "event": "holding",
            "seq": active["seq"],
            "leg": active["leg"],
            "axis": active["axis"],
            "sign": active["sign_name"],
            "start_step": active["start_step"],
            "end_step": int(step),
            "age": age,
            "phase": active["phase"],
            "q_start": q_start[0].detach().cpu().numpy(),
            "q_cmd": q_cmd[0].detach().cpu().numpy(),
            "dq_cmd": active["dq_cmd"].detach().cpu().numpy(),
            "reason": "holding_probe_target",
        })
        return info

    # Start new probe.
    if int(step) < int(args.b32_start_step):
        info.update({"event": "idle", "reason": "before_start_step", "leg": probe_leg})
        return info
    if int(_B32_PROBE_STATE.get("completed", 0)) >= int(args.b32_max_probes):
        info.update({"event": "idle", "reason": "max_probes_done", "leg": probe_leg})
        return info
    if int(step) - int(_B32_PROBE_STATE.get("last_end_step", -1000000000)) < int(args.b32_gap_steps):
        info.update({"event": "idle", "reason": "gap_wait", "leg": probe_leg})
        return info

    allowed, phase = _b32_phase_allows_for_leg(probe_leg, active_swing_names, capture_leg, capture_count)
    if not allowed:
        info.update({"event": "idle", "reason": "phase_not_allowed", "leg": probe_leg, "phase": phase})
        return info

    queue = _B32_PROBE_STATE.get("queue", [])
    if not queue:
        info.update({"event": "idle", "reason": "empty_probe_queue", "leg": probe_leg})
        return info

    probe = queue.pop(0)
    _B32_PROBE_STATE["queue"] = queue

    leg_id = LEG_TO_ID[probe_leg]
    joint_ids = _b12_get_leg_joint_ids(leg_id)
    q_start = robot.data.joint_pos.detach()[:, joint_ids]
    dq_cmd = torch.zeros_like(q_start)
    delta_mag = abs(float(args.b32_delta))
    dq_cmd[0, int(probe["axis_idx"])] = float(probe["sign"]) * delta_mag
    q_cmd = q_start + dq_cmd

    ids = torch.tensor(joint_ids, device=q_cmd.device, dtype=torch.long)
    robot.set_joint_position_target(q_cmd, joint_ids=ids)
    if args.b32_zero_target_velocity:
        robot.set_joint_velocity_target(torch.zeros_like(q_cmd), joint_ids=ids)

    _B32_PROBE_STATE["seq"] = int(_B32_PROBE_STATE.get("seq", 0)) + 1
    seq = int(_B32_PROBE_STATE["seq"])
    _B32_PROBE_STATE["active"] = {
        "seq": seq,
        "leg": probe_leg,
        "leg_id": leg_id,
        "joint_ids": joint_ids,
        "axis": probe["axis"],
        "axis_idx": int(probe["axis_idx"]),
        "sign_name": probe["sign_name"],
        "sign": float(probe["sign"]),
        "delta_mag": delta_mag,
        "start_step": int(step),
        "phase": phase,
        "q_start": q_start.detach().clone(),
        "q_cmd": q_cmd.detach().clone(),
        "dq_cmd": dq_cmd[0].detach().clone(),
        "foot_start": foot_pos_arg[0, leg_id, :].detach().clone(),
    }

    info.update({
        "event": "start",
        "seq": seq,
        "leg": probe_leg,
        "axis": probe["axis"],
        "sign": probe["sign_name"],
        "start_step": int(step),
        "age": 0,
        "phase": phase,
        "q_start": q_start[0].detach().cpu().numpy(),
        "q_cmd": q_cmd[0].detach().cpu().numpy(),
        "dq_cmd": dq_cmd[0].detach().cpu().numpy(),
        "foot_start": foot_pos_arg[0, leg_id, :].detach().cpu().numpy(),
        "reason": "started_probe",
    })
    return info


_B33_PROBE_STATE = {
    "queue": None,
    "phase": None,   # None | "baseline" | "perturb"
    "active": None,
    "last_end_step": -1000000000,
    "completed": 0,
    "seq": 0,
}


def _b33_parse_probe_queue():
    axes_raw = [x.strip().lower() for x in str(args.b33_axes).split(",") if x.strip()]
    signs_raw = [x.strip().lower() for x in str(args.b33_signs).split(",") if x.strip()]
    valid_axes = {"hx": 0, "hy": 1, "kn": 2}
    valid_signs = {"plus": 1.0, "+": 1.0, "pos": 1.0, "minus": -1.0, "-": -1.0, "neg": -1.0}
    queue = []
    for ax in axes_raw:
        if ax not in valid_axes:
            continue
        for sg in signs_raw:
            if sg not in valid_signs:
                continue
            queue.append({
                "axis": ax,
                "axis_idx": valid_axes[ax],
                "sign_name": "plus" if valid_signs[sg] > 0 else "minus",
                "sign": valid_signs[sg],
            })
    return queue


def _b33_phase_allows_for_leg(probe_leg, active_swing_names, capture_leg, capture_count):
    in_swing = probe_leg in active_swing_names if probe_leg is not None else False
    in_capture = (capture_leg == probe_leg and int(capture_count) > 0)
    if not args.b33_require_swing_or_capture:
        return True, "forced_probe"
    if in_swing:
        return True, "swing"
    if in_capture:
        return True, "capture"
    return False, "not_swing_or_capture"


def _b33_set_joint_target(robot, joint_ids, q_cmd, zero_vel):
    ids = torch.tensor(joint_ids, device=q_cmd.device, dtype=torch.long)
    robot.set_joint_position_target(q_cmd, joint_ids=ids)
    if zero_vel:
        robot.set_joint_velocity_target(torch.zeros_like(q_cmd), joint_ids=ids)


def _b33_update_baseline_cancelled_response_probe(robot, foot_pos_arg, active_swing_names,
                                                  capture_leg, capture_count, step):
    """Measure joint-target response with natural drift cancellation.

    Per probe pair:
      A) baseline: hold the current joint target at q_start for baseline_steps
         and record baseline_delta = foot_end - foot_start.
      B) perturb: command q_start + one-axis delta for perturb_steps
         and record perturb_delta = foot_end - foot_start.
      C) corrected_delta = perturb_delta - baseline_delta.

    This is still not perfect because the baseline and perturb windows are sequential,
    but it removes much of the swing-phase natural drift seen in B32.
    """
    info = {
        "event": "disabled",
        "seq": None,
        "leg": None,
        "axis": None,
        "sign": None,
        "phase": None,
        "start_step": None,
        "end_step": None,
        "age": None,
        "q_start": None,
        "q_cmd": None,
        "dq_cmd": None,
        "baseline_delta": None,
        "perturb_delta": None,
        "corrected_delta": None,
        "corrected_norm": None,
        "corrected_delta_per_rad": None,
        "foot_start": None,
        "foot_now": None,
        "reason": None,
    }
    if not args.enable_b33_baseline_cancelled_response_probe:
        return info

    if _B33_PROBE_STATE["queue"] is None:
        _B33_PROBE_STATE["queue"] = _b33_parse_probe_queue()

    probe_leg = str(args.b33_probe_leg).upper()
    if probe_leg not in LEG_TO_ID:
        info.update({"event": "idle", "reason": "invalid_probe_leg", "leg": probe_leg})
        return info

    active = _B33_PROBE_STATE.get("active", None)

    # Continue active baseline/perturb stage.
    if active is not None:
        leg_id = int(active["leg_id"])
        joint_ids = active["joint_ids"]
        q_cmd = active["q_cmd"]
        _b33_set_joint_target(robot, joint_ids, q_cmd, args.b33_zero_target_velocity)

        age = int(step) - int(active["stage_start_step"])
        needed = int(args.b33_baseline_steps) if active["stage"] == "baseline" else int(args.b33_perturb_steps)

        if age < needed:
            info.update({
                "event": "holding",
                "seq": active["seq"],
                "leg": active["leg"],
                "axis": active["axis"],
                "sign": active["sign_name"],
                "phase": active["stage"],
                "start_step": active["stage_start_step"],
                "end_step": int(step),
                "age": age,
                "q_start": active["q_start"][0].detach().cpu().numpy(),
                "q_cmd": q_cmd[0].detach().cpu().numpy(),
                "dq_cmd": active["dq_cmd"].detach().cpu().numpy(),
                "foot_start": active["foot_start"].detach().cpu().numpy(),
                "reason": "holding_" + active["stage"],
            })
            return info

        foot_now = foot_pos_arg[0, leg_id, :].detach()
        foot_start = active["foot_start"].to(device=foot_now.device, dtype=foot_now.dtype)
        stage_delta = foot_now - foot_start

        # Baseline done -> immediately start perturb stage from the current state.
        if active["stage"] == "baseline":
            q_start_pert = robot.data.joint_pos.detach()[:, joint_ids]
            dq_cmd = torch.zeros_like(q_start_pert)
            dq_cmd[0, int(active["axis_idx"])] = float(active["sign"]) * float(active["delta_mag"])
            q_cmd_pert = q_start_pert + dq_cmd
            _b33_set_joint_target(robot, joint_ids, q_cmd_pert, args.b33_zero_target_velocity)

            active.update({
                "stage": "perturb",
                "stage_start_step": int(step),
                "baseline_delta": stage_delta.detach().clone(),
                "baseline_end_step": int(step),
                "q_start": q_start_pert.detach().clone(),
                "q_cmd": q_cmd_pert.detach().clone(),
                "dq_cmd": dq_cmd[0].detach().clone(),
                "foot_start": foot_now.detach().clone(),
            })
            _B33_PROBE_STATE["active"] = active

            info.update({
                "event": "baseline_complete",
                "seq": active["seq"],
                "leg": active["leg"],
                "axis": active["axis"],
                "sign": active["sign_name"],
                "phase": "baseline",
                "start_step": active["start_step"],
                "end_step": int(step),
                "age": age,
                "q_start": q_start_pert[0].detach().cpu().numpy(),
                "q_cmd": q_cmd_pert[0].detach().cpu().numpy(),
                "dq_cmd": dq_cmd[0].detach().cpu().numpy(),
                "baseline_delta": stage_delta.detach().cpu().numpy(),
                "foot_now": foot_now.detach().cpu().numpy(),
                "reason": "baseline_done_starting_perturb",
            })
            return info

        # Perturb done -> report corrected response.
        baseline_delta = active["baseline_delta"].to(device=foot_now.device, dtype=foot_now.dtype)
        perturb_delta = stage_delta
        corrected = perturb_delta - baseline_delta
        corrected_norm = float(torch.linalg.norm(corrected).item())
        delta_mag = max(float(active["delta_mag"]), 1.0e-9)
        corrected_per_rad = corrected / delta_mag

        info.update({
            "event": "complete",
            "seq": active["seq"],
            "leg": active["leg"],
            "axis": active["axis"],
            "sign": active["sign_name"],
            "phase": "perturb",
            "start_step": active["start_step"],
            "end_step": int(step),
            "age": age,
            "q_start": active["q_start"][0].detach().cpu().numpy(),
            "q_cmd": active["q_cmd"][0].detach().cpu().numpy(),
            "dq_cmd": active["dq_cmd"].detach().cpu().numpy(),
            "baseline_delta": baseline_delta.detach().cpu().numpy(),
            "perturb_delta": perturb_delta.detach().cpu().numpy(),
            "corrected_delta": corrected.detach().cpu().numpy(),
            "corrected_norm": corrected_norm,
            "corrected_delta_per_rad": corrected_per_rad.detach().cpu().numpy(),
            "foot_start": foot_start.detach().cpu().numpy(),
            "foot_now": foot_now.detach().cpu().numpy(),
            "reason": "completed_pair",
        })

        _B33_PROBE_STATE["active"] = None
        _B33_PROBE_STATE["last_end_step"] = int(step)
        _B33_PROBE_STATE["completed"] = int(_B33_PROBE_STATE.get("completed", 0)) + 1
        return info

    # Start new baseline stage.
    if int(step) < int(args.b33_start_step):
        info.update({"event": "idle", "reason": "before_start_step", "leg": probe_leg})
        return info
    if int(_B33_PROBE_STATE.get("completed", 0)) >= int(args.b33_max_pairs):
        info.update({"event": "idle", "reason": "max_pairs_done", "leg": probe_leg})
        return info
    if int(step) - int(_B33_PROBE_STATE.get("last_end_step", -1000000000)) < int(args.b33_gap_steps):
        info.update({"event": "idle", "reason": "gap_wait", "leg": probe_leg})
        return info

    allowed, phase = _b33_phase_allows_for_leg(probe_leg, active_swing_names, capture_leg, capture_count)
    if not allowed:
        info.update({"event": "idle", "reason": "phase_not_allowed", "leg": probe_leg, "phase": phase})
        return info

    queue = _B33_PROBE_STATE.get("queue", [])
    if not queue:
        info.update({"event": "idle", "reason": "empty_probe_queue", "leg": probe_leg})
        return info

    probe = queue.pop(0)
    _B33_PROBE_STATE["queue"] = queue

    leg_id = LEG_TO_ID[probe_leg]
    joint_ids = _b12_get_leg_joint_ids(leg_id)
    q_start = robot.data.joint_pos.detach()[:, joint_ids]
    q_cmd = q_start.detach().clone()
    dq_cmd = torch.zeros_like(q_start)[0]

    _b33_set_joint_target(robot, joint_ids, q_cmd, args.b33_zero_target_velocity)

    _B33_PROBE_STATE["seq"] = int(_B33_PROBE_STATE.get("seq", 0)) + 1
    seq = int(_B33_PROBE_STATE["seq"])
    active = {
        "seq": seq,
        "leg": probe_leg,
        "leg_id": leg_id,
        "joint_ids": joint_ids,
        "axis": probe["axis"],
        "axis_idx": int(probe["axis_idx"]),
        "sign_name": probe["sign_name"],
        "sign": float(probe["sign"]),
        "delta_mag": abs(float(args.b33_delta)),
        "stage": "baseline",
        "start_step": int(step),
        "stage_start_step": int(step),
        "phase_context": phase,
        "q_start": q_start.detach().clone(),
        "q_cmd": q_cmd.detach().clone(),
        "dq_cmd": dq_cmd.detach().clone(),
        "foot_start": foot_pos_arg[0, leg_id, :].detach().clone(),
        "baseline_delta": None,
    }
    _B33_PROBE_STATE["active"] = active

    info.update({
        "event": "start_baseline",
        "seq": seq,
        "leg": probe_leg,
        "axis": probe["axis"],
        "sign": probe["sign_name"],
        "phase": "baseline",
        "start_step": int(step),
        "age": 0,
        "q_start": q_start[0].detach().cpu().numpy(),
        "q_cmd": q_cmd[0].detach().cpu().numpy(),
        "dq_cmd": dq_cmd.detach().cpu().numpy(),
        "foot_start": foot_pos_arg[0, leg_id, :].detach().cpu().numpy(),
        "reason": "started_baseline",
    })
    return info


_B34_BRIDGE_STATE = {
    "active": False,
    "leg_name": None,
    "q_latch": None,
    "q_goal": None,
    "target_w": None,
    "last_active_step": -1,
    "source": None,
}


def _b34_pick_leg_and_target(b23_info, b18_info):
    """Pick the committed foothold target used by the existing stack."""
    if isinstance(b23_info, dict) and b23_info.get("active", False):
        leg_name = b23_info.get("leg")
        target = b23_info.get("committed", None)
        if target is None:
            target = b23_info.get("target_after", None)
        if leg_name in LEG_TO_ID and target is not None:
            if not torch.is_tensor(target):
                target = torch.as_tensor(target)
            return leg_name, target.reshape(3), "b23_committed"

    if isinstance(b18_info, dict) and b18_info.get("active", False):
        leg_name = b18_info.get("leg")
        target = b18_info.get("target_w", None)
        if target is None:
            target = b18_info.get("committed", None)
        if target is None:
            target = b18_info.get("target", None)
        if leg_name in LEG_TO_ID and target is not None:
            if not torch.is_tensor(target):
                target = torch.as_tensor(target)
            return leg_name, target.reshape(3), "b18_target"

    return None, None, "no_target"


def _b34_phase_allows(leg_name, active_swing_names, capture_leg, capture_count, post_capture_pin_count):
    in_swing = leg_name in active_swing_names if leg_name is not None else False
    in_capture = (capture_leg == leg_name and int(capture_count) > 0)
    in_post = (leg_name is not None and int(post_capture_pin_count) > 0)

    allowed = (
        (args.b34_apply_during_swing and in_swing)
        or (args.b34_apply_during_capture and in_capture)
        or (args.b34_apply_post_capture_extra and in_post)
    )
    phase = "swing" if in_swing else ("capture" if in_capture else ("post_capture" if in_post else "not_allowed"))
    return allowed, phase


def _b34_apply_empirical_rf_x_bridge(robot, foot_pos_arg, b23_info, b18_info,
                                     active_swing_names, capture_leg, capture_count,
                                     post_capture_pin_count, step):
    """Empirical directional bridge.

    B33 diagnostic gave the local RF empirical rule:
        hy+ moves RF foot x negative
        hy- moves RF foot x positive

    Therefore:
        e_x = committed_x - foot_x
        if e_x > 0, q_hy should move negative.

    This is deliberately not a full inverse-Jacobian controller. It is a bounded,
    latched directional assist to push RF forward when it lands behind the committed foothold.
    """
    info = {
        "active": False,
        "reason": "disabled",
        "leg": None,
        "source": None,
        "phase": None,
        "target_w": None,
        "foot_w": None,
        "foot_err": None,
        "e_x_raw": None,
        "e_x_used": None,
        "dq_des": None,
        "q_now": None,
        "q_latch": None,
        "q_goal": None,
        "q_cmd": None,
        "delta_from_now": None,
        "delta_from_latch": None,
        "hold_age": None,
        "latched_new": None,
    }

    if not args.enable_b34_empirical_rf_x_bridge:
        return info

    target_leg = str(args.b34_leg).upper()
    leg_name, target_w, source = _b34_pick_leg_and_target(b23_info, b18_info)

    if leg_name is None or target_w is None:
        # Hold recent goal through short target gaps.
        if _B34_BRIDGE_STATE.get("active", False) and _B34_BRIDGE_STATE.get("q_goal") is not None:
            hold_age = int(step) - int(_B34_BRIDGE_STATE.get("last_active_step", -1))
            if hold_age <= int(args.b34_hold_without_target_steps):
                leg_hold = _B34_BRIDGE_STATE.get("leg_name")
                if leg_hold in LEG_TO_ID:
                    leg_id = LEG_TO_ID[leg_hold]
                    joint_ids = _b12_get_leg_joint_ids(leg_id)
                    q_now = robot.data.joint_pos.detach()[:, joint_ids]
                    q_goal = _B34_BRIDGE_STATE["q_goal"].to(device=q_now.device, dtype=q_now.dtype)
                    delta_from_now = torch.clamp(
                        q_goal - q_now,
                        -float(args.b34_max_joint_delta),
                        float(args.b34_max_joint_delta),
                    )
                    q_cmd = q_now + delta_from_now
                    ids = torch.tensor(joint_ids, device=q_now.device, dtype=torch.long)
                    robot.set_joint_position_target(q_cmd, joint_ids=ids)
                    if args.b34_zero_target_velocity:
                        robot.set_joint_velocity_target(torch.zeros_like(q_cmd), joint_ids=ids)
                    info.update({
                        "active": True,
                        "reason": "hold_latched_gap",
                        "leg": leg_hold,
                        "source": _B34_BRIDGE_STATE.get("source", "held"),
                        "phase": "hold_gap",
                        "q_now": q_now[0].detach().cpu().numpy(),
                        "q_latch": _B34_BRIDGE_STATE["q_latch"][0].detach().cpu().numpy(),
                        "q_goal": q_goal[0].detach().cpu().numpy(),
                        "q_cmd": q_cmd[0].detach().cpu().numpy(),
                        "delta_from_now": delta_from_now[0].detach().cpu().numpy(),
                        "hold_age": hold_age,
                    })
                    return info

        info.update({"reason": "no_target", "source": source, "phase": "no_target"})
        return info

    if leg_name != target_leg:
        info.update({"reason": "not_target_leg", "leg": leg_name, "source": source})
        return info

    allowed, phase = _b34_phase_allows(leg_name, active_swing_names, capture_leg, capture_count, post_capture_pin_count)
    if not allowed:
        info.update({"reason": "phase_not_allowed", "leg": leg_name, "source": source, "phase": phase})
        return info

    leg_id = LEG_TO_ID[leg_name]
    joint_ids = _b12_get_leg_joint_ids(leg_id)
    q_now = robot.data.joint_pos.detach()[:, joint_ids]
    foot_w = foot_pos_arg[0, leg_id, :].detach()
    target_w = target_w.to(device=foot_w.device, dtype=foot_w.dtype)
    foot_err = target_w - foot_w

    e_x_raw = float(foot_err[0].detach().cpu().item())
    e_y_raw = float(foot_err[1].detach().cpu().item())

    # Deadband and saturation on x error.
    if abs(e_x_raw) < float(args.b34_deadband_x):
        e_x_used = 0.0
    else:
        e_x_used = max(-float(args.b34_max_x_err), min(float(args.b34_max_x_err), e_x_raw))

    dq_des = torch.zeros_like(q_now)
    # Joint order is [hx, hy, kn].
    # Empirical rule: +x desired -> hy negative.
    dq_des[0, 1] = -float(args.b34_k_hy_per_x) * float(e_x_used)
    dq_des[0, 1] = torch.clamp(
        dq_des[0, 1],
        -float(args.b34_max_joint_delta),
        float(args.b34_max_joint_delta),
    )

    # Optional tiny lateral correction using hx. This is conservative because hx response was weak/noisy.
    if args.b34_include_small_hx:
        dq_hx = -float(args.b34_k_hx_per_y) * float(e_y_raw)
        dq_hx = max(-float(args.b34_max_hx_delta), min(float(args.b34_max_hx_delta), dq_hx))
        dq_des[0, 0] = float(dq_hx)

    need_latch = (not _B34_BRIDGE_STATE.get("active", False)) or (_B34_BRIDGE_STATE.get("leg_name") != leg_name)
    if need_latch:
        q_latch = q_now.detach().clone()
    else:
        q_latch = _B34_BRIDGE_STATE["q_latch"].to(device=q_now.device, dtype=q_now.dtype)

    q_goal_raw = q_now + dq_des
    delta_from_latch = torch.clamp(
        q_goal_raw - q_latch,
        -float(args.b34_max_abs_from_latch),
        float(args.b34_max_abs_from_latch),
    )
    q_goal = q_latch + delta_from_latch

    delta_from_now = torch.clamp(
        q_goal - q_now,
        -float(args.b34_max_joint_delta),
        float(args.b34_max_joint_delta),
    )
    q_cmd = q_now + delta_from_now

    ids = torch.tensor(joint_ids, device=q_now.device, dtype=torch.long)
    robot.set_joint_position_target(q_cmd, joint_ids=ids)
    if args.b34_zero_target_velocity:
        robot.set_joint_velocity_target(torch.zeros_like(q_cmd), joint_ids=ids)

    _B34_BRIDGE_STATE.update({
        "active": True,
        "leg_name": leg_name,
        "q_latch": q_latch.detach().clone(),
        "q_goal": q_goal.detach().clone(),
        "target_w": target_w.detach().clone(),
        "last_active_step": int(step),
        "source": source,
    })

    info.update({
        "active": True,
        "reason": "empirical_rf_x_correction",
        "leg": leg_name,
        "source": source,
        "phase": phase,
        "target_w": target_w.detach().cpu().numpy(),
        "foot_w": foot_w.detach().cpu().numpy(),
        "foot_err": foot_err.detach().cpu().numpy(),
        "e_x_raw": e_x_raw,
        "e_x_used": e_x_used,
        "dq_des": dq_des[0].detach().cpu().numpy(),
        "q_now": q_now[0].detach().cpu().numpy(),
        "q_latch": q_latch[0].detach().cpu().numpy(),
        "q_goal": q_goal[0].detach().cpu().numpy(),
        "q_cmd": q_cmd[0].detach().cpu().numpy(),
        "delta_from_now": delta_from_now[0].detach().cpu().numpy(),
        "delta_from_latch": delta_from_latch[0].detach().cpu().numpy(),
        "hold_age": 0,
        "latched_new": bool(need_latch),
    })
    return info


def _b35_pick_leg_target_sigma(b23_info, b18_info):
    """Pick committed target and sigma, preferring the main B23 foot ref."""
    if isinstance(b23_info, dict) and b23_info.get("active", False):
        leg_name = b23_info.get("leg")
        target = b23_info.get("committed", None)
        if target is None:
            target = b23_info.get("target_after", None)
        sigma = b23_info.get("sigma", None)
        if leg_name in LEG_TO_ID and target is not None:
            if not torch.is_tensor(target):
                target = torch.as_tensor(target)
            return leg_name, target.reshape(3), sigma, "b23_committed"

    if isinstance(b18_info, dict) and b18_info.get("active", False):
        leg_name = b18_info.get("leg")
        target = b18_info.get("target_w", None)
        if target is None:
            target = b18_info.get("committed", None)
        if target is None:
            target = b18_info.get("target", None)
        sigma = b18_info.get("sigma", None)
        if leg_name in LEG_TO_ID and target is not None:
            if not torch.is_tensor(target):
                target = torch.as_tensor(target)
            return leg_name, target.reshape(3), sigma, "b18_target"

    return None, None, None, "no_target"


def _b35_phase_allows(leg_name, active_swing_names, capture_leg, capture_count, post_capture_pin_count):
    in_swing = leg_name in active_swing_names if leg_name is not None else False
    in_capture = (capture_leg == leg_name and int(capture_count) > 0)
    in_post = (leg_name is not None and int(post_capture_pin_count) > 0)

    allowed = (
        (args.b35_apply_during_swing and in_swing)
        or (args.b35_apply_during_capture and in_capture)
        or (args.b35_apply_post_capture_extra and in_post)
    )
    phase = "swing" if in_swing else ("capture" if in_capture else ("post_capture" if in_post else "not_allowed"))
    return allowed, phase


def _b35_apply_pure_empirical_hy_assist(robot, foot_pos_arg, b23_info, b18_info,
                                        active_swing_names, capture_leg, capture_count,
                                        post_capture_pin_count, step):
    """Pure additive HY assist without latch/hold contamination.

    B34 improved RF x placement, but q_latch/q_goal caused HX/KN contamination and
    long hold_latched_gap behavior. B35 removes that structure.

    It only writes the HY joint for the selected leg:
        e_x = target_x - foot_x
        dq_hy = -k * clamp_deadband(e_x)

    Then:
        q_cmd = q_now
        q_cmd[hy] += clamp(dq_hy)

    HX and KN targets are explicitly left at q_now, so the bridge cannot inject
    accumulated latch drift into those axes.
    """
    info = {
        "active": False,
        "reason": "disabled",
        "leg": None,
        "source": None,
        "phase": None,
        "sigma": None,
        "target_w": None,
        "foot_w": None,
        "foot_err": None,
        "e_x_raw": None,
        "e_x_used": None,
        "dq_hy": None,
        "q_now": None,
        "q_cmd": None,
        "delta_from_now": None,
    }

    if not args.enable_b35_pure_empirical_hy_assist:
        return info

    target_leg = str(args.b35_leg).upper()
    leg_name, target_w, sigma, source = _b35_pick_leg_target_sigma(b23_info, b18_info)

    if leg_name is None or target_w is None:
        info.update({"reason": "no_target", "source": source, "phase": "no_target"})
        return info

    if leg_name != target_leg:
        info.update({"reason": "not_target_leg", "leg": leg_name, "source": source})
        return info

    allowed, phase = _b35_phase_allows(leg_name, active_swing_names, capture_leg, capture_count, post_capture_pin_count)
    if not allowed:
        info.update({"reason": "phase_not_allowed", "leg": leg_name, "source": source, "phase": phase})
        return info

    # Optional sigma window, only when sigma is available. Capture/post-capture often has no
    # meaningful sigma, so do not block those phases.
    sigma_val = None
    try:
        if sigma is not None:
            sigma_val = float(sigma)
    except Exception:
        sigma_val = None
    if phase == "swing" and sigma_val is not None:
        if sigma_val < float(args.b35_min_sigma) or sigma_val > float(args.b35_max_sigma):
            info.update({
                "reason": "sigma_out_of_window",
                "leg": leg_name,
                "source": source,
                "phase": phase,
                "sigma": sigma_val,
            })
            return info

    leg_id = LEG_TO_ID[leg_name]
    joint_ids = _b12_get_leg_joint_ids(leg_id)
    q_now = robot.data.joint_pos.detach()[:, joint_ids]
    foot_w = foot_pos_arg[0, leg_id, :].detach()
    target_w = target_w.to(device=foot_w.device, dtype=foot_w.dtype)
    foot_err = target_w - foot_w

    e_x_raw = float(foot_err[0].detach().cpu().item())
    if abs(e_x_raw) < float(args.b35_deadband_x):
        e_x_used = 0.0
    else:
        e_x_used = max(-float(args.b35_max_x_err), min(float(args.b35_max_x_err), e_x_raw))

    # Empirical RF rule from B33:
    #   hy+ -> x negative, hy- -> x positive.
    # Therefore positive x error requires negative hy command.
    dq_hy = -float(args.b35_k_hy_per_x) * float(e_x_used)
    dq_hy = max(-float(args.b35_max_hy_delta), min(float(args.b35_max_hy_delta), dq_hy))

    q_cmd = q_now.detach().clone()
    q_cmd[0, 1] = q_now[0, 1] + float(dq_hy)  # [hx, hy, kn], modify HY only.

    ids = torch.tensor(joint_ids, device=q_now.device, dtype=torch.long)
    robot.set_joint_position_target(q_cmd, joint_ids=ids)
    if args.b35_zero_target_velocity:
        # Zero only HY velocity target; leave the command vector shape matching joint_ids.
        v_cmd = torch.zeros_like(q_cmd)
        robot.set_joint_velocity_target(v_cmd, joint_ids=ids)

    delta_from_now = q_cmd - q_now

    info.update({
        "active": True,
        "reason": "pure_empirical_hy_assist",
        "leg": leg_name,
        "source": source,
        "phase": phase,
        "sigma": sigma_val,
        "target_w": target_w.detach().cpu().numpy(),
        "foot_w": foot_w.detach().cpu().numpy(),
        "foot_err": foot_err.detach().cpu().numpy(),
        "e_x_raw": e_x_raw,
        "e_x_used": e_x_used,
        "dq_hy": dq_hy,
        "q_now": q_now[0].detach().cpu().numpy(),
        "q_cmd": q_cmd[0].detach().cpu().numpy(),
        "delta_from_now": delta_from_now[0].detach().cpu().numpy(),
    })
    return info


def _b36_pick_leg_target_sigma(b23_info, b18_info):
    if isinstance(b23_info, dict) and b23_info.get("active", False):
        leg_name = b23_info.get("leg")
        target = b23_info.get("committed", None)
        if target is None:
            target = b23_info.get("target_after", None)
        sigma = b23_info.get("sigma", None)
        if leg_name in LEG_TO_ID and target is not None:
            if not torch.is_tensor(target):
                target = torch.as_tensor(target)
            return leg_name, target.reshape(3), sigma, "b23_committed"
    if isinstance(b18_info, dict) and b18_info.get("active", False):
        leg_name = b18_info.get("leg")
        target = b18_info.get("target_w", None)
        if target is None:
            target = b18_info.get("committed", None)
        if target is None:
            target = b18_info.get("target", None)
        sigma = b18_info.get("sigma", None)
        if leg_name in LEG_TO_ID and target is not None:
            if not torch.is_tensor(target):
                target = torch.as_tensor(target)
            return leg_name, target.reshape(3), sigma, "b18_target"
    return None, None, None, "no_target"


def _b36_phase_allows(leg_name, active_swing_names, capture_leg, capture_count, post_capture_pin_count):
    in_swing = leg_name in active_swing_names if leg_name is not None else False
    in_capture = (capture_leg == leg_name and int(capture_count) > 0)
    in_post = (leg_name is not None and int(post_capture_pin_count) > 0)
    allowed = ((args.b36_apply_during_swing and in_swing)
               or (args.b36_apply_during_capture and in_capture)
               or (args.b36_apply_post_capture_extra and in_post))
    phase = "swing" if in_swing else ("capture" if in_capture else ("post_capture" if in_post else "not_allowed"))
    return allowed, phase


def _b36_apply_pure_empirical_hy_kn_assist(robot, foot_pos_arg, b23_info, b18_info,
                                           active_swing_names, capture_leg, capture_count,
                                           post_capture_pin_count, step):
    info = {"active": False, "reason": "disabled", "leg": None, "source": None,
            "phase": None, "sigma": None, "target_w": None, "foot_w": None,
            "foot_err": None, "e_x_raw": None, "e_x_used": None,
            "dq_hy": None, "dq_kn": None, "q_now": None, "q_cmd": None,
            "delta_from_now": None}
    if not args.enable_b36_pure_empirical_hy_kn_assist:
        return info

    target_leg = str(args.b36_leg).upper()
    leg_name, target_w, sigma, source = _b36_pick_leg_target_sigma(b23_info, b18_info)
    if leg_name is None or target_w is None:
        info.update({"reason": "no_target", "source": source, "phase": "no_target"})
        return info
    if leg_name != target_leg:
        info.update({"reason": "not_target_leg", "leg": leg_name, "source": source})
        return info

    allowed, phase = _b36_phase_allows(leg_name, active_swing_names, capture_leg, capture_count, post_capture_pin_count)
    if not allowed:
        info.update({"reason": "phase_not_allowed", "leg": leg_name, "source": source, "phase": phase})
        return info

    sigma_val = None
    try:
        if sigma is not None:
            sigma_val = float(sigma)
    except Exception:
        sigma_val = None
    if phase == "swing" and sigma_val is not None:
        if sigma_val < float(args.b36_min_sigma) or sigma_val > float(args.b36_max_sigma):
            info.update({"reason": "sigma_out_of_window", "leg": leg_name, "source": source,
                         "phase": phase, "sigma": sigma_val})
            return info

    leg_id = LEG_TO_ID[leg_name]
    joint_ids = _b12_get_leg_joint_ids(leg_id)
    q_now = robot.data.joint_pos.detach()[:, joint_ids]
    foot_w = foot_pos_arg[0, leg_id, :].detach()
    target_w = target_w.to(device=foot_w.device, dtype=foot_w.dtype)
    foot_err = target_w - foot_w
    e_x_raw = float(foot_err[0].detach().cpu().item())
    if abs(e_x_raw) < float(args.b36_deadband_x):
        e_x_used = 0.0
    else:
        e_x_used = max(-float(args.b36_max_x_err), min(float(args.b36_max_x_err), e_x_raw))

    # B33/B34 empirical rule: x target ahead => HY negative, small KN positive.
    dq_hy = -float(args.b36_k_hy_per_x) * float(e_x_used)
    dq_hy = max(-float(args.b36_max_hy_delta), min(float(args.b36_max_hy_delta), dq_hy))
    dq_kn = float(args.b36_kn_sign) * float(args.b36_k_kn_per_x) * float(e_x_used)
    dq_kn = max(-float(args.b36_max_kn_delta), min(float(args.b36_max_kn_delta), dq_kn))

    q_cmd = q_now.detach().clone()
    q_cmd[0, 1] = q_now[0, 1] + float(dq_hy)
    q_cmd[0, 2] = q_now[0, 2] + float(dq_kn)
    ids = torch.tensor(joint_ids, device=q_now.device, dtype=torch.long)
    robot.set_joint_position_target(q_cmd, joint_ids=ids)
    if args.b36_zero_target_velocity:
        robot.set_joint_velocity_target(torch.zeros_like(q_cmd), joint_ids=ids)
    delta_from_now = q_cmd - q_now

    info.update({"active": True, "reason": "pure_empirical_hy_kn_assist",
                 "leg": leg_name, "source": source, "phase": phase, "sigma": sigma_val,
                 "target_w": target_w.detach().cpu().numpy(),
                 "foot_w": foot_w.detach().cpu().numpy(),
                 "foot_err": foot_err.detach().cpu().numpy(),
                 "e_x_raw": e_x_raw, "e_x_used": e_x_used,
                 "dq_hy": dq_hy, "dq_kn": dq_kn,
                 "q_now": q_now[0].detach().cpu().numpy(),
                 "q_cmd": q_cmd[0].detach().cpu().numpy(),
                 "delta_from_now": delta_from_now[0].detach().cpu().numpy()})
    return info

def _b37_parse_leg_list(s):
    out = []
    for item in str(s).split(","):
        name = item.strip().upper()
        if name in LEG_TO_ID and name not in out:
            out.append(name)
    return out


def _b37_parse_sign_map(s, default=1.0):
    m = {}
    for item in str(s).split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        k, v = item.split(":", 1)
        leg = k.strip().upper()
        if leg not in LEG_TO_ID:
            continue
        try:
            val = float(v.strip())
            m[leg] = 1.0 if val >= 0 else -1.0
        except Exception:
            pass
    for leg in ["LF", "RF", "LH", "RH"]:
        if leg not in m:
            m[leg] = float(default)
    return m



def _b38_leg_allowed(candidate):
    try:
        cand_name = ID_TO_LEG.get(int(candidate), str(candidate)).upper()
    except Exception:
        return False
    allowed = []
    for item in str(args.b38_candidate_legs).split(','):
        name = item.strip().upper()
        if name in LEG_TO_ID and name not in allowed:
            allowed.append(name)
    return cand_name in allowed

def _b39_leg_allowed(candidate):
    try:
        cand_name = ID_TO_LEG.get(int(candidate), str(candidate)).upper()
    except Exception:
        return False
    allowed = []
    for item in str(args.b39_candidate_legs).split(','):
        name = item.strip().upper()
        if name in LEG_TO_ID and name not in allowed:
            allowed.append(name)
    return cand_name in allowed

_B37_MOVEMENT_STATE = {
    "last_foot_w": {},
    "last_step": {},
    "active_counts": {"LF": 0, "RF": 0, "LH": 0, "RH": 0},
    "max_step_delta": {"LF": 0.0, "RF": 0.0, "LH": 0.0, "RH": 0.0},
    "max_err_x": {"LF": 0.0, "RF": 0.0, "LH": 0.0, "RH": 0.0},
}

_B43_MOVEMENT_STATE = {
    "last_foot_w": {},
    "last_step": {},
    "active_counts": {"LF": 0, "RF": 0, "LH": 0, "RH": 0},
    "max_step_delta": {"LF": 0.0, "RF": 0.0, "LH": 0.0, "RH": 0.0},
    "max_err_y": {"LF": 0.0, "RF": 0.0, "LH": 0.0, "RH": 0.0},
}

_B44_LATCH_STATE = {
    "q_latch": {},
    "target_latch": {},
    "last_foot_w": {},
    "last_step": {},
    "last_phase": {},
    "active_counts": {"LF": 0, "RF": 0, "LH": 0, "RH": 0},
    "latch_counts": {"LF": 0, "RF": 0, "LH": 0, "RH": 0},
    "max_step_delta": {"LF": 0.0, "RF": 0.0, "LH": 0.0, "RH": 0.0},
    "max_err_y": {"LF": 0.0, "RF": 0.0, "LH": 0.0, "RH": 0.0},
}


_B45_PUSH_STATE = {
    "q_latch": {},
    "last_phase": {},
    "last_step": {},
    "last_foot_w": {},
    "active_counts": {"LF": 0, "RF": 0, "LH": 0, "RH": 0},
    "latch_counts": {"LF": 0, "RF": 0, "LH": 0, "RH": 0},
    "max_step_delta": {"LF": 0.0, "RF": 0.0, "LH": 0.0, "RH": 0.0},
    "max_err_y": {"LF": 0.0, "RF": 0.0, "LH": 0.0, "RH": 0.0},
    "blocked_counts": {"LF": 0, "RF": 0, "LH": 0, "RH": 0},
}



def _b37_pick_leg_target_sigma(b23_info, b18_info):
    """Pick target and sigma, preferring B23 main foot ref.

    B49 note:
      For foothold planning, the committed target is the touchdown point
      at ground height.  For swing execution, however, bridges/assists
      should often track the elevated swing reference (B23 target_after)
      so the leg actually lifts.  When requested, prefer target_after.
    """
    if isinstance(b23_info, dict) and b23_info.get("active", False):
        leg_name = b23_info.get("leg")
        if bool(getattr(args, "enable_b49_elevated_swing_tracking_reference", False)) and bool(getattr(args, "b49_apply_to_b37", False)):
            target = b23_info.get("target_after", None)
            source_name = "b23_target_after_elevated"
            if target is None:
                target = b23_info.get("committed", None)
                source_name = "b23_committed"
        else:
            target = b23_info.get("committed", None)
            source_name = "b23_committed"
            if target is None:
                target = b23_info.get("target_after", None)
                source_name = "b23_target_after"
        sigma = b23_info.get("sigma", None)
        if leg_name in LEG_TO_ID and target is not None:
            if not torch.is_tensor(target):
                target = torch.as_tensor(target)
            return leg_name, target.reshape(3), sigma, source_name

    if isinstance(b18_info, dict) and b18_info.get("active", False):
        leg_name = b18_info.get("leg")
        target = b18_info.get("target_w", None)
        if target is None:
            target = b18_info.get("committed", None)
        if target is None:
            target = b18_info.get("target", None)
        sigma = b18_info.get("sigma", None)
        if leg_name in LEG_TO_ID and target is not None:
            if not torch.is_tensor(target):
                target = torch.as_tensor(target)
            return leg_name, target.reshape(3), sigma, "b18_target"

    return None, None, None, "no_target"


def _b37_phase_allows(leg_name, active_swing_names, capture_leg, capture_count, post_capture_pin_count):
    in_swing = leg_name in active_swing_names if leg_name is not None else False
    in_capture = (capture_leg == leg_name and int(capture_count) > 0)
    in_post = (leg_name is not None and int(post_capture_pin_count) > 0)

    allowed = (
        (args.b37_apply_during_swing and in_swing)
        or (args.b37_apply_during_capture and in_capture)
        or (args.b37_apply_post_capture_extra and in_post)
    )
    phase = "swing" if in_swing else ("capture" if in_capture else ("post_capture" if in_post else "not_allowed"))
    return allowed, phase


def _b37_apply_all_leg_empirical_hy_kn_assist(robot, foot_pos_arg, b23_info, b18_info,
                                              active_swing_names, capture_leg, capture_count,
                                              post_capture_pin_count, step):
    """All-leg clean additive HY+KN assist.

    This is a movement/execution check:
      - keep the clean B36 additive structure,
      - allow all legs,
      - log whether each committed leg receives a clean command and physically moves.
    """
    info = {
        "active": False, "reason": "disabled", "leg": None, "allowed_legs": None,
        "source": None, "phase": None, "sigma": None, "target_w": None,
        "foot_w": None, "foot_err": None, "e_x_raw": None, "e_x_used": None,
        "hy_sign": None, "kn_sign": None, "dq_hy": None, "dq_kn": None,
        "q_now": None, "q_cmd": None, "delta_from_now": None,
        "step_foot_delta": None, "step_foot_delta_norm": None,
        "active_counts": None, "max_step_delta": None, "max_err_x": None,
    }

    if not args.enable_b37_all_leg_empirical_hy_kn_assist:
        return info

    allowed_legs = _b37_parse_leg_list(args.b37_legs)
    hy_sign_map = _b37_parse_sign_map(args.b37_hy_signs, default=-1.0)
    kn_sign_map = _b37_parse_sign_map(args.b37_kn_signs, default=1.0)

    leg_name, target_w, sigma, source = _b37_pick_leg_target_sigma(b23_info, b18_info)
    if leg_name is None or target_w is None:
        info.update({"reason": "no_target", "source": source, "phase": "no_target", "allowed_legs": allowed_legs})
        return info

    if leg_name not in allowed_legs:
        info.update({"reason": "leg_not_allowed", "leg": leg_name, "source": source, "allowed_legs": allowed_legs})
        return info

    allowed, phase = _b37_phase_allows(leg_name, active_swing_names, capture_leg, capture_count, post_capture_pin_count)
    if not allowed:
        info.update({"reason": "phase_not_allowed", "leg": leg_name, "source": source, "phase": phase, "allowed_legs": allowed_legs})
        return info

    sigma_val = None
    try:
        if sigma is not None:
            sigma_val = float(sigma)
    except Exception:
        sigma_val = None
    if phase == "swing" and sigma_val is not None:
        if sigma_val < float(args.b37_min_sigma) or sigma_val > float(args.b37_max_sigma):
            info.update({"reason": "sigma_out_of_window", "leg": leg_name, "source": source,
                         "phase": phase, "sigma": sigma_val, "allowed_legs": allowed_legs})
            return info

    leg_id = LEG_TO_ID[leg_name]
    joint_ids = _b12_get_leg_joint_ids(leg_id)
    q_now = robot.data.joint_pos.detach()[:, joint_ids]
    foot_w = foot_pos_arg[0, leg_id, :].detach()
    target_w = target_w.to(device=foot_w.device, dtype=foot_w.dtype)
    foot_err = target_w - foot_w

    e_x_raw = float(foot_err[0].detach().cpu().item())
    if abs(e_x_raw) < float(args.b37_deadband_x):
        e_x_used = 0.0
    else:
        e_x_used = max(-float(args.b37_max_x_err), min(float(args.b37_max_x_err), e_x_raw))

    hy_sign = float(hy_sign_map.get(leg_name, -1.0))
    kn_sign = float(kn_sign_map.get(leg_name, 1.0))
    dq_hy = hy_sign * float(args.b37_k_hy_per_x) * float(e_x_used)
    dq_hy = max(-float(args.b37_max_hy_delta), min(float(args.b37_max_hy_delta), dq_hy))
    dq_kn = kn_sign * float(args.b37_k_kn_per_x) * float(e_x_used)
    dq_kn = max(-float(args.b37_max_kn_delta), min(float(args.b37_max_kn_delta), dq_kn))

    q_cmd = q_now.detach().clone()
    q_cmd[0, 1] = q_now[0, 1] + float(dq_hy)
    q_cmd[0, 2] = q_now[0, 2] + float(dq_kn)

    ids = torch.tensor(joint_ids, device=q_now.device, dtype=torch.long)
    robot.set_joint_position_target(q_cmd, joint_ids=ids)
    if args.b37_zero_target_velocity:
        robot.set_joint_velocity_target(torch.zeros_like(q_cmd), joint_ids=ids)

    delta_from_now = q_cmd - q_now

    last = _B37_MOVEMENT_STATE["last_foot_w"].get(leg_name, None)
    if last is not None:
        last = last.to(device=foot_w.device, dtype=foot_w.dtype)
        step_foot_delta = foot_w - last
        step_foot_delta_norm = float(torch.linalg.norm(step_foot_delta).item())
    else:
        step_foot_delta = torch.zeros_like(foot_w)
        step_foot_delta_norm = 0.0

    _B37_MOVEMENT_STATE["last_foot_w"][leg_name] = foot_w.detach().clone()
    _B37_MOVEMENT_STATE["last_step"][leg_name] = int(step)
    _B37_MOVEMENT_STATE["active_counts"][leg_name] = int(_B37_MOVEMENT_STATE["active_counts"].get(leg_name, 0)) + 1
    _B37_MOVEMENT_STATE["max_step_delta"][leg_name] = max(float(_B37_MOVEMENT_STATE["max_step_delta"].get(leg_name, 0.0)), step_foot_delta_norm)
    _B37_MOVEMENT_STATE["max_err_x"][leg_name] = max(float(_B37_MOVEMENT_STATE["max_err_x"].get(leg_name, 0.0)), abs(e_x_raw))

    info.update({
        "active": True, "reason": "all_leg_empirical_hy_kn_assist", "leg": leg_name,
        "allowed_legs": allowed_legs, "source": source, "phase": phase, "sigma": sigma_val,
        "target_w": target_w.detach().cpu().numpy(), "foot_w": foot_w.detach().cpu().numpy(),
        "foot_err": foot_err.detach().cpu().numpy(), "e_x_raw": e_x_raw, "e_x_used": e_x_used,
        "hy_sign": hy_sign, "kn_sign": kn_sign, "dq_hy": dq_hy, "dq_kn": dq_kn,
        "q_now": q_now[0].detach().cpu().numpy(), "q_cmd": q_cmd[0].detach().cpu().numpy(),
        "delta_from_now": delta_from_now[0].detach().cpu().numpy(),
        "step_foot_delta": step_foot_delta.detach().cpu().numpy(),
        "step_foot_delta_norm": step_foot_delta_norm,
        "active_counts": dict(_B37_MOVEMENT_STATE["active_counts"]),
        "max_step_delta": dict(_B37_MOVEMENT_STATE["max_step_delta"]),
        "max_err_x": dict(_B37_MOVEMENT_STATE["max_err_x"]),
    })
    return info


def _b43_phase_allows(leg_name, active_swing_names, capture_leg, capture_count, post_capture_pin_count):
    in_swing = leg_name in active_swing_names if leg_name is not None else False
    in_capture = (capture_leg == leg_name and int(capture_count) > 0)
    in_post = (leg_name is not None and int(post_capture_pin_count) > 0)
    allowed = (
        (args.b43_apply_during_swing and in_swing)
        or (args.b43_apply_during_capture and in_capture)
        or (args.b43_apply_post_capture_extra and in_post)
    )
    phase = "swing" if in_swing else ("capture" if in_capture else ("post_capture" if in_post else "not_allowed"))
    return allowed, phase


def _b43_apply_hind_empirical_lateral_y_bridge(robot, foot_pos_arg, b23_info, b18_info,
                                               active_swing_names, capture_leg, capture_count,
                                               post_capture_pin_count, step):
    """Empirical hind lateral-y bridge from B42 response probe.

    It is intentionally independent of analytic J^T F. It uses the B42-observed mapping
    to write bounded joint-position targets for HX/HY from lateral foot error.
    e_y = target_y - foot_y. Default signs yield for LH: e_y<0 -> hx+, hy-.
    """
    info = {
        "active": False, "reason": "disabled", "leg": None, "allowed_legs": None,
        "source": None, "phase": None, "sigma": None, "target_w": None,
        "foot_w": None, "foot_err": None, "e_y_raw": None, "e_y_used": None,
        "hx_sign": None, "hy_sign": None, "dq_hx": None, "dq_hy": None,
        "q_now": None, "q_cmd": None, "delta_from_now": None,
        "step_foot_delta": None, "step_foot_delta_norm": None,
        "active_counts": None, "max_step_delta": None, "max_err_y": None,
    }
    if not args.enable_b43_hind_empirical_lateral_y_bridge:
        return info

    allowed_legs = _b37_parse_leg_list(args.b43_legs)
    hx_sign_map = _b37_parse_sign_map(args.b43_hx_signs, default=-1.0)
    hy_sign_map = _b37_parse_sign_map(args.b43_hy_signs, default=1.0)

    leg_name, target_w, sigma, source = _b37_pick_leg_target_sigma(b23_info, b18_info)
    if leg_name is None or target_w is None:
        info.update({"reason": "no_target", "source": source, "phase": "no_target", "allowed_legs": allowed_legs})
        return info
    if leg_name not in allowed_legs:
        info.update({"reason": "leg_not_allowed", "leg": leg_name, "source": source, "allowed_legs": allowed_legs})
        return info

    allowed, phase = _b43_phase_allows(leg_name, active_swing_names, capture_leg, capture_count, post_capture_pin_count)
    if not allowed:
        info.update({"reason": "phase_not_allowed", "leg": leg_name, "source": source, "phase": phase, "allowed_legs": allowed_legs})
        return info

    sigma_val = None
    try:
        if sigma is not None:
            sigma_val = float(sigma)
    except Exception:
        sigma_val = None
    if phase == "swing" and sigma_val is not None:
        if sigma_val < float(args.b43_min_sigma) or sigma_val > float(args.b43_max_sigma):
            info.update({"reason": "sigma_out_of_window", "leg": leg_name, "source": source,
                         "phase": phase, "sigma": sigma_val, "allowed_legs": allowed_legs})
            return info

    leg_id = LEG_TO_ID[leg_name]
    joint_ids = _b12_get_leg_joint_ids(leg_id)
    q_now = robot.data.joint_pos.detach()[:, joint_ids]
    foot_w = foot_pos_arg[0, leg_id, :].detach()
    target_w = target_w.to(device=foot_w.device, dtype=foot_w.dtype)
    foot_err = target_w - foot_w

    e_y_raw = float(foot_err[1].detach().cpu().item())
    if abs(e_y_raw) < float(args.b43_deadband_y):
        e_y_used = 0.0
    else:
        e_y_used = max(-float(args.b43_max_y_err), min(float(args.b43_max_y_err), e_y_raw))

    hx_sign = float(hx_sign_map.get(leg_name, -1.0))
    hy_sign = float(hy_sign_map.get(leg_name, 1.0))
    dq_hx = hx_sign * float(args.b43_k_hx_per_y) * float(e_y_used)
    dq_hx = max(-float(args.b43_max_hx_delta), min(float(args.b43_max_hx_delta), dq_hx))
    dq_hy = hy_sign * float(args.b43_k_hy_per_y) * float(e_y_used)
    dq_hy = max(-float(args.b43_max_hy_delta), min(float(args.b43_max_hy_delta), dq_hy))

    q_cmd = q_now.detach().clone()
    q_cmd[0, 0] = q_now[0, 0] + float(dq_hx)
    q_cmd[0, 1] = q_now[0, 1] + float(dq_hy)

    ids = torch.tensor(joint_ids, device=q_now.device, dtype=torch.long)
    robot.set_joint_position_target(q_cmd, joint_ids=ids)
    if args.b43_zero_target_velocity:
        robot.set_joint_velocity_target(torch.zeros_like(q_cmd), joint_ids=ids)

    delta_from_now = q_cmd - q_now

    last = _B43_MOVEMENT_STATE["last_foot_w"].get(leg_name, None)
    if last is not None:
        last = last.to(device=foot_w.device, dtype=foot_w.dtype)
        step_foot_delta = foot_w - last
        step_foot_delta_norm = float(torch.linalg.norm(step_foot_delta).item())
    else:
        step_foot_delta = torch.zeros_like(foot_w)
        step_foot_delta_norm = 0.0

    _B43_MOVEMENT_STATE["last_foot_w"][leg_name] = foot_w.detach().clone()
    _B43_MOVEMENT_STATE["last_step"][leg_name] = int(step)
    _B43_MOVEMENT_STATE["active_counts"][leg_name] = int(_B43_MOVEMENT_STATE["active_counts"].get(leg_name, 0)) + 1
    _B43_MOVEMENT_STATE["max_step_delta"][leg_name] = max(float(_B43_MOVEMENT_STATE["max_step_delta"].get(leg_name, 0.0)), step_foot_delta_norm)
    _B43_MOVEMENT_STATE["max_err_y"][leg_name] = max(float(_B43_MOVEMENT_STATE["max_err_y"].get(leg_name, 0.0)), abs(e_y_raw))

    info.update({
        "active": True, "reason": "hind_empirical_lateral_y_bridge", "leg": leg_name,
        "allowed_legs": allowed_legs, "source": source, "phase": phase, "sigma": sigma_val,
        "target_w": target_w.detach().cpu().numpy(), "foot_w": foot_w.detach().cpu().numpy(),
        "foot_err": foot_err.detach().cpu().numpy(), "e_y_raw": e_y_raw, "e_y_used": e_y_used,
        "hx_sign": hx_sign, "hy_sign": hy_sign, "dq_hx": dq_hx, "dq_hy": dq_hy,
        "q_now": q_now[0].detach().cpu().numpy(), "q_cmd": q_cmd[0].detach().cpu().numpy(),
        "delta_from_now": delta_from_now[0].detach().cpu().numpy(),
        "step_foot_delta": step_foot_delta.detach().cpu().numpy(),
        "step_foot_delta_norm": step_foot_delta_norm,
        "active_counts": dict(_B43_MOVEMENT_STATE["active_counts"]),
        "max_step_delta": dict(_B43_MOVEMENT_STATE["max_step_delta"]),
        "max_err_y": dict(_B43_MOVEMENT_STATE["max_err_y"]),
    })
    return info



def _b44_phase_allows(leg_name, active_swing_names, capture_leg, capture_count, post_capture_pin_count):
    in_swing = leg_name in active_swing_names if leg_name is not None else False
    in_capture = (capture_leg == leg_name and int(capture_count) > 0)
    in_post = (leg_name is not None and int(post_capture_pin_count) > 0)
    allowed = (
        (args.b44_apply_during_swing and in_swing)
        or (args.b44_apply_during_capture and in_capture)
        or (args.b44_apply_post_capture_extra and in_post)
    )
    phase = "swing" if in_swing else ("capture" if in_capture else ("post_capture" if in_post else "not_allowed"))
    return allowed, phase


def _b45_phase_allows(leg_name, active_swing_names, capture_leg, capture_count, post_capture_pin_count):
    """B46 fix: B45 must use B45 phase flags, not B44 phase flags.

    In the first B45 run, logs showed:
      [B45 ...] reason=phase_not_allowed phase=swing leg=LH
    even though --b45_apply_during_swing was passed.

    Cause:
      B45 called _b44_phase_allows(), which checks args.b44_apply_during_*.
      Since B44 was disabled in the intended B45 run, B45 never became active.

    This helper keeps the same phase detection but uses args.b45_apply_during_*.
    """
    in_swing = leg_name in active_swing_names if leg_name is not None else False
    in_capture = (capture_leg == leg_name and int(capture_count) > 0)
    in_post = (leg_name is not None and int(post_capture_pin_count) > 0)
    allowed = (
        (args.b45_apply_during_swing and in_swing)
        or (args.b45_apply_during_capture and in_capture)
        or (args.b45_apply_post_capture_extra and in_post)
    )
    phase = "swing" if in_swing else ("capture" if in_capture else ("post_capture" if in_post else "not_allowed"))
    return allowed, phase


def _b44_apply_latched_hind_y_bridge(robot, foot_pos_arg, b23_info, b18_info,
                                     active_swing_names, capture_leg, capture_count,
                                     post_capture_pin_count, step):
    """B44 latched hind lateral-y bridge.

    This is the direct follow-up to B43. B43 produced a correct-sign command,
    but because it was q_now + tiny offset, the command did not persist enough
    to pull LH foot_y back from the wrong lateral basin. B44 latches q at the
    beginning of the allowed phase and computes q_goal from q_latch.

    Error convention:
      e_y = target_y - foot_y.
    Default signs:
      LH/RH e_y < 0 -> hx positive, hy negative.
    """
    info = {
        "active": False, "reason": "disabled", "leg": None, "allowed_legs": None,
        "source": None, "phase": None, "sigma": None, "target_w": None,
        "foot_w": None, "foot_err": None, "e_y_raw": None, "e_y_used": None,
        "hx_sign": None, "hy_sign": None, "q_latch": None, "q_goal": None,
        "q_now": None, "q_cmd": None, "delta_goal_from_latch": None,
        "delta_goal_from_now": None, "delta_cmd_from_now": None,
        "dq_goal_hx": None, "dq_goal_hy": None, "dq_cmd_hx": None, "dq_cmd_hy": None,
        "latched_now": False, "latch_counts": None,
        "step_foot_delta": None, "step_foot_delta_norm": None,
        "active_counts": None, "max_step_delta": None, "max_err_y": None,
    }
    if not args.enable_b44_latched_hind_y_bridge:
        return info

    allowed_legs = _b37_parse_leg_list(args.b44_legs)
    hx_sign_map = _b37_parse_sign_map(args.b44_hx_signs, default=-1.0)
    hy_sign_map = _b37_parse_sign_map(args.b44_hy_signs, default=1.0)

    leg_name, target_w, sigma, source = _b37_pick_leg_target_sigma(b23_info, b18_info)
    if leg_name is None or target_w is None:
        info.update({"reason": "no_target", "source": source, "phase": "no_target", "allowed_legs": allowed_legs})
        return info
    if leg_name not in allowed_legs:
        info.update({"reason": "leg_not_allowed", "leg": leg_name, "source": source, "allowed_legs": allowed_legs})
        return info

    allowed, phase = _b44_phase_allows(leg_name, active_swing_names, capture_leg, capture_count, post_capture_pin_count)
    if not allowed:
        # End-of-phase cleanup: do not keep pushing stale hind targets after the allowed phase.
        _B44_LATCH_STATE["last_phase"][leg_name] = phase
        info.update({"reason": "phase_not_allowed", "leg": leg_name, "source": source, "phase": phase, "allowed_legs": allowed_legs})
        return info

    sigma_val = None
    try:
        if sigma is not None:
            sigma_val = float(sigma)
    except Exception:
        sigma_val = None
    if phase == "swing" and sigma_val is not None:
        if sigma_val < float(args.b44_min_sigma) or sigma_val > float(args.b44_max_sigma):
            info.update({"reason": "sigma_out_of_window", "leg": leg_name, "source": source,
                         "phase": phase, "sigma": sigma_val, "allowed_legs": allowed_legs})
            return info

    leg_id = LEG_TO_ID[leg_name]
    joint_ids = _b12_get_leg_joint_ids(leg_id)
    q_now = robot.data.joint_pos.detach()[:, joint_ids]
    foot_w = foot_pos_arg[0, leg_id, :].detach()
    target_w = target_w.to(device=foot_w.device, dtype=foot_w.dtype)
    foot_err = target_w - foot_w

    e_y_raw = float(foot_err[1].detach().cpu().item())
    if abs(e_y_raw) < float(args.b44_deadband_y):
        e_y_used = 0.0
    else:
        e_y_used = max(-float(args.b44_max_y_err), min(float(args.b44_max_y_err), e_y_raw))

    last_step = _B44_LATCH_STATE["last_step"].get(leg_name, None)
    last_phase = _B44_LATCH_STATE["last_phase"].get(leg_name, None)
    need_new_latch = (
        leg_name not in _B44_LATCH_STATE["q_latch"]
        or last_step is None
        or int(step) != int(last_step) + 1
        or (args.b44_reset_latch_on_phase_change and last_phase != phase)
    )
    if need_new_latch:
        _B44_LATCH_STATE["q_latch"][leg_name] = q_now.detach().clone()
        _B44_LATCH_STATE["target_latch"][leg_name] = target_w.detach().clone()
        _B44_LATCH_STATE["latch_counts"][leg_name] = int(_B44_LATCH_STATE["latch_counts"].get(leg_name, 0)) + 1

    q_latch = _B44_LATCH_STATE["q_latch"][leg_name].to(device=q_now.device, dtype=q_now.dtype)

    hx_sign = float(hx_sign_map.get(leg_name, -1.0))
    hy_sign = float(hy_sign_map.get(leg_name, 1.0))
    dq_goal_hx = hx_sign * float(args.b44_k_hx_per_y) * float(e_y_used)
    dq_goal_hx = max(-float(args.b44_max_abs_hx_from_latch), min(float(args.b44_max_abs_hx_from_latch), dq_goal_hx))
    dq_goal_hy = hy_sign * float(args.b44_k_hy_per_y) * float(e_y_used)
    dq_goal_hy = max(-float(args.b44_max_abs_hy_from_latch), min(float(args.b44_max_abs_hy_from_latch), dq_goal_hy))

    q_goal = q_latch.detach().clone()
    q_goal[0, 0] = q_latch[0, 0] + float(dq_goal_hx)
    q_goal[0, 1] = q_latch[0, 1] + float(dq_goal_hy)

    delta_goal_from_now = q_goal - q_now
    dq_cmd_hx = float(delta_goal_from_now[0, 0].detach().cpu().item())
    dq_cmd_hy = float(delta_goal_from_now[0, 1].detach().cpu().item())
    dq_cmd_hx = max(-float(args.b44_max_step_hx_delta), min(float(args.b44_max_step_hx_delta), dq_cmd_hx))
    dq_cmd_hy = max(-float(args.b44_max_step_hy_delta), min(float(args.b44_max_step_hy_delta), dq_cmd_hy))

    q_cmd = q_now.detach().clone()
    q_cmd[0, 0] = q_now[0, 0] + float(dq_cmd_hx)
    q_cmd[0, 1] = q_now[0, 1] + float(dq_cmd_hy)
    # Do not touch KN for lateral-y in B44. B42 showed little KN authority for y.

    ids = torch.tensor(joint_ids, device=q_now.device, dtype=torch.long)
    robot.set_joint_position_target(q_cmd, joint_ids=ids)
    if args.b44_zero_target_velocity:
        robot.set_joint_velocity_target(torch.zeros_like(q_cmd), joint_ids=ids)

    delta_goal_from_latch = q_goal - q_latch
    delta_cmd_from_now = q_cmd - q_now

    last = _B44_LATCH_STATE["last_foot_w"].get(leg_name, None)
    if last is not None:
        last = last.to(device=foot_w.device, dtype=foot_w.dtype)
        step_foot_delta = foot_w - last
        step_foot_delta_norm = float(torch.linalg.norm(step_foot_delta).item())
    else:
        step_foot_delta = torch.zeros_like(foot_w)
        step_foot_delta_norm = 0.0

    _B44_LATCH_STATE["last_foot_w"][leg_name] = foot_w.detach().clone()
    _B44_LATCH_STATE["last_step"][leg_name] = int(step)
    _B44_LATCH_STATE["last_phase"][leg_name] = phase
    _B44_LATCH_STATE["active_counts"][leg_name] = int(_B44_LATCH_STATE["active_counts"].get(leg_name, 0)) + 1
    _B44_LATCH_STATE["max_step_delta"][leg_name] = max(float(_B44_LATCH_STATE["max_step_delta"].get(leg_name, 0.0)), step_foot_delta_norm)
    _B44_LATCH_STATE["max_err_y"][leg_name] = max(float(_B44_LATCH_STATE["max_err_y"].get(leg_name, 0.0)), abs(e_y_raw))

    info.update({
        "active": True, "reason": "latched_hind_y_bridge", "leg": leg_name,
        "allowed_legs": allowed_legs, "source": source, "phase": phase, "sigma": sigma_val,
        "target_w": target_w.detach().cpu().numpy(), "foot_w": foot_w.detach().cpu().numpy(),
        "foot_err": foot_err.detach().cpu().numpy(), "e_y_raw": e_y_raw, "e_y_used": e_y_used,
        "hx_sign": hx_sign, "hy_sign": hy_sign,
        "q_latch": q_latch[0].detach().cpu().numpy(), "q_goal": q_goal[0].detach().cpu().numpy(),
        "q_now": q_now[0].detach().cpu().numpy(), "q_cmd": q_cmd[0].detach().cpu().numpy(),
        "delta_goal_from_latch": delta_goal_from_latch[0].detach().cpu().numpy(),
        "delta_goal_from_now": delta_goal_from_now[0].detach().cpu().numpy(),
        "delta_cmd_from_now": delta_cmd_from_now[0].detach().cpu().numpy(),
        "dq_goal_hx": dq_goal_hx, "dq_goal_hy": dq_goal_hy,
        "dq_cmd_hx": dq_cmd_hx, "dq_cmd_hy": dq_cmd_hy,
        "latched_now": bool(need_new_latch),
        "latch_counts": dict(_B44_LATCH_STATE["latch_counts"]),
        "step_foot_delta": step_foot_delta.detach().cpu().numpy(),
        "step_foot_delta_norm": step_foot_delta_norm,
        "active_counts": dict(_B44_LATCH_STATE["active_counts"]),
        "max_step_delta": dict(_B44_LATCH_STATE["max_step_delta"]),
        "max_err_y": dict(_B44_LATCH_STATE["max_err_y"]),
    })
    return info



def _b45_clip_direction_with_latch_guard(q_now_axis, q_latch_axis, step_delta, max_abs_from_latch):
    """Clip a directional step so it does not push farther beyond latch guard.

    B47 fix:
      If step_delta is zero, return zero immediately.

    Why:
      In B46, HY was disabled, so raw_step_hy=0.0. However the old guard still
      clipped q_now back toward the latch boundary, producing huge positive
      cmd_step_hy values such as +0.8 rad. That reintroduced the B44-style
      latch-restoring behavior and invalidated the intended HX-only test.

    Rule:
      - This guard may clip a requested nonzero directional push.
      - It must never create a command on an axis whose requested step is zero.
      - It must never restore toward q_latch by itself.
    """
    q_now_f = float(q_now_axis)
    q_latch_f = float(q_latch_axis)
    step_f = float(step_delta)
    max_abs = abs(float(max_abs_from_latch))

    # Critical B47 fix: no requested motion means no command.
    if abs(step_f) < 1e-12:
        return 0.0, False

    if max_abs <= 0.0:
        return step_f, False

    cur = q_now_f - q_latch_f
    proposed = cur + step_f

    if abs(proposed) <= max_abs:
        return step_f, False

    # If already beyond the guard and the requested step pushes farther out, block it.
    if abs(cur) >= max_abs and (cur * step_f) > 0.0:
        return 0.0, True

    # Otherwise clip to the nearest boundary without reversing direction.
    boundary = max_abs if proposed > 0.0 else -max_abs
    clipped = boundary - cur

    # Do not allow numerical clipping to reverse the originally requested sign.
    if clipped * step_f < 0.0:
        return 0.0, True

    return clipped, True


def _b45_apply_directional_hind_y_push_bridge(robot, foot_pos_arg, b23_info, b18_info,
                                              active_swing_names, capture_leg, capture_count,
                                              post_capture_pin_count, step):
    """B45 directional hind lateral-y push bridge.

    B44 showed that q_latch -> q_goal restoration can reverse the actual
    per-step command when q_now has already drifted past q_goal. B45 goes back
    to the empirical B42 sign rule and commands a current-state directional push:

      e_y = target_y - foot_y
      LH/RH default: e_y < 0 -> hx+, optional hy-

    It uses a phase latch only as a runaway guard, not as a restoring target.
    """
    info = {
        "active": False, "reason": "disabled", "leg": None, "allowed_legs": None,
        "source": None, "phase": None, "sigma": None, "target_w": None,
        "foot_w": None, "foot_err": None, "e_y_raw": None, "e_y_used": None,
        "hx_sign": None, "hy_sign": None, "q_latch": None, "q_now": None, "q_cmd": None,
        "raw_step_hx": None, "raw_step_hy": None, "cmd_step_hx": None, "cmd_step_hy": None,
        "delta_now_from_latch": None, "delta_cmd_from_now": None,
        "latched_now": False, "latch_counts": None, "blocked_hx": False, "blocked_hy": False,
        "step_foot_delta": None, "step_foot_delta_norm": None,
        "active_counts": None, "blocked_counts": None, "max_step_delta": None, "max_err_y": None,
    }
    if not args.enable_b45_directional_hind_y_push_bridge:
        return info

    allowed_legs = _b37_parse_leg_list(args.b45_legs)
    hx_sign_map = _b37_parse_sign_map(args.b45_hx_signs, default=-1.0)
    hy_sign_map = _b37_parse_sign_map(args.b45_hy_signs, default=1.0)

    leg_name, target_w, sigma, source = _b37_pick_leg_target_sigma(b23_info, b18_info)
    if leg_name is None or target_w is None:
        info.update({"reason": "no_target", "source": source, "phase": "no_target", "allowed_legs": allowed_legs})
        return info
    if leg_name not in allowed_legs:
        info.update({"reason": "leg_not_allowed", "leg": leg_name, "source": source, "allowed_legs": allowed_legs})
        return info

    allowed, phase = _b45_phase_allows(leg_name, active_swing_names, capture_leg, capture_count, post_capture_pin_count)
    if not allowed:
        _B45_PUSH_STATE["last_phase"][leg_name] = phase
        info.update({"reason": "phase_not_allowed", "leg": leg_name, "source": source, "phase": phase, "allowed_legs": allowed_legs})
        return info

    sigma_val = None
    try:
        if sigma is not None:
            sigma_val = float(sigma)
    except Exception:
        sigma_val = None
    if phase == "swing" and sigma_val is not None:
        if sigma_val < float(args.b45_min_sigma) or sigma_val > float(args.b45_max_sigma):
            info.update({"reason": "sigma_out_of_window", "leg": leg_name, "source": source,
                         "phase": phase, "sigma": sigma_val, "allowed_legs": allowed_legs})
            return info

    leg_id = LEG_TO_ID[leg_name]
    joint_ids = _b12_get_leg_joint_ids(leg_id)
    q_now = robot.data.joint_pos.detach()[:, joint_ids]
    foot_w = foot_pos_arg[0, leg_id, :].detach()
    target_w = target_w.to(device=foot_w.device, dtype=foot_w.dtype)
    foot_err = target_w - foot_w

    e_y_raw = float(foot_err[1].detach().cpu().item())
    if abs(e_y_raw) < float(args.b45_deadband_y):
        e_y_used = 0.0
    else:
        e_y_used = max(-float(args.b45_max_y_err), min(float(args.b45_max_y_err), e_y_raw))

    last_step = _B45_PUSH_STATE["last_step"].get(leg_name, None)
    last_phase = _B45_PUSH_STATE["last_phase"].get(leg_name, None)
    need_new_latch = (
        leg_name not in _B45_PUSH_STATE["q_latch"]
        or last_step is None
        or int(step) != int(last_step) + 1
        or (args.b45_reset_latch_on_phase_change and last_phase != phase)
    )
    if need_new_latch:
        _B45_PUSH_STATE["q_latch"][leg_name] = q_now.detach().clone()
        _B45_PUSH_STATE["latch_counts"][leg_name] = int(_B45_PUSH_STATE["latch_counts"].get(leg_name, 0)) + 1

    q_latch = _B45_PUSH_STATE["q_latch"][leg_name].to(device=q_now.device, dtype=q_now.dtype)

    hx_sign = float(hx_sign_map.get(leg_name, -1.0))
    hy_sign = float(hy_sign_map.get(leg_name, 1.0))
    raw_step_hx = hx_sign * float(args.b45_k_hx_per_y) * float(e_y_used)
    raw_step_hx = max(-float(args.b45_max_step_hx_delta), min(float(args.b45_max_step_hx_delta), raw_step_hx))
    raw_step_hy = 0.0
    if args.b45_enable_hy:
        raw_step_hy = hy_sign * float(args.b45_k_hy_per_y) * float(e_y_used)
        raw_step_hy = max(-float(args.b45_max_step_hy_delta), min(float(args.b45_max_step_hy_delta), raw_step_hy))

    cmd_step_hx, blocked_hx = _b45_clip_direction_with_latch_guard(
        float(q_now[0, 0].detach().cpu().item()),
        float(q_latch[0, 0].detach().cpu().item()),
        raw_step_hx,
        float(args.b45_max_abs_hx_from_latch),
    )
    cmd_step_hy, blocked_hy = _b45_clip_direction_with_latch_guard(
        float(q_now[0, 1].detach().cpu().item()),
        float(q_latch[0, 1].detach().cpu().item()),
        raw_step_hy,
        float(args.b45_max_abs_hy_from_latch),
    )

    q_cmd = q_now.detach().clone()
    q_cmd[0, 0] = q_now[0, 0] + float(cmd_step_hx)
    q_cmd[0, 1] = q_now[0, 1] + float(cmd_step_hy)

    ids = torch.tensor(joint_ids, device=q_now.device, dtype=torch.long)
    robot.set_joint_position_target(q_cmd, joint_ids=ids)
    if args.b45_zero_target_velocity:
        robot.set_joint_velocity_target(torch.zeros_like(q_cmd), joint_ids=ids)

    delta_now_from_latch = q_now - q_latch
    delta_cmd_from_now = q_cmd - q_now

    last = _B45_PUSH_STATE["last_foot_w"].get(leg_name, None)
    if last is not None:
        last = last.to(device=foot_w.device, dtype=foot_w.dtype)
        step_foot_delta = foot_w - last
        step_foot_delta_norm = float(torch.linalg.norm(step_foot_delta).item())
    else:
        step_foot_delta = torch.zeros_like(foot_w)
        step_foot_delta_norm = 0.0

    _B45_PUSH_STATE["last_foot_w"][leg_name] = foot_w.detach().clone()
    _B45_PUSH_STATE["last_step"][leg_name] = int(step)
    _B45_PUSH_STATE["last_phase"][leg_name] = phase
    _B45_PUSH_STATE["active_counts"][leg_name] = int(_B45_PUSH_STATE["active_counts"].get(leg_name, 0)) + 1
    if blocked_hx or blocked_hy:
        _B45_PUSH_STATE["blocked_counts"][leg_name] = int(_B45_PUSH_STATE["blocked_counts"].get(leg_name, 0)) + 1
    _B45_PUSH_STATE["max_step_delta"][leg_name] = max(float(_B45_PUSH_STATE["max_step_delta"].get(leg_name, 0.0)), step_foot_delta_norm)
    _B45_PUSH_STATE["max_err_y"][leg_name] = max(float(_B45_PUSH_STATE["max_err_y"].get(leg_name, 0.0)), abs(e_y_raw))

    info.update({
        "active": True, "reason": "directional_hind_y_push", "leg": leg_name,
        "allowed_legs": allowed_legs, "source": source, "phase": phase, "sigma": sigma_val,
        "target_w": target_w.detach().cpu().numpy(), "foot_w": foot_w.detach().cpu().numpy(),
        "foot_err": foot_err.detach().cpu().numpy(), "e_y_raw": e_y_raw, "e_y_used": e_y_used,
        "hx_sign": hx_sign, "hy_sign": hy_sign,
        "q_latch": q_latch[0].detach().cpu().numpy(), "q_now": q_now[0].detach().cpu().numpy(),
        "q_cmd": q_cmd[0].detach().cpu().numpy(),
        "raw_step_hx": raw_step_hx, "raw_step_hy": raw_step_hy,
        "cmd_step_hx": cmd_step_hx, "cmd_step_hy": cmd_step_hy,
        "delta_now_from_latch": delta_now_from_latch[0].detach().cpu().numpy(),
        "delta_cmd_from_now": delta_cmd_from_now[0].detach().cpu().numpy(),
        "latched_now": bool(need_new_latch),
        "latch_counts": dict(_B45_PUSH_STATE["latch_counts"]),
        "blocked_hx": bool(blocked_hx), "blocked_hy": bool(blocked_hy),
        "step_foot_delta": step_foot_delta.detach().cpu().numpy(),
        "step_foot_delta_norm": step_foot_delta_norm,
        "active_counts": dict(_B45_PUSH_STATE["active_counts"]),
        "blocked_counts": dict(_B45_PUSH_STATE["blocked_counts"]),
        "max_step_delta": dict(_B45_PUSH_STATE["max_step_delta"]),
        "max_err_y": dict(_B45_PUSH_STATE["max_err_y"]),
    })
    return info

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

    dx = float(torch.abs(foot_pos[0, leg, 0] - foothold_target_w[0, 0]).detach().cpu())
    dy = float(torch.abs(foot_pos[0, leg, 1] - foothold_target_w[0, 1]).detach().cpu())
    xy_ok = xy_err <= float(args.late_touchdown_xy_tol)
    if args.touchdown_xy_tol_x >= 0.0:
        xy_ok = bool(xy_ok and dx <= float(args.touchdown_xy_tol_x))
    if args.touchdown_xy_tol_y >= 0.0:
        xy_ok = bool(xy_ok and dy <= float(args.touchdown_xy_tol_y))
    z_ok = z_err <= float(args.late_touchdown_z_tol)
    speed_ok = speed <= float(args.late_touchdown_max_foot_speed)

    return bool(xy_ok and z_ok and speed_ok), {
        "has_foothold": True,
        "xy_err": xy_err,
        "x_err": dx,
        "y_err": dy,
        "z_err": z_err,
        "speed": speed,
        "xy_ok": bool(xy_ok),
        "z_ok": bool(z_ok),
        "speed_ok": bool(speed_ok),
    }


def apply_committed_foothold_pin_target(robot, Jfeet_full, foot_pos, leg, foothold_target_w, active):
    """B8-bq: pin a committed stance foot at its accepted foothold."""
    _device = foot_pos.device
    _dtype = foot_pos.dtype
    if not active or leg is None or foothold_target_w is None:
        return None, None, {
            "active": False, "leg": leg, "joint_ids": [],
            "foot_err": torch.zeros((args.num_envs, 3), device=_device, dtype=_dtype),
            "dq_cmd": torch.zeros((args.num_envs, 3), device=_device, dtype=_dtype),
            "target_minus_q": torch.zeros((args.num_envs, 3), device=_device, dtype=_dtype),
        }

    jid = [HX[int(leg)], HY[int(leg)], KN[int(leg)]]
    foot_err = foothold_target_w[:, :] - foot_pos[:, int(leg), :]
    foot_err_scaled = foot_err.clone()
    foot_err_scaled[:, 0:2] *= float(args.committed_pin_target_scale_xy)
    foot_err_scaled[:, 2] *= float(args.committed_pin_target_scale_z)

    jid_tensor = torch.tensor(jid, device=_device, dtype=torch.long)
    J_leg = Jfeet_full[:, int(leg), :, 6 + jid_tensor]
    JT = J_leg.transpose(1, 2)
    A = J_leg @ JT + float(args.committed_pin_damping) * torch.eye(3, device=_device, dtype=_dtype).unsqueeze(0)
    y = torch.linalg.solve(A, foot_err_scaled.unsqueeze(-1)).squeeze(-1)
    dq_cmd = float(args.committed_pin_gain) * (JT @ y.unsqueeze(-1)).squeeze(-1)
    dq_cmd = torch.clamp(dq_cmd, -float(args.committed_pin_max_joint_delta), float(args.committed_pin_max_joint_delta))

    q = robot.data.joint_pos.detach().clone()
    q_target = q.clone()
    q_target[:, jid] = q[:, jid] + dq_cmd
    robot.set_joint_position_target(q_target)
    if args.committed_pin_zero_velocity:
        qd = robot.data.joint_vel.detach().clone()
        qd[:, jid] = 0.0
        robot.set_joint_velocity_target(qd)

    delta_full = torch.zeros_like(q)
    delta_full[:, jid] = dq_cmd
    return q_target, delta_full, {
        "active": True,
        "leg": int(leg),
        "joint_ids": jid,
        "foot_err": foot_err.detach().clone(),
        "dq_cmd": dq_cmd.detach().clone(),
        "target_minus_q": (q_target[:, jid] - q[:, jid]).detach().clone(),
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


def apply_jacobian_ik_swing_target(robot, Jfeet_full, foot_pos, swing_target, phase, profile, swing_enabled, active_leg_override=None):
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

    leg = int(active_leg_override) if active_leg_override is not None else LEG_TO_ID[args.test_leg]
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



def apply_canonical_committed_foothold_pin(robot, Jfeet_full, foot_pos, commit_target_w, commit_leg, commit_count):
    """Keep a just-landed foothold near its committed target after contact is restored."""
    q_now = robot.data.joint_pos.detach()
    q_delta_full = torch.zeros_like(q_now)
    info = {
        "active": False,
        "leg": commit_leg,
        "count": int(commit_count),
        "joint_ids": [],
        "foot_err": torch.zeros((q_now.shape[0], 3), device=q_now.device, dtype=q_now.dtype),
        "dq_cmd": torch.zeros((q_now.shape[0], 3), device=q_now.device, dtype=q_now.dtype),
        "target_minus_q": torch.zeros((q_now.shape[0], 3), device=q_now.device, dtype=q_now.dtype),
    }

    if not args.canonical_enable_touchdown_foothold_commit:
        return None, q_delta_full, info
    if commit_target_w is None or commit_leg is None or int(commit_count) <= 0:
        return None, q_delta_full, info

    leg = int(commit_leg)
    foot_err_3 = commit_target_w[:, leg, :] - foot_pos[:, leg, :]
    foot_err_3 = foot_err_3.clone()
    foot_err_3[:, 0:2] = float(args.canonical_commit_target_scale_xy) * foot_err_3[:, 0:2]
    foot_err_3[:, 2] = float(args.canonical_commit_target_scale_z) * foot_err_3[:, 2]

    joint_ids = [int(HX[leg]), int(HY[leg]), int(KN[leg])]
    full_cols = [6 + jid for jid in joint_ids]
    Jleg_full = Jfeet_full[:, leg, :, full_cols]

    dq = damped_least_squares_delta(Jleg_full, foot_err_3, float(args.canonical_commit_ik_damping))
    dq = float(args.canonical_commit_ik_gain) * dq
    dq = torch.clamp(dq, -float(args.canonical_commit_max_joint_delta), float(args.canonical_commit_max_joint_delta))

    q_selected = q_now[:, joint_ids] + dq
    joint_ids_tensor = torch.tensor(joint_ids, device=q_now.device, dtype=torch.long)
    robot.set_joint_position_target(q_selected, joint_ids=joint_ids_tensor)
    if args.canonical_commit_zero_velocity:
        robot.set_joint_velocity_target(torch.zeros_like(q_selected), joint_ids=joint_ids_tensor)

    q_delta_full[:, joint_ids] = q_selected - q_now[:, joint_ids]
    q_target_full = q_now.clone()
    q_target_full[:, joint_ids] = q_selected

    info["active"] = True
    info["joint_ids"] = joint_ids
    info["foot_err"] = foot_err_3
    info["dq_cmd"] = dq
    info["target_minus_q"] = q_selected - q_now[:, joint_ids]
    return q_target_full, q_delta_full, info



def apply_canonical_touchdown_capture_q_hold(robot, capture_leg, capture_q_ref, capture_count):
    """Hold the actual joint configuration captured at touchdown.

    Unlike Cartesian commit pinning, this does not drag the stance foot toward a desired
    target after contact is restored. It accepts the actual touchdown as the new stance
    configuration and prevents the implicit/default posture from snapping it back.
    """
    q_now = robot.data.joint_pos.detach()
    q_delta_full = torch.zeros_like(q_now)
    info = {
        "active": False,
        "leg": capture_leg,
        "count": int(capture_count),
        "joint_ids": [],
        "target_minus_q": torch.zeros((q_now.shape[0], 3), device=q_now.device, dtype=q_now.dtype),
    }
    if not args.canonical_enable_touchdown_capture:
        return None, q_delta_full, info
    if capture_leg is None or capture_q_ref is None or int(capture_count) <= 0:
        return None, q_delta_full, info

    leg = int(capture_leg)
    joint_ids = [int(HX[leg]), int(HY[leg]), int(KN[leg])]
    joint_ids_tensor = torch.tensor(joint_ids, device=q_now.device, dtype=torch.long)
    q_selected = capture_q_ref[:, joint_ids].detach().clone()
    robot.set_joint_position_target(q_selected, joint_ids=joint_ids_tensor)
    if args.canonical_capture_zero_velocity:
        robot.set_joint_velocity_target(torch.zeros_like(q_selected), joint_ids=joint_ids_tensor)

    q_delta_full[:, joint_ids] = q_selected - q_now[:, joint_ids]
    q_target_full = q_now.clone()
    q_target_full[:, joint_ids] = q_selected

    info["active"] = True
    info["joint_ids"] = joint_ids
    info["target_minus_q"] = q_selected - q_now[:, joint_ids]
    return q_target_full, q_delta_full, info



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
                previous_load_bearing_valid=True,
                swing_latched=False, swing_latched_leg=None, swing_latched_started_step=-1,
                post_touchdown_lock_count=0, post_touchdown_lock_info=None,
                touchdown_committed=False, touchdown_committed_leg=None,
                touchdown_committed_step_key=None, touchdown_committed_step=-1,
                committed_pin_count=0, committed_pin_info=None,
                canonical_commit_info=None):
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
    print(f"[B39 LATCHED-SUPPORT-RECENTER] step={step}")
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
    print("latched_swing_touchdown_priority:",
          "swing_latched:", swing_latched,
          "swing_latched_leg:", swing_latched_leg,
          "swing_latched_started_step:", swing_latched_started_step,
          "post_touchdown_lock_count:", post_touchdown_lock_count,
          "post_touchdown_lock_info:",
          None if post_touchdown_lock_info is None else {
              "active": post_touchdown_lock_info.get("active"),
              "leg": post_touchdown_lock_info.get("leg"),
              "foot_err": post_touchdown_lock_info.get("foot_err")[0].detach().cpu().numpy() if post_touchdown_lock_info.get("foot_err") is not None else None,
              "dq_cmd": post_touchdown_lock_info.get("dq_cmd")[0].detach().cpu().numpy() if post_touchdown_lock_info.get("dq_cmd") is not None else None,
          })
    print("touchdown_committed_state:",
          "committed:", touchdown_committed,
          "committed_leg:", touchdown_committed_leg,
          "committed_step_key:", touchdown_committed_step_key,
          "committed_step:", touchdown_committed_step)
    print("committed_foothold_pinning:",
          "pin_count:", committed_pin_count,
          "pin_info:",
          None if committed_pin_info is None else {
              "active": committed_pin_info.get("active"),
              "leg": committed_pin_info.get("leg"),
              "foot_err": committed_pin_info.get("foot_err")[0].detach().cpu().numpy() if committed_pin_info.get("foot_err") is not None else None,
              "dq_cmd": committed_pin_info.get("dq_cmd")[0].detach().cpu().numpy() if committed_pin_info.get("dq_cmd") is not None else None,
              "target_minus_q": committed_pin_info.get("target_minus_q")[0].detach().cpu().numpy() if committed_pin_info.get("target_minus_q") is not None else None,
          })
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
        kp_swing_xy=args.kp_swing_xy,
        kd_swing_xy=args.kd_swing_xy,
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
    swing_latched = False
    swing_latched_leg = None
    swing_latched_foothold_target_w = None
    swing_latched_stance_q_ref = None
    swing_latched_started_step = -1
    post_touchdown_lock_count = 0
    post_touchdown_lock_leg = None
    post_touchdown_lock_foothold_target_w = None
    touchdown_committed = False
    touchdown_committed_leg = None
    touchdown_committed_step_key = None
    touchdown_committed_step = -1
    touchdown_committed_foothold_target_w = None
    committed_pin_count = 0
    committed_pin_target = None
    committed_pin_delta_full = None
    committed_pin_info = None
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
    print("[INFO] Starting B50 fixed B49 elevated swing tracking indexing + B48 planner")
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
    print("latched_swing_touchdown_priority:",
          "enabled:", args.enable_latched_swing_until_touchdown,
          "min_phase:", args.latched_swing_min_phase,
          "keep_base_freeze:", args.latched_swing_keep_base_freeze,
          "force_contact_open:", args.latched_swing_force_contact_open,
          "force_ik:", args.latched_swing_force_ik,
          "release_on_late_touchdown:", args.latched_swing_release_on_late_touchdown,
          "post_touchdown_lock_steps:", args.post_touchdown_lock_steps)
    print("touchdown_committed_state:",
          "enabled:", args.enable_touchdown_committed_state,
          "blocks_same_step_relatch:", args.touchdown_committed_blocks_same_step_relatch,
          "force_all_stance:", args.touchdown_committed_force_all_stance,
          "keep_base_freeze:", args.touchdown_committed_keep_base_freeze,
          "until_next_step:", args.touchdown_committed_until_next_step)
    print("committed_foothold_pinning:",
          "enabled:", args.enable_committed_foothold_pinning,
          "until_next_step:", args.committed_pin_until_next_step,
          "ignore_late_touchdown_logic:", args.committed_pin_ignore_late_touchdown_logic,
          "gain:", args.committed_pin_gain,
          "damping:", args.committed_pin_damping,
          "max_joint_delta:", args.committed_pin_max_joint_delta,
          "target_scale_xy:", args.committed_pin_target_scale_xy,
          "target_scale_z:", args.committed_pin_target_scale_z,
          "zero_velocity:", args.committed_pin_zero_velocity,
          "extra_steps:", args.committed_pin_extra_steps,
          "axis_tol_x:", args.touchdown_xy_tol_x,
          "axis_tol_y:", args.touchdown_xy_tol_y)
    if args.enable_b24_wbc_swing_authority_boost:
        wbc_cfg.w_swing_acc = float(args.b24_w_swing_acc)
        wbc_cfg.w_stance_acc = float(args.b24_w_stance_acc)
        wbc_cfg.w_base_acc = float(args.b24_w_base_acc)
        wbc_cfg.kp_swing_xy = float(args.b24_kp_swing_xy)
        wbc_cfg.kd_swing_xy = float(args.b24_kd_swing_xy)
        wbc_cfg.kp_swing_z = float(args.b24_kp_swing_z)
        wbc_cfg.kd_swing_z = float(args.b24_kd_swing_z)
        wbc_cfg.max_swing_acc = float(args.b24_max_swing_acc)
        wbc_cfg.w_swing_force_zero = float(args.b24_w_swing_force_zero)
        wbc_cfg.w_tau_reg = float(args.b24_w_tau_reg)
        wbc_cfg.w_tau_posture = float(args.b24_w_tau_posture)
        if args.b24_print:
            print("[B37 WBC CFG OVERRIDE]",
                  "w_swing_acc=", wbc_cfg.w_swing_acc,
                  "w_stance_acc=", wbc_cfg.w_stance_acc,
                  "w_base_acc=", wbc_cfg.w_base_acc,
                  "kp_swing_xy=", wbc_cfg.kp_swing_xy,
                  "kd_swing_xy=", wbc_cfg.kd_swing_xy,
                  "kp_swing_z=", wbc_cfg.kp_swing_z,
                  "kd_swing_z=", wbc_cfg.kd_swing_z,
                  "max_swing_acc=", wbc_cfg.max_swing_acc,
                  "w_swing_force_zero=", wbc_cfg.w_swing_force_zero,
                  "w_tau_reg=", wbc_cfg.w_tau_reg,
                  "w_tau_posture=", wbc_cfg.w_tau_posture)
    print("wbc_cfg:", wbc_cfg)
    print("canonical_gait_schedule:",
          "enabled:", args.enable_canonical_gait_schedule,
          "period_steps:", args.gait_period_steps,
          "duty:", args.gait_duty,
          "phase_offsets:", args.gait_phase_offsets,
          "swing_height:", args.canonical_gait_swing_height,
          "forward_step:", args.canonical_gait_forward_step,
          "disable_event_overrides:", args.canonical_gait_disable_event_overrides,
          "start_step:", args.canonical_gait_start_step,
          "force_all_stance_before_start:", args.canonical_gait_force_all_stance_before_start,
          "relative_time:", args.canonical_gait_use_relative_time,
          "debug_active_swing_leg:", args.canonical_debug_use_active_swing_leg)
    print("canonical_execution_gate:",
          "single_swing:", args.canonical_single_swing_gate,
          "gap_steps:", args.canonical_min_all_stance_gap_steps,
          "liftoff_min_margin:", args.canonical_liftoff_min_margin,
          "max_roll:", args.canonical_liftoff_max_abs_roll,
          "max_pitch:", args.canonical_liftoff_max_abs_pitch,
          "min_base_z:", args.canonical_liftoff_min_base_z,
          "first_swing_leg:", args.canonical_first_swing_leg,
          "priority:", args.canonical_crawl_priority)
    print("canonical_feasible_trunk_shift:",
          "enabled:", args.canonical_enable_feasible_trunk_shift,
          "before_liftoff:", args.canonical_shift_before_liftoff,
          "during_swing:", args.canonical_shift_during_swing,
          "max_step:", args.canonical_shift_max_step,
          "lateral_bias:", args.canonical_shift_lateral_bias,
          "forward_bias:", args.canonical_shift_forward_bias,
          "track_actual_base:", args.canonical_shift_track_actual_base,
          "max_base_ref_xy_err:", args.canonical_shift_max_base_ref_xy_err,
          "hold_when_ref_error:", args.canonical_shift_hold_when_ref_error,
          "allow_after_shift_steps:", args.canonical_liftoff_allow_after_shift_steps,
          "relaxed_margin:", args.canonical_liftoff_margin_relaxed)
    print("canonical_latched_executed_swing:",
          "enabled:", args.canonical_enable_latched_executed_swing,
          "duration:", args.canonical_executed_swing_duration_steps,
          "min_duration:", args.canonical_executed_swing_min_duration_steps,
          "gap_after_touchdown:", args.canonical_executed_swing_force_gap_after_touchdown)
    print("swing_ik_clearance:",
          "enable_jacobian_ik_swing:", args.enable_jacobian_ik_swing,
          "use_canonical_active_leg:", args.ik_use_canonical_active_leg,
          "ik_include_xy:", args.ik_include_xy,
          "ik_use_hx:", args.ik_use_hx,
          "ik_gain:", args.ik_gain,
          "ik_max_joint_delta:", args.ik_max_joint_delta,
          "ik_target_scale_z:", args.ik_target_scale_z,
          "kp_swing_xy:", args.kp_swing_xy,
          "kd_swing_xy:", args.kd_swing_xy)
    print("touchdown_foothold_commit:",
          "enabled:", args.canonical_enable_touchdown_foothold_commit,
          "steps:", args.canonical_commit_steps,
          "ik_gain:", args.canonical_commit_ik_gain,
          "max_joint_delta:", args.canonical_commit_max_joint_delta,
          "target_scale_xy:", args.canonical_commit_target_scale_xy,
          "target_scale_z:", args.canonical_commit_target_scale_z)
    print("event_crawl_queue:",
          "enabled:", args.canonical_enable_event_crawl_queue,
          "queue:", args.canonical_event_queue,
          "shift_min_future_margin:", args.canonical_event_shift_min_future_margin,
          "commit_anchor_to_target:", args.canonical_event_commit_anchor_to_target)
    print("touchdown_capture:",
          "enabled:", args.canonical_enable_touchdown_capture,
          "steps:", args.canonical_capture_steps,
          "update_anchor_actual:", args.canonical_capture_update_anchor_actual,
          "freeze_base_ref:", args.canonical_freeze_base_ref_during_swing_commit,
          "require_stable_before_next:", args.canonical_event_require_stable_before_next,
          "stable_steps:", args.canonical_event_stable_steps)
    print("support_region_target_planner:",
          "enabled:", args.canonical_enable_support_region_target_planner,
          "target_min_margin:", args.canonical_fr_target_min_margin,
          "liftoff_min_geom_margin:", args.canonical_fr_liftoff_min_geom_margin,
          "target_err_tol:", args.canonical_fr_liftoff_target_err_tol,
          "min_shift_steps:", args.canonical_fr_liftoff_min_shift_steps,
          "hind_forward_bias:", args.canonical_fr_hind_forward_bias,
          "hind_lateral_bias:", args.canonical_fr_hind_lateral_bias,
          "projected_target:", args.canonical_fr_use_projected_target,
          "projection_grid:", args.canonical_fr_projection_grid,
          "projection_max_step:", args.canonical_fr_projection_max_step_from_current,
          "projection_inner_push:", args.canonical_fr_projection_inner_push)
    print("b10_fsm:",
          "enabled:", args.canonical_enable_b10_fsm_diagnostics,
          "allow_liftoff_on_b10_gate:", args.b10_allow_liftoff_on_b10_gate,
          "target_err_tol:", args.b10_relocation_target_err_tol,
          "geom_margin_tol:", args.b10_relocation_geom_margin_tol,
          "min_shift_steps:", args.b10_relocation_min_shift_steps)
    print("b11_wbc_relocation:",
          "enabled:", args.enable_b11_wbc_relocation_task,
          "kp_xy:", args.b11_reloc_kp_base_xy,
          "kd_xy:", args.b11_reloc_kd_base_xy,
          "w_base_acc:", args.b11_reloc_w_base_acc,
          "w_stance_acc:", args.b11_reloc_w_stance_acc,
          "max_base_acc_lin:", args.b11_reloc_max_base_acc_lin,
          "max_base_ref_xy_err:", args.b11_reloc_max_base_ref_xy_err)
    print("b12_pd_target_alignment:",
          "enabled:", args.enable_b12_pd_target_alignment,
          "gain:", args.b12_align_gain,
          "include_hx:", args.b12_align_include_hx,
          "sign:", args.b12_align_sign,
          "max_joint_delta:", args.b12_align_max_joint_delta,
          "max_foot_xy_cmd:", args.b12_align_max_foot_xy_cmd,
          "torque_kp:", args.b12_align_torque_kp,
          "torque_kd:", args.b12_align_torque_kd,
          "max_tau:", args.b12_align_max_tau)
    print("b13_adaptive_next_leg:",
          "enabled:", args.enable_b13_adaptive_next_leg,
          "start_index:", args.b13_adaptive_start_index,
          "candidates:", args.b13_candidate_legs,
          "exclude_last_completed:", args.b13_exclude_last_completed,
          "score_weights:", (args.b13_score_margin_weight, args.b13_score_geom_weight, args.b13_score_target_err_weight))
    print("b14_no_queue_mutation:",
          "enabled:", args.enable_b14_no_queue_mutation,
          "completed_cooldown_steps:", args.b14_completed_cooldown_steps,
          "require_positive_best_margin:", args.b14_require_positive_best_margin)
    print("b15_recenter_fallback:",
          "enabled:", args.enable_b15_recenter_fallback,
          "trigger_steps:", args.b15_recenter_trigger_steps,
          "duration_steps:", args.b15_recenter_duration_steps,
          "future_margin_threshold:", args.b15_recenter_future_margin_threshold,
          "geom_margin_threshold:", args.b15_recenter_geom_margin_threshold,
          "max_step:", args.b15_recenter_max_step,
          "blend_current:", args.b15_recenter_target_blend_current)
    print("b16_recenter_all_stance_pd_alignment:",
          "enabled:", args.enable_b16_recenter_all_stance_pd_alignment,
          "gain:", args.b16_recenter_align_gain,
          "include_hx:", args.b16_recenter_align_include_hx,
          "sign:", args.b16_recenter_align_sign,
          "max_joint_delta:", args.b16_recenter_align_max_joint_delta,
          "max_foot_xy_cmd:", args.b16_recenter_align_max_foot_xy_cmd,
          "torque_kp:", args.b16_recenter_align_torque_kp,
          "torque_kd:", args.b16_recenter_align_torque_kd,
          "max_tau:", args.b16_recenter_align_max_tau)
    print("b17_support_aware_foothold_mapper:",
          "enabled:", args.enable_b17_support_aware_foothold_mapper,
          "forward_candidates:", args.b17_foothold_forward_candidates,
          "lateral_candidates:", args.b17_foothold_lateral_candidates,
          "max_extra_xy:", args.b17_foothold_max_extra_xy,
          "next_candidates:", args.b17_foothold_next_candidates,
          "apply_only_to:", args.b17_foothold_apply_only_to)
    print("b18_committed_foothold_pinning:",
          "enabled:", args.enable_b18_committed_foothold_pinning,
          "commit_min_sigma:", args.b18_commit_min_sigma,
          "pin_during_swing:", args.b18_pin_during_swing,
          "pin_during_capture:", args.b18_pin_during_capture,
          "extra_after_capture_steps:", args.b18_pin_extra_after_capture_steps,
          "pin_gain:", args.b18_pin_gain,
          "pin_max_joint_delta:", args.b18_pin_max_joint_delta,
          "pin_max_tau:", args.b18_pin_max_tau)
    print("b19_forced_exploratory_foothold_bias:",
          "enabled:", args.enable_b19_forced_exploratory_foothold_bias,
          "legs:", args.b19_force_bias_legs,
          "forward_bias:", args.b19_force_forward_bias,
          "outward_bias:", args.b19_force_outward_bias,
          "max_extra_xy:", args.b19_force_max_extra_xy,
          "min_sigma:", args.b19_force_min_sigma,
          "override_b17:", args.b19_force_override_b17_selection)
    print("b20_committed_foothold_direct_swing_ik:",
          "enabled:", args.enable_b20_committed_foothold_direct_swing_ik,
          "start_sigma:", args.b20_direct_ik_start_sigma,
          "end_sigma:", args.b20_direct_ik_end_sigma,
          "gain:", args.b20_direct_ik_gain,
          "max_joint_delta:", args.b20_direct_ik_max_joint_delta,
          "max_tau:", args.b20_direct_ik_max_tau)
    print("b21_cartesian_committed_swing_torque:",
          "enabled:", args.enable_b21_cartesian_committed_swing_torque,
          "kp_xy:", args.b21_cart_kp_xy,
          "kp_z:", args.b21_cart_kp_z,
          "kd_xy:", args.b21_cart_kd_xy,
          "kd_z:", args.b21_cart_kd_z,
          "max_tau:", args.b21_cart_max_tau,
          "sign:", args.b21_cart_force_sign,
          "during_swing:", args.b21_cart_apply_during_swing,
          "during_capture:", args.b21_cart_apply_during_capture,
          "post_extra:", args.b21_cart_apply_post_capture_extra)
    print("b22_wbc_authority_calibration:",
          "enabled:", args.enable_b22_wbc_authority_calibration,
          "mode:", args.b22_authority_mode,
          "wbc_scale:", args.b22_wbc_scale,
          "assist_scale:", args.b22_assist_scale,
          "use_ramp:", args.b22_use_ramp,
          "ramp_start:", args.b22_ramp_start_step,
          "ramp_steps:", args.b22_ramp_steps)
    print("b23_wbc_main_committed_footref:",
          "enabled:", args.enable_b23_wbc_main_committed_footref,
          "during_swing:", args.b23_apply_during_swing,
          "during_capture:", args.b23_apply_during_capture,
          "post_extra:", args.b23_apply_post_capture_extra,
          "start_sigma:", args.b23_start_sigma,
          "end_sigma:", args.b23_end_sigma,
          "xy_mode:", args.b23_xy_mode,
          "preserve_swing_z:", args.b23_preserve_swing_z,
          "commit_anchor_on_capture_start:", args.b23_commit_anchor_on_capture_start)
    print("b24_wbc_swing_authority_boost:",
          "enabled:", args.enable_b24_wbc_swing_authority_boost,
          "w_swing_acc:", args.b24_w_swing_acc,
          "w_stance_acc:", args.b24_w_stance_acc,
          "w_base_acc:", args.b24_w_base_acc,
          "kp_swing_xy:", args.b24_kp_swing_xy,
          "kd_swing_xy:", args.b24_kd_swing_xy,
          "kp_swing_z:", args.b24_kp_swing_z,
          "kd_swing_z:", args.b24_kd_swing_z,
          "max_swing_acc:", args.b24_max_swing_acc,
          "w_swing_force_zero:", args.b24_w_swing_force_zero,
          "w_tau_reg:", args.b24_w_tau_reg,
          "w_tau_posture:", args.b24_w_tau_posture)
    print("b25_canonical_swing_pd_bridge:",
          "enabled:", args.enable_b25_canonical_swing_pd_bridge,
          "use_b20:", args.b25_use_b20_direct_ik_target,
          "use_b18:", args.b25_use_b18_pin_target,
          "zero_vel:", args.b25_zero_target_velocity,
          "max_joint_delta:", args.b25_max_joint_delta)
    print("b26_latched_safe_swing_pd_bridge:",
          "enabled:", args.enable_b26_latched_safe_swing_pd_bridge,
          "use_b20:", args.b26_use_b20_direct_ik_target,
          "use_b18:", args.b26_use_b18_pin_target,
          "zero_vel:", args.b26_zero_target_velocity,
          "max_joint_delta:", args.b26_max_joint_delta,
          "max_abs_from_latch:", args.b26_max_abs_from_latch,
          "retarget_interval:", args.b26_retarget_interval)
    print("b27_signed_latched_swing_pd_bridge:",
          "enabled:", args.enable_b27_signed_latched_swing_pd_bridge,
          "use_b20:", args.b27_use_b20_direct_ik_target,
          "zero_vel:", args.b27_zero_target_velocity,
          "signs(hx,hy,kn):", (args.b27_hx_sign, args.b27_hy_sign, args.b27_kn_sign),
          "max_joint_delta:", args.b27_max_joint_delta,
          "max_abs_from_latch:", args.b27_max_abs_from_latch,
          "retarget_interval:", args.b27_retarget_interval)
    print("b28_hold_latched_swing_pd_bridge:",
          "enabled:", args.enable_b28_hold_latched_swing_pd_bridge,
          "use_b20:", args.b28_use_b20_direct_ik_target,
          "zero_vel:", args.b28_zero_target_velocity,
          "signs(hx,hy,kn):", (args.b28_hx_sign, args.b28_hy_sign, args.b28_kn_sign),
          "max_joint_delta:", args.b28_max_joint_delta,
          "max_abs_from_latch:", args.b28_max_abs_from_latch,
          "hold_without_target_steps:", args.b28_hold_without_target_steps,
          "retarget_interval:", args.b28_retarget_interval)
    print("b29_task_space_foot_error_bridge:",
          "enabled:", args.enable_b29_task_space_foot_error_bridge,
          "use_committed:", args.b29_use_committed_target,
          "during_swing:", args.b29_apply_during_swing,
          "during_capture:", args.b29_apply_during_capture,
          "post_extra:", args.b29_apply_post_capture_extra,
          "kp_xy:", args.b29_kp_xy,
          "kp_z:", args.b29_kp_z,
          "damping:", args.b29_damping,
          "max_foot_err_xy:", args.b29_max_foot_err_xy,
          "max_foot_err_z:", args.b29_max_foot_err_z,
          "max_joint_delta:", args.b29_max_joint_delta,
          "max_abs_from_latch:", args.b29_max_abs_from_latch,
          "latch_goal:", args.b29_latch_goal,
          "hold_without_target_steps:", args.b29_hold_without_target_steps)
    print("b30_jfeet_task_space_bridge:",
          "enabled:", args.enable_b30_jfeet_task_space_bridge,
          "use_committed:", args.b30_use_committed_target,
          "during_swing:", args.b30_apply_during_swing,
          "during_capture:", args.b30_apply_during_capture,
          "post_extra:", args.b30_apply_post_capture_extra,
          "kp_xy:", args.b30_kp_xy,
          "kp_z:", args.b30_kp_z,
          "damping:", args.b30_damping,
          "max_foot_err_xy:", args.b30_max_foot_err_xy,
          "max_foot_err_z:", args.b30_max_foot_err_z,
          "max_joint_delta:", args.b30_max_joint_delta,
          "max_abs_from_latch:", args.b30_max_abs_from_latch,
          "latch_goal:", args.b30_latch_goal,
          "hold_without_target_steps:", args.b30_hold_without_target_steps)
    print("b31_jacobian_response_probe:",
          "enabled:", args.enable_b31_jacobian_response_probe,
          "delay_steps:", args.b31_delay_steps,
          "min_gap_steps:", args.b31_min_gap_steps,
          "min_delta_norm:", args.b31_min_delta_norm)
    print("b32_direct_joint_response_probe:",
          "enabled:", args.enable_b32_direct_joint_response_probe,
          "probe_leg:", args.b32_probe_leg,
          "start_step:", args.b32_start_step,
          "axes:", args.b32_axes,
          "signs:", args.b32_signs,
          "delta:", args.b32_delta,
          "hold_steps:", args.b32_hold_steps,
          "gap_steps:", args.b32_gap_steps,
          "max_probes:", args.b32_max_probes,
          "zero_vel:", args.b32_zero_target_velocity,
          "require_swing_or_capture:", args.b32_require_swing_or_capture,
          "disable_b30_during_probe:", args.b32_disable_b30_during_probe)
    print("b33_baseline_cancelled_response_probe:",
          "enabled:", args.enable_b33_baseline_cancelled_response_probe,
          "probe_leg:", args.b33_probe_leg,
          "start_step:", args.b33_start_step,
          "axes:", args.b33_axes,
          "signs:", args.b33_signs,
          "delta:", args.b33_delta,
          "baseline_steps:", args.b33_baseline_steps,
          "perturb_steps:", args.b33_perturb_steps,
          "gap_steps:", args.b33_gap_steps,
          "max_pairs:", args.b33_max_pairs,
          "zero_vel:", args.b33_zero_target_velocity,
          "require_swing_or_capture:", args.b33_require_swing_or_capture)
    print("b34_empirical_rf_x_bridge:",
          "enabled:", args.enable_b34_empirical_rf_x_bridge,
          "leg:", args.b34_leg,
          "use_committed:", args.b34_use_committed_target,
          "during_swing:", args.b34_apply_during_swing,
          "during_capture:", args.b34_apply_during_capture,
          "post_extra:", args.b34_apply_post_capture_extra,
          "k_hy_per_x:", args.b34_k_hy_per_x,
          "deadband_x:", args.b34_deadband_x,
          "max_x_err:", args.b34_max_x_err,
          "max_joint_delta:", args.b34_max_joint_delta,
          "max_abs_from_latch:", args.b34_max_abs_from_latch,
          "hold_without_target_steps:", args.b34_hold_without_target_steps,
          "zero_vel:", args.b34_zero_target_velocity,
          "include_small_hx:", args.b34_include_small_hx,
          "k_hx_per_y:", args.b34_k_hx_per_y,
          "max_hx_delta:", args.b34_max_hx_delta)
    print("b35_pure_empirical_hy_assist:",
          "enabled:", args.enable_b35_pure_empirical_hy_assist,
          "leg:", args.b35_leg,
          "use_committed:", args.b35_use_committed_target,
          "during_swing:", args.b35_apply_during_swing,
          "during_capture:", args.b35_apply_during_capture,
          "post_extra:", args.b35_apply_post_capture_extra,
          "k_hy_per_x:", args.b35_k_hy_per_x,
          "deadband_x:", args.b35_deadband_x,
          "max_x_err:", args.b35_max_x_err,
          "max_hy_delta:", args.b35_max_hy_delta,
          "sigma_window:", (args.b35_min_sigma, args.b35_max_sigma),
          "zero_vel:", args.b35_zero_target_velocity,
          "no_hold_without_target:", args.b35_no_hold_without_target)
    print("b36_pure_empirical_hy_kn_assist:",
          "enabled:", args.enable_b36_pure_empirical_hy_kn_assist,
          "leg:", args.b36_leg,
          "use_committed:", args.b36_use_committed_target,
          "during_swing:", args.b36_apply_during_swing,
          "during_capture:", args.b36_apply_during_capture,
          "post_extra:", args.b36_apply_post_capture_extra,
          "k_hy_per_x:", args.b36_k_hy_per_x,
          "k_kn_per_x:", args.b36_k_kn_per_x,
          "kn_sign:", args.b36_kn_sign,
          "deadband_x:", args.b36_deadband_x,
          "max_x_err:", args.b36_max_x_err,
          "max_hy_delta:", args.b36_max_hy_delta,
          "max_kn_delta:", args.b36_max_kn_delta,
          "sigma_window:", (args.b36_min_sigma, args.b36_max_sigma),
          "zero_vel:", args.b36_zero_target_velocity)
    print("b37_all_leg_empirical_hy_kn_assist:",
          "enabled:", args.enable_b37_all_leg_empirical_hy_kn_assist,
          "legs:", args.b37_legs,
          "use_committed:", args.b37_use_committed_target,
          "during_swing:", args.b37_apply_during_swing,
          "during_capture:", args.b37_apply_during_capture,
          "post_extra:", args.b37_apply_post_capture_extra,
          "k_hy_per_x:", args.b37_k_hy_per_x,
          "k_kn_per_x:", args.b37_k_kn_per_x,
          "deadband_x:", args.b37_deadband_x,
          "max_x_err:", args.b37_max_x_err,
          "max_hy_delta:", args.b37_max_hy_delta,
          "max_kn_delta:", args.b37_max_kn_delta,
          "sigma_window:", (args.b37_min_sigma, args.b37_max_sigma),
          "hy_signs:", args.b37_hy_signs,
          "kn_signs:", args.b37_kn_signs,
          "zero_vel:", args.b37_zero_target_velocity)
    print("b38_candidate_support_recenter:",
          "enabled:", args.enable_b38_candidate_support_recenter,
          "candidate_legs:", args.b38_candidate_legs,
          "force_shift_even_if_future_bad:", args.b38_force_shift_even_if_future_bad,
          "use_actual_base_start:", args.b38_use_actual_base_start,
          "max_step:", args.b38_max_step,
          "target_err_tol:", args.b38_target_err_tol,
          "geom_margin_tol:", args.b38_geom_margin_tol,
          "min_shift_steps:", args.b38_min_shift_steps,
          "allow_liftoff_on_b38_gate:", args.b38_allow_liftoff_on_b38_gate)
    print("b39_latched_support_recenter:",
          "enabled:", args.enable_b39_latched_support_recenter,
          "candidate_legs:", args.b39_candidate_legs,
          "force_shift_even_if_future_bad:", args.b39_force_shift_even_if_future_bad,
          "max_step:", args.b39_max_step,
          "target_err_tol:", args.b39_target_err_tol,
          "geom_margin_tol:", args.b39_geom_margin_tol,
          "min_shift_steps:", args.b39_min_shift_steps,
          "allow_liftoff_on_b39_gate:", args.b39_allow_liftoff_on_b39_gate)
    print("b41_swing_phase_hind_target_clamp:",
          "enabled:", args.enable_b41_swing_phase_hind_target_clamp,
          "clamp_legs:", args.b41_clamp_legs,
          "max_forward:", args.b41_max_forward_from_anchor,
          "max_lateral:", args.b41_max_lateral_from_anchor,
          "max_xy:", args.b41_max_xy_from_anchor,
          "clamp_z:", args.b41_clamp_z,
          "max_z:", args.b41_max_z_from_anchor)
    print("b42_lh_taskspace_authority_probe:",
          "enabled:", args.enable_b42_lh_taskspace_authority_probe,
          "probe_leg:", args.b42_probe_leg,
          "start_step:", args.b42_start_step,
          "axes:", args.b42_axes,
          "signs:", args.b42_signs,
          "delta:", args.b42_delta,
          "hold_steps:", args.b42_hold_steps,
          "gap_steps:", args.b42_gap_steps,
          "max_probes:", args.b42_max_probes,
          "require_swing_or_capture:", args.b42_require_swing_or_capture,
          "zero_target_velocity:", args.b42_zero_target_velocity)
    print("b43_hind_empirical_lateral_y_bridge:",
          "enabled:", args.enable_b43_hind_empirical_lateral_y_bridge,
          "legs:", args.b43_legs,
          "use_committed:", args.b43_use_committed_target,
          "during_swing:", args.b43_apply_during_swing,
          "during_capture:", args.b43_apply_during_capture,
          "post_extra:", args.b43_apply_post_capture_extra,
          "k_hx_per_y:", args.b43_k_hx_per_y,
          "k_hy_per_y:", args.b43_k_hy_per_y,
          "deadband_y:", args.b43_deadband_y,
          "max_y_err:", args.b43_max_y_err,
          "max_hx_delta:", args.b43_max_hx_delta,
          "max_hy_delta:", args.b43_max_hy_delta,
          "sigma_window:", (args.b43_min_sigma, args.b43_max_sigma),
          "hx_signs:", args.b43_hx_signs,
          "hy_signs:", args.b43_hy_signs,
          "zero_vel:", args.b43_zero_target_velocity)
    print("b44_latched_hind_y_bridge:",
          "enabled:", args.enable_b44_latched_hind_y_bridge,
          "legs:", args.b44_legs,
          "use_committed:", args.b44_use_committed_target,
          "during_swing:", args.b44_apply_during_swing,
          "during_capture:", args.b44_apply_during_capture,
          "post_extra:", args.b44_apply_post_capture_extra,
          "k_hx_per_y:", args.b44_k_hx_per_y,
          "k_hy_per_y:", args.b44_k_hy_per_y,
          "deadband_y:", args.b44_deadband_y,
          "max_y_err:", args.b44_max_y_err,
          "max_abs_from_latch(hx,hy):", (args.b44_max_abs_hx_from_latch, args.b44_max_abs_hy_from_latch),
          "max_step_delta(hx,hy):", (args.b44_max_step_hx_delta, args.b44_max_step_hy_delta),
          "sigma_window:", (args.b44_min_sigma, args.b44_max_sigma),
          "hx_signs:", args.b44_hx_signs,
          "hy_signs:", args.b44_hy_signs,
          "zero_vel:", args.b44_zero_target_velocity,
          "reset_latch_on_phase_change:", args.b44_reset_latch_on_phase_change)
    print("b48_post_landing_feasibility_foothold_planner:",
          "enabled:", args.enable_b48_post_landing_feasibility_foothold_planner,
          "apply_only_to:", args.b48_apply_only_to,
          "next_mode:", args.b48_next_leg_mode,
          "fixed_next:", args.b48_fixed_next_leg,
          "next_candidates:", args.b48_next_candidates,
          "forward_candidates:", args.b48_forward_candidates,
          "lateral_candidates:", args.b48_lateral_candidates,
          "max_extra_xy:", args.b48_max_extra_xy,
          "weights(margin,geom,target_err,step,progress):",
          (args.b48_score_margin_weight, args.b48_score_geom_weight, args.b48_score_target_err_weight, args.b48_score_step_penalty, args.b48_score_progress_weight),
          "min_post_margin:", args.b48_min_post_future_margin,
          "disable_b19_when_active:", args.b48_disable_b19_when_active)
    print("b49_elevated_swing_tracking_reference:",
          "enabled:", args.enable_b49_elevated_swing_tracking_reference,
          "apply_to_b20:", args.b49_apply_to_b20,
          "apply_to_b21:", args.b49_apply_to_b21,
          "apply_to_b37:", args.b49_apply_to_b37,
          "sigma_window:", (args.b49_min_sigma, args.b49_max_sigma),
          "min_clearance:", args.b49_min_clearance,
          "z_boost:", args.b49_z_boost)

    print("b45_directional_hind_y_push_bridge:",
          "enabled:", args.enable_b45_directional_hind_y_push_bridge,
          "legs:", args.b45_legs,
          "use_committed:", args.b45_use_committed_target,
          "during_swing:", args.b45_apply_during_swing,
          "during_capture:", args.b45_apply_during_capture,
          "post_extra:", args.b45_apply_post_capture_extra,
          "k_hx_per_y:", args.b45_k_hx_per_y,
          "k_hy_per_y:", args.b45_k_hy_per_y,
          "enable_hy:", args.b45_enable_hy,
          "deadband_y:", args.b45_deadband_y,
          "max_y_err:", args.b45_max_y_err,
          "max_step_delta(hx,hy):", (args.b45_max_step_hx_delta, args.b45_max_step_hy_delta),
          "max_abs_from_latch(hx,hy):", (args.b45_max_abs_hx_from_latch, args.b45_max_abs_hy_from_latch),
          "sigma_window:", (args.b45_min_sigma, args.b45_max_sigma),
          "hx_signs:", args.b45_hx_signs,
          "hy_signs:", args.b45_hy_signs,
          "zero_vel:", args.b45_zero_target_velocity,
          "reset_latch_on_phase_change:", args.b45_reset_latch_on_phase_change)
    print("=" * 150)

    canonical_foot_anchor_w = None
    canonical_prev_s_t = None
    canonical_requested_s_t = None
    canonical_executed_s_t = None
    canonical_exec_prev_s_t = None
    canonical_phi_t = None
    canonical_s_t = None
    canonical_sigma_t = None
    canonical_foot_target_w = None
    canonical_current_swing_leg = None
    canonical_all_stance_gap_count = 0
    canonical_exec_info = {"gate_reason": "init", "future_margin": 0.0}
    canonical_shift_info = {"active": False}
    canonical_shift_hold_count = 0
    canonical_last_shift_target_xy = None
    canonical_latched_swing_leg = None
    canonical_latched_swing_start_step = -1
    canonical_latched_swing_elapsed = 0
    canonical_latched_swing_active = False
    canonical_latched_swing_just_closed = False
    canonical_committed_leg = None
    canonical_committed_target_w = None
    canonical_commit_count = 0
    canonical_commit_info = {"active": False}
    canonical_event_queue_ids = parse_canonical_event_queue()
    canonical_event_queue_index = 0
    canonical_event_last_completed_leg = None
    canonical_event_completed_step_by_leg = {0: -10**9, 1: -10**9, 2: -10**9, 3: -10**9}
    b15_infeasible_count = 0
    b15_recenter_count = 0
    b15_recenter_prev_target_xy = None
    b15_recenter_info = {"active": False}
    b16_recenter_align_info = {"active": False}
    b16_recenter_q_align_target = None
    b17_foothold_info = {"active": False}
    b18_committed_foothold_w = torch.zeros((4, 3), device=device, dtype=dtype)
    b18_committed_valid = [False, False, False, False]
    b18_committed_step = [-10**9, -10**9, -10**9, -10**9]
    b18_pin_extra_count_by_leg = {0: 0, 1: 0, 2: 0, 3: 0}
    b18_commit_info = {"active": False}
    b18_pin_info = {"active": False}
    b19_force_info = {"active": False}
    b20_direct_ik_info = {"active": False}
    b21_cart_info = {"active": False}
    canonical_event_info = {
        "enabled": bool(args.canonical_enable_event_crawl_queue),
        "queue": [ID_TO_LEG[i] for i in canonical_event_queue_ids],
        "index": 0,
        "next_leg": ID_TO_LEG[canonical_event_queue_ids[0]] if len(canonical_event_queue_ids) > 0 else None,
        "blocked": False,
        "reason": "init",
    }

    canonical_capture_leg = None
    canonical_capture_q_ref = None
    canonical_capture_count = 0
    canonical_capture_info = {"active": False}
    canonical_stable_count = 0
    canonical_frozen_base_ref = None

    for step in range(total_steps):
        if not simulation_app.is_running():
            break

        root_pos_before_step = robot.data.root_pos_w.detach().clone()
        x_hat = make_x_hat(robot, velocity_frame="world")
        foot_pos = robot.data.body_pos_w[:, foot_indices, :]
        foot_vel = get_foot_vel(robot, foot_indices)
        if args.enable_canonical_gait_schedule:
            if canonical_foot_anchor_w is None:
                canonical_foot_anchor_w = foot_pos.detach().clone()
            canonical_phi_t, canonical_requested_s_t, canonical_sigma_t = compute_canonical_gait_schedule(step, args.num_envs, device, dtype)

            # B9-h: replace analytical phase requests with an event-driven queue.
            # While a swing is latched or a touchdown commit is active, request all-stance.
            # Only after the previous foothold is committed do we request the next queued leg.
            if args.canonical_enable_event_crawl_queue and step >= int(args.canonical_gait_start_step):
                stable_now = (
                    abs(float(x_hat[0, 3].detach().cpu())) <= float(args.canonical_event_stable_roll)
                    and abs(float(x_hat[0, 4].detach().cpu())) <= float(args.canonical_event_stable_pitch)
                    and float(x_hat[0, 2].detach().cpu()) >= float(args.canonical_event_stable_min_z)
                )
                if stable_now:
                    canonical_stable_count += 1
                else:
                    canonical_stable_count = 0
                stable_gate_blocked = bool(
                    args.canonical_event_require_stable_before_next
                    and int(canonical_event_queue_index) > 0
                    and int(canonical_stable_count) < int(args.canonical_event_stable_steps)
                )
                queue_blocked = (
                    canonical_latched_swing_leg is not None
                    or int(canonical_commit_count) > 0
                    or int(canonical_capture_count) > 0
                    or stable_gate_blocked
                    or (canonical_exec_prev_s_t is not None and bool((canonical_exec_prev_s_t[0] < 0.5).any().detach().cpu()))
                )
                raw_requested_s_t = canonical_requested_s_t.detach().clone()
                raw_sigma_t = canonical_sigma_t.detach().clone()
                b13_scores = []
                b13_selected_leg_id = None
                if queue_blocked:
                    canonical_requested_s_t = torch.ones_like(canonical_requested_s_t)
                    canonical_sigma_t = torch.zeros_like(canonical_sigma_t)
                    q_reason = "hold_during_swing_or_commit"
                else:
                    # If B15 fallback is active, force all stance and let base_ref recenter below.
                    if args.enable_b15_recenter_fallback and b15_recenter_count > 0:
                        canonical_requested_s_t = torch.ones((args.num_envs, 4), device=device, dtype=dtype)
                        canonical_sigma_t = torch.zeros((args.num_envs, 4), device=device, dtype=dtype)
                        b15_recenter_count -= 1
                        q_reason = "b15_recenter_fallback"
                        if args.b15_recenter_print and (step % max(args.print_every, 1) == 0):
                            print("[B25 RECENTER ACTIVE]",
                                  "step=", step,
                                  "remaining=", b15_recenter_count)
                    else:
                        b13_selected_leg_id, b13_scores = _b13_select_adaptive_next_leg(
                            x_hat, foot_pos, ref_cfg, device, dtype,
                            canonical_event_queue_index,
                            last_completed_leg=canonical_event_last_completed_leg,
                            completed_step_by_leg=canonical_event_completed_step_by_leg,
                            step_now=step,
                        )

                        if _b15_candidates_all_infeasible(b13_scores):
                            b15_infeasible_count += 1
                        else:
                            b15_infeasible_count = 0

                        if args.enable_b15_recenter_fallback and b15_infeasible_count >= int(args.b15_recenter_trigger_steps):
                            b15_recenter_count = int(args.b15_recenter_duration_steps)
                            b15_infeasible_count = 0
                            b15_recenter_prev_target_xy = None
                            canonical_requested_s_t = torch.ones((args.num_envs, 4), device=device, dtype=dtype)
                            canonical_sigma_t = torch.zeros((args.num_envs, 4), device=device, dtype=dtype)
                            q_reason = "b15_recenter_triggered"
                            if args.b15_recenter_print:
                                print("[B25 RECENTER TRIGGER]",
                                      "step=", step,
                                      "scores=", b13_scores)
                        elif b13_selected_leg_id is not None:
                            if args.enable_b14_no_queue_mutation:
                                # Temporary request only. The persistent queue is not modified.
                                canonical_requested_s_t = torch.ones((args.num_envs, 4), device=device, dtype=dtype)
                                canonical_sigma_t = torch.zeros((args.num_envs, 4), device=device, dtype=dtype)
                                canonical_requested_s_t[:, int(b13_selected_leg_id)] = 0.0
                                q_reason = "adaptive_candidate_request"
                            else:
                                # Legacy B13 behavior: mutates the queue slot.
                                canonical_event_queue_ids[int(canonical_event_queue_index) % len(canonical_event_queue_ids)] = int(b13_selected_leg_id)
                                canonical_requested_s_t, canonical_sigma_t = event_queue_requested_mask(
                                    canonical_event_queue_ids, canonical_event_queue_index, args.num_envs, device, dtype
                                )
                                q_reason = "adaptive_queue_request"

                            if (args.b13_print or args.b14_print) and (step % max(args.print_every, 1) == 0):
                                print("[B25 ADAPTIVE CANDIDATE]",
                                      "step=", step,
                                      "selected=", ID_TO_LEG[int(b13_selected_leg_id)],
                                      "scores=", b13_scores,
                                      "queue_mutated=", (not args.enable_b14_no_queue_mutation),
                                      "infeasible_count=", b15_infeasible_count)
                        else:
                            canonical_requested_s_t, canonical_sigma_t = event_queue_requested_mask(
                                canonical_event_queue_ids, canonical_event_queue_index, args.num_envs, device, dtype
                            )
                            q_reason = "queue_request"

                next_leg_id = int(b13_selected_leg_id) if (args.enable_b14_no_queue_mutation and b13_selected_leg_id is not None) else int(canonical_event_queue_ids[int(canonical_event_queue_index) % len(canonical_event_queue_ids)])
                canonical_event_info = {
                    "enabled": True,
                    "queue": [ID_TO_LEG[i] for i in canonical_event_queue_ids],
                    "index": int(canonical_event_queue_index),
                    "next_leg": ID_TO_LEG[next_leg_id],
                    "blocked": bool(queue_blocked),
                    "reason": q_reason if not stable_gate_blocked else "wait_stable",
                    "stable_count": int(canonical_stable_count),
                    "stable_now": bool(stable_now),
                    "capture_count": int(canonical_capture_count),
                    "raw_requested_s": raw_requested_s_t.detach().clone(),
                    "raw_sigma": raw_sigma_t.detach().clone(),
                    "b13_selected_leg": ID_TO_LEG[int(b13_selected_leg_id)] if b13_selected_leg_id is not None else None,
                    "b13_scores": b13_scores,
                    "last_completed_leg": ID_TO_LEG.get(canonical_event_last_completed_leg, None) if canonical_event_last_completed_leg is not None else None,
                }
            canonical_s_t = canonical_requested_s_t.detach().clone()
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
                    post_touchdown_lock_count = 0
                    post_touchdown_lock_leg = None
                    post_touchdown_lock_foothold_target_w = None
                    if args.enable_touchdown_committed_state and not args.touchdown_committed_until_next_step:
                        touchdown_committed = False
                        touchdown_committed_leg = None
                        touchdown_committed_step_key = None
                        touchdown_committed_step = -1
                        touchdown_committed_foothold_target_w = None
                        committed_pin_count = 0

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
            if args.enable_touchdown_committed_state and touchdown_committed and args.touchdown_committed_until_next_step:
                touchdown_committed = False
                touchdown_committed_leg = None
                touchdown_committed_step_key = None
                touchdown_committed_step = -1
                touchdown_committed_foothold_target_w = None
                committed_pin_count = 0
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
            if not swing_latched:
                swing_latched_leg = None
                swing_latched_foothold_target_w = None
                swing_latched_stance_q_ref = None
                swing_latched_started_step = -1
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

        if args.enable_committed_foothold_pinning and args.committed_pin_ignore_late_touchdown_logic and touchdown_committed:
            late_touchdown_candidate = False
            late_touchdown_hold_swing = False
            late_touchdown_ready = True
            late_touchdown_diag = {"ignored": "touchdown_committed"}

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

        if args.enable_canonical_gait_schedule and args.canonical_gait_disable_event_overrides:
            canonical_executed_s_t, canonical_current_swing_leg, canonical_all_stance_gap_count, canonical_exec_info = apply_canonical_execution_gate(
                canonical_requested_s_t,
                canonical_sigma_t,
                canonical_exec_prev_s_t,
                canonical_current_swing_leg,
                canonical_all_stance_gap_count,
                x_hat,
                foot_pos,
                ref_cfg,
                device,
                dtype,
                canonical_shift_hold_count,
            )
            canonical_s_t = canonical_executed_s_t.detach().clone()

            # B9-e: latch the actual executed swing phase.
            # The analytical/requested sigma may be near 1.0 when the gate finally opens.
            # Once opened, the executed swing must start from local sigma=0 and run for a full duration.
            canonical_latched_swing_just_closed = False
            if args.canonical_enable_latched_executed_swing:
                opened_ids = torch.nonzero(canonical_s_t[0] < 0.5, as_tuple=False).flatten()

                if canonical_latched_swing_leg is None and int(opened_ids.numel()) > 0:
                    canonical_latched_swing_leg = int(opened_ids[0].detach().cpu())
                    canonical_latched_swing_start_step = int(step)
                    canonical_latched_swing_elapsed = 0
                    if args.canonical_print_latched_swing:
                        print("[B37 LATCH START] step=", step, "leg=", ID_TO_LEG[canonical_latched_swing_leg])

                if canonical_latched_swing_leg is not None:
                    canonical_latched_swing_elapsed = int(step) - int(canonical_latched_swing_start_step)
                    duration = max(1, int(args.canonical_executed_swing_duration_steps))
                    min_duration = max(1, int(args.canonical_executed_swing_min_duration_steps))

                    if canonical_latched_swing_elapsed < duration:
                        # Override executed contact schedule and local sigma.
                        canonical_s_t = torch.ones_like(canonical_s_t)
                        canonical_s_t[:, canonical_latched_swing_leg] = 0.0
                        canonical_sigma_t = torch.zeros_like(canonical_sigma_t)
                        local_sigma = float(canonical_latched_swing_elapsed) / float(duration)
                        local_sigma = max(0.0, min(1.0, local_sigma))
                        canonical_sigma_t[:, canonical_latched_swing_leg] = local_sigma
                        canonical_current_swing_leg = canonical_latched_swing_leg
                        canonical_exec_info["gate_reason"] = str(canonical_exec_info.get("gate_reason", "pass")) + "_latched"
                        canonical_exec_info["latched_leg"] = canonical_latched_swing_leg
                        canonical_exec_info["executed_sigma"] = local_sigma
                        canonical_exec_info["latched_elapsed"] = int(canonical_latched_swing_elapsed)
                    else:
                        # Close the executed swing and enforce an all-stance gap.
                        if args.canonical_print_latched_swing:
                            print("[B37 LATCH END] step=", step, "leg=", ID_TO_LEG[canonical_latched_swing_leg])
                        canonical_s_t = torch.ones_like(canonical_s_t)
                        canonical_sigma_t = torch.zeros_like(canonical_sigma_t)
                        canonical_current_swing_leg = None
                        canonical_all_stance_gap_count = max(
                            int(canonical_all_stance_gap_count),
                            int(args.canonical_executed_swing_force_gap_after_touchdown),
                        )
                        canonical_exec_info["gate_reason"] = "latched_touchdown_gap"
                        canonical_exec_info["latched_leg"] = canonical_latched_swing_leg
                        canonical_exec_info["executed_sigma"] = 1.0
                        canonical_exec_info["latched_elapsed"] = int(canonical_latched_swing_elapsed)
                        if args.canonical_enable_touchdown_capture:
                            canonical_capture_leg = int(canonical_latched_swing_leg)
                            canonical_capture_q_ref = robot.data.joint_pos.detach().clone()
                            canonical_capture_count = int(args.canonical_capture_steps)
                            canonical_stable_count = 0
                            if args.canonical_capture_update_anchor_actual:
                                if args.enable_b23_wbc_main_committed_footref and args.b23_commit_anchor_on_capture_start and bool(b18_committed_valid[int(canonical_capture_leg)]):
                                    canonical_foot_anchor_w[:, canonical_capture_leg, :] = b18_committed_foothold_w[int(canonical_capture_leg), :].view(1, 3).detach()
                                    if args.b23_print:
                                        print("[B23 COMMIT ANCHOR ON CAPTURE]",
                                              "step=", step,
                                              "leg=", ID_TO_LEG[int(canonical_capture_leg)],
                                              "anchor_committed=", b18_committed_foothold_w[int(canonical_capture_leg), :].detach().cpu().numpy(),
                                              "actual_foot=", foot_pos[0, canonical_capture_leg, :].detach().cpu().numpy())
                                else:
                                    canonical_foot_anchor_w[:, canonical_capture_leg, :] = foot_pos[:, canonical_capture_leg, :].detach()
                            if args.canonical_capture_print:
                                print("[B37 CAPTURE START] step=", step,
                                      "leg=", ID_TO_LEG[canonical_capture_leg],
                                      "anchor_actual=", foot_pos[0, canonical_capture_leg, :].detach().cpu().numpy(),
                                      "q=", canonical_capture_q_ref[0, [HX[canonical_capture_leg], HY[canonical_capture_leg], KN[canonical_capture_leg]]].detach().cpu().numpy())
                        elif args.canonical_enable_touchdown_foothold_commit:
                            canonical_committed_leg = int(canonical_latched_swing_leg)
                            canonical_committed_target_w = canonical_foot_target_w.detach().clone() if canonical_foot_target_w is not None else foot_pos.detach().clone()
                            canonical_commit_count = int(args.canonical_commit_steps)
                            if args.canonical_enable_event_crawl_queue and args.canonical_event_commit_anchor_to_target:
                                canonical_foot_anchor_w[:, canonical_committed_leg, :] = canonical_committed_target_w[:, canonical_committed_leg, :].detach()
                            if args.canonical_commit_print:
                                print("[B9-i COMMIT START] step=", step,
                                      "leg=", ID_TO_LEG[canonical_committed_leg],
                                      "target=", canonical_committed_target_w[0, canonical_committed_leg, :].detach().cpu().numpy(),
                                      "foot=", foot_pos[0, canonical_committed_leg, :].detach().cpu().numpy())
                        canonical_latched_swing_leg = None
                        canonical_latched_swing_start_step = -1
                        canonical_latched_swing_elapsed = 0
                        canonical_latched_swing_just_closed = True

                canonical_executed_s_t = canonical_s_t.detach().clone()
                canonical_exec_info["executed_s"] = canonical_s_t.detach().clone()
                canonical_latched_swing_active = canonical_latched_swing_leg is not None
            else:
                canonical_latched_swing_active = False

            # Update anchors on executed S_t edges only, not requested S_t.
            if canonical_exec_prev_s_t is not None:
                liftoff = (canonical_exec_prev_s_t > 0.5) & (canonical_s_t < 0.5)
                touchdown_edge = (canonical_exec_prev_s_t < 0.5) & (canonical_s_t > 0.5)
                for li in range(4):
                    if bool(liftoff[0, li].detach().cpu()):
                        canonical_foot_anchor_w[:, li, :] = foot_pos[:, li, :].detach()
                    if bool(touchdown_edge[0, li].detach().cpu()):
                        canonical_foot_anchor_w[:, li, :] = foot_pos[:, li, :].detach()
                if args.canonical_gait_print_edges and bool((liftoff | touchdown_edge).any().detach().cpu()):
                    print("[B17 EDGE] step=", step,
                          "liftoff=", liftoff[0].detach().cpu().numpy(),
                          "touchdown=", touchdown_edge[0].detach().cpu().numpy(),
                          "requested_s=", canonical_requested_s_t[0].detach().cpu().numpy(),
                          "executed_s=", canonical_s_t[0].detach().cpu().numpy(),
                          "phi=", canonical_phi_t[0].detach().cpu().numpy(),
                          "reason=", canonical_exec_info.get("gate_reason"))
            canonical_exec_prev_s_t = canonical_s_t.detach().clone()
            canonical_foot_target_w = canonical_swing_ref_from_schedule(
                canonical_foot_anchor_w, robot.data.root_quat_w, canonical_s_t, canonical_sigma_t
            )
            canonical_foot_target_w, b17_foothold_info = _b17_apply_support_aware_foothold_mapper(
                canonical_foot_target_w,
                canonical_foot_anchor_w,
                robot.data.root_quat_w,
                canonical_s_t,
                canonical_sigma_t,
                x_hat,
                foot_pos,
                ref_cfg,
                device,
                dtype,
                step,
            )
            canonical_foot_target_w, b48_foothold_info = _b48_apply_post_landing_feasibility_foothold_planner(
                canonical_foot_target_w,
                canonical_foot_anchor_w,
                robot.data.root_quat_w,
                canonical_s_t,
                canonical_sigma_t,
                x_hat,
                foot_pos,
                ref_cfg,
                device,
                dtype,
                step,
            )
            if isinstance(b48_foothold_info, dict) and b48_foothold_info.get("active", False):
                b17_foothold_info = b48_foothold_info
            canonical_foot_target_w, b17_foothold_info, b19_force_info = _b19_apply_forced_exploratory_foothold_bias(
                canonical_foot_target_w,
                canonical_foot_anchor_w,
                robot.data.root_quat_w,
                canonical_s_t,
                canonical_sigma_t,
                b17_foothold_info,
                step,
            )
            canonical_foot_target_w, b18_committed_foothold_w, b18_committed_valid, b18_committed_step, b18_commit_info = _b18_commit_or_reuse_foothold_target(
                canonical_foot_target_w, canonical_foot_anchor_w, b17_foothold_info, canonical_s_t, canonical_sigma_t,
                b18_committed_foothold_w, b18_committed_valid, b18_committed_step, step
            )
            if args.b18_print and b18_commit_info.get("active", False):
                print("[B25 FOOTHOLD COMMIT]",
                      "step=", step,
                      "leg=", b18_commit_info.get("leg"),
                      "reason=", b18_commit_info.get("reason"),
                      "committed_target=", b18_commit_info.get("committed_target"),
                      "selected_delta=", b18_commit_info.get("selected_delta"))
            elif args.b18_print_reuse and b18_commit_info.get("reuse", False) and (step % max(args.print_every, 1) == 0):
                print("[B25 FOOTHOLD REUSE]",
                      "step=", step,
                      "leg=", b18_commit_info.get("leg"),
                      "reason=", b18_commit_info.get("reason"),
                      "committed_target=", b18_commit_info.get("committed_target"))
            canonical_foot_target_w, b23_footref_info = _b23_apply_wbc_main_committed_footref(
                canonical_foot_target_w,
                canonical_foot_anchor_w,
                canonical_s_t,
                canonical_sigma_t,
                canonical_capture_leg,
                canonical_capture_count,
                b18_pin_extra_count_by_leg,
                b18_committed_foothold_w,
                b18_committed_valid,
                step,
            )
            if args.b23_print and (b23_footref_info.get("active", False) or step % max(args.print_every, 1) == 0):
                print("[B23 MAIN FOOT REF]",
                      "step=", step,
                      "active=", b23_footref_info.get("active", False),
                      "reason=", b23_footref_info.get("reason"),
                      "leg=", b23_footref_info.get("leg"),
                      "sigma=", b23_footref_info.get("sigma"),
                      "xy_mode=", b23_footref_info.get("xy_mode"),
                      "target_before=", b23_footref_info.get("target_before"),
                      "target_after=", b23_footref_info.get("target_after"),
                      "committed=", b23_footref_info.get("committed"),
                      "target_minus_anchor=", b23_footref_info.get("target_minus_anchor"))

            b49_tracking_targets_w, b49_tracking_info = _b49_make_elevated_tracking_targets(
                b18_committed_foothold_w,
                b18_committed_valid,
                canonical_foot_target_w,
                canonical_foot_anchor_w,
                canonical_s_t,
                canonical_sigma_t,
                step,
            )
            if bool(getattr(args, "b49_print", False)) and (b49_tracking_info.get("active", False) or step % max(args.print_every, 1) == 0):
                print("[B49 ELEVATED SWING TRACKING REF]",
                      "step=", step,
                      "active=", b49_tracking_info.get("active", False),
                      "reason=", b49_tracking_info.get("reason"),
                      "leg=", b49_tracking_info.get("leg"),
                      "sigma=", b49_tracking_info.get("sigma"),
                      "old_target=", b49_tracking_info.get("old_target"),
                      "new_target=", b49_tracking_info.get("new_target"),
                      "z_ref=", b49_tracking_info.get("z_ref"),
                      "z_min=", b49_tracking_info.get("z_min"),
                      "error=", b49_tracking_info.get("error"))

            base_ref, canonical_shift_info = apply_canonical_feasible_trunk_shift(
                base_ref, x_hat, foot_pos, canonical_exec_info, prev_base_ref, canonical_last_shift_target_xy
            )
            if canonical_shift_info.get("active", False):
                canonical_shift_hold_count += 1
                canonical_last_shift_target_xy = base_ref[:, 0:2].detach().clone()
            else:
                # Reset the tracked shift count only when no swing is requested.
                if not bool((canonical_requested_s_t[0] < 0.5).any().detach().cpu()):
                    canonical_shift_hold_count = 0
                    canonical_last_shift_target_xy = None

            # B15 fallback: if all next-leg candidates are infeasible for a while,
            # keep all stance and recenter the trunk toward the full support center.
            b15_recenter_info = {"active": False}
            if args.enable_b15_recenter_fallback and b15_recenter_count > 0:
                base_ref, b15_recenter_info = _b15_apply_all_stance_recenter(
                    base_ref, x_hat, foot_pos, prev_target_xy=b15_recenter_prev_target_xy
                )
                if b15_recenter_info.get("active", False):
                    b15_recenter_prev_target_xy = base_ref[:, 0:2].detach().clone()
                    canonical_shift_info = {
                        "active": True,
                        "candidate_leg": -1,
                        "base_ref_xy_err": b15_recenter_info.get("base_ref_xy_err"),
                        "base_ref": base_ref,
                        "shift_hold_count": int(args.b15_recenter_duration_steps - b15_recenter_count),
                        "b15_recenter_elapsed": int(args.b15_recenter_duration_steps - b15_recenter_count),
                        "b15_recenter": True,
                        "target_xy": b15_recenter_info.get("target_xy"),
                    }
                    if args.b15_recenter_print and (step % max(args.print_every, 1) == 0):
                        print("[B25 RECENTER REF]",
                              "step=", step,
                              "remaining=", b15_recenter_count,
                              "base_ref_xy_err=", float(b15_recenter_info["base_ref_xy_err"][0].detach().cpu()),
                              "target_xy=", b15_recenter_info["target_xy"][0].detach().cpu().numpy(),
                              "next_xy=", b15_recenter_info["next_xy"][0].detach().cpu().numpy())
            elif args.enable_b15_recenter_fallback:
                b15_recenter_prev_target_xy = None
            canonical_exec_info["shift_hold_count"] = int(canonical_shift_hold_count)
            canonical_exec_info["base_ref_xy_err"] = (
                float(canonical_shift_info.get("base_ref_xy_err", torch.zeros((1,), device=device))[0].detach().cpu())
                if canonical_shift_info.get("base_ref_xy_err", None) is not None else 0.0
            )
            if args.b38_print and isinstance(canonical_shift_info, dict) and canonical_shift_info.get("b38_candidate_support_recenter", False) and (step % max(args.print_every, 1) == 0):
                print("[B38 CANDIDATE SUPPORT RECENTER]",
                      "step=", step,
                      "candidate=", ID_TO_LEG.get(int(canonical_shift_info.get("candidate_leg", -1)), canonical_shift_info.get("candidate_leg")),
                      "future_margin=", canonical_shift_info.get("future_margin"),
                      "geom_current=", float(canonical_shift_info.get("b9k_geom_margin_current", torch.zeros((1,), device=device))[0].detach().cpu()),
                      "geom_target=", float(canonical_shift_info.get("b9k_geom_margin_target", torch.zeros((1,), device=device))[0].detach().cpu()),
                      "target_err=", float(canonical_shift_info.get("b9k_target_err", torch.zeros((1,), device=device))[0].detach().cpu()),
                      "target_xy=", canonical_shift_info.get("target_xy")[0].detach().cpu().numpy() if canonical_shift_info.get("target_xy", None) is not None else None,
                      "next_xy=", canonical_shift_info.get("next_xy")[0].detach().cpu().numpy() if canonical_shift_info.get("next_xy", None) is not None else None,
                      "base_ref_xy_err=", canonical_exec_info.get("base_ref_xy_err"))
            if args.b39_print and isinstance(canonical_shift_info, dict) and canonical_shift_info.get("b39_latched_support_recenter", False) and (step % max(args.print_every, 1) == 0):
                print("[B39 LATCHED SUPPORT RECENTER]",
                      "step=", step,
                      "candidate=", ID_TO_LEG.get(int(canonical_shift_info.get("candidate_leg", -1)), canonical_shift_info.get("candidate_leg")),
                      "future_margin=", canonical_shift_info.get("future_margin"),
                      "geom_current=", float(canonical_shift_info.get("b9k_geom_margin_current", torch.zeros((1,), device=device))[0].detach().cpu()),
                      "geom_target=", float(canonical_shift_info.get("b9k_geom_margin_target", torch.zeros((1,), device=device))[0].detach().cpu()),
                      "target_err=", float(canonical_shift_info.get("b9k_target_err", torch.zeros((1,), device=device))[0].detach().cpu()),
                      "target_xy=", canonical_shift_info.get("target_xy")[0].detach().cpu().numpy() if canonical_shift_info.get("target_xy", None) is not None else None,
                      "next_xy=", canonical_shift_info.get("next_xy")[0].detach().cpu().numpy() if canonical_shift_info.get("next_xy", None) is not None else None,
                      "applied_delta=", canonical_shift_info.get("applied_delta_xy")[0].detach().cpu().numpy() if canonical_shift_info.get("applied_delta_xy", None) is not None else None,
                      "base_ref_xy_err=", canonical_exec_info.get("base_ref_xy_err"))
            # B9-i: freeze the canonical base reference during swing/capture/commit.
            # This prevents legacy phase/base_ref motion from moving the trunk while a
            # canonical leg is still in the air or just landed.
            canonical_freeze_now = bool(
                args.canonical_freeze_base_ref_during_swing_commit
                and (
                    canonical_latched_swing_leg is not None
                    or int(canonical_capture_count) > 0
                    or int(canonical_commit_count) > 0
                    or bool((canonical_s_t[0] < 0.5).any().detach().cpu())
                )
            )
            if canonical_freeze_now:
                if canonical_frozen_base_ref is None:
                    canonical_frozen_base_ref = base_ref.detach().clone()
                    if args.canonical_freeze_base_ref_keep_current_xy:
                        canonical_frozen_base_ref[:, 0:2] = x_hat[:, 0:2]
                    if float(args.canonical_freeze_base_ref_height) > 0.0:
                        canonical_frozen_base_ref[:, 2] = float(args.canonical_freeze_base_ref_height)
                    canonical_frozen_base_ref[:, 3:6] = 0.0
                base_ref = canonical_frozen_base_ref.detach().clone()
            else:
                canonical_frozen_base_ref = None

            base_ref_raw = base_ref.detach().clone()

        # B8-bj: separate trunk shift from swing/recovery.
        freeze_base_now = (
            (args.freeze_base_ref_during_swing and phase in ["lift", "hold_lift", "lower"] and (swing_enabled or late_touchdown_hold_swing or (args.latched_swing_keep_base_freeze and swing_latched)))
            or (args.freeze_base_ref_during_touchdown_recovery and (touchdown_gate_active or freeze_step_key_for_touchdown or late_touchdown_hold_swing or (args.latched_swing_keep_base_freeze and swing_latched) or post_touchdown_lock_count > 0 or (args.touchdown_committed_keep_base_freeze and touchdown_committed)))
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

        swing_enabled_pre_latch = gate_swing(phase, phase_step, trial_out, x_hat)
        swing_enabled = swing_enabled_pre_latch
        old_forced, controlled_forced = gate_debug_flags(phase, phase_step, trial_out, x_hat)

        # Pre-detach gates. These can prevent a new swing from starting, but must not cancel a latched swing.
        pre_detach_blocked = False
        if args.enable_feasible_region_lite_shift and phase in ["lift", "hold_lift", "lower"]:
            if float(trial_out.margin_to_edge[0].detach().cpu()) < float(args.fr_lite_min_margin_for_detach):
                pre_detach_blocked = True
        if step_idx >= 1 and phase in ["lift", "hold_lift", "lower"]:
            _margin_now = float(trial_out.margin_to_edge[0].detach().cpu())
            _needed_margin = float(args.fr_lite_min_margin_for_detach) if args.enable_feasible_region_lite_shift else float(args.second_shift_requires_margin)
            if _margin_now < _needed_margin:
                pre_detach_blocked = True
        if args.block_recenter_until_previous_valid and step_idx >= 1 and not previous_load_bearing_valid:
            pre_detach_blocked = True
        if args.force_no_swing_in_shift_phase and phase == "shift":
            pre_detach_blocked = True

        if pre_detach_blocked and not swing_latched:
            swing_enabled = False

        if phase in ["shift", "settle", "recenter", "warmup", "done"] or late_touchdown or (args.recenter_require_safe_for_next and step_idx >= 1 and not recenter_safe_seen):
            # These phases normally keep all contacts. B8-bo overrides this only for an already latched swing.
            if not swing_latched:
                swing_enabled = False

        if late_touchdown_hold_swing and args.late_touchdown_keep_swing_controller:
            # Keep selected leg in swing/contact-free mode while lower phase waits for foothold commit.
            swing_enabled = True

        if args.enable_canonical_gait_schedule and args.canonical_gait_disable_event_overrides:
            # Do not let event-driven B8 state machine own controller switching.
            late_touchdown = False
            late_touchdown_candidate = False
            late_touchdown_hold_swing = False
            touchdown_committed = False
            swing_latched = False
            post_touchdown_lock_count = 0
            committed_pin_count = 0
            touchdown_gate_active = False
            freeze_step_key_for_touchdown = False

        # B8-bp: touchdown committed state has priority over any lower/lift phase still remaining.
        same_step_touchdown_committed = (
            args.enable_touchdown_committed_state
            and touchdown_committed
            and (
                (touchdown_committed_step_key is None)
                or (current_step_key == touchdown_committed_step_key)
                or (not args.touchdown_committed_until_next_step)
            )
        )
        if same_step_touchdown_committed and args.touchdown_committed_force_all_stance:
            swing_enabled = False

        # B8-bo latch: once a leg really starts contact-free swing, keep it contact-free until touchdown commit.
        can_start_latch = (
            args.enable_latched_swing_until_touchdown
            and (not swing_latched)
            and (not (same_step_touchdown_committed and args.touchdown_committed_blocks_same_step_relatch))
            and swing_enabled
            and phase in ["lift", "hold_lift", "lower"]
        )
        if can_start_latch:
            swing_latched = True
            swing_latched_leg = LEG_TO_ID[args.test_leg]
            swing_latched_foothold_target_w = foothold_target_w.detach().clone() if foothold_target_w is not None else None
            swing_latched_stance_q_ref = current_swing_stance_q_ref.detach().clone() if current_swing_stance_q_ref is not None else None
            swing_latched_started_step = step

        if args.enable_latched_swing_until_touchdown and swing_latched:
            # Latch follows the physical swing leg, not a later schedule leg.
            if swing_latched_leg is not None:
                # Keep both schedule-facing and mask-facing leg identifiers as names.
                # future_swing_mask_for_leg expects "LF/RF/LH/RH", not integer ids.
                args.test_leg = ID_TO_LEG[int(swing_latched_leg)]
                active_leg = ID_TO_LEG[int(swing_latched_leg)]
            if args.latched_swing_force_contact_open and not late_touchdown:
                swing_enabled = True
            if late_touchdown_hold_swing:
                swing_enabled = True
            if args.latched_swing_release_on_late_touchdown and late_touchdown:
                # Touchdown accepted. Contact can be restored now, but keep post-touchdown lock.
                post_touchdown_lock_count = int(args.post_touchdown_lock_steps)
                post_touchdown_lock_leg = swing_latched_leg
                post_touchdown_lock_foothold_target_w = (
                    swing_latched_foothold_target_w.detach().clone()
                    if swing_latched_foothold_target_w is not None
                    else (foothold_target_w.detach().clone() if foothold_target_w is not None else None)
                )
                if args.enable_touchdown_committed_state:
                    touchdown_committed = True
                    touchdown_committed_leg = post_touchdown_lock_leg
                    touchdown_committed_step_key = current_step_key
                    touchdown_committed_step = step
                    touchdown_committed_foothold_target_w = (
                        post_touchdown_lock_foothold_target_w.detach().clone()
                        if post_touchdown_lock_foothold_target_w is not None
                        else None
                    )
                    committed_pin_count = int(args.committed_pin_extra_steps)
                swing_latched = False
                swing_enabled = False

        contact_mask = _future_mask_for_active_leg() if swing_enabled else all_stance_mask(args.num_envs, device, dtype)
        if args.enable_touchdown_committed_state and touchdown_committed and args.touchdown_committed_force_all_stance:
            swing_enabled = False
            contact_mask = all_stance_mask(args.num_envs, device, dtype)
        if args.enable_canonical_gait_schedule and args.canonical_gait_disable_event_overrides:
            contact_mask = canonical_s_t.clone()
            swing_enabled = bool((canonical_s_t < 0.5).any().detach().cpu())

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
                if swing_latched_foothold_target_w is not None:
                    real_swing_foothold_target_w = swing_latched_foothold_target_w.detach().clone()
                if swing_latched_stance_q_ref is not None:
                    real_swing_stance_q_ref = swing_latched_stance_q_ref.detach().clone()

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
            if args.enable_latched_swing_until_touchdown and args.latched_swing_release_on_late_touchdown:
                post_touchdown_lock_count = int(args.post_touchdown_lock_steps)
                post_touchdown_lock_leg = previous_swing_leg
                post_touchdown_lock_foothold_target_w = (
                    previous_swing_foothold_target_w.detach().clone()
                    if previous_swing_foothold_target_w is not None
                    else None
                )
                if args.enable_touchdown_committed_state:
                    touchdown_committed = True
                    touchdown_committed_leg = previous_swing_leg
                    touchdown_committed_step_key = current_step_key
                    touchdown_committed_step = step
                    touchdown_committed_foothold_target_w = (
                        previous_swing_foothold_target_w.detach().clone()
                        if previous_swing_foothold_target_w is not None
                        else None
                    )
                    committed_pin_count = int(args.committed_pin_extra_steps)
                swing_latched = False

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
        if args.enable_canonical_gait_schedule and args.canonical_gait_disable_event_overrides:
            swing_target = canonical_foot_target_w.clone()

        try:
            if isinstance(canonical_shift_info, dict):
                canonical_shift_info["base_ref"] = base_ref
                if isinstance(canonical_exec_info, dict):
                    canonical_shift_info["shift_hold_count"] = canonical_exec_info.get("shift_hold_count", 0)
        except Exception:
            pass

        wbc_cfg_active, b11_reloc_info = _b11_make_relocation_wbc_cfg(wbc_cfg, canonical_shift_info, x_hat)
        if isinstance(canonical_shift_info, dict) and canonical_shift_info.get("b15_recenter", False):
            wbc_cfg_active = copy.copy(wbc_cfg_active)
            wbc_cfg_active.kp_base_xy = float(args.b15_recenter_wbc_kp_xy)
            wbc_cfg_active.kd_base_xy = float(args.b15_recenter_wbc_kd_xy)
            wbc_cfg_active.w_base_acc = float(args.b15_recenter_w_base_acc)
            wbc_cfg_active.w_stance_acc = float(args.b15_recenter_w_stance_acc)
            b11_reloc_info["active"] = True
            b11_reloc_info["reason"] = "b15_recenter"
            b11_reloc_info["candidate"] = -1
            try:
                b11_reloc_info["base_ref_xy_err"] = float(canonical_shift_info.get("base_ref_xy_err")[0].detach().cpu())
            except Exception:
                pass
        b16_recenter_align_info = {"active": False}
        b16_recenter_q_align_target = None
        if isinstance(canonical_shift_info, dict) and canonical_shift_info.get("b15_recenter", False):
            rec_elapsed = int(canonical_shift_info.get("b15_recenter_elapsed", 0))
            b16_recenter_q_align_target, b16_recenter_align_info = _b16_compute_recenter_all_stance_target(
                robot, Jfeet_full, base_ref, x_hat, rec_elapsed
            )
            q_align_target = b16_recenter_q_align_target
            b12_align_info = b16_recenter_align_info
        else:
            q_align_target, b12_align_info = _b12_compute_pd_alignment_target(robot, Jfeet_full, canonical_shift_info, x_hat)
        q_nom_active = q_align_target if b12_align_info.get("active", False) else q_nom

        b18_pin_info = {"active": False}
        b18_pin_target = None
        b18_pin_leg = None
        try:
            if args.enable_b18_committed_foothold_pinning:
                # B18 hotfix: infer active swing leg directly from canonical_s_t.
                # The previous B18 referenced canonical_active_swing_leg, which is not defined in this script scope.
                _active_b18 = []
                try:
                    if canonical_s_t is not None:
                        _active_b18 = [int(i) for i in torch.nonzero(canonical_s_t[0] < 0.5, as_tuple=False).flatten()]
                except Exception:
                    _active_b18 = []

                if args.b18_pin_during_swing and len(_active_b18) == 1:
                    b18_pin_leg = int(_active_b18[0])
                elif args.b18_pin_during_capture and canonical_capture_count > 0 and canonical_capture_leg is not None:
                    b18_pin_leg = int(canonical_capture_leg)
                else:
                    for _li, _cnt in list(b18_pin_extra_count_by_leg.items()):
                        if int(_cnt) > 0 and bool(b18_committed_valid[int(_li)]):
                            b18_pin_leg = int(_li)
                            b18_pin_extra_count_by_leg[int(_li)] = max(0, int(_cnt) - 1)
                            break
                if b18_pin_leg is not None and bool(b18_committed_valid[int(b18_pin_leg)]):
                    b18_pin_target, b18_pin_info = _b18_foothold_pin_target(robot, Jfeet_full, foot_pos, b18_committed_foothold_w, b18_committed_valid, b18_pin_leg)
                    if b18_pin_info.get("active", False):
                        q_nom_active = b18_pin_target
                        if args.b18_print and (step % max(args.print_every, 1) == 0):
                            print("[B29 FOOTHOLD PIN]",
                                  "step=", step,
                                  "leg=", b18_pin_info.get("leg"),
                                  "foot_err=", b18_pin_info.get("foot_err"),
                                  "max_delta=", b18_pin_info.get("max_delta"),
                                  "extra_counts=", b18_pin_extra_count_by_leg)
        except Exception as _e:
            if args.b18_print and (step % max(args.print_every, 1) == 0):
                print("[B25 FOOTHOLD PIN ERROR]", "step=", step, "err=", repr(_e))

        b20_direct_ik_info = {"active": False}
        b20_direct_ik_target = None
        try:
            if args.enable_b20_committed_foothold_direct_swing_ik:
                b20_direct_ik_target, b20_direct_ik_info = _b20_committed_direct_swing_ik_target(
                    robot, Jfeet_full, foot_pos, canonical_s_t, canonical_sigma_t,
                    (b49_tracking_targets_w if bool(getattr(args, "b49_apply_to_b20", False)) and 'b49_tracking_targets_w' in locals() else b18_committed_foothold_w),
                    b18_committed_valid
                )
                if b20_direct_ik_info.get("active", False):
                    q_nom_active = b20_direct_ik_target
                    if args.b20_direct_ik_print and (step % max(args.print_every, 1) == 0):
                        print("[B21 DIRECT SWING IK]",
                              "step=", step,
                              "leg=", b20_direct_ik_info.get("leg"),
                              "sigma=", b20_direct_ik_info.get("sigma"),
                              "foot_err=", b20_direct_ik_info.get("foot_err"),
                              "max_delta=", b20_direct_ik_info.get("max_delta"),
                              "committed_target=", b20_direct_ik_info.get("committed_target"))
                elif args.b20_direct_ik_print and (step % max(args.print_every, 1) == 0):
                    print("[B20 DIRECT SWING IK HOLD]",
                          "step=", step,
                          "reason=", b20_direct_ik_info.get("reason"),
                          "leg=", b20_direct_ik_info.get("leg"),
                          "sigma=", b20_direct_ik_info.get("sigma"))
        except Exception as _e:
            if args.b20_direct_ik_print and (step % max(args.print_every, 1) == 0):
                print("[B20 DIRECT SWING IK ERROR]", "step=", step, "err=", repr(_e))

        b25_pd_bridge_info = _b25_apply_canonical_swing_pd_bridge(
            robot,
            b20_direct_ik_target, b20_direct_ik_info,
            b18_pin_target, b18_pin_info,
            step,
        )
        if args.b25_print and (b25_pd_bridge_info.get("active", False) or step % max(args.print_every, 1) == 0):
            print("[B25 CANONICAL SWING PD BRIDGE]",
                  "step=", step,
                  "active=", b25_pd_bridge_info.get("active", False),
                  "reason=", b25_pd_bridge_info.get("reason"),
                  "source=", b25_pd_bridge_info.get("source"),
                  "leg=", b25_pd_bridge_info.get("leg"),
                  "joint_ids=", b25_pd_bridge_info.get("joint_ids"),
                  "q_now=", b25_pd_bridge_info.get("q_now"),
                  "q_goal=", b25_pd_bridge_info.get("q_goal"),
                  "delta=", b25_pd_bridge_info.get("delta"))

        b26_pd_bridge_info = _b26_apply_latched_safe_swing_pd_bridge(
            robot,
            b20_direct_ik_target, b20_direct_ik_info,
            b18_pin_target, b18_pin_info,
            step,
        )
        if args.b26_print and (b26_pd_bridge_info.get("active", False) or step % max(args.print_every, 1) == 0):
            print("[B26 LATCHED SAFE SWING PD BRIDGE]",
                  "step=", step,
                  "active=", b26_pd_bridge_info.get("active", False),
                  "reason=", b26_pd_bridge_info.get("reason"),
                  "source=", b26_pd_bridge_info.get("source"),
                  "leg=", b26_pd_bridge_info.get("leg"),
                  "latched_new=", b26_pd_bridge_info.get("latched_new"),
                  "joint_ids=", b26_pd_bridge_info.get("joint_ids"),
                  "q_now=", b26_pd_bridge_info.get("q_now"),
                  "q_latch=", b26_pd_bridge_info.get("q_latch"),
                  "q_goal=", b26_pd_bridge_info.get("q_goal"),
                  "q_cmd=", b26_pd_bridge_info.get("q_cmd"),
                  "delta_from_now=", b26_pd_bridge_info.get("delta_from_now"),
                  "delta_from_latch=", b26_pd_bridge_info.get("delta_from_latch"))

        b27_pd_bridge_info = _b27_apply_signed_latched_swing_pd_bridge(
            robot,
            b20_direct_ik_target, b20_direct_ik_info,
            step,
        )
        if args.b27_print and (b27_pd_bridge_info.get("active", False) or step % max(args.print_every, 1) == 0):
            print("[B27 SIGNED LATCHED SWING PD BRIDGE]",
                  "step=", step,
                  "active=", b27_pd_bridge_info.get("active", False),
                  "reason=", b27_pd_bridge_info.get("reason"),
                  "source=", b27_pd_bridge_info.get("source"),
                  "leg=", b27_pd_bridge_info.get("leg"),
                  "latched_new=", b27_pd_bridge_info.get("latched_new"),
                  "joint_ids=", b27_pd_bridge_info.get("joint_ids"),
                  "q_now=", b27_pd_bridge_info.get("q_now"),
                  "q_latch=", b27_pd_bridge_info.get("q_latch"),
                  "q_goal_raw=", b27_pd_bridge_info.get("q_goal_raw"),
                  "q_goal=", b27_pd_bridge_info.get("q_goal"),
                  "q_cmd=", b27_pd_bridge_info.get("q_cmd"),
                  "raw_delta=", b27_pd_bridge_info.get("raw_delta"),
                  "signed_delta=", b27_pd_bridge_info.get("signed_delta"),
                  "delta_from_now=", b27_pd_bridge_info.get("delta_from_now"))

        b28_pd_bridge_info = _b28_apply_hold_latched_swing_pd_bridge(
            robot,
            b20_direct_ik_target, b20_direct_ik_info,
            step,
        )
        if args.b28_print and (b28_pd_bridge_info.get("active", False) or step % max(args.print_every, 1) == 0):
            print("[B28 HOLD LATCHED SWING PD BRIDGE]",
                  "step=", step,
                  "active=", b28_pd_bridge_info.get("active", False),
                  "reason=", b28_pd_bridge_info.get("reason"),
                  "source=", b28_pd_bridge_info.get("source"),
                  "leg=", b28_pd_bridge_info.get("leg"),
                  "latched_new=", b28_pd_bridge_info.get("latched_new"),
                  "hold_age=", b28_pd_bridge_info.get("hold_age"),
                  "joint_ids=", b28_pd_bridge_info.get("joint_ids"),
                  "q_now=", b28_pd_bridge_info.get("q_now"),
                  "q_latch=", b28_pd_bridge_info.get("q_latch"),
                  "q_goal_raw=", b28_pd_bridge_info.get("q_goal_raw"),
                  "q_goal=", b28_pd_bridge_info.get("q_goal"),
                  "q_cmd=", b28_pd_bridge_info.get("q_cmd"),
                  "raw_delta=", b28_pd_bridge_info.get("raw_delta"),
                  "signed_delta=", b28_pd_bridge_info.get("signed_delta"),
                  "delta_from_now=", b28_pd_bridge_info.get("delta_from_now"))

        b29_active_swing_names = []
        if args.enable_canonical_gait_schedule and canonical_s_t is not None:
            _b29_active_swing_idx = torch.nonzero(canonical_s_t[0] < 0.5, as_tuple=False).flatten()
            b29_active_swing_names = [ID_TO_LEG[int(x.detach().cpu())] for x in _b29_active_swing_idx]

        b29_capture_leg = locals().get("capture_leg", None)
        b29_capture_count = locals().get("capture_count", 0)
        b29_post_capture_pin_count = locals().get("post_capture_pin_count", 0)

        b29_pd_bridge_info = _b29_apply_task_space_foot_error_bridge(
            robot,
            foot_pos,
            b23_footref_info,
            b18_pin_info,
            b29_active_swing_names,
            b29_capture_leg,
            b29_capture_count,
            b29_post_capture_pin_count,
            step,
        )
        if args.b29_print and (b29_pd_bridge_info.get("active", False) or step % max(args.print_every, 1) == 0):
            print("[B29 TASK SPACE FOOT ERROR BRIDGE]",
                  "step=", step,
                  "active=", b29_pd_bridge_info.get("active", False),
                  "reason=", b29_pd_bridge_info.get("reason"),
                  "source=", b29_pd_bridge_info.get("source"),
                  "phase=", b29_pd_bridge_info.get("phase"),
                  "leg=", b29_pd_bridge_info.get("leg"),
                  "latched_new=", b29_pd_bridge_info.get("latched_new"),
                  "hold_age=", b29_pd_bridge_info.get("hold_age"),
                  "target_w=", b29_pd_bridge_info.get("target_w"),
                  "foot_w=", b29_pd_bridge_info.get("foot_w"),
                  "foot_err_raw=", b29_pd_bridge_info.get("foot_err_raw"),
                  "foot_err_used=", b29_pd_bridge_info.get("foot_err_used"),
                  "dq=", b29_pd_bridge_info.get("dq"),
                  "q_now=", b29_pd_bridge_info.get("q_now"),
                  "q_latch=", b29_pd_bridge_info.get("q_latch"),
                  "q_goal=", b29_pd_bridge_info.get("q_goal"),
                  "q_cmd=", b29_pd_bridge_info.get("q_cmd"),
                  "delta_from_now=", b29_pd_bridge_info.get("delta_from_now"))

        b30_capture_leg = locals().get("canonical_capture_leg", locals().get("capture_leg", None))
        b30_capture_count = locals().get("canonical_capture_count", locals().get("capture_count", 0))
        b30_post_capture_pin_count = locals().get("post_capture_pin_count", 0)
        try:
            b30_post_capture_pin_count = max(int(b30_post_capture_pin_count), int(b18_pin_extra_count_by_leg[LEG_TO_ID.get(b29_active_swing_names[0], 0)]))
        except Exception:
            pass

        b30_pd_bridge_info = _b30_apply_jfeet_task_space_bridge(
            robot,
            Jfeet_full,
            foot_pos,
            b23_footref_info,
            b18_pin_info,
            b29_active_swing_names,
            b30_capture_leg,
            b30_capture_count,
            b30_post_capture_pin_count,
            step,
        )
        if args.b30_print and (b30_pd_bridge_info.get("active", False) or step % max(args.print_every, 1) == 0):
            print("[B30 JFEET TASK SPACE FOOT ERROR BRIDGE]",
                  "step=", step,
                  "active=", b30_pd_bridge_info.get("active", False),
                  "reason=", b30_pd_bridge_info.get("reason"),
                  "source=", b30_pd_bridge_info.get("source"),
                  "phase=", b30_pd_bridge_info.get("phase"),
                  "leg=", b30_pd_bridge_info.get("leg"),
                  "latched_new=", b30_pd_bridge_info.get("latched_new"),
                  "hold_age=", b30_pd_bridge_info.get("hold_age"),
                  "target_w=", b30_pd_bridge_info.get("target_w"),
                  "foot_w=", b30_pd_bridge_info.get("foot_w"),
                  "foot_err_raw=", b30_pd_bridge_info.get("foot_err_raw"),
                  "foot_err_used=", b30_pd_bridge_info.get("foot_err_used"),
                  "dq=", b30_pd_bridge_info.get("dq"),
                  "q_now=", b30_pd_bridge_info.get("q_now"),
                  "q_latch=", b30_pd_bridge_info.get("q_latch"),
                  "q_goal=", b30_pd_bridge_info.get("q_goal"),
                  "q_cmd=", b30_pd_bridge_info.get("q_cmd"),
                  "delta_from_now=", b30_pd_bridge_info.get("delta_from_now"))

        b31_probe_info = _b31_update_jacobian_response_probe(b30_pd_bridge_info, foot_pos, step)
        if args.b31_print and b31_probe_info.get("event") in ("start", "complete"):
            print("[B31 JACOBIAN RESPONSE PROBE]",
                  "event=", b31_probe_info.get("event"),
                  "seq=", b31_probe_info.get("seq"),
                  "step=", step,
                  "leg=", b31_probe_info.get("leg"),
                  "start_step=", b31_probe_info.get("start_step"),
                  "end_step=", b31_probe_info.get("end_step"),
                  "dt_steps=", b31_probe_info.get("dt_steps"),
                  "reason=", b31_probe_info.get("reason"),
                  "dq=", b31_probe_info.get("dq"),
                  "pred_delta=", b31_probe_info.get("pred_delta"),
                  "actual_delta=", b31_probe_info.get("actual_delta"),
                  "error=", b31_probe_info.get("error"),
                  "pred_norm=", b31_probe_info.get("pred_norm"),
                  "actual_norm=", b31_probe_info.get("actual_norm"),
                  "cosine=", b31_probe_info.get("cosine"),
                  "axis_sign_match=", b31_probe_info.get("axis_sign_match"))

        b32_capture_leg = locals().get("canonical_capture_leg", locals().get("capture_leg", None))
        b32_capture_count = locals().get("canonical_capture_count", locals().get("capture_count", 0))

        # If requested, let the direct probe override B30 only while it is active/starting.
        # B32 is called after B30, so its joint target is the last writer for the probed leg.
        b32_probe_info = _b32_update_direct_joint_response_probe(
            robot,
            foot_pos,
            b29_active_swing_names,
            b32_capture_leg,
            b32_capture_count,
            step,
        )
        if (args.b32_print or args.b42_print) and b32_probe_info.get("event") in ("start", "complete", "holding"):
            _b32_label = "[B42 LH TASKSPACE AUTHORITY PROBE]" if args.enable_b42_lh_taskspace_authority_probe else "[B32 DIRECT JOINT RESPONSE PROBE]"
            print(_b32_label,
                  "event=", b32_probe_info.get("event"),
                  "seq=", b32_probe_info.get("seq"),
                  "step=", step,
                  "leg=", b32_probe_info.get("leg"),
                  "axis=", b32_probe_info.get("axis"),
                  "sign=", b32_probe_info.get("sign"),
                  "start_step=", b32_probe_info.get("start_step"),
                  "end_step=", b32_probe_info.get("end_step"),
                  "age=", b32_probe_info.get("age"),
                  "phase=", b32_probe_info.get("phase"),
                  "reason=", b32_probe_info.get("reason"),
                  "q_start=", b32_probe_info.get("q_start"),
                  "q_cmd=", b32_probe_info.get("q_cmd"),
                  "dq_cmd=", b32_probe_info.get("dq_cmd"),
                  "foot_start=", b32_probe_info.get("foot_start"),
                  "foot_now=", b32_probe_info.get("foot_now"),
                  "actual_delta=", b32_probe_info.get("actual_delta"),
                  "actual_norm=", b32_probe_info.get("actual_norm"),
                  "delta_per_rad=", b32_probe_info.get("delta_per_rad"))

        b33_capture_leg = locals().get("canonical_capture_leg", locals().get("capture_leg", None))
        b33_capture_count = locals().get("canonical_capture_count", locals().get("capture_count", 0))

        # B33 is called after B32; if both are enabled, B33 is the final writer.
        # Recommended: disable B32 when running B33.
        b33_probe_info = _b33_update_baseline_cancelled_response_probe(
            robot,
            foot_pos,
            b29_active_swing_names,
            b33_capture_leg,
            b33_capture_count,
            step,
        )
        if args.b33_print and b33_probe_info.get("event") in ("start_baseline", "baseline_complete", "complete", "holding"):
            print("[B33 BASELINE CANCELLED RESPONSE PROBE]",
                  "event=", b33_probe_info.get("event"),
                  "seq=", b33_probe_info.get("seq"),
                  "step=", step,
                  "leg=", b33_probe_info.get("leg"),
                  "axis=", b33_probe_info.get("axis"),
                  "sign=", b33_probe_info.get("sign"),
                  "start_step=", b33_probe_info.get("start_step"),
                  "end_step=", b33_probe_info.get("end_step"),
                  "age=", b33_probe_info.get("age"),
                  "phase=", b33_probe_info.get("phase"),
                  "reason=", b33_probe_info.get("reason"),
                  "q_start=", b33_probe_info.get("q_start"),
                  "q_cmd=", b33_probe_info.get("q_cmd"),
                  "dq_cmd=", b33_probe_info.get("dq_cmd"),
                  "baseline_delta=", b33_probe_info.get("baseline_delta"),
                  "perturb_delta=", b33_probe_info.get("perturb_delta"),
                  "corrected_delta=", b33_probe_info.get("corrected_delta"),
                  "corrected_norm=", b33_probe_info.get("corrected_norm"),
                  "corrected_delta_per_rad=", b33_probe_info.get("corrected_delta_per_rad"),
                  "foot_start=", b33_probe_info.get("foot_start"),
                  "foot_now=", b33_probe_info.get("foot_now"))

        b34_capture_leg = locals().get("canonical_capture_leg", locals().get("capture_leg", None))
        b34_capture_count = locals().get("canonical_capture_count", locals().get("capture_count", 0))
        b34_post_capture_pin_count = locals().get("post_capture_pin_count", 0)
        try:
            _b34_leg_for_post = LEG_TO_ID.get(str(args.b34_leg).upper(), 0)
            b34_post_capture_pin_count = max(int(b34_post_capture_pin_count), int(b18_pin_extra_count_by_leg[_b34_leg_for_post]))
        except Exception:
            pass

        # B34 should normally be run with B30/B31/B32/B33 probes disabled.
        # It is placed late so it becomes the final joint-target writer for the selected leg.
        b34_bridge_info = _b34_apply_empirical_rf_x_bridge(
            robot,
            foot_pos,
            b23_footref_info,
            b18_pin_info,
            b29_active_swing_names,
            b34_capture_leg,
            b34_capture_count,
            b34_post_capture_pin_count,
            step,
        )
        if args.b34_print and (b34_bridge_info.get("active", False) or step % max(args.print_every, 1) == 0):
            print("[B34 EMPIRICAL RF X BRIDGE]",
                  "step=", step,
                  "active=", b34_bridge_info.get("active", False),
                  "reason=", b34_bridge_info.get("reason"),
                  "source=", b34_bridge_info.get("source"),
                  "phase=", b34_bridge_info.get("phase"),
                  "leg=", b34_bridge_info.get("leg"),
                  "latched_new=", b34_bridge_info.get("latched_new"),
                  "hold_age=", b34_bridge_info.get("hold_age"),
                  "target_w=", b34_bridge_info.get("target_w"),
                  "foot_w=", b34_bridge_info.get("foot_w"),
                  "foot_err=", b34_bridge_info.get("foot_err"),
                  "e_x_raw=", b34_bridge_info.get("e_x_raw"),
                  "e_x_used=", b34_bridge_info.get("e_x_used"),
                  "dq_des=", b34_bridge_info.get("dq_des"),
                  "q_now=", b34_bridge_info.get("q_now"),
                  "q_latch=", b34_bridge_info.get("q_latch"),
                  "q_goal=", b34_bridge_info.get("q_goal"),
                  "q_cmd=", b34_bridge_info.get("q_cmd"),
                  "delta_from_now=", b34_bridge_info.get("delta_from_now"),
                  "delta_from_latch=", b34_bridge_info.get("delta_from_latch"))

        b35_capture_leg = locals().get("canonical_capture_leg", locals().get("capture_leg", None))
        b35_capture_count = locals().get("canonical_capture_count", locals().get("capture_count", 0))
        b35_post_capture_pin_count = locals().get("post_capture_pin_count", 0)
        try:
            _b35_leg_for_post = LEG_TO_ID.get(str(args.b35_leg).upper(), 0)
            b35_post_capture_pin_count = max(int(b35_post_capture_pin_count), int(b18_pin_extra_count_by_leg[_b35_leg_for_post]))
        except Exception:
            pass

        # B35 is intentionally placed after B34 and all probes. Recommended:
        # disable B34 and enable B35 only.
        b35_assist_info = _b35_apply_pure_empirical_hy_assist(
            robot,
            foot_pos,
            b23_footref_info,
            b18_pin_info,
            b29_active_swing_names,
            b35_capture_leg,
            b35_capture_count,
            b35_post_capture_pin_count,
            step,
        )
        if args.b35_print and (b35_assist_info.get("active", False) or step % max(args.print_every, 1) == 0):
            print("[B35 PURE EMPIRICAL HY ASSIST]",
                  "step=", step,
                  "active=", b35_assist_info.get("active", False),
                  "reason=", b35_assist_info.get("reason"),
                  "source=", b35_assist_info.get("source"),
                  "phase=", b35_assist_info.get("phase"),
                  "leg=", b35_assist_info.get("leg"),
                  "sigma=", b35_assist_info.get("sigma"),
                  "target_w=", b35_assist_info.get("target_w"),
                  "foot_w=", b35_assist_info.get("foot_w"),
                  "foot_err=", b35_assist_info.get("foot_err"),
                  "e_x_raw=", b35_assist_info.get("e_x_raw"),
                  "e_x_used=", b35_assist_info.get("e_x_used"),
                  "dq_hy=", b35_assist_info.get("dq_hy"),
                  "q_now=", b35_assist_info.get("q_now"),
                  "q_cmd=", b35_assist_info.get("q_cmd"),
                  "delta_from_now=", b35_assist_info.get("delta_from_now"))

        b36_capture_leg = locals().get("canonical_capture_leg", locals().get("capture_leg", None))
        b36_capture_count = locals().get("canonical_capture_count", locals().get("capture_count", 0))
        b36_post_capture_pin_count = locals().get("post_capture_pin_count", 0)
        try:
            _b36_leg_for_post = LEG_TO_ID.get(str(args.b36_leg).upper(), 0)
            b36_post_capture_pin_count = max(int(b36_post_capture_pin_count), int(b18_pin_extra_count_by_leg[_b36_leg_for_post]))
        except Exception:
            pass

        b36_assist_info = _b36_apply_pure_empirical_hy_kn_assist(
            robot, foot_pos, b23_footref_info, b18_pin_info,
            b29_active_swing_names, b36_capture_leg, b36_capture_count,
            b36_post_capture_pin_count, step)
        if args.b36_print and (b36_assist_info.get("active", False) or step % max(args.print_every, 1) == 0):
            print("[B36 PURE EMPIRICAL HY KN ASSIST]",
                  "step=", step,
                  "active=", b36_assist_info.get("active", False),
                  "reason=", b36_assist_info.get("reason"),
                  "source=", b36_assist_info.get("source"),
                  "phase=", b36_assist_info.get("phase"),
                  "leg=", b36_assist_info.get("leg"),
                  "sigma=", b36_assist_info.get("sigma"),
                  "target_w=", b36_assist_info.get("target_w"),
                  "foot_w=", b36_assist_info.get("foot_w"),
                  "foot_err=", b36_assist_info.get("foot_err"),
                  "e_x_raw=", b36_assist_info.get("e_x_raw"),
                  "e_x_used=", b36_assist_info.get("e_x_used"),
                  "dq_hy=", b36_assist_info.get("dq_hy"),
                  "dq_kn=", b36_assist_info.get("dq_kn"),
                  "q_now=", b36_assist_info.get("q_now"),
                  "q_cmd=", b36_assist_info.get("q_cmd"),
                  "delta_from_now=", b36_assist_info.get("delta_from_now"))

        b37_capture_leg = locals().get("canonical_capture_leg", locals().get("capture_leg", None))
        b37_capture_count = locals().get("canonical_capture_count", locals().get("capture_count", 0))
        b37_post_capture_pin_count = locals().get("post_capture_pin_count", 0)
        try:
            for _lid in range(4):
                b37_post_capture_pin_count = max(int(b37_post_capture_pin_count), int(b18_pin_extra_count_by_leg[_lid]))
        except Exception:
            pass

        # B37 should normally be run alone among B34/B35/B36-style bridges.
        # It is placed last, so it becomes the final joint-target writer.
        b37_assist_info = _b37_apply_all_leg_empirical_hy_kn_assist(
            robot,
            foot_pos,
            b23_footref_info,
            b18_pin_info,
            b29_active_swing_names,
            b37_capture_leg,
            b37_capture_count,
            b37_post_capture_pin_count,
            step,
        )
        if args.b37_print and (b37_assist_info.get("active", False) or step % max(args.print_every, 1) == 0):
            print("[B37 ALL LEG EMPIRICAL HY KN ASSIST]",
                  "step=", step,
                  "active=", b37_assist_info.get("active", False),
                  "reason=", b37_assist_info.get("reason"),
                  "source=", b37_assist_info.get("source"),
                  "phase=", b37_assist_info.get("phase"),
                  "leg=", b37_assist_info.get("leg"),
                  "allowed_legs=", b37_assist_info.get("allowed_legs"),
                  "sigma=", b37_assist_info.get("sigma"),
                  "target_w=", b37_assist_info.get("target_w"),
                  "foot_w=", b37_assist_info.get("foot_w"),
                  "foot_err=", b37_assist_info.get("foot_err"),
                  "e_x_raw=", b37_assist_info.get("e_x_raw"),
                  "e_x_used=", b37_assist_info.get("e_x_used"),
                  "hy_sign=", b37_assist_info.get("hy_sign"),
                  "kn_sign=", b37_assist_info.get("kn_sign"),
                  "dq_hy=", b37_assist_info.get("dq_hy"),
                  "dq_kn=", b37_assist_info.get("dq_kn"),
                  "q_now=", b37_assist_info.get("q_now"),
                  "q_cmd=", b37_assist_info.get("q_cmd"),
                  "delta_from_now=", b37_assist_info.get("delta_from_now"),
                  "step_foot_delta=", b37_assist_info.get("step_foot_delta"),
                  "step_foot_delta_norm=", b37_assist_info.get("step_foot_delta_norm"),
                  "active_counts=", b37_assist_info.get("active_counts"),
                  "max_step_delta=", b37_assist_info.get("max_step_delta"),
                  "max_err_x=", b37_assist_info.get("max_err_x"))

        b43_capture_leg = locals().get("canonical_capture_leg", locals().get("capture_leg", None))
        b43_capture_count = locals().get("canonical_capture_count", locals().get("capture_count", 0))
        b43_post_capture_pin_count = locals().get("post_capture_pin_count", 0)
        try:
            for _lid in range(4):
                b43_post_capture_pin_count = max(int(b43_post_capture_pin_count), int(b18_pin_extra_count_by_leg[_lid]))
        except Exception:
            pass

        # B43 should run after B37 so the empirical lateral-y bridge is the final
        # joint-target writer for LH/RH when enabled.
        b43_lateral_info = _b43_apply_hind_empirical_lateral_y_bridge(
            robot,
            foot_pos,
            b23_footref_info,
            b18_pin_info,
            b29_active_swing_names,
            b43_capture_leg,
            b43_capture_count,
            b43_post_capture_pin_count,
            step,
        )
        if args.b43_print and (b43_lateral_info.get("active", False) or step % max(args.print_every, 1) == 0):
            print("[B43 HIND EMPIRICAL LATERAL-Y BRIDGE]",
                  "step=", step,
                  "active=", b43_lateral_info.get("active", False),
                  "reason=", b43_lateral_info.get("reason"),
                  "source=", b43_lateral_info.get("source"),
                  "phase=", b43_lateral_info.get("phase"),
                  "leg=", b43_lateral_info.get("leg"),
                  "allowed_legs=", b43_lateral_info.get("allowed_legs"),
                  "sigma=", b43_lateral_info.get("sigma"),
                  "target_w=", b43_lateral_info.get("target_w"),
                  "foot_w=", b43_lateral_info.get("foot_w"),
                  "foot_err=", b43_lateral_info.get("foot_err"),
                  "e_y_raw=", b43_lateral_info.get("e_y_raw"),
                  "e_y_used=", b43_lateral_info.get("e_y_used"),
                  "hx_sign=", b43_lateral_info.get("hx_sign"),
                  "hy_sign=", b43_lateral_info.get("hy_sign"),
                  "dq_hx=", b43_lateral_info.get("dq_hx"),
                  "dq_hy=", b43_lateral_info.get("dq_hy"),
                  "q_now=", b43_lateral_info.get("q_now"),
                  "q_cmd=", b43_lateral_info.get("q_cmd"),
                  "delta_from_now=", b43_lateral_info.get("delta_from_now"),
                  "step_foot_delta=", b43_lateral_info.get("step_foot_delta"),
                  "step_foot_delta_norm=", b43_lateral_info.get("step_foot_delta_norm"),
                  "active_counts=", b43_lateral_info.get("active_counts"),
                  "max_step_delta=", b43_lateral_info.get("max_step_delta"),
                  "max_err_y=", b43_lateral_info.get("max_err_y"))


        # B44 runs after B43. If both are enabled, B44 is the final hind-leg
        # joint-target writer and therefore tests persistent latched authority.
        b44_latched_info = _b44_apply_latched_hind_y_bridge(
            robot,
            foot_pos,
            b23_footref_info,
            b18_pin_info,
            b29_active_swing_names,
            b43_capture_leg,
            b43_capture_count,
            b43_post_capture_pin_count,
            step,
        )
        if args.b44_print and (b44_latched_info.get("active", False) or step % max(args.print_every, 1) == 0):
            print("[B44 LATCHED HIND Y BRIDGE]",
                  "step=", step,
                  "active=", b44_latched_info.get("active"),
                  "reason=", b44_latched_info.get("reason"),
                  "source=", b44_latched_info.get("source"),
                  "phase=", b44_latched_info.get("phase"),
                  "leg=", b44_latched_info.get("leg"),
                  "allowed_legs=", b44_latched_info.get("allowed_legs"),
                  "sigma=", b44_latched_info.get("sigma"),
                  "target_w=", b44_latched_info.get("target_w"),
                  "foot_w=", b44_latched_info.get("foot_w"),
                  "foot_err=", b44_latched_info.get("foot_err"),
                  "e_y_raw=", b44_latched_info.get("e_y_raw"),
                  "e_y_used=", b44_latched_info.get("e_y_used"),
                  "hx_sign=", b44_latched_info.get("hx_sign"),
                  "hy_sign=", b44_latched_info.get("hy_sign"),
                  "latched_now=", b44_latched_info.get("latched_now"),
                  "q_latch=", b44_latched_info.get("q_latch"),
                  "q_goal=", b44_latched_info.get("q_goal"),
                  "q_now=", b44_latched_info.get("q_now"),
                  "q_cmd=", b44_latched_info.get("q_cmd"),
                  "delta_goal_from_latch=", b44_latched_info.get("delta_goal_from_latch"),
                  "delta_goal_from_now=", b44_latched_info.get("delta_goal_from_now"),
                  "delta_cmd_from_now=", b44_latched_info.get("delta_cmd_from_now"),
                  "dq_goal_hx=", b44_latched_info.get("dq_goal_hx"),
                  "dq_goal_hy=", b44_latched_info.get("dq_goal_hy"),
                  "dq_cmd_hx=", b44_latched_info.get("dq_cmd_hx"),
                  "dq_cmd_hy=", b44_latched_info.get("dq_cmd_hy"),
                  "step_foot_delta=", b44_latched_info.get("step_foot_delta"),
                  "step_foot_delta_norm=", b44_latched_info.get("step_foot_delta_norm"),
                  "active_counts=", b44_latched_info.get("active_counts"),
                  "latch_counts=", b44_latched_info.get("latch_counts"),
                  "max_step_delta=", b44_latched_info.get("max_step_delta"),
                  "max_err_y=", b44_latched_info.get("max_err_y"))

        # B45 runs after B44. In the recommended run B44 is disabled and B45 is
        # the final hind-leg y bridge. If both are enabled, B45 intentionally
        # overrides B44's q_latch-restoring command with a direction-only push.
        b45_push_info = _b45_apply_directional_hind_y_push_bridge(
            robot,
            foot_pos,
            b23_footref_info,
            b18_pin_info,
            b29_active_swing_names,
            b43_capture_leg,
            b43_capture_count,
            b43_post_capture_pin_count,
            step,
        )
        if args.b45_print and (b45_push_info.get("active", False) or step % max(args.print_every, 1) == 0):
            print("[B45 DIRECTIONAL HIND Y PUSH]",
                  "step=", step,
                  "active=", b45_push_info.get("active"),
                  "reason=", b45_push_info.get("reason"),
                  "source=", b45_push_info.get("source"),
                  "phase=", b45_push_info.get("phase"),
                  "leg=", b45_push_info.get("leg"),
                  "allowed_legs=", b45_push_info.get("allowed_legs"),
                  "sigma=", b45_push_info.get("sigma"),
                  "target_w=", b45_push_info.get("target_w"),
                  "foot_w=", b45_push_info.get("foot_w"),
                  "foot_err=", b45_push_info.get("foot_err"),
                  "e_y_raw=", b45_push_info.get("e_y_raw"),
                  "e_y_used=", b45_push_info.get("e_y_used"),
                  "hx_sign=", b45_push_info.get("hx_sign"),
                  "hy_sign=", b45_push_info.get("hy_sign"),
                  "latched_now=", b45_push_info.get("latched_now"),
                  "q_latch=", b45_push_info.get("q_latch"),
                  "q_now=", b45_push_info.get("q_now"),
                  "q_cmd=", b45_push_info.get("q_cmd"),
                  "raw_step_hx=", b45_push_info.get("raw_step_hx"),
                  "raw_step_hy=", b45_push_info.get("raw_step_hy"),
                  "cmd_step_hx=", b45_push_info.get("cmd_step_hx"),
                  "cmd_step_hy=", b45_push_info.get("cmd_step_hy"),
                  "delta_now_from_latch=", b45_push_info.get("delta_now_from_latch"),
                  "delta_cmd_from_now=", b45_push_info.get("delta_cmd_from_now"),
                  "blocked_hx=", b45_push_info.get("blocked_hx"),
                  "blocked_hy=", b45_push_info.get("blocked_hy"),
                  "step_foot_delta=", b45_push_info.get("step_foot_delta"),
                  "step_foot_delta_norm=", b45_push_info.get("step_foot_delta_norm"),
                  "active_counts=", b45_push_info.get("active_counts"),
                  "blocked_counts=", b45_push_info.get("blocked_counts"),
                  "max_step_delta=", b45_push_info.get("max_step_delta"),
                  "max_err_y=", b45_push_info.get("max_err_y"))

        if args.b11_reloc_print and b11_reloc_info.get("active", False) and (step % max(args.print_every, 1) == 0):
            print("[B38 WBC RELOCATION]",
                  "step=", step,
                  "candidate=", ID_TO_LEG.get(b11_reloc_info.get("candidate"), b11_reloc_info.get("candidate")),
                  "base_ref_xy_err=", b11_reloc_info.get("base_ref_xy_err"),
                  "kp_xy=", wbc_cfg_active.kp_base_xy,
                  "kd_xy=", wbc_cfg_active.kd_base_xy,
                  "w_base_acc=", wbc_cfg_active.w_base_acc,
                  "w_stance_acc=", wbc_cfg_active.w_stance_acc,
                  "max_base_acc_lin=", wbc_cfg_active.max_base_acc_lin)
        if (args.b12_align_print or args.b16_recenter_align_print) and b12_align_info.get("active", False) and (step % max(args.print_every, 1) == 0):
            align_label = "[B25 RECENTER PD ALIGN]" if (isinstance(canonical_shift_info, dict) and canonical_shift_info.get("b15_recenter", False)) else "[B38 PD TARGET ALIGN]"
            print(align_label,
                  "step=", step,
                  "candidate=", ID_TO_LEG.get(b12_align_info.get("candidate"), b12_align_info.get("candidate")),
                  "base_ref_xy_err=", b12_align_info.get("base_ref_xy_err"),
                  "foot_cmd_xy=", b12_align_info.get("foot_cmd_xy"),
                  "max_delta=", b12_align_info.get("max_delta"),
                  "ramp=", b12_align_info.get("ramp"),
                  "reason=", b12_align_info.get("reason"))

        tau_cmd, qpd = solve_full_wbc_qp_v1(
            M_full=M,
            Jfeet_full=Jfeet_full,
            f_mpc=f_mpc,
            q=robot.data.joint_pos,
            qd=robot.data.joint_vel,
            q_nom=q_nom_active,
            x_hat=x_hat,
            base_ref=base_ref,
            foot_pos_w=foot_pos,
            foot_vel_w=foot_vel,
            swing_target_pos_w=swing_target,
            stance_mask=contact_mask,
            gravity_forces=gravity,
            coriolis_forces=coriolis,
            cfg=wbc_cfg_active,
        )

        tau_cmd = args.tau_cmd_scale * tau_cmd
        tau_wbc_component_b22 = tau_cmd.detach().clone()
        if isinstance(canonical_shift_info, dict) and canonical_shift_info.get("b15_recenter", False):
            b12_tau_assist = _b16_recenter_alignment_torque_assist(robot, q_align_target, b12_align_info)
        else:
            b12_tau_assist = _b12_alignment_torque_assist(robot, q_align_target, b12_align_info)
        if b12_align_info.get("active", False):
            tau_cmd = tau_cmd + b12_tau_assist
            tau_cmd = torch.clamp(tau_cmd, -float(args.max_tau), float(args.max_tau))
        if isinstance(b18_pin_info, dict) and b18_pin_info.get("active", False) and b18_pin_target is not None:
            tau_cmd = tau_cmd + _b18_foothold_pin_torque_assist(robot, b18_pin_target, b18_pin_info)
            tau_cmd = torch.clamp(tau_cmd, -float(args.max_tau), float(args.max_tau))
        if isinstance(b20_direct_ik_info, dict) and b20_direct_ik_info.get("active", False) and b20_direct_ik_target is not None:
            tau_cmd = tau_cmd + _b20_direct_swing_ik_torque_assist(robot, b20_direct_ik_target, b20_direct_ik_info)
            tau_cmd = torch.clamp(tau_cmd, -float(args.max_tau), float(args.max_tau))
        b21_cart_info = {"active": False}
        try:
            if args.enable_b21_cartesian_committed_swing_torque:
                b21_tau_assist, b21_cart_info = _b21_cartesian_committed_swing_torque_assist(
                    robot, Jfeet_full, foot_pos, foot_vel,
                    canonical_s_t, canonical_sigma_t,
                    canonical_capture_leg, canonical_capture_count,
                    b18_pin_extra_count_by_leg,
                    (b49_tracking_targets_w if bool(getattr(args, "b49_apply_to_b21", False)) and 'b49_tracking_targets_w' in locals() else b18_committed_foothold_w),
                    b18_committed_valid
                )
                if b21_cart_info.get("active", False):
                    tau_cmd = tau_cmd + b21_tau_assist
                    tau_cmd = torch.clamp(tau_cmd, -float(args.max_tau), float(args.max_tau))
                    if args.b21_cart_print and (step % max(args.print_every, 1) == 0):
                        print("[B22 CARTESIAN COMMITTED TORQUE]",
                              "step=", step,
                              "leg=", b21_cart_info.get("leg"),
                              "reason=", b21_cart_info.get("reason"),
                              "sigma=", b21_cart_info.get("sigma"),
                              "foot_err=", b21_cart_info.get("foot_err"),
                              "foot_vel=", b21_cart_info.get("foot_vel"),
                              "f_cmd=", b21_cart_info.get("f_cmd"),
                              "tau_leg=", b21_cart_info.get("tau_leg"),
                              "max_tau=", b21_cart_info.get("max_tau"))
                elif args.b21_cart_print and (step % max(args.print_every, 1) == 0):
                    print("[B21 CARTESIAN COMMITTED TORQUE HOLD]",
                          "step=", step,
                          "reason=", b21_cart_info.get("reason"))
        except Exception as _e:
            if args.b21_cart_print and (step % max(args.print_every, 1) == 0):
                print("[B21 CARTESIAN COMMITTED TORQUE ERROR]", "step=", step, "err=", repr(_e))
        tau_pre_b22 = tau_cmd.detach().clone()
        b22_authority_info = {"active": False}
        if args.enable_b22_wbc_authority_calibration:
            tau_cmd, b22_authority_info = _b22_apply_authority_mode(tau_wbc_component_b22, tau_pre_b22, step)
            _b22_print_authority(step, tau_wbc_component_b22, tau_pre_b22, tau_cmd, b22_authority_info, robot)

        crouch_target, crouch_delta_full, crouch_ramp = apply_crouch_selected_joint_targets(robot, q_initial, step)

        canonical_ik_leg_override = None
        if args.ik_use_canonical_active_leg and args.enable_canonical_gait_schedule:
            if canonical_latched_swing_leg is not None:
                canonical_ik_leg_override = int(canonical_latched_swing_leg)
            elif canonical_current_swing_leg is not None:
                canonical_ik_leg_override = int(canonical_current_swing_leg)
            elif canonical_s_t is not None and bool((canonical_s_t[0] < 0.5).any().detach().cpu()):
                canonical_ik_leg_override = int(torch.nonzero(canonical_s_t[0] < 0.5, as_tuple=False).flatten()[0].detach().cpu())

        ik_target, ik_delta_full, ik_info = apply_jacobian_ik_swing_target(
            robot, Jfeet_full, foot_pos, swing_target, phase, profile, swing_enabled,
            active_leg_override=canonical_ik_leg_override
        )

        canonical_commit_target, canonical_commit_delta_full, canonical_commit_info = apply_canonical_committed_foothold_pin(
            robot, Jfeet_full, foot_pos, canonical_committed_target_w, canonical_committed_leg, canonical_commit_count
        )
        if canonical_commit_info.get("active", False):
            if args.canonical_commit_print and (step % max(args.print_every, 1) == 0):
                print("[B9-i COMMIT PIN]",
                      "step=", step,
                      "leg=", ID_TO_LEG.get(canonical_committed_leg, None),
                      "count=", canonical_commit_count,
                      "foot_err=", canonical_commit_info["foot_err"][0].detach().cpu().numpy(),
                      "dq=", canonical_commit_info["dq_cmd"][0].detach().cpu().numpy())
            canonical_commit_count -= 1
            if canonical_commit_count <= 0:
                if args.canonical_commit_print:
                    print("[B9-i COMMIT END] step=", step, "leg=", ID_TO_LEG.get(canonical_committed_leg, None))
                if args.canonical_enable_event_crawl_queue and canonical_committed_leg is not None:
                    expected_leg = int(canonical_event_queue_ids[int(canonical_event_queue_index) % len(canonical_event_queue_ids)])
                    # In B14 the executed/captured leg may be adaptively selected without queue mutation.
                    if args.enable_b14_no_queue_mutation:
                        expected_leg = int(canonical_committed_leg)
                    if int(canonical_committed_leg) == expected_leg:
                        canonical_event_last_completed_leg = int(canonical_committed_leg)
                        canonical_event_completed_step_by_leg[int(canonical_committed_leg)] = int(step)
                        canonical_event_queue_index = (int(canonical_event_queue_index) + 1) % len(canonical_event_queue_ids)
                        if args.canonical_event_print:
                            print("[B37 QUEUE ADVANCE] step=", step,
                                  "completed=", ID_TO_LEG.get(canonical_committed_leg, None),
                                  "next=", ID_TO_LEG[canonical_event_queue_ids[canonical_event_queue_index]])
                    else:
                        if args.canonical_event_print:
                            print("[B25 QUEUE HOLD] step=", step,
                                  "committed=", ID_TO_LEG.get(canonical_committed_leg, None),
                                  "expected=", ID_TO_LEG.get(expected_leg, None))
                canonical_committed_leg = None
                canonical_committed_target_w = None
                canonical_commit_count = 0

        canonical_capture_target, canonical_capture_delta_full, canonical_capture_info = apply_canonical_touchdown_capture_q_hold(
            robot, canonical_capture_leg, canonical_capture_q_ref, canonical_capture_count
        )
        if canonical_capture_info.get("active", False):
            if args.canonical_capture_print and (step % max(args.print_every, 1) == 0):
                print("[B37 CAPTURE HOLD]",
                      "step=", step,
                      "leg=", ID_TO_LEG.get(canonical_capture_leg, None),
                      "count=", canonical_capture_count,
                      "target_minus_q=", canonical_capture_info["target_minus_q"][0].detach().cpu().numpy())
            canonical_capture_count -= 1
            if canonical_capture_count <= 0:
                if args.canonical_capture_print:
                    print("[B37 CAPTURE END] step=", step, "leg=", ID_TO_LEG.get(canonical_capture_leg, None))
                    if args.enable_b18_committed_foothold_pinning and canonical_capture_leg is not None:
                        try:
                            _leg_tmp = int(canonical_capture_leg)
                            if bool(b18_committed_valid[_leg_tmp]):
                                b18_pin_extra_count_by_leg[_leg_tmp] = int(args.b18_pin_extra_after_capture_steps)
                                if args.b18_print:
                                    _err_tmp = (b18_committed_foothold_w[_leg_tmp, :] - foot_pos[0, _leg_tmp, :]).detach().cpu().numpy()
                                    print("[B37 POST-CAPTURE PIN START]",
                                          "step=", step,
                                          "leg=", ID_TO_LEG.get(_leg_tmp, _leg_tmp),
                                          "foot_minus_committed=", -_err_tmp,
                                          "remaining=", b18_pin_extra_count_by_leg[_leg_tmp])
                                _b18_print_post_capture_geometry(step, x_hat, foot_pos, ref_cfg, device, dtype, label="capture_end")
                        except Exception as _e:
                            if args.b18_print:
                                print("[B18 POST-CAPTURE PIN ERROR]", "step=", step, "err=", repr(_e))
                if args.canonical_enable_event_crawl_queue and canonical_capture_leg is not None:
                    expected_leg = int(canonical_event_queue_ids[int(canonical_event_queue_index) % len(canonical_event_queue_ids)])
                    # In B14 the executed/captured leg may be adaptively selected without queue mutation.
                    if args.enable_b14_no_queue_mutation:
                        expected_leg = int(canonical_capture_leg)
                    if int(canonical_capture_leg) == expected_leg:
                        canonical_event_last_completed_leg = int(canonical_capture_leg)
                        canonical_event_completed_step_by_leg[int(canonical_capture_leg)] = int(step)
                        canonical_event_queue_index = (int(canonical_event_queue_index) + 1) % len(canonical_event_queue_ids)
                        if args.canonical_event_print:
                            print("[B37 QUEUE ADVANCE] step=", step,
                                  "completed=", ID_TO_LEG.get(canonical_capture_leg, None),
                                  "next=", ID_TO_LEG[canonical_event_queue_ids[canonical_event_queue_index]])
                canonical_capture_leg = None
                canonical_capture_q_ref = None
                canonical_capture_count = 0
                canonical_stable_count = 0

        pre_touchdown_lock_target, pre_touchdown_lock_delta_full, pre_touchdown_lock_info = apply_foothold_lock_recovery_target(
            robot, Jfeet_full, foot_pos, LEG_TO_ID[args.test_leg], 
            swing_latched_foothold_target_w if (swing_latched and swing_latched_foothold_target_w is not None) else foothold_target_w,
            bool((args.late_touchdown_force_foothold_lock and late_touchdown_hold_swing)
                 or (args.latched_swing_force_ik and swing_latched and phase in ["lift", "hold_lift", "lower"]))
        )

        post_touchdown_lock_target = None
        post_touchdown_lock_delta_full = None
        post_touchdown_lock_info = None
        if post_touchdown_lock_count > 0 and post_touchdown_lock_leg is not None:
            post_touchdown_lock_target, post_touchdown_lock_delta_full, post_touchdown_lock_info = apply_foothold_lock_recovery_target(
                robot, Jfeet_full, foot_pos, int(post_touchdown_lock_leg), 
                touchdown_committed_foothold_target_w if touchdown_committed_foothold_target_w is not None else post_touchdown_lock_foothold_target_w,
                True
            )
            post_touchdown_lock_count -= 1

        committed_pin_target = None
        committed_pin_delta_full = None
        committed_pin_info = None
        committed_pin_active = bool(
            args.enable_committed_foothold_pinning
            and touchdown_committed
            and touchdown_committed_leg is not None
            and touchdown_committed_foothold_target_w is not None
            and (args.committed_pin_until_next_step or committed_pin_count > 0)
        )
        if committed_pin_active:
            committed_pin_target, committed_pin_delta_full, committed_pin_info = apply_committed_foothold_pin_target(
                robot, Jfeet_full, foot_pos, int(touchdown_committed_leg), touchdown_committed_foothold_target_w, True
            )
            if not args.committed_pin_until_next_step:
                committed_pin_count = max(0, committed_pin_count - 1)

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
        if args.enable_canonical_gait_schedule and args.canonical_gait_disable_event_overrides:
            committed_pin_target = None
            post_touchdown_lock_target = None
            pre_touchdown_lock_target = None
            foothold_lock_target = None
            touchdown_recovery_target = None
            ik_target = None
            crouch_target = None
            q_target = None

        q_target = (
            committed_pin_target
            if committed_pin_target is not None
            else (
                post_touchdown_lock_target
                if post_touchdown_lock_target is not None
                else (
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
            )
        )

        combined_delta_full = (
            crouch_delta_full
            + ik_delta_full
            + (pre_touchdown_lock_delta_full if pre_touchdown_lock_delta_full is not None else torch.zeros_like(crouch_delta_full))
            + (post_touchdown_lock_delta_full if post_touchdown_lock_delta_full is not None else torch.zeros_like(crouch_delta_full))
            + (committed_pin_delta_full if committed_pin_delta_full is not None else torch.zeros_like(crouch_delta_full))
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
                swing_latched = False
                swing_latched_leg = None
                swing_latched_foothold_target_w = None
                swing_latched_stance_q_ref = None
                swing_latched_started_step = -1
                post_touchdown_lock_count = 0
                post_touchdown_lock_leg = None
                post_touchdown_lock_foothold_target_w = None
                touchdown_committed = False
                touchdown_committed_leg = None
                touchdown_committed_step_key = None
                touchdown_committed_step = -1
                touchdown_committed_foothold_target_w = None
                committed_pin_count = 0
                committed_pin_target = None
                committed_pin_delta_full = None
                committed_pin_info = None
                recenter_safe_seen = False
                shift_gate_safe_count = 0
                lift_unlocked_by_shift_gate = {}
                q_initial = robot.data.joint_pos.detach().clone()
                q_nom = q_initial.clone()
                if args.enable_crouch_target and args.use_crouch_q_nom:
                    q_nom = build_crouch_target(q_initial, q_initial).detach().clone()

        if step % max(args.print_every, 1) == 0:
            if args.enable_canonical_gait_schedule and canonical_s_t is not None:
                _active_swing = torch.nonzero(canonical_s_t[0] < 0.5, as_tuple=False).flatten()
                _active_swing_names = [ID_TO_LEG[int(x.detach().cpu())] for x in _active_swing]
                print("[B38 canonical]",
                      "step=", step,
                      "start_step=", args.canonical_gait_start_step,
                      "active_swing=", _active_swing_names,
                      "phi=", canonical_phi_t[0].detach().cpu().numpy(),
                      "requested_s=", canonical_requested_s_t[0].detach().cpu().numpy() if canonical_requested_s_t is not None else None,
                      "executed_s=", canonical_s_t[0].detach().cpu().numpy(),
                      "sigma=", canonical_sigma_t[0].detach().cpu().numpy(),
                      "contact_mask=", contact_mask[0].detach().cpu().numpy(),
                      "gate_reason=", canonical_exec_info.get("gate_reason", None),
                      "candidate=", ID_TO_LEG.get(canonical_exec_info.get("candidate_leg"), None) if canonical_exec_info.get("candidate_leg") is not None else None,
                      "future_margin=", canonical_exec_info.get("future_margin", None),
                      "relaxed_ok=", canonical_exec_info.get("relaxed_margin_ok", None),
                      "gap_count=", canonical_exec_info.get("gap_count", None),
                      "shift_hold_count=", canonical_exec_info.get("shift_hold_count", None),
                      "latched_leg=", ID_TO_LEG.get(canonical_exec_info.get("latched_leg"), None) if canonical_exec_info.get("latched_leg", None) is not None else None,
                      "executed_sigma=", canonical_exec_info.get("executed_sigma", None),
                      "latched_elapsed=", canonical_exec_info.get("latched_elapsed", None),
                      "event=", canonical_event_info if args.canonical_enable_event_crawl_queue else None,
                      "capture_count=", int(canonical_capture_count),
                      "capture_leg=", ID_TO_LEG.get(canonical_capture_leg, None) if canonical_capture_leg is not None else None,
                      "shift_active=", canonical_shift_info.get("active", False),
                      "base_ref_xy_err=", canonical_exec_info.get("base_ref_xy_err", None),
                      "b9k_gate=", canonical_exec_info.get("b9k_support_gate_ok", None),
                      "b9k_geom=", canonical_exec_info.get("b9k_geom_margin_current", None),
                      "b9k_target_err=", canonical_exec_info.get("b9k_target_err", None),
                      "hold_due_to_error=", canonical_shift_info.get("hold_due_to_error", None),
                      "shift_next=", canonical_shift_info.get("next_xy", None)[0].detach().cpu().numpy() if canonical_shift_info.get("next_xy", None) is not None else None,
                      "shift_target=", canonical_shift_info.get("target_xy", None)[0].detach().cpu().numpy() if canonical_shift_info.get("target_xy", None) is not None else None)
            _debug_original_test_leg = args.test_leg
            if (
                args.enable_canonical_gait_schedule
                and args.canonical_debug_use_active_swing_leg
                and canonical_s_t is not None
                and bool((canonical_s_t < 0.5).any().detach().cpu())
            ):
                _swing_ids = torch.nonzero(canonical_s_t[0] < 0.5, as_tuple=False).flatten()
                if int(_swing_ids.numel()) > 0:
                    args.test_leg = ID_TO_LEG[int(_swing_ids[0].detach().cpu())]
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
                previous_load_bearing_valid,
                swing_latched, swing_latched_leg, swing_latched_started_step,
                post_touchdown_lock_count, post_touchdown_lock_info,
                touchdown_committed, touchdown_committed_leg,
                touchdown_committed_step_key, touchdown_committed_step,
                committed_pin_count, committed_pin_info,
                canonical_commit_info
            )
            args.test_leg = _debug_original_test_leg

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
