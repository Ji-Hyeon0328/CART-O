# isaaclab_carto/lowlevel/spot_swing_visual.py
#
# B7-a: visual swing gait bridge.
#
# Purpose:
#   Connect Ref.S / Ref.phase to actual visible leg motion.
#
# This is NOT a dynamic walking controller.
# It uses Isaac Lab JointPositionActionCfg and small relative joint offsets.
#
# Expected behavior:
#   - conservative: one leg swings at a time
#   - aggressive: diagonal legs swing together
#   - swing legs receive small hip/knee offsets
#   - stance legs receive zero action offset
#
# If the leg moves in the wrong direction, change lift_sign or knee_sign
# from the script arguments.

from __future__ import annotations

from typing import Dict, Tuple

import torch


LEG_NAMES = ["fl", "fr", "hl", "hr"]

# Native action order observed from Spot in Isaac:
# [fl_hx, fr_hx, hl_hx, hr_hx,
#  fl_hy, fr_hy, hl_hy, hr_hy,
#  fl_kn, fr_kn, hl_kn, hr_kn]
HX_IDX = torch.tensor([0, 1, 2, 3], dtype=torch.long)
HY_IDX = torch.tensor([4, 5, 6, 7], dtype=torch.long)
KN_IDX = torch.tensor([8, 9, 10, 11], dtype=torch.long)


def compute_swing_phase(
    ref: Dict[str, torch.Tensor],
    theta,
    k: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute swing mask and normalized swing progress.

    Args:
        ref["S"]: [N,4,H], 1 stance, 0 swing.
        ref["phase"]: [N,4,H]
        theta.gait["duty_i"]: [N,4]

    Returns:
        swing_mask: [N,4], 1 swing, 0 stance
        swing_progress: [N,4] in [0,1]
        lift_profile: [N,4] = sin(pi * swing_progress), zero on stance
    """
    S = ref["S"]
    H = S.shape[-1]
    k = min(max(k, 0), H - 1)

    phase = ref["phase"][:, :, k]
    duty = theta.gait["duty_i"]

    swing_mask = (S[:, :, k] < 0.5).to(S.dtype)

    denom = torch.clamp(1.0 - duty, min=1e-5)
    progress = torch.clamp((phase - duty) / denom, min=0.0, max=1.0)
    progress = progress * swing_mask

    lift = torch.sin(torch.pi * progress) * swing_mask

    return swing_mask, progress, lift


def make_swing_visual_joint_action(
    ref: Dict[str, torch.Tensor],
    theta,
    k: int = 0,
    hy_lift_delta: float = -0.08,
    kn_lift_delta: float = 0.18,
    hy_sweep_delta: float = 0.04,
    max_action_abs: float = 0.30,
    lift_sign: float = 1.0,
    knee_sign: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Build small relative joint-position action for swing visualization.

    This assumes JointPositionActionCfg where zero action corresponds to the
    nominal/default standing pose. Earlier A/B tests showed zero action stable.

    Args:
        hy_lift_delta:
            Hip pitch offset magnitude during swing.
        kn_lift_delta:
            Knee offset magnitude during swing.
        hy_sweep_delta:
            Small front/back sweep over swing progress.
        lift_sign:
            Flip hip-pitch lift direction if motion looks wrong.
        knee_sign:
            Flip knee lift direction if motion looks wrong.

    Returns:
        action: [N,12]
        info
    """
    device = ref["S"].device
    dtype = ref["S"].dtype
    N = ref["S"].shape[0]

    swing_mask, progress, lift = compute_swing_phase(ref, theta, k=k)

    action = torch.zeros((N, 12), device=device, dtype=dtype)

    hx_idx = HX_IDX.to(device=device)
    hy_idx = HY_IDX.to(device=device)
    kn_idx = KN_IDX.to(device=device)

    # Smooth sweep: back → front over swing phase.
    # This is intentionally small. The purpose is visible motion, not walking.
    sweep = (2.0 * progress - 1.0) * swing_mask

    hy_offset = lift_sign * hy_lift_delta * lift + hy_sweep_delta * sweep
    kn_offset = knee_sign * kn_lift_delta * lift

    action[:, hy_idx] = hy_offset
    action[:, kn_idx] = kn_offset
    # hx remains zero for now; avoid lateral destabilization.

    action = torch.clamp(action, -max_action_abs, max_action_abs)

    info = {
        "swing_mask_env0": swing_mask[0].detach().cpu().tolist(),
        "swing_progress_env0": progress[0].detach().cpu().tolist(),
        "lift_profile_env0": lift[0].detach().cpu().tolist(),
        "hy_offset_env0": hy_offset[0].detach().cpu().tolist(),
        "kn_offset_env0": kn_offset[0].detach().cpu().tolist(),
        "action_env0": action[0].detach().cpu().tolist(),
        "hy_lift_delta": hy_lift_delta,
        "kn_lift_delta": kn_lift_delta,
        "hy_sweep_delta": hy_sweep_delta,
        "lift_sign": lift_sign,
        "knee_sign": knee_sign,
        "max_action_abs": max_action_abs,
    }

    return action, info
