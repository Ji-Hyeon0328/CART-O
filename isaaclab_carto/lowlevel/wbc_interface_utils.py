# isaaclab_carto/lowlevel/wbc_interface_utils.py
#
# B8-a: WBC interface utilities.
#
# This does NOT solve WBC yet.
# It checks whether the ingredients for WBC are available:
#   q, dq, foot Jacobians, stance/swing masks, f_ref, tau=J^T f,
#   foot positions/velocities, and mass/gravity/coriolis if Isaac exposes them.

from __future__ import annotations

from typing import Dict, Tuple, Any

import torch

from isaaclab_carto.lowlevel.support_force_control import (
    extract_foot_jacobians_action_order,
    compute_tau_jtf,
)


FOOT_NAMES = ["fl_foot", "fr_foot", "hl_foot", "hr_foot"]


def get_root_physx_view(robot):
    if hasattr(robot, "root_physx_view"):
        return robot.root_physx_view
    if hasattr(robot, "_root_physx_view"):
        return robot._root_physx_view
    return None


def get_dynamics_terms(robot) -> Dict[str, object]:
    view = get_root_physx_view(robot)
    out: Dict[str, object] = {
        "has_root_physx_view": view is not None,
        "M": None,
        "gravity": None,
        "coriolis": None,
    }
    if view is None:
        return out

    for out_key, names in [
        ("M", ["get_mass_matrices", "get_generalized_mass_matrices", "get_mass_matrix"]),
        ("gravity", ["get_generalized_gravity_forces", "get_gravity_compensation_forces", "get_gravity_forces"]),
        ("coriolis", ["get_coriolis_and_centrifugal_forces", "get_generalized_coriolis_and_centrifugal_forces", "get_coriolis_forces"]),
    ]:
        for name in names:
            if hasattr(view, name):
                try:
                    out[out_key] = getattr(view, name)()
                    break
                except Exception:
                    pass
    return out


def get_body_indices(robot, body_names=FOOT_NAMES):
    name_to_idx = {name: i for i, name in enumerate(robot.body_names)}
    return [name_to_idx[name] for name in body_names]


def extract_ref_masks(ref: Dict[str, torch.Tensor], k: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    S = ref["S"]
    H = S.shape[-1]
    k = min(max(k, 0), H - 1)
    stance = (S[:, :, k] > 0.5).to(S.dtype)
    swing = 1.0 - stance
    return stance, swing


def summarize_tensor(name: str, x) -> Dict[str, object]:
    if x is None:
        return {f"{name}_available": False, f"{name}_shape": None}
    if not torch.is_tensor(x):
        try:
            x = torch.as_tensor(x)
        except Exception:
            return {f"{name}_available": True, f"{name}_shape": "non_tensor"}
    return {
        f"{name}_available": True,
        f"{name}_shape": list(x.shape),
        f"{name}_dtype": str(x.dtype),
        f"{name}_device": str(x.device),
        f"{name}_mean_abs": float(x.abs().mean().detach().cpu()),
        f"{name}_max_abs": float(x.abs().max().detach().cpu()),
    }


def build_wbc_interface_packet(
    robot,
    ref: Dict[str, torch.Tensor],
    f_ref: torch.Tensor,
    tau_stance: torch.Tensor,
    x_hat: torch.Tensor,
    k: int = 0,
    linear_rows: str = "0_3",
) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    qj = robot.data.joint_pos
    dqj = robot.data.joint_vel

    stance, swing = extract_ref_masks(ref, k=k)

    Jv_feet, j_info = extract_foot_jacobians_action_order(robot=robot, linear_rows=linear_rows)
    tau_jtf_check = compute_tau_jtf(Jv_feet, f_ref)

    dynamics = get_dynamics_terms(robot)
    M = dynamics.get("M")
    gravity = dynamics.get("gravity")
    coriolis = dynamics.get("coriolis")

    body_indices = get_body_indices(robot)
    foot_pos_w = robot.data.body_pos_w[:, body_indices, :]
    foot_vel_w = robot.data.body_lin_vel_w[:, body_indices, :]

    packet: Dict[str, object] = {
        "num_envs": int(qj.shape[0]),
        "num_joints": int(qj.shape[1]),
        "body_names_feet": FOOT_NAMES,
        "body_indices_feet": body_indices,
        "linear_rows": linear_rows,
        "stance_mask_env0": stance[0].detach().cpu().tolist(),
        "swing_mask_env0": swing[0].detach().cpu().tolist(),
        "ref_S_env0": ref["S"][0, :, k].detach().cpu().tolist(),
        "ref_phase_env0": ref["phase"][0, :, k].detach().cpu().tolist(),
        "qj_env0": qj[0].detach().cpu().tolist(),
        "dqj_env0": dqj[0].detach().cpu().tolist(),
        "x_hat_env0": x_hat[0].detach().cpu().tolist(),
        "foot_pos_w_env0": foot_pos_w[0].detach().cpu().tolist(),
        "foot_vel_w_env0": foot_vel_w[0].detach().cpu().tolist(),
        "f_ref_env0": f_ref[0].detach().cpu().tolist(),
        "tau_stance_env0": tau_stance[0].detach().cpu().tolist(),
        "tau_jtf_check_env0": tau_jtf_check[0].detach().cpu().tolist(),
        "tau_jtf_diff_max_abs": float((tau_stance - tau_jtf_check).abs().max().detach().cpu()),
        "Jv_feet_shape": list(Jv_feet.shape),
        "Jv_feet_mean_abs": float(Jv_feet.abs().mean().detach().cpu()),
        "Jv_feet_max_abs": float(Jv_feet.abs().max().detach().cpu()),
        "has_root_physx_view": dynamics["has_root_physx_view"],
    }

    packet.update(j_info)
    packet.update(summarize_tensor("M", M))
    packet.update(summarize_tensor("gravity", gravity))
    packet.update(summarize_tensor("coriolis", coriolis))

    tensors: Dict[str, torch.Tensor] = {
        "qj": qj,
        "dqj": dqj,
        "stance": stance,
        "swing": swing,
        "Jv_feet": Jv_feet,
        "f_ref": f_ref,
        "tau_jtf_check": tau_jtf_check,
        "foot_pos_w": foot_pos_w,
        "foot_vel_w": foot_vel_w,
    }
    if torch.is_tensor(M):
        tensors["M"] = M
    if torch.is_tensor(gravity):
        tensors["gravity"] = gravity
    if torch.is_tensor(coriolis):
        tensors["coriolis"] = coriolis

    return packet, tensors


def print_wbc_packet_summary(packet: Dict[str, object]) -> None:
    print("\n[WBC interface packet summary]")
    keys = [
        "num_envs", "num_joints",
        "body_names_feet", "body_indices_feet",
        "linear_rows",
        "stance_mask_env0", "swing_mask_env0",
        "ref_S_env0", "ref_phase_env0",
        "Jv_feet_shape", "Jv_feet_mean_abs", "Jv_feet_max_abs",
        "tau_jtf_diff_max_abs",
        "M_available", "M_shape", "M_mean_abs", "M_max_abs",
        "gravity_available", "gravity_shape", "gravity_mean_abs", "gravity_max_abs",
        "coriolis_available", "coriolis_shape", "coriolis_mean_abs", "coriolis_max_abs",
    ]
    for key in keys:
        if key in packet:
            print(f"{key}: {packet[key]}")
