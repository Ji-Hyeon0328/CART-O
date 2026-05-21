# isaaclab_carto/lowlevel/jacobian_utils.py
#
# Utilities for inspecting and extracting Jacobians from Isaac Lab / PhysX articulation.
#
# This file is intentionally defensive because Jacobian APIs can differ slightly
# across Isaac Lab / Isaac Sim versions.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch


SPOT_FOOT_NAMES = ["fl_foot", "fr_foot", "hl_foot", "hr_foot"]
SPOT_LEG_NAMES = ["fl", "fr", "hl", "hr"]
SPOT_JOINT_NAMES_BY_LEG = {
    "fl": ["fl_hx", "fl_hy", "fl_kn"],
    "fr": ["fr_hx", "fr_hy", "fr_kn"],
    "hl": ["hl_hx", "hl_hy", "hl_kn"],
    "hr": ["hr_hx", "hr_hy", "hr_kn"],
}


def get_name_to_index(names: List[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(names)}


def get_spot_foot_body_indices(robot) -> Tuple[List[int], List[str]]:
    body_names = getattr(robot, "body_names", None)
    if body_names is None:
        raise RuntimeError("robot.body_names not found.")

    name_to_idx = get_name_to_index(body_names)
    missing = [name for name in SPOT_FOOT_NAMES if name not in name_to_idx]
    if missing:
        raise RuntimeError(f"Missing Spot foot body names: {missing}. Available body names: {body_names}")

    return [name_to_idx[name] for name in SPOT_FOOT_NAMES], SPOT_FOOT_NAMES


def get_spot_actuated_joint_indices(robot) -> Tuple[List[int], List[str]]:
    joint_names = getattr(robot, "joint_names", None)
    if joint_names is None:
        raise RuntimeError("robot.joint_names not found.")

    name_to_idx = get_name_to_index(joint_names)

    ordered_joint_names: List[str] = []
    for leg in SPOT_LEG_NAMES:
        ordered_joint_names.extend(SPOT_JOINT_NAMES_BY_LEG[leg])

    missing = [name for name in ordered_joint_names if name not in name_to_idx]
    if missing:
        raise RuntimeError(f"Missing Spot joint names: {missing}. Available joint names: {joint_names}")

    return [name_to_idx[name] for name in ordered_joint_names], ordered_joint_names


def list_robot_view_methods(robot) -> Dict[str, List[str]]:
    """
    Return useful method/property names from robot and root_physx_view.
    """
    out: Dict[str, List[str]] = {}

    out["robot_has"] = [
        name for name in [
            "root_physx_view",
            "_root_physx_view",
            "body_names",
            "joint_names",
            "data",
            "find_bodies",
            "find_joints",
        ]
        if hasattr(robot, name)
    ]

    view = getattr(robot, "root_physx_view", None)
    if view is None:
        view = getattr(robot, "_root_physx_view", None)

    if view is None:
        out["root_physx_view_has"] = []
        return out

    candidates = [
        "get_jacobians",
        "get_jacobian",
        "get_generalized_gravity_forces",
        "get_coriolis_and_centrifugal_forces",
        "get_mass_matrices",
        "get_dof_limits",
        "get_dof_max_forces",
        "get_link_transforms",
        "get_link_velocities",
    ]
    out["root_physx_view_has"] = [name for name in candidates if hasattr(view, name)]

    # Also include any method containing "jacob".
    out["root_physx_view_jacobian_like"] = [
        name for name in dir(view)
        if "jacob" in name.lower()
    ]

    return out


def try_get_root_physx_view(robot):
    view = getattr(robot, "root_physx_view", None)
    if view is None:
        view = getattr(robot, "_root_physx_view", None)
    return view


def try_get_jacobians(robot) -> Optional[torch.Tensor]:
    """
    Try to obtain raw Jacobian tensor.

    Common PhysX pattern:
        robot.root_physx_view.get_jacobians()

    Returns:
        jacobians tensor or None.
    """
    view = try_get_root_physx_view(robot)
    if view is None:
        print("[WARN] root_physx_view not found on robot.")
        return None

    if hasattr(view, "get_jacobians"):
        try:
            J = view.get_jacobians()
            return J
        except Exception as exc:
            print(f"[WARN] root_physx_view.get_jacobians() failed: {exc}")

    if hasattr(view, "get_jacobian"):
        try:
            J = view.get_jacobian()
            return J
        except Exception as exc:
            print(f"[WARN] root_physx_view.get_jacobian() failed: {exc}")

    print("[WARN] No Jacobian getter found on root_physx_view.")
    return None


def infer_jacobian_layout(J: torch.Tensor, robot) -> Dict[str, Any]:
    """
    Infer likely Jacobian tensor layout.

    Common shape in Isaac Sim articulation view:
        [num_envs, num_bodies, 6, num_dofs]
    But some versions may include floating-base columns:
        [num_envs, num_bodies, 6, 6 + num_dofs]
    """
    info: Dict[str, Any] = {}
    info["shape"] = tuple(J.shape)
    info["ndim"] = J.ndim

    num_envs = robot.data.joint_pos.shape[0]
    num_joints = robot.data.joint_pos.shape[1]
    num_bodies = len(getattr(robot, "body_names", []))

    info["num_envs"] = num_envs
    info["num_joints"] = num_joints
    info["num_bodies"] = num_bodies

    if J.ndim == 4:
        info["looks_like"] = "[env, body, spatial6, columns]"
        info["body_dim"] = 1
        info["spatial_dim"] = 2
        info["column_dim"] = 3
        info["has_floating_base_columns"] = (J.shape[-1] == num_joints + 6)
        info["has_joint_only_columns"] = (J.shape[-1] == num_joints)
    elif J.ndim == 3:
        info["looks_like"] = "[body, spatial6, columns] or [env, ?, ?]"
        info["body_dim"] = None
        info["spatial_dim"] = None
        info["column_dim"] = None
        info["has_floating_base_columns"] = (J.shape[-1] == num_joints + 6)
        info["has_joint_only_columns"] = (J.shape[-1] == num_joints)
    else:
        info["looks_like"] = "unknown"

    return info


def extract_spot_foot_linear_jacobians(
    J: torch.Tensor,
    robot,
    assume_floating_base: Optional[bool] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Extract foot linear Jacobians for actuated 12 joints.

    Args:
        J:
            raw Jacobian tensor, expected [N, B, 6, C].
        robot:
            Isaac Lab articulation.
        assume_floating_base:
            If None, infer from column count.
            If True, use columns 6:18.
            If False, use columns 0:12.

    Returns:
        Jv_feet:
            [num_envs, 4, 3, 12]
            Linear part of foot Jacobian for feet [fl, fr, hl, hr].
        info:
            extraction metadata.
    """
    if J.ndim != 4:
        raise ValueError(f"Expected raw J shape [N, B, 6, C], got {tuple(J.shape)}")

    foot_indices, foot_names = get_spot_foot_body_indices(robot)
    joint_indices, joint_names = get_spot_actuated_joint_indices(robot)

    num_joints = robot.data.joint_pos.shape[1]
    C = J.shape[-1]

    if assume_floating_base is None:
        if C == num_joints + 6:
            assume_floating_base = True
        elif C == num_joints:
            assume_floating_base = False
        else:
            # Conservative default: if enough columns, assume base columns exist.
            assume_floating_base = C >= num_joints + 6

    if assume_floating_base:
        # Usually columns: [base 6, actuated joints]
        col_indices = [6 + i for i in joint_indices]
    else:
        col_indices = joint_indices

    max_col = max(col_indices)
    if max_col >= C:
        raise RuntimeError(
            f"Column index out of range. C={C}, col_indices={col_indices}, "
            f"assume_floating_base={assume_floating_base}"
        )

    # Spatial rows: commonly [linear xyz, angular xyz] or [angular, linear].
    # For now we extract both variants so the debug script can compare.
    J_feet = J[:, foot_indices, :, :][:, :, :, col_indices]  # [N, 4, 6, 12]

    J_rows_0_3 = J_feet[:, :, 0:3, :]
    J_rows_3_6 = J_feet[:, :, 3:6, :]

    # We return rows 0:3 as default, but include both in info.
    info = {
        "foot_indices": foot_indices,
        "foot_names": foot_names,
        "joint_indices": joint_indices,
        "joint_names": joint_names,
        "assume_floating_base": assume_floating_base,
        "col_indices": col_indices,
        "raw_shape": tuple(J.shape),
        "J_rows_0_3_norm": float(torch.linalg.norm(J_rows_0_3).detach().cpu()),
        "J_rows_3_6_norm": float(torch.linalg.norm(J_rows_3_6).detach().cpu()),
        "note": "Need verify whether rows 0:3 or 3:6 are linear velocity rows in your Isaac version.",
    }

    return J_rows_0_3, info


def compute_tau_from_foot_forces(
    Jv_feet: torch.Tensor,
    f_feet: torch.Tensor,
) -> torch.Tensor:
    """
    Compute tau = J^T f.

    Args:
        Jv_feet:
            [num_envs, 4, 3, 12]
        f_feet:
            [num_envs, 4, 3]

    Returns:
        tau:
            [num_envs, 12]
    """
    if Jv_feet.ndim != 4:
        raise ValueError(f"Jv_feet must be [N,4,3,12], got {tuple(Jv_feet.shape)}")
    if f_feet.ndim != 3:
        raise ValueError(f"f_feet must be [N,4,3], got {tuple(f_feet.shape)}")

    tau = torch.einsum("nlik,nli->nk", Jv_feet, f_feet)
    return tau
