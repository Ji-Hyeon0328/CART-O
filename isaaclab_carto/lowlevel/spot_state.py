# isaaclab_carto/lowlevel/spot_state.py
#
# Isaac Lab Spot state helpers.
#
# Current confirmed Spot order from debug:
#
# Joint names:
#   00 fl_hx
#   01 fr_hx
#   02 hl_hx
#   03 hr_hx
#   04 fl_hy
#   05 fr_hy
#   06 hl_hy
#   07 hr_hy
#   08 fl_kn
#   09 fr_kn
#   10 hl_kn
#   11 hr_kn
#
# Body names:
#   ...
#   13 fl_foot
#   14 fr_foot
#   15 hl_foot
#   16 hr_foot
#
# Project leg order:
#   [LF, RF, LH, RH] == [fl, fr, hl, hr]

from __future__ import annotations

from typing import Dict, List, Tuple, Any

import torch


SPOT_LEG_ORDER = ["fl", "fr", "hl", "hr"]
PROJECT_LEG_ORDER = ["LF", "RF", "LH", "RH"]


def quat_to_rpy_xyz(q: torch.Tensor) -> torch.Tensor:
    """
    Quaternion to roll-pitch-yaw.

    Assumption:
        Isaac Lab root_quat_w is [w, x, y, z].

    Args:
        q: [..., 4]

    Returns:
        rpy: [..., 3]
    """
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


def _safe_get_attr(obj: Any, names: List[str]):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def make_x_hat(robot, velocity_frame: str = "world") -> torch.Tensor:
    """
    Build x_hat for low-level controller.

    Output:
        x_hat = [x y z roll pitch yaw vx vy vz wx wy wz]
        shape: [num_envs, 12]

    Args:
        robot:
            Isaac Lab Articulation object.
        velocity_frame:
            "world" or "body".

    Notes:
        For theta_ref_mapper(), recommended convention is:
            pose/orientation: world frame
            velocity: world frame
            command u_cmd: body frame
    """
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    rpy = quat_to_rpy_xyz(root_quat_w)

    if velocity_frame == "world":
        lin_vel = _safe_get_attr(
            robot.data,
            ["root_lin_vel_w", "root_com_lin_vel_w"],
        )
        ang_vel = _safe_get_attr(
            robot.data,
            ["root_ang_vel_w", "root_com_ang_vel_w"],
        )
    elif velocity_frame == "body":
        lin_vel = _safe_get_attr(
            robot.data,
            ["root_lin_vel_b", "root_com_lin_vel_b"],
        )
        ang_vel = _safe_get_attr(
            robot.data,
            ["root_ang_vel_b", "root_com_ang_vel_b"],
        )
    else:
        raise ValueError(f"velocity_frame must be 'world' or 'body', got {velocity_frame}")

    # Fallback to body-frame if world-frame is not available.
    if lin_vel is None:
        lin_vel = _safe_get_attr(robot.data, ["root_lin_vel_b", "root_com_lin_vel_b"])
    if ang_vel is None:
        ang_vel = _safe_get_attr(robot.data, ["root_ang_vel_b", "root_com_ang_vel_b"])

    if lin_vel is None:
        raise RuntimeError("Could not find root linear velocity in robot.data.")
    if ang_vel is None:
        raise RuntimeError("Could not find root angular velocity in robot.data.")

    return torch.cat([root_pos_w, rpy, lin_vel, ang_vel], dim=-1)


def build_spot_ref_params(
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    dt: float = 0.02,
    horizon: int = 20,
) -> Dict[str, torch.Tensor | float | int]:
    """
    Approximate Spot reference parameters for theta_ref_mapper().

    Leg order:
        [LF, RF, LH, RH] == [fl, fr, hl, hr]

    These are not exact dynamics parameters.
    They are only used for early reference generation and debugging.
    """
    hip_offset_body = torch.tensor(
        [
            [0.30, 0.30, -0.30, -0.30],
            [0.18, -0.18, 0.18, -0.18],
            [0.00, 0.00, 0.00, 0.00],
        ],
        device=device,
        dtype=dtype,
    )

    p_foot_now = torch.tensor(
        [
            [0.34, 0.34, -0.32, -0.32],
            [0.20, -0.20, 0.20, -0.20],
            [0.00, 0.00, 0.00, 0.00],
        ],
        device=device,
        dtype=dtype,
    )

    return {
        "dt": dt,
        "N": horizon,
        "hip_offset_body": hip_offset_body,
        "p_foot_now": p_foot_now,
    }


def get_spot_joint_indices(robot) -> Dict[str, List[int]]:
    """
    Return Spot joint indices by leg and by joint type.

    Expected joint names:
        fl_hx, fr_hx, hl_hx, hr_hx,
        fl_hy, fr_hy, hl_hy, hr_hy,
        fl_kn, fr_kn, hl_kn, hr_kn

    Returns:
        {
            "fl": [fl_hx, fl_hy, fl_kn],
            "fr": [fr_hx, fr_hy, fr_kn],
            "hl": [hl_hx, hl_hy, hl_kn],
            "hr": [hr_hx, hr_hy, hr_kn],
            "by_project_order": [fl_hx, fl_hy, fl_kn, fr_hx, ...]
        }
    """
    joint_names = getattr(robot, "joint_names", None)
    if joint_names is None:
        raise RuntimeError("robot.joint_names not found.")

    name_to_idx = {name: i for i, name in enumerate(joint_names)}

    out: Dict[str, List[int]] = {}
    by_project_order: List[int] = []

    for leg in SPOT_LEG_ORDER:
        keys = [f"{leg}_hx", f"{leg}_hy", f"{leg}_kn"]
        missing = [k for k in keys if k not in name_to_idx]
        if missing:
            raise RuntimeError(f"Missing Spot joint names: {missing}. Available: {joint_names}")

        idxs = [name_to_idx[k] for k in keys]
        out[leg] = idxs
        by_project_order.extend(idxs)

    out["by_project_order"] = by_project_order
    return out


def get_spot_foot_indices(robot) -> Tuple[List[int], List[str]]:
    """
    Return foot body indices in project leg order:
        [fl_foot, fr_foot, hl_foot, hr_foot]
    """
    body_names = getattr(robot, "body_names", None)
    if body_names is None:
        raise RuntimeError("robot.body_names not found.")

    target_names = ["fl_foot", "fr_foot", "hl_foot", "hr_foot"]
    name_to_idx = {name: i for i, name in enumerate(body_names)}

    missing = [name for name in target_names if name not in name_to_idx]
    if missing:
        raise RuntimeError(f"Missing Spot foot body names: {missing}. Available: {body_names}")

    indices = [name_to_idx[name] for name in target_names]
    return indices, target_names


def get_current_foot_positions_w(robot) -> torch.Tensor:
    """
    Current foot positions in world frame.

    Returns:
        p_foot_w: [num_envs, 3, 4]
        leg order: [fl, fr, hl, hr]
    """
    foot_indices, _ = get_spot_foot_indices(robot)
    body_pos_w = robot.data.body_pos_w[:, foot_indices, :]  # [N, 4, 3]
    return body_pos_w.transpose(1, 2).contiguous()           # [N, 3, 4]


def get_current_foot_velocities_w(robot) -> torch.Tensor:
    """
    Current foot linear velocities in world frame.

    Returns:
        v_foot_w: [num_envs, 3, 4]
        leg order: [fl, fr, hl, hr]
    """
    foot_indices, _ = get_spot_foot_indices(robot)
    body_lin_vel_w = robot.data.body_lin_vel_w[:, foot_indices, :]  # [N, 4, 3]
    return body_lin_vel_w.transpose(1, 2).contiguous()              # [N, 3, 4]


def make_standing_action(robot, mode: str = "zero") -> torch.Tensor:
    """
    Build action tensor for current JointPositionActionCfg environment.

    Args:
        mode:
            "zero":
                zero action. In many Isaac Lab JointPositionActionCfg setups,
                this means default pose if default offset is internally used.
            "default":
                default_joint_pos.
            "current":
                current joint_pos.

    Returns:
        action tensor with shape [num_envs, num_actions].
    """
    if mode == "zero":
        return torch.zeros_like(robot.data.joint_pos)

    if mode == "default":
        return robot.data.default_joint_pos.clone()

    if mode == "current":
        return robot.data.joint_pos.clone()

    raise ValueError(f"Unknown standing action mode: {mode}")


def print_robot_debug_info(robot) -> None:
    """
    Print joint/body/foot debug information.
    """
    print("\n" + "=" * 80)
    print("[DEBUG] Robot info")
    print("=" * 80)

    joint_names = getattr(robot, "joint_names", None)
    body_names = getattr(robot, "body_names", None)

    if joint_names is not None:
        print("\n[Joint names]")
        for i, name in enumerate(joint_names):
            print(f"{i:02d}: {name}")
    else:
        print("[WARN] robot.joint_names not found.")

    if body_names is not None:
        print("\n[Body names]")
        for i, name in enumerate(body_names):
            print(f"{i:02d}: {name}")
    else:
        print("[WARN] robot.body_names not found.")

    print("\n[Find foot bodies]")
    for pattern in [".*_foot", ".*foot.*", ".*_leg", ".*lleg.*"]:
        try:
            indices, names = robot.find_bodies(pattern)
            print(f"pattern={pattern} -> indices={indices}, names={names}")
        except Exception as exc:
            print(f"pattern={pattern} -> failed: {exc}")

    print("\n[Parsed Spot mapping]")
    try:
        joint_map = get_spot_joint_indices(robot)
        print("joint_map:")
        for k, v in joint_map.items():
            print(f"  {k}: {v}")
    except Exception as exc:
        print(f"[WARN] get_spot_joint_indices failed: {exc}")

    try:
        foot_indices, foot_names = get_spot_foot_indices(robot)
        print(f"foot_indices: {foot_indices}")
        print(f"foot_names:   {foot_names}")
    except Exception as exc:
        print(f"[WARN] get_spot_foot_indices failed: {exc}")

    print("\n[State shapes]")
    print(f"joint_pos:         {tuple(robot.data.joint_pos.shape)}")
    print(f"default_joint_pos: {tuple(robot.data.default_joint_pos.shape)}")
    print(f"root_pos_w:        {tuple(robot.data.root_pos_w.shape)}")
    print(f"root_quat_w:       {tuple(robot.data.root_quat_w.shape)}")

    print("=" * 80 + "\n")

def make_gait_joint_position_action(
    robot,
    ref: dict,
    lift_scale: float = 0.12,
    knee_scale: float = -0.18,
    use_k: int = 0,
) -> torch.Tensor:
    """
    Build a small gait-modulated joint-position action for selection A final test.

    This is NOT real IK and NOT final control.
    It only checks whether Ref.S / swing schedule can be connected to
    Isaac Lab JointPositionActionCfg safely.

    Current Isaac Lab config seems to interpret zero action as default pose.
    Therefore this function starts from zero action and adds small offsets.

    Leg order:
        Ref leg order: [LF, RF, LH, RH]
        Spot order:    [fl, fr, hl, hr]

    Joint order confirmed:
        0  fl_hx
        1  fr_hx
        2  hl_hx
        3  hr_hx
        4  fl_hy
        5  fr_hy
        6  hl_hy
        7  hr_hy
        8  fl_kn
        9  fr_kn
        10 hl_kn
        11 hr_kn

    Args:
        robot:
            Isaac Lab robot articulation.
        ref:
            Output from theta_ref_mapper().
        lift_scale:
            Hip pitch offset for swing legs.
        knee_scale:
            Knee offset for swing legs.
            Sign may need adjustment after visual check.
        use_k:
            Which horizon index to use from Ref.S.
            For visual testing, k=0 may often be all stance.
            k=10 is useful because swing usually appears there.

    Returns:
        actions: [num_envs, 12]
    """
    actions = torch.zeros_like(robot.data.joint_pos)

    S = ref["S"]  # [num_envs, 4, H]
    H = S.shape[-1]
    k = min(max(use_k, 0), H - 1)

    # swing = 1 when S == 0
    swing = (S[:, :, k] < 0.5).to(actions.dtype)  # [num_envs, 4]

    # Joint indices by leg in Ref order [fl, fr, hl, hr]
    hx_idx = [0, 1, 2, 3]
    hy_idx = [4, 5, 6, 7]
    kn_idx = [8, 9, 10, 11]

    # Keep hx unchanged for this test.
    _ = hx_idx

    for leg in range(4):
        actions[:, hy_idx[leg]] += lift_scale * swing[:, leg]
        actions[:, kn_idx[leg]] += knee_scale * swing[:, leg]

    return actions