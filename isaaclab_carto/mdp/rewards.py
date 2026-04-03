# import torch
# from isaaclab.envs import ManagerBasedRLEnv
# from isaaclab.managers import SceneEntityCfg
# from isaaclab.assets import Articulation

# def carto_reward_total(
#     env: ManagerBasedRLEnv, 
#     beta_weights: torch.Tensor, 
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """
#     CARTO의 가중치 기반 통합 보상 함수
#     R = beta_v * r_vel + beta_s * r_slip + beta_e * r_energy
#     """
#     robot: Articulation = env.scene[asset_cfg.name]
    
#     # 1. Velocity Tracking Reward (r_vel)
#     # 목표 속도(Body frame)와 현재 속도의 차이 계산
#     target_vel = env.command_manager.get_command("base_velocity")[:, :2]
#     current_vel = robot.data.root_lin_vel_b[:, :2]
#     lin_vel_error = torch.sum(torch.square(target_vel - current_vel), dim=1)
#     r_vel = torch.exp(-lin_vel_error / 0.25)
    
#     # 2. Slip Prevention Reward (r_slip)
#     # 발(Feet)의 월드 좌표계 기준 선속도를 페널티로 부여
#     # env.scene.sensors["feet_contact"]가 설정되어 있어야 합니다.
#     foot_indices, _ = robot.find_bodies(".*_foot")
#     foot_velocities = robot.data.body_lin_vel_w[:, foot_indices, :]
#     r_slip = -torch.sum(torch.norm(foot_velocities, dim=-1), dim=1)
    
#     # 3. Energy Cost (r_energy) - 수정됨!
#     # Isaac Lab에서는 applied_torque 속성을 사용하여 실제 가해진 토크를 가져옵니다.
#     r_energy = -torch.sum(torch.square(robot.data.applied_torque), dim=1)
    
#     # Objective Selector에서 온 가중치 적용
#     total_reward = (
#         beta_weights[:, 0] * r_vel + 
#         beta_weights[:, 1] * r_slip + 
#         beta_weights[:, 2] * r_energy
#     )
    
#     return total_reward
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation


def _get_robot(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Articulation:
    """Fetch robot articulation from the scene."""
    return env.scene[asset_cfg.name]


def reward_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    std: float = 0.5,
) -> torch.Tensor:
    """
    Velocity tracking reward.

    Higher is better.
    Uses XY linear velocity tracking in body frame.
    """
    robot = _get_robot(env, asset_cfg)

    target_vel = env.command_manager.get_command(command_name)[:, :2]
    current_vel = robot.data.root_lin_vel_b[:, :2]

    lin_vel_error = torch.sum(torch.square(target_vel - current_vel), dim=1)
    reward = torch.exp(-lin_vel_error / (std ** 2))

    return reward


def reward_slip(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_pattern: str = ".*_foot",
) -> torch.Tensor:
    """
    Slip prevention reward.

    Higher is better.
    Penalizes large foot linear velocity in world frame.
    Returns a negative or near-zero value.
    """
    robot = _get_robot(env, asset_cfg)

    foot_indices, _ = robot.find_bodies(foot_pattern)
    foot_velocities = robot.data.body_lin_vel_w[:, foot_indices, :]
    slip_penalty = torch.sum(torch.norm(foot_velocities, dim=-1), dim=1)

    return -slip_penalty


def reward_stability(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_pattern: str = ".*_foot",
) -> torch.Tensor:
    """
    Backward-compatible alias for reward_slip().
    """
    return reward_slip(env, asset_cfg=asset_cfg, foot_pattern=foot_pattern)


def reward_energy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Energy-aware reward.

    Higher is better.
    Penalizes large applied torque magnitude.
    Returns a negative or near-zero value.
    """
    robot = _get_robot(env, asset_cfg)

    torque = robot.data.applied_torque
    energy_penalty = torch.sum(torch.square(torque), dim=1)

    return -energy_penalty

def reward_body_height_violation(env, target_height: float = 0.38, scale: float = 10.0):
    """
    Penalize body height below target.
    Returns negative penalty.
    """
    robot = env.scene["robot"]
    base_h = robot.data.root_pos_w[:, 2]
    violation = torch.clamp(target_height - base_h, min=0.0)
    return -scale * violation

def reward_base_tilt_penalty(env, scale: float = 2.0):
    """
    Penalize roll/pitch tilt using projected gravity.
    Returns negative penalty.
    """
    robot = env.scene["robot"]
    g_proj = robot.data.projected_gravity_b  # [N, 3]
    tilt_xy = torch.norm(g_proj[:, :2], dim=-1)
    return -scale * tilt_xy

def reward_stand_pose_penalty(
    env,
    hy_target: float = 0.65,
    kn_target: float = -1.20,
    scale: float = 0.5,
):
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos

    # adjust these indices if needed after checking joint order
    # ideally map by names once you verify the exact joint ordering
    # here we assume all hy and kn joints can be sliced or indexed consistently

    # safer version: just use all joints and penalize large deviations from default pose
    default_joint_pos = robot.data.default_joint_pos
    diff = joint_pos - default_joint_pos
    return -scale * torch.sum(diff * diff, dim=-1)

def reward_components(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    """
    Return reward components in a dictionary.

    Keys:
        - velocity
        - slip
        - energy
    """
    return {
        "velocity": reward_velocity(env, asset_cfg=asset_cfg),
        "slip": reward_slip(env, asset_cfg=asset_cfg),
        "energy": reward_energy(env, asset_cfg=asset_cfg),
    }


def reward_tensor(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Return stacked reward components as [num_envs, 3].

    Order:
        [velocity, slip, energy]
    """
    comps = reward_components(env, asset_cfg=asset_cfg)
    return torch.stack(
        [comps["velocity"], comps["slip"], comps["energy"]],
        dim=-1,
    )

def reward_forward_progress(env, scale: float = 5.0):
    """
    Small reward for forward base velocity in body/world x direction.
    Clamp negative values so standing still is better than going backward.
    """
    robot = env.scene["robot"]
    vx = robot.data.root_lin_vel_w[:, 0]
    return scale * torch.clamp(vx, min=0.0)

def combine_reward_components(
    beta_weights: torch.Tensor,
    reward_values: torch.Tensor,
) -> torch.Tensor:
    """
    Weighted sum of reward components.

    Args:
        beta_weights: [num_envs, 3]
        reward_values: [num_envs, 3]

    Returns:
        total_reward: [num_envs]
    """
    if beta_weights.ndim != 2 or beta_weights.shape[-1] != 3:
        raise ValueError(
            f"beta_weights must have shape [num_envs, 3], got {beta_weights.shape}"
        )

    if reward_values.ndim != 2 or reward_values.shape[-1] != 3:
        raise ValueError(
            f"reward_values must have shape [num_envs, 3], got {reward_values.shape}"
        )

    return torch.sum(beta_weights * reward_values, dim=-1)


def carto_reward_total(
    env: ManagerBasedRLEnv,
    beta_weights: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Total weighted reward for CART/CART-O.

    Reward order:
        beta[:, 0] -> velocity
        beta[:, 1] -> slip/stability
        beta[:, 2] -> energy
    """
    comps = reward_tensor(env, asset_cfg=asset_cfg)
    return combine_reward_components(beta_weights, comps)