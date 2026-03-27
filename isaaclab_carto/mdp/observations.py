# import torch
# from isaaclab.assets import Articulation, RigidObject
# from isaaclab.envs import ManagerBasedRLEnv
# from isaaclab.managers import SceneEntityCfg

# ##
# # Proprioceptive Observations (for BiLSTM)
# ##

# def joint_states_history_input(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """
#     BiLSTM의 입력으로 사용될 로봇의 고유 수용 감각(Proprioception)을 추출합니다.
#     구성: 관절 각도(12) + 관절 속도(12) + 베이스 선속도(3) + 베이스 각속도(3) + 투영된 중력(3) + 발 슬립(3) = 총 36차원
#     """
#     asset: Articulation = env.scene[asset_cfg.name]
    
#     # 1. 관절 상태 (기본 포즈 대비 상대값)
#     joint_pos = asset.data.joint_pos - asset.data.default_joint_pos
#     joint_vel = asset.data.joint_vel
    
#     # 2. 베이스 동역학 (Body frame 기준)
#     base_lin_vel = asset.data.root_lin_vel_b
#     base_ang_vel = asset.data.root_ang_vel_b
#     projected_gravity = asset.data.projected_gravity_b
    
#     # 3. 발의 상태 (슬립 등 - 필요 시 추가 계산)
#     # 여기서는 간단히 0으로 채우거나 contact_sensor 데이터를 활용할 수 있습니다.
#     foot_slip = torch.zeros((env.num_envs, 3), device=env.device) 

#     # 모든 데이터를 하나로 결합 [Batch, 36]
#     return torch.cat([joint_pos, joint_vel, base_lin_vel, base_ang_vel, projected_gravity, foot_slip], dim=-1)

# ##
# # Visual Observations (for CNN)
# ##

# def processed_height_scan(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#     """
#     지형의 높이 정보를 추출합니다. CNN 인코더(Sv)의 입력이 됩니다.
#     """
#     sensor = env.scene.sensors[sensor_cfg.name]
#     # 레이캐스터(RayCaster) 데이터에서 높이값 추출
#     heights = sensor.data.ray_hits_w[..., 2] - env.scene["robot"].data.root_pos_w[:, 2:3]
#     # [Batch, 160] (16x10 그리드 기준)
#     return torch.clamp(heights, -1.0, 1.0)

# ##
# # Surface/Mesh Observations (for SEL)
# ##

# def terrain_surface_info(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """
#     지면의 물리적 특성(마찰력, 경사도 등)을 추출합니다. (Sm)
#     """
#     # 현재는 간단하게 로봇이 느끼는 지면의 기울기(Roll/Pitch)를 활용합니다.
#     # 나중에 진짜 SEL이 통합되면 마찰 계수 등을 추가할 예정입니다.
#     return env.scene["robot"].data.projected_gravity_b[:, :2]

# def joint_pos_target(env, targets, asset_cfg=SceneEntityCfg("robot")):
#     return targets

# def time_out(env):
#     return env.episode_length_buf >= env.max_episode_length

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def _get_robot(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Articulation:
    return env.scene[asset_cfg.name]


def base_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Robot base height in world frame.
    shape: [num_envs, 1]
    """
    robot = _get_robot(env, asset_cfg)
    return robot.data.root_pos_w[:, 2:3]


def feet_slip_per_foot(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_pattern: str = ".*_foot",
) -> torch.Tensor:
    """
    Slip proxy for each foot using XY linear velocity norm in world frame.
    shape: [num_envs, num_feet]
    """
    robot = _get_robot(env, asset_cfg)

    foot_indices, _ = robot.find_bodies(foot_pattern)
    foot_vel_w = robot.data.body_lin_vel_w[:, foot_indices, :]   # [N, F, 3]

    slip_xy = torch.norm(foot_vel_w[..., :2], dim=-1)            # [N, F]
    return slip_xy


def feet_slip_summary(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_pattern: str = ".*_foot",
) -> torch.Tensor:
    """
    Summary of foot slip.
    Returns [mean, max, std].
    shape: [num_envs, 3]
    """
    slip = feet_slip_per_foot(env, asset_cfg=asset_cfg, foot_pattern=foot_pattern)

    mean_slip = slip.mean(dim=1, keepdim=True)
    max_slip = slip.max(dim=1, keepdim=True).values
    std_slip = slip.std(dim=1, keepdim=True)

    return torch.cat([mean_slip, max_slip, std_slip], dim=-1)


def reference_friction(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    if hasattr(env, "reference_friction_buffer"):
        return env.reference_friction_buffer

    return torch.ones((env.num_envs, 1), device=env.device, dtype=torch.float32)

def selector_aux(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_pattern: str = ".*_foot",
) -> torch.Tensor:
    """
    Auxiliary observation for objective selector.

    구성:
    - reference_friction: [N, 1]
    - base_height:        [N, 1]
    - feet_slip_summary:  [N, 3]

    total shape: [N, 5]
    """
    friction = reference_friction(env)                                            # [N, 1]
    height = base_height(env, asset_cfg=asset_cfg)                                # [N, 1]
    slip = feet_slip_summary(env, asset_cfg=asset_cfg, foot_pattern=foot_pattern) # [N, 3]

    return torch.cat([friction, height, slip], dim=-1)