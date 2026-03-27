# 이 파일은 단순히 내부 함수들을 외부로 노출하는 역할을 합니다.
from .observations import (
    base_height,
    feet_slip_per_foot,
    feet_slip_summary,
    reference_friction, # type: ignore
    selector_aux,
)

from .rewards import (
    reward_velocity,
    reward_slip,
    reward_stability,
    reward_energy,
    reward_components,
    reward_tensor,
    combine_reward_components,
    carto_reward_total,
)

import isaaclab.envs.mdp as mdp_base
# Isaac Lab의 기본 MDP 함수들을 통합하여 mdp.base_lin_vel 등을 쓸 수 있게 합니다.

import isaaclab.envs.mdp as mdp_base
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import (
    ObservationGroupCfg, # 'G'입니다!
    ObservationTermCfg, 
    RewardTermCfg, 
    SceneEntityCfg
)
import isaaclab.envs.mdp.events as events_mdp

from isaaclab.envs.mdp import (
    joint_pos_rel, 
    joint_vel_rel, 
    base_lin_vel, 
    base_ang_vel, 
    projected_gravity,
    generated_commands,
    last_action,
    height_scan,
    time_out
)

def randomize_rigid_body_material(env: ManagerBasedRLEnv, env_ids: torch.Tensor, **kwargs) -> None:
    """지형 마찰력을 랜덤하게 변경하는 이벤트 함수의 래퍼입니다."""
    # Isaac Lab Manager는 (env, env_ids, **params) 순서로 호출합니다.
    # mdp_base에 있는 원본 함수에 모든 인자를 그대로 넘겨줍니다.
    return mdp_base.randomize_rigid_body_material(env, env_ids, **kwargs) # type: ignore



# # mdp 임포트 시 빨간 줄이 뜬다면, 아래와 같이 시도해 보세요.
# try:
#     import isaaclab_carto.isaaclab_carto.mdp as mdp
# except ImportError:
#     # 절대 경로 인식이 안 될 경우 상대 경로로 시도
#     from .. import mdp

def camera_rgb(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """카메라 센서에서 RGB 데이터를 추출합니다."""
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.output["rgb"]

def camera_depth(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """카메라 센서에서 Depth(거리) 데이터를 추출합니다."""
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.output["distance_to_image_plane"]

# def joint_efforts(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """각 관절의 노력(Effort/Torque) 데이터를 추출합니다."""
#     return env.scene[asset_cfg.name].data.joint_effort

def joint_efforts(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """각 관절의 effort/torque 데이터를 추출합니다."""
    return env.scene[asset_cfg.name].data.applied_torque

def base_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """로봇 몸체의 현재 높이(Z축)를 추출합니다."""
    # root_pos_w의 3번째 값(index 2)이 높이입니다.
    return env.scene[asset_cfg.name].data.root_pos_w[:, 2].unsqueeze(1)

def feet_slip_per_foot(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    foot_ids = robot.find_bodies(".*foot.*")[0]
    foot_vel = robot.data.body_lin_vel_w[:, foot_ids]
    slip = torch.norm(foot_vel[..., :2], dim=-1)   # [N, 4]
    return slip

def reference_friction(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """시뮬레이션 설정(cfg)에서 마찰 계수를 직접 가져옵니다. (빨간 줄 완벽 해결 버전)"""
    try:
        # 1. getattr을 사용하면 Pylance가 'terrain' 속성이 없다고 불평하지 않습니다.
        terrain_cfg = getattr(env.cfg.scene, "terrain", None)
        
        if terrain_cfg is not None and hasattr(terrain_cfg.spawn, "static_friction"):
            friction_value = terrain_cfg.spawn.static_friction
        else:
            # 기본값 (마찰력이 보통 수준인 0.5)
            friction_value = 0.5
            
    except Exception:
        # 어떤 에러가 나더라도 시뮬레이션이 멈추지 않게 방어적으로 코딩합니다.
        friction_value = 0.5
    
    # [Batch, 4] 형태로 텐서를 만들어 반환합니다.
    return torch.full((env.num_envs, 4), friction_value, device=env.device)

# def reward_stability(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """안정성 보상: 몸체 높이 유지 및 미끄러짐 방지"""
#     # 1. 높이 유지 (0.5m 목표)
#     height = base_height(env)
#     r_height = torch.exp(-torch.abs(height - 0.5) / 0.1)
#     # 2. 미끄러짐 패널티
#     slip = feet_slip(env)
#     r_slip = torch.exp(-torch.sum(slip, dim=-1, keepdim=True) / 1.0)
#     return (r_height + r_slip).squeeze(1)

def reward_stability(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]

    # 높이 안정성
    base_height = robot.data.root_pos_w[:, 2]
    target_height = 0.5
    height_error = torch.square(base_height - target_height)
    r_height = torch.exp(-height_error / 0.05)

    # slip 안정성
    foot_ids = robot.find_bodies(".*foot.*")[0]
    foot_vel = robot.data.body_lin_vel_w[:, foot_ids]
    foot_slip = torch.norm(foot_vel[..., :2], dim=-1)
    r_slip = torch.exp(-torch.mean(foot_slip, dim=-1) / 0.5)

    # scale 맞추기
    return 0.5 * (r_height + r_slip)

# def reward_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """속도 보상: 목표 명령(Command) 추종"""
#     base_vel = env.scene["robot"].data.root_lin_vel_w[:, :2] # XY 선속도
#     target_vel = env.command_manager.get_command("base_velocity")[:, :2]
#     # 속도 오차에 대한 가우시안 보상
#     vel_error = torch.sum(torch.square(target_vel - base_vel), dim=-1)
#     return torch.exp(-vel_error / 1.0)

def reward_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    target_vel = env.command_manager.get_command("base_velocity")[:, :2]
    current_vel = robot.data.root_lin_vel_b[:, :2]
    vel_error = torch.sum(torch.square(target_vel - current_vel), dim=-1)
    return torch.exp(-vel_error / 0.25)

# def reward_energy(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """에너지 효율 보상: 관절 토크 최소화"""
#     torques = env.scene["robot"].data.applied_torque
#     # 토크의 제곱 합에 대한 패널티를 보상 형태로 변환
#     # torque_penalty = torch.sum(torch.square(torques), dim=-1)
#     # return torch.exp(-torque_penalty / 100.0)
#     torque_penalty = torch.mean(torch.square(torques), dim=-1)
#     return torch.exp(-torque_penalty / 10.0)

def reward_energy(env: ManagerBasedRLEnv) -> torch.Tensor:
    torques = env.scene["robot"].data.applied_torque
    torque_penalty = torch.mean(torch.square(torques), dim=-1)
    return 1.0 / (1.0 + torque_penalty)

def feet_slip_summary(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    foot_ids = robot.find_bodies(".*foot.*")[0]
    foot_vel = robot.data.body_lin_vel_w[:, foot_ids]
    slip = torch.norm(foot_vel[..., :2], dim=-1)   # [N, 4] 가정

    front_mean = slip[:, :2].mean(dim=-1, keepdim=True)
    rear_mean = slip[:, 2:].mean(dim=-1, keepdim=True)

    return torch.cat([front_mean, rear_mean], dim=-1)  # [N, 2]

def base_height_below_threshold(
    env: ManagerBasedRLEnv, 
    threshold: float, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """로봇 몸체의 높이가 일정 임계값(threshold) 이하인지 확인합니다."""
    # 현재 높이 추출
    height = env.scene[asset_cfg.name].data.root_pos_w[:, 2]
    # 임계값 이하인 환경들을 True로 반환 (종료 신호)
    return height < threshold

# mdp/__init__.py 에 추가

def reward_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """로봇이 목표 방향(X축)으로 실제로 이동한 거리에 비례한 보상"""
    # 현재 선속도 (World Frame)
    lin_vel_x = env.scene["robot"].data.root_lin_vel_w[:, 0]
    # 앞으로 가고 있다면 (+) 보상, 뒤로 가거나 멈춰있으면 0
    return torch.clamp(lin_vel_x, min=0.0)