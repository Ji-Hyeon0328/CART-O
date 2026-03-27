import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation

def carto_reward_total(
    env: ManagerBasedRLEnv, 
    beta_weights: torch.Tensor, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    CARTO의 가중치 기반 통합 보상 함수
    R = beta_v * r_vel + beta_s * r_slip + beta_e * r_energy
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # 1. Velocity Tracking Reward (r_vel)
    # 목표 속도(Body frame)와 현재 속도의 차이 계산
    target_vel = env.command_manager.get_command("base_velocity")[:, :2]
    current_vel = robot.data.root_lin_vel_b[:, :2]
    lin_vel_error = torch.sum(torch.square(target_vel - current_vel), dim=1)
    r_vel = torch.exp(-lin_vel_error / 0.25)
    
    # 2. Slip Prevention Reward (r_slip)
    # 발(Feet)의 월드 좌표계 기준 선속도를 페널티로 부여
    # env.scene.sensors["feet_contact"]가 설정되어 있어야 합니다.
    foot_indices, _ = robot.find_bodies(".*_foot")
    foot_velocities = robot.data.body_lin_vel_w[:, foot_indices, :]
    r_slip = -torch.sum(torch.norm(foot_velocities, dim=-1), dim=1)
    
    # 3. Energy Cost (r_energy) - 수정됨!
    # Isaac Lab에서는 applied_torque 속성을 사용하여 실제 가해진 토크를 가져옵니다.
    r_energy = -torch.sum(torch.square(robot.data.applied_torque), dim=1)
    
    # Objective Selector에서 온 가중치 적용
    total_reward = (
        beta_weights[:, 0] * r_vel + 
        beta_weights[:, 1] * r_slip + 
        beta_weights[:, 2] * r_energy
    )
    
    return total_reward