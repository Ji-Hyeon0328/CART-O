import os
import sys
import torch
import argparse
from isaaclab.app import AppLauncher
from typing import Dict

# 1. 앱 런처 설정
parser = argparse.ArgumentParser(description="Train Spot with CART Framework")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
simulation_app = launcher.app

# 2. 필요한 라이브러리 임포트
from isaaclab.envs import ManagerBasedRLEnv
#from isaaclab_carto.isaaclab_carto.envs.carto_env_cfg import CartoEnvCfg
from isaaclab_carto.envs.carto_env_cfg import CartoEnvCfg
from spawn_spot import SpotActor # 같은 폴더에 있으므로 바로 임포트 가능

from torch.utils.tensorboard import SummaryWriter
import datetime
from spawn_spot import SpotActor

try:
    import isaaclab_carto.isaaclab_carto.mdp as mdp
except ImportError:
    import isaaclab_carto.mdp as mdp
def sanitize(x, nan=0.0, posinf=0.0, neginf=0.0):
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)

def normalize_reward(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + eps)

def main():
    # 환경 설정
    env_cfg = CartoEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 텐서보드 기록기 초기화 (logs/carto_spot 폴더에 시간별로 저장)
    log_dir = os.path.join("logs", "carto_spot", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)

    # Figure 175 아키텍처 모델 생성
    # num_proprio=36 (CART 논문 기준)
    policy = SpotActor(num_proprio=36, num_map=187, num_actions=12).to(env.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    obs_dict, _ = env.reset()

    print("-" * 50)
    print("[INFO]: Figure 175 Hierarchical Training Started!")
    print(f"[INFO]: Number of Envs: {env.num_envs}")
    print("-" * 50)

    # old_actions=None

    while simulation_app.is_running():
        
        obs_dict: Dict[str, torch.Tensor] = env.observation_manager.compute()
        proprio = obs_dict["policy"].clone().detach() # [4, 36]
        proprio=sanitize(proprio).float()
        selector_aux = obs_dict["selector_aux"].clone().detach()

        rgb = obs_dict["rgb_image"].clone().detach()
        rgb=sanitize(rgb).float()
        depth = obs_dict["depth_image"].clone().detach()
        depth=sanitize(depth, nan=5.0, posinf=5.0, neginf=0.0).clamp(0.0, 5.0).float()

        rgbd = torch.cat([rgb, depth], dim=-1)
        
        ele_map = obs_dict["elevation_map"].clone().detach()
        ele_map=sanitize(ele_map, nan=0.0, posinf=0.0, neginf=0.0).float()

        cmd = env.command_manager.get_command("base_velocity").clone().detach()
        cmd=sanitize(cmd).float()

        selector_aux = selector_aux / selector_aux.abs().mean()
        latent = policy.forward_latent(proprio, rgbd, ele_map, cmd)
        R = policy.objective_selector(latent, cmd, proprio, selector_aux)

        # Forward 실행
        raw_actions = policy(proprio, rgbd, ele_map, cmd,R)
        raw_actions = torch.clamp(raw_actions, min=-1.0, max=1.0)

        dist = torch.distributions.Normal(raw_actions, 0.1)
        actions=dist.rsample()
        actions = torch.clamp(actions, min=-1.1, max=1.1)


        # 환경 업데이트 및 보상 수집
        obs_dict, rewards, terminated, truncated, extras = env.step(actions.detach())#env.step(actions)
        


        # [핵심] 개별 보상 계산 및 가중치 결합 (미분 경로 생성)
        r_s = mdp.reward_stability(env)
        r_v = mdp.reward_velocity(env)
        r_e = mdp.reward_energy(env)

        r_s_n = normalize_reward(r_s)
        r_v_n = normalize_reward(r_v)
        r_e_n = normalize_reward(r_e)

        #latent = policy.forward_latent(proprio, rgbd, ele_map, cmd)
        #R = policy.objective_selector(torch.cat([latent, cmd], dim=-1)) # [num_envs, 3]

        

        #weighted_reward = R[:, 0] * r_s + R[:, 1] * r_v + R[:, 2] * r_e
        weighted_reward = R[:, 0] * r_s_n + R[:, 1] * r_v_n + R[:, 2] * r_e_n
        

        reward_scale=1.0 #1e-3
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = -torch.sum(R * torch.log(R + 1e-8), dim=-1).mean()
        loss = -(log_prob * weighted_reward.detach()).mean() * reward_scale - 0.01 * entropy


        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        # old_actions = actions.detach()


        # [데이터 기록] 100 스텝마다 텐서보드에 저장
        if env.common_step_counter % 100 == 0:
            step = env.common_step_counter
            
            # 1. 기본 훈련 지표 기록
            writer.add_scalar("Train/Loss", loss.item(), step)
            writer.add_scalar("Train/Avg_Reward", rewards.mean().item(), step)
            
            # 2. Figure 175: Objective Selector 가중치(R) 기록
            with torch.no_grad():
                #latent = policy.forward_latent(proprio, rgbd, ele_map, cmd)
                #R_weights = policy.objective_selector(torch.cat([latent, cmd], dim=-1))[0]
                latent = policy.forward_latent(proprio, rgbd, ele_map, cmd)
                R_weights = policy.objective_selector(latent, cmd, proprio, selector_aux)[0]
                
                writer.add_scalar("R_Weights/Stability", R_weights[0].item(), step)
                writer.add_scalar("R_Weights/Velocity", R_weights[1].item(), step)
                writer.add_scalar("R_Weights/Energy", R_weights[2].item(), step)
                writer.add_scalar("Component_Reward/Stability", r_s.mean().item(), env.common_step_counter)
                writer.add_scalar("Component_Reward/Velocity", r_v.mean().item(), env.common_step_counter)
                writer.add_scalar("Component_Reward/Energy", r_e.mean().item(), env.common_step_counter)

            print(f"Step: {step} | Data logged to TensorBoard")
        
        if env.common_step_counter % 1000 == 0:
            torch.save(policy.state_dict(), f"logs/model_{env.common_step_counter}.pt")

    env.close()

if __name__ == "__main__":
    main()