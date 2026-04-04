import os
import sys
import argparse
import torch
import torch.nn as nn
from typing import Dict
from isaaclab.app import AppLauncher
from isaaclab_carto.networks.tss import TSSModule

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if SOURCE_DIR not in sys.path:
    sys.path.append(SOURCE_DIR)

# parser = argparse.ArgumentParser()
# AppLauncher.add_app_launcher_args(parser)
# args = parser.parse_args()
# app_launcher = AppLauncher(args)
# simulation_app = app_launcher.app 

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg
#from isaaclab_carto.isaaclab_carto.envs.carto_env_cfg import CartoEnvCfg
from isaaclab_carto.envs.carto_env_cfg import CartoEnvCfg
from isaaclab_carto.networks.objective_selector import ObjectiveSelector

class SpotActor(nn.Module):
    def __init__(self, num_proprio=36, num_map=187, num_actions=12):#num_actions = 12 -> 8
        super().__init__()
        
        # 1. Encoders (Sv, Sm, Sp)
        self.cnn = nn.Sequential(nn.Conv2d(4, 32, 8, 4), nn.ReLU(), nn.Flatten(), nn.Linear(32*31*31, 128))
        self.sel = nn.Sequential(nn.Linear(num_map, 128), nn.ReLU(), nn.Linear(128, 64))
        self.bilstm = nn.LSTM(num_proprio, 64, batch_first=True, bidirectional=True)
        
        # 2. Attention (c_t 생성)
        self.cmd_proj = nn.Linear(3, 128)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        
        # 3. Objective Selector (R 결정)
        # 입력: [Sv, Sm, Sp, ct] + Cmd = 451차원
        # self.objective_selector = nn.Sequential(
        #     nn.Linear(451, 256), nn.ReLU(),
        #     nn.Linear(256, 3), nn.Softmax(dim=-1) # R: 보상 가중치 벡터
        # )
        self.objective_selector = ObjectiveSelector(
            context_dim=448,   # forward_latent가 반환하는 차원
            cmd_dim=3,
            state_dim=num_proprio,   # 현재 36
            aux_dim=5,               # 현재 selector_aux: friction 4 + height 1 + slip_summary 2 ->5
            output_dim=3,
            temperature=1.5,
            min_weight=0.02,
        )
        # 4. TSS (sequence-based mode selection)
        self.num_sequences = 8
        self.theta_dim = 128

        self.tss = TSSModule(
            theta_dim=self.theta_dim,
            state_dim=num_proprio,   # 36
            context_dim=448,         # forward_latent output dim
            hidden_dim=128,
        )

        # learnable sequence library (논문의 S_theta를 latent prototype으로 근사)
        self.sequence_library = nn.Parameter(
            torch.randn(self.num_sequences, self.theta_dim) * 0.02
        )
        
        # 5. [핵심] RL Block (최종 pi_theta* 생성)
        # 설계도상 입력 결합: theta* + R + ct + Cmd
        # 차원: 128(theta*) + 3(R) + 128(ct) + 3(Cmd) = 262차원
        self.rl_block = nn.Sequential(
            nn.Linear(262, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, num_actions) # 최종 관절 토크/위치 명령
        )

    def forward(self, p_t, rgbd, ele_map, cmd, R, z_t=None):
        # A. Feature Extraction & Attention
        s_v = self.cnn(rgbd.permute(0, 3, 1, 2).float() / 255.0)
        s_m = self.sel(ele_map)
        h_t_seq, _ = self.bilstm(p_t.unsqueeze(1))
        s_p = h_t_seq.squeeze(1)
        
        c_t_raw, _ = self.attention(self.cmd_proj(cmd).unsqueeze(1), s_p.unsqueeze(1), s_p.unsqueeze(1))
        c_t = c_t_raw.squeeze(1)
        
        # # B. Objective Selection (R)
        # # 설계도 Circle: [Sv, Sm, Sp, ct] 융합 데이터 활용
        # latent_features = torch.cat([s_v, s_m, s_p, c_t], dim=-1)
        # R = self.objective_selector(torch.cat([latent_features, cmd], dim=-1))
        
        # # C. TSS (theta*)
        # # 논문에서 TSS는 사후 선택이지만, 설계도 구조상 '파라미터 결정기'로 구현
        # theta_star = self.tss(torch.cat([latent_features, R], dim=-1))
        latent_features = torch.cat([s_v, s_m, s_p, c_t], dim=-1)

        if z_t is None:
            best_seq_indices, scores = self.tss(
                sequence_library=self.sequence_library,
                s_t=p_t,
                c_t=latent_features,
            )
            theta_star = self.sequence_library[best_seq_indices]
        else:
            theta_star = z_t

        rl_input = torch.cat([theta_star, R, c_t, cmd], dim=-1)
        actions = self.rl_block(rl_input)
        return actions
    
    def forward_latent(self, p_t, rgbd, ele_map, cmd):
        """Figure out the latent feature"""
        # A. Feature Extraction
        s_v = self.cnn(rgbd.permute(0, 3, 1, 2).float() / 255.0)
        s_m = self.sel(ele_map)
        h_t_seq, _ = self.bilstm(p_t.unsqueeze(1))
        s_p = h_t_seq.squeeze(1)
        
        # B. Attention Mechanism
        c_t_raw, _ = self.attention(self.cmd_proj(cmd).unsqueeze(1), s_p.unsqueeze(1), s_p.unsqueeze(1))
        c_t = c_t_raw.squeeze(1)
        
        # C. 융합 (448차원: 128 + 64 + 128 + 128)
        return torch.cat([s_v, s_m, s_p, c_t], dim=-1)
    
    def select_sequence(self, p_t, rgbd, ele_map, cmd):
        """
        TSS가 현재 proprio/context에 맞는 sequence index와 z_t를 고른다.
        """
        latent_features = self.forward_latent(p_t, rgbd, ele_map, cmd)   # [N, 448]

        best_seq_indices, scores = self.tss(
            sequence_library=self.sequence_library,   # [K, 128]
            s_t=p_t,                                  # [N, 36]
            c_t=latent_features,                      # [N, 448]
        )
        z_t = self.sequence_library[best_seq_indices]  # [N, 128]
        return z_t, best_seq_indices, scores, latent_features

def main():
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    simulation_app = launcher.app

    env_cfg = CartoEnvCfg()
    env_cfg.scene.num_envs = 4 
    env = ManagerBasedRLEnv(cfg=env_cfg)
     
    policy = SpotActor(num_proprio=36, num_map=187, num_actions=12).to(env.device)
    policy.eval()

    teleop_interface = Se2Keyboard(Se2KeyboardCfg())
    obs_dict, _ = env.reset() 

    print("-" * 50)
    print("[INFO]: Encoder-Planner mode")
    print("-" * 50)

    while simulation_app.is_running():
        with torch.inference_mode():
            # (1) 키보드 명령 주입 (3차원 정밀 매칭)
            delta_pose = teleop_interface.advance() 
            targets = torch.as_tensor(delta_pose, device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
            env.command_manager._terms["base_velocity"].command[:] = targets

            # (2) 관측값 수집 및 정밀 슬라이싱
            obs_dict: Dict[str, torch.Tensor] = env.observation_manager.compute()
            # 데이터 추출
            proprio = obs_dict["policy"][:, :36] # 36차원 신체 정보
            rgb = obs_dict["rgb_image"]   # 이미지 데이터 
            depth = obs_dict["depth_image"] # depth data
            rgbd = torch.cat([rgb, depth], dim=-1)
            ele_map = obs_dict["elevation_map"] # 지형 데이터
            
            # 3. 인코더 기반 정책 실행
            actions = policy(proprio, rgbd, ele_map, targets)
            env.step(actions)
            
            '''           
            # 카메라 데이터가 제대로 들어오는지 터미널에 shape 출력
            if "rgb_image" in obs_dict:
                rgb_data = obs_dict["rgb_image"] # [num_envs, height, width, 3]
                print(f"Camera RGB Shape: {rgb_data.shape}") 
                
            # 지형 맵 데이터 확인
            if "elevation_map" in obs_dict:
                map_data = obs_dict["elevation_map"]
                print(f"Elevation Map Shape: {map_data.shape}")
            '''

    env.close()

if __name__ == "__main__":
    main()