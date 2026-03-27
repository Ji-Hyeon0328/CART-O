import torch
import torch.nn as nn

class CARTOActionPolicy(nn.Module):
    def __init__(self, state_dim=36, context_dim=256, action_dim=4):
        """
        TSS 이후 최종 Action을 결정하는 RL Policy 블록
        
        Args:
            state_dim: 로봇 상태 (36)
            context_dim: Attention에서 온 c_t (256)
            action_dim: [v_x, v_y, yaw_vel, height] (4)
        """
        super(CARTOActionPolicy, self).__init__()
        
        # 입력: 상태(36) + 맥락(256) = 292
        input_dim = state_dim + context_dim
        
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(), # RL에서 자주 쓰이는 활성화 함수
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
            nn.Tanh() # 속도 범위를 -1 ~ 1 사이로 정규화 (이후 스케일링)
        )

    def forward(self, state, c_t):
        """
        Args:
            state: 현재 로봇의 상태 [Batch, 36]
            c_t: Attention 컨텍스트 벡터 [Batch, 256]
        """
        x = torch.cat([state, c_t], dim=-1)
        action = self.actor(x)
        return action