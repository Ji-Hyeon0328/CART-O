import torch
import torch.nn as nn
import torch.nn.functional as F

class CARTOAttention(nn.Module):
    def __init__(self, cmd_dim=3, h_t_dim=128, encoder_feature_dim=256, internal_dim=128):
        """
        RL 디코더 대신 Cmd와 h_t를 사용하여 Context를 생성하는 CARTO용 Attention
        
        Args:
            cmd_dim: [x_vel, y_vel, yaw_vel] (3차원)
            h_t_dim: BiLSTM의 은닉 상태 차원 (128)
            encoder_feature_dim: h_i (Sv, Sm, Sp 각각의 차원)
        """
        super(CARTOAttention, self).__init__()
        
        # 1. Query Generator: 명령과 현재 동작 상태를 결합하여 '의도(Intent)' 추출
        self.query_gen = nn.Sequential(
            nn.Linear(cmd_dim + h_t_dim, internal_dim),
            nn.ReLU(),
            nn.Linear(internal_dim, internal_dim)
        )
        
        # 2. Attention Wa & v (Bahdanau 스타일)
        self.W_a = nn.Linear(internal_dim + encoder_feature_dim, internal_dim)
        self.v = nn.Linear(internal_dim, 1, bias=False)

    def forward(self, cmd, h_t, encoder_features):
        """
        Args:
            cmd: 로봇 주행 명령 [Batch, 3]
            h_t: BiLSTM의 현재 은닉 상태 [Batch, 128]
            encoder_features: [s_v, s_m, s_p] 특징들 [Batch, 3, 256]
        """
        # 1. Query(q_t) 생성: Cmd와 h_t를 융합
        q_t = self.query_gen(torch.cat([cmd, h_t], dim=-1)) # [Batch, internal_dim]

        # 2. Attention 점수 계산
        num_features = encoder_features.size(1)
        q_t_expanded = q_t.unsqueeze(1).repeat(1, num_features, 1) # [Batch, 3, internal_dim]

        # [q_t; h_i] 결합
        combined = torch.cat([q_t_expanded, encoder_features], dim=-1)
        energy = torch.tanh(self.W_a(combined))
        score = self.v(energy).squeeze(-1) # [Batch, 3]

        # 3. 가중치 및 Context Vector 산출
        alpha_ti = F.softmax(score, dim=-1)
        c_t = torch.bmm(alpha_ti.unsqueeze(1), encoder_features).squeeze(1)

        return c_t, alpha_ti