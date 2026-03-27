import torch
import torch.nn as nn
import torch.nn.functional as F

class TSSModule(nn.Module):
    def __init__(self, theta_dim=512, state_dim=256, context_dim=256, hidden_dim=128):
        """
        논문 수식 (7)~(9)를 구현한 TSS 모듈
        
        Args:
            theta_dim: 정책 파라미터 시퀀스의 임베딩 차원
            state_dim: 현재 관측값(S_t)의 차원
            context_dim: 컨텍스트 벡터(c_t)의 차원
        """
        super(TSSModule, self).__init__()
        
        # 1. Parameter Embedding Network (phi)
        # 정책 시퀀스를 잠재 공간으로 투영
        self.phi = nn.Sequential(
            nn.Linear(theta_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. Context Similarity Scorer (W_c)
        # [phi(theta); S_v; S_p; c_t]를 입력받아 유사도 점수 산출
        # 입력: phi(128) + S_t(256) + c_t(256) = 640
        input_dim = hidden_dim + state_dim + context_dim
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def compute_context_score(self, theta_seq, s_t, c_t):
        """
        논문 수식 (8): C(theta, S_v, S_p) = tanh(W_c[phi(theta); S_v; S_p])
        """
        # 파라미터 시퀀스 임베딩
        theta_emb = self.phi(theta_seq)
        
        # 모든 정보 결합
        combined = torch.cat([theta_emb, s_t, c_t], dim=-1)
        
        # 유사도 점수 계산
        score = torch.tanh(self.scorer(combined))
        return score

    def forward(self, sequence_library, s_t, c_t):
        """
        논문 수식 (9): 최적의 시퀀스 theta* 선택
        
        Args:
            sequence_library: [Num_Sequences, theta_dim]
            s_t: 현재 상태 [Batch, state_dim]
            c_t: 현재 컨텍스트 [Batch, context_dim]
        """
        batch_size = s_t.size(0)
        num_seqs = sequence_library.size(0)
        
        # 각 배치별로 라이브러리의 모든 시퀀스에 대해 점수 계산
        # (간략화를 위해 브로드캐스팅 활용)
        s_t_exp = s_t.unsqueeze(1).repeat(1, num_seqs, 1) # [Batch, N, state_dim]
        c_t_exp = c_t.unsqueeze(1).repeat(1, num_seqs, 1) # [Batch, N, context_dim]
        lib_exp = sequence_library.unsqueeze(0).repeat(batch_size, 1, 1) # [Batch, N, theta_dim]
        
        # [Batch, N, total_input] 형태로 결합하여 점수 산출
        theta_emb = self.phi(lib_exp)
        combined = torch.cat([theta_emb, s_t_exp, c_t_exp], dim=-1)
        scores = torch.tanh(self.scorer(combined)).squeeze(-1) # [Batch, N]
        
        # 가장 높은 점수를 가진 시퀀스의 인덱스 선택
        best_seq_indices = torch.argmax(scores, dim=-1)
        
        return best_seq_indices, scores