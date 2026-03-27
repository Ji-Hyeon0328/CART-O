import torch

class HistoryBuffer:
    def __init__(self, num_envs: int, horizon: int, feature_dim: int, device: str = "cuda"):
        """
        buffer for states in history.
        
        Args:
            num_envs: number of envs (Batch size)
            horizon: length of past steps (Sequence length, such as: 10)
            feature_dim: dim of feature vector (ex: 36)
            device: computation device
        """
        self.num_envs = num_envs
        self.horizon = horizon
        self.feature_dim = feature_dim
        self.device = device
        
        # init as 0 tensor sized: [Batch, Horizon, Feature_Dim] 
        self.buffer = torch.zeros((num_envs, horizon, feature_dim), device=device)

    def update(self, current_state: torch.Tensor):
        """
        adding new states on buffer - Sliding Window
        current_state: [Batch, Feature_Dim]
        """
        # 기존 데이터를 한 칸씩 왼쪽(과거)으로 밉니다.
        self.buffer[:, :-1, :] = self.buffer[:, 1:, :].clone()
        
        # 가장 마지막 칸(최신)에 현재 상태를 넣습니다.
        self.buffer[:, -1, :] = current_state

    def get_history(self):
        """
        BiLSTM의 입력으로 사용할 전체 시퀀스 반환
        Returns: [Batch, Horizon, Feature_Dim]
        """
        return self.buffer

    def reset(self, env_ids: torch.Tensor = None):
        """
        로봇이 넘어지거나 에피소드가 끝났을 때 해당 환경의 버퍼를 초기화
        """
        if env_ids is None:
            self.buffer.zero_()
        else:
            self.buffer[env_ids] = 0.0