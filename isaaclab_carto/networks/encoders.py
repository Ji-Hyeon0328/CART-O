import torch
import torch.nn as nn

##
# 1. Visual Encoder (Sv): RGB-D 또는 Height Scan 처리
##
class VisualEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super(VisualEncoder, self). __init__()
        # 논문 수식 (1) 기반: CNN 구조
        # 입력: [Batch, 4, H, W] (RGB 3채널 + Depth 1채널)
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, feature_dim) # 입력 이미지 크기에 따라 조정 필요
        )

    def forward(self, x):
        return self.net(x)

##
# 2. Surface Encoder (Sm): SEL 기반 지형 메시 처리
##
class SurfaceEncoder(nn.Module):
    def __init__(self, input_dim=128, feature_dim=256):
        super(SurfaceEncoder, self). __init__()
        # 논문 수식 (2) 기반: MLP 구조 (Friction Mesh Encoding)
        # 입력: Flattened terrain mesh vertices [Batch, 128]
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

##
# 3. Proprioceptive Encoder (Sp): BiLSTM 기반 시계열 처리
##
class ProprioEncoder(nn.Module):
    def __init__(self, input_dim=36, hidden_dim=128, feature_dim=256):
        super(ProprioEncoder, self). __init__()
        # 논문 수식 (3) 기반: BiLSTM 구조
        # 입력: [Batch, Horizon(10), Proprio_Dim(36)]
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        # 양방향 hidden state를 합쳐서 최종 특징량 생성
        self.fc = nn.Linear(hidden_dim * 2, feature_dim)

    def forward(self, x):
        # x: [Batch, 10, 36]
        lstm_out, (h_n, c_n) = self.bilstm(x)
        
        # h_n shape: [num_layers * num_directions, batch, hidden_dim]
        # 마지막 타임스텝의 forward(0)와 backward(1) hidden state 결합
        h_combined = torch.cat([h_n[0], h_n[1]], dim=-1)
        
        s_p = self.fc(h_combined)
        # h_combined는 나중에 Attention의 Query(h_t)로도 사용될 수 있습니다.
        return s_p, h_combined