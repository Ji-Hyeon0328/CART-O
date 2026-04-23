from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthCNN(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),   # 480x640 -> 240x320
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # -> 120x160
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # -> 60x80
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # -> 30x40
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, H, W)
        feat = self.conv(x)              # (N, 64, 1, 1)
        feat = feat.flatten(1)           # (N, 64)
        feat = self.fc(feat)             # (N, out_dim)
        return feat


class ProprioMLP(nn.Module):
    def __init__(self, in_dim: int = 58, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 58)
        return self.net(x)


class SpotHighLevelEncoder(nn.Module):
    def __init__(
        self,
        proprio_dim: int = 58,
        depth_feat_dim: int = 128,
        proprio_feat_dim: int = 128,
        fusion_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 1,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.depth_encoder = DepthCNN(out_dim=depth_feat_dim)
        self.proprio_encoder = ProprioMLP(in_dim=proprio_dim, out_dim=proprio_feat_dim)

        self.fusion = nn.Sequential(
            nn.Linear(depth_feat_dim + proprio_feat_dim, fusion_dim),
            nn.ReLU(),
        )

        self.temporal = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        context_dim = hidden_dim * (2 if bidirectional else 1)
        self.context_dim = context_dim

    def forward(self, proprio_seq: torch.Tensor, depth_seq: torch.Tensor):
        """
        proprio_seq: (B, T, 58)
        depth_seq:   (B, T, 1, H, W)
        """
        B, T, _, H, W = depth_seq.shape

        depth_flat = depth_seq.reshape(B * T, 1, H, W)
        proprio_flat = proprio_seq.reshape(B * T, -1)

        depth_feat = self.depth_encoder(depth_flat)         # (B*T, Dd)
        proprio_feat = self.proprio_encoder(proprio_flat)   # (B*T, Dp)

        fused = torch.cat([depth_feat, proprio_feat], dim=-1)
        fused = self.fusion(fused)                          # (B*T, fusion_dim)
        fused = fused.reshape(B, T, -1)                    # (B, T, fusion_dim)

        context_seq, (h_n, c_n) = self.temporal(fused)     # (B, T, context_dim)

        context_last = context_seq[:, -1]                  # (B, context_dim)

        return {
            "context_seq": context_seq,
            "context_last": context_last,
        }


if __name__ == "__main__":
    B, T = 2, 20
    proprio = torch.randn(B, T, 58)
    depth = torch.randn(B, T, 1, 480, 640)

    model = SpotHighLevelEncoder()
    out = model(proprio, depth)

    print("context_seq shape:", out["context_seq"].shape)
    print("context_last shape:", out["context_last"].shape)