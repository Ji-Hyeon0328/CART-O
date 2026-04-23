from __future__ import annotations

import torch
import torch.nn as nn

from isaaclab_carto.networks.spot_highlevel_encoder import SpotHighLevelEncoder


class SpotTSSModel(nn.Module):
    def __init__(self, num_modes: int = 2, freeze_encoder: bool = True):
        super().__init__()

        self.encoder = SpotHighLevelEncoder()
        context_dim = self.encoder.context_dim

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_modes),
        )

    def forward(self, proprio: torch.Tensor, depth: torch.Tensor):
        out = self.encoder(proprio, depth)
        context_last = out["context_last"] # (B, D)
        logits = self.classifier(context_last) # (B, K)

        return {
            "context_last": context_last,
            "logits": logits,
        }


if __name__ == "__main__":
    B, T = 2, 20
    proprio = torch.randn(B, T, 58)
    depth = torch.randn(B, T, 1, 480, 640)

    model = SpotTSSModel(num_modes=2, freeze_encoder=True)
    out = model(proprio, depth)

    print("context_last:", out["context_last"].shape)
    print("logits:", out["logits"].shape)

