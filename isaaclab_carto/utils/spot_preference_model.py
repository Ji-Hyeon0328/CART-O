from __future__ import annotations

import torch
import torch.nn as nn

from isaaclab_carto.networks.spot_highlevel_encoder import SpotHighLevelEncoder


class SpotPreferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SpotHighLevelEncoder()
        context_dim = self.encoder.context_dim

        # scalar reward / preference score
        self.reward_head = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # optional beta head (later)
        self.beta_head = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def encode_window(self, proprio_seq: torch.Tensor, depth_seq: torch.Tensor):
        out = self.encoder(proprio_seq, depth_seq)
        context_last = out["context_last"] # (B, D)
        reward_score = self.reward_head(context_last).squeeze(-1) # (B,)
        beta_logit = self.beta_head(context_last) # (B, 3)
        beta = torch.softmax(beta_logit, dim=-1)
        return {
            "context_last": context_last,
            "reward_score": reward_score,
            "beta": beta,
        }

    def forward(
        self,
        better_proprio: torch.Tensor,
        better_depth: torch.Tensor,
        worse_proprio: torch.Tensor,
        worse_depth: torch.Tensor,
    ):
        better = self.encode_window(better_proprio, better_depth)
        worse = self.encode_window(worse_proprio, worse_depth)

        return {
            "better_reward": better["reward_score"],
            "worse_reward": worse["reward_score"],
            "better_beta": better["beta"],
            "worse_beta": worse["beta"],
            "better_context": better["context_last"],
            "worse_context": worse["context_last"],
        }


if __name__ == "__main__":
    B, T = 2, 20
    better_proprio = torch.randn(B, T, 58)
    better_depth = torch.randn(B, T, 1, 480, 640)
    worse_proprio = torch.randn(B, T, 58)
    worse_depth = torch.randn(B, T, 1, 480, 640)

    model = SpotPreferenceModel()
    out = model(
        better_proprio, better_depth,
        worse_proprio, worse_depth,
    )

    print("better_reward:", out["better_reward"].shape)
    print("worse_reward:", out["worse_reward"].shape)
    print("better_beta:", out["better_beta"].shape)
    print("worse_beta:", out["worse_beta"].shape)