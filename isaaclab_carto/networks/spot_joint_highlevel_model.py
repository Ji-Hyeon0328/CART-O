from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from isaaclab_carto.networks.spot_highlevel_encoder import SpotHighLevelEncoder


class SpotJointHighLevelModel(nn.Module):
    """
    Option 2 integrated high-level planner

    shared encoder/context
        -> beta_head
        -> z_head

    reward = beta · objective_components
    """

    def __init__(self, num_modes: int = 2, freeze_encoder: bool = False):
        super().__init__()

        self.encoder = SpotHighLevelEncoder()
        context_dim = self.encoder.context_dim

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.beta_head = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

        self.z_head = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_modes),
        )

    def encode(self, proprio: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        out = self.encoder(proprio, depth)
        return out["context_last"] # (B, D)

    def compute_beta(self, context: torch.Tensor) -> torch.Tensor:
        beta_logits = self.beta_head(context) # (B, 3)
        beta = F.softmax(beta_logits, dim=-1) # (B, 3)
        return beta

    def compute_reward(
        self,
        context: torch.Tensor,
        objective_components: torch.Tensor,
    ):
        beta = self.compute_beta(context) # (B, 3)
        reward = (beta * objective_components).sum(dim=-1) # (B,)
        return reward, beta

    def compute_z_logits(self, context: torch.Tensor) -> torch.Tensor:
        return self.z_head(context) # (B, K)

    def forward(
        self,
        better_proprio: torch.Tensor,
        better_depth: torch.Tensor,
        better_obj: torch.Tensor,
        worse_proprio: torch.Tensor,
        worse_depth: torch.Tensor,
        worse_obj: torch.Tensor,
    ):
        better_ctx = self.encode(better_proprio, better_depth)
        worse_ctx = self.encode(worse_proprio, worse_depth)

        better_reward, better_beta = self.compute_reward(better_ctx, better_obj)
        worse_reward, worse_beta = self.compute_reward(worse_ctx, worse_obj)

        better_z_logits = self.compute_z_logits(better_ctx)

        return {
            "better_context": better_ctx,
            "worse_context": worse_ctx,

            "better_reward": better_reward,
            "worse_reward": worse_reward,

            "better_beta": better_beta,
            "worse_beta": worse_beta,

            "better_z_logits": better_z_logits,
        }


if __name__ == "__main__":
    B, T = 2, 20

    better_proprio = torch.randn(B, T, 58)
    better_depth = torch.randn(B, T, 1, 480, 640)
    better_obj = torch.randn(B, 3)

    worse_proprio = torch.randn(B, T, 58)
    worse_depth = torch.randn(B, T, 1, 480, 640)
    worse_obj = torch.randn(B, 3)

    model = SpotJointHighLevelModel(num_modes=2, freeze_encoder=False)
    out = model(
        better_proprio=better_proprio,
        better_depth=better_depth,
        better_obj=better_obj,
        worse_proprio=worse_proprio,
        worse_depth=worse_depth,
        worse_obj=worse_obj,
    )

    print("better_reward:", out["better_reward"].shape)
    print("worse_reward :", out["worse_reward"].shape)
    print("better_beta :", out["better_beta"].shape)
    print("worse_beta :", out["worse_beta"].shape)
    print("better_z_logits:", out["better_z_logits"].shape)