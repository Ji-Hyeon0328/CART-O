import torch
import torch.nn as nn
import torch.nn.functional as F

from isaaclab_carto.networks.spot_highlevel_encoder import SpotHighLevelEncoder


class SpotBetaPreferenceModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = SpotHighLevelEncoder()

        # beta head (3-dim)
        self.beta_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def encode(self, proprio, depth):
        """
        proprio: (B, T, 58)
        depth: (B, T, 1, H, W)
        """
        #context_seq, context_last = self.encoder(proprio, depth)
        out = self.encoder(proprio,depth)
        #return context_last # (B, 512)
        return out["context_last"] # (B,context_dim)

    def compute_reward(self, context, objective_components):
        """
        context: (B, 512)
        objective_components: (B, 3)
        """
        beta_logits = self.beta_head(context) # (B, 3)
        beta = F.softmax(beta_logits, dim=-1) # (B, 3)

        reward = (beta * objective_components).sum(dim=-1) # (B,)

        return reward, beta

    def forward(
        self,
        better_proprio,
        better_depth,
        better_obj,
        worse_proprio,
        worse_depth,
        worse_obj,
    ):
        # encode
        better_ctx = self.encode(better_proprio, better_depth)
        worse_ctx = self.encode(worse_proprio, worse_depth)

        # compute reward
        better_reward, better_beta = self.compute_reward(better_ctx, better_obj)
        worse_reward, worse_beta = self.compute_reward(worse_ctx, worse_obj)

        return {
            "better_reward": better_reward,
            "worse_reward": worse_reward,
            "better_beta": better_beta,
            "worse_beta": worse_beta,
        }
    
    
if __name__ == "__main__":
    B, T = 2, 20
    better_proprio = torch.randn(B, T, 58)
    better_depth = torch.randn(B, T, 1, 480, 640)
    better_obj = torch.randn(B, 3)

    worse_proprio = torch.randn(B, T, 58)
    worse_depth = torch.randn(B, T, 1, 480, 640)
    worse_obj = torch.randn(B, 3)

    model = SpotBetaPreferenceModel()
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