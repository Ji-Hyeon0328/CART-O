from __future__ import annotations

import torch
import torch.nn as nn

from isaaclab_carto.networks.spot_highlevel_encoder import SpotHighLevelEncoder


class SpotProxyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SpotHighLevelEncoder()
        context_dim = self.encoder.context_dim

        self.vx_head = nn.Linear(context_dim, 1)
        self.height_head = nn.Linear(context_dim, 1)
        self.contact_head = nn.Linear(context_dim, 4)

    def forward(self, proprio_seq: torch.Tensor, depth_seq: torch.Tensor):
        enc = self.encoder(proprio_seq, depth_seq)
        context_seq = enc["context_seq"]   # (B, T, D)

        vx_pred = self.vx_head(context_seq)              # (B, T, 1)
        height_pred = self.height_head(context_seq)      # (B, T, 1)
        contact_logit = self.contact_head(context_seq)   # (B, T, 4)

        return {
            "context_seq": context_seq,
            "context_last": enc["context_last"],
            "vx_pred": vx_pred,
            "height_pred": height_pred,
            "contact_logit": contact_logit,
        }


if __name__ == "__main__":
    B, T = 2, 20
    proprio = torch.randn(B, T, 58)
    depth = torch.randn(B, T, 1, 480, 640)

    model = SpotProxyModel()
    out = model(proprio, depth)

    print("vx_pred:", out["vx_pred"].shape)
    print("height_pred:", out["height_pred"].shape)
    print("contact_logit:", out["contact_logit"].shape)