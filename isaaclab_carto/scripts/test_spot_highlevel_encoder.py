from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from isaaclab_carto.utils.spot_carto_dataset import SpotCartoDataset
from isaaclab_carto.networks.spot_highlevel_encoder import SpotHighLevelEncoder


def collate_fn(batch):
    proprio = torch.stack([b["proprio"] for b in batch], dim=0)   # (B, T, 58)
    depth = torch.stack([b["depth"] for b in batch], dim=0)       # (B, T, 1, H, W)

    forward_velocity = torch.stack(
        [b["targets"]["forward_velocity"] for b in batch], dim=0
    )
    body_height = torch.stack(
        [b["targets"]["body_height"] for b in batch], dim=0
    )

    return {
        "proprio": proprio,
        "depth": depth,
        "targets": {
            "forward_velocity": forward_velocity,
            "body_height": body_height,
        }
    }


def main():
    data_dir = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2"

    dataset = SpotCartoDataset(
        data_dir=data_dir,
        seq_len=20,
        stride=5,
        use_rgb=False,
        use_depth=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
    )

    batch = next(iter(loader))
    print("proprio batch:", batch["proprio"].shape)
    print("depth batch:", batch["depth"].shape)

    model = SpotHighLevelEncoder()
    out = model(batch["proprio"], batch["depth"])

    print("context_seq:", out["context_seq"].shape)
    print("context_last:", out["context_last"].shape)


if __name__ == "__main__":
    main()