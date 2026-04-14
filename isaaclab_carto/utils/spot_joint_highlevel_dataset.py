from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from isaaclab_carto.utils.spot_preference_dataset import SpotPreferenceDataset


class SpotJointHighLevelDataset(Dataset):
    """
    Joint dataset for Option 2 integrated high-level planner.

    Returns:
      - better/worse pair for preference learning
      - objective components for beta-dependent reward
      - z_label for TSS supervision (currently attached to better branch)
    """

    def __init__(
        self,
        data_dir: str | Path,
        scores_csv: str | Path,
        tss_csv: str | Path,
        seq_len: int = 20,
        stride: int = 5,
        use_depth: bool = True,
        top_ratio: float = 0.2,
        bottom_ratio: float = 0.2,
        max_pairs: int = 1000,
    ):
        self.pref_dataset = SpotPreferenceDataset(
            data_dir=data_dir,
            scores_csv=scores_csv,
            seq_len=seq_len,
            stride=stride,
            use_depth=use_depth,
            top_ratio=top_ratio,
            bottom_ratio=bottom_ratio,
            max_pairs=max_pairs,
        )

        tss_df = pd.read_csv(Path(tss_csv).expanduser())

        # better_idx -> z_label 매핑
        self.idx_to_z = {}
        for _, row in tss_df.iterrows():
            better_idx = int(row["better_idx"])
            z_label = int(row["z_label"])
            self.idx_to_z[better_idx] = z_label

    def __len__(self) -> int:
        return len(self.pref_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.pref_dataset[idx]

        better_idx = int(sample["better_idx"])
        worse_idx = int(sample["worse_idx"])

        if better_idx not in self.idx_to_z:
            raise KeyError(f"better_idx={better_idx} not found in TSS label map")

        z_label = self.idx_to_z[better_idx]

        return {
            "better": sample["better"],
            "worse": sample["worse"],
            "label": sample["label"], # preference label
            "better_idx": torch.tensor(better_idx, dtype=torch.long),
            "worse_idx": torch.tensor(worse_idx, dtype=torch.long),
            "z_label": torch.tensor(z_label, dtype=torch.long),
        }


def collate_joint_highlevel(batch):
    better_proprio = torch.stack([b["better"]["proprio"] for b in batch], dim=0)
    better_depth = torch.stack([b["better"]["depth"] for b in batch], dim=0)
    better_obj = torch.stack([b["better"]["objective_components"] for b in batch], dim=0)

    worse_proprio = torch.stack([b["worse"]["proprio"] for b in batch], dim=0)
    worse_depth = torch.stack([b["worse"]["depth"] for b in batch], dim=0)
    worse_obj = torch.stack([b["worse"]["objective_components"] for b in batch], dim=0)

    labels = torch.stack([b["label"] for b in batch], dim=0)
    z_label = torch.stack([b["z_label"] for b in batch], dim=0)

    better_idx = torch.stack([b["better_idx"] for b in batch], dim=0)
    worse_idx = torch.stack([b["worse_idx"] for b in batch], dim=0)

    return {
        "better": {
            "proprio": better_proprio,
            "depth": better_depth,
            "objective_components": better_obj,
        },
        "worse": {
            "proprio": worse_proprio,
            "depth": worse_depth,
            "objective_components": worse_obj,
        },
        "label": labels,
        "z_label": z_label,
        "better_idx": better_idx,
        "worse_idx": worse_idx,
    }


if __name__ == "__main__":
    data_dir = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2"
    scores_csv = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2/pseudo_expert/window_scores.csv"
    tss_csv = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2/tss_labels/beta_analysis_with_z.csv"

    dataset = SpotJointHighLevelDataset(
        data_dir=data_dir,
        scores_csv=scores_csv,
        tss_csv=tss_csv,
        seq_len=20,
        stride=5,
        use_depth=True,
        top_ratio=0.2,
        bottom_ratio=0.2,
        max_pairs=1000,
    )

    print("len(dataset) =", len(dataset))
    sample = dataset[0]

    print("better proprio:", sample["better"]["proprio"].shape)
    print("better depth:", sample["better"]["depth"].shape)
    print("better objective_components:", sample["better"]["objective_components"].shape)

    print("worse proprio:", sample["worse"]["proprio"].shape)
    print("worse depth:", sample["worse"]["depth"].shape)
    print("worse objective_components:", sample["worse"]["objective_components"].shape)

    print("label:", sample["label"])
    print("z_label:", sample["z_label"])
    print("better_idx:", sample["better_idx"])
    print("worse_idx:", sample["worse_idx"])