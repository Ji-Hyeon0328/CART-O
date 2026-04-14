from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from isaaclab_carto.utils.spot_carto_dataset import SpotCartoDataset


class SpotTSSDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        tss_csv: str | Path,
        seq_len: int = 20,
        stride: int = 5,
        use_depth: bool = True,
        use_better_only: bool = True,
    ):
        self.base_dataset = SpotCartoDataset(
            data_dir=data_dir,
            seq_len=seq_len,
            stride=stride,
            use_rgb=False,
            use_depth=use_depth,
        )

        self.df = pd.read_csv(Path(tss_csv).expanduser())
        self.use_better_only = use_better_only

        # better branch만 써서 context -> z 분류를 먼저 배움
        if self.use_better_only:
            self.rows = self.df[["better_idx", "z_label"]].copy()
            self.rows = self.rows.rename(columns={"better_idx": "sample_idx"})
        else:
            # 필요하면 나중에 better/worse 둘 다 쓰도록 확장 가능
            rows_b = self.df[["better_idx", "z_label"]].copy().rename(columns={"better_idx": "sample_idx"})
            rows_w = self.df[["worse_idx", "z_label"]].copy().rename(columns={"worse_idx": "sample_idx"})
            self.rows = pd.concat([rows_b, rows_w], ignore_index=True)

        self.rows = self.rows.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows.iloc[idx]
        sample_idx = int(row["sample_idx"])
        z_label = int(row["z_label"])

        sample = self.base_dataset[sample_idx]

        return {
            "proprio": sample["proprio"], # (T, 58)
            "depth": sample["depth"], # (T, 1, H, W)
            "z_label": torch.tensor(z_label, dtype=torch.long),
            "sample_idx": torch.tensor(sample_idx, dtype=torch.long),
            "objective_components": sample["objective_components"], # optional, 분석용
        }


def collate_tss(batch):
    proprio = torch.stack([b["proprio"] for b in batch], dim=0) # (B, T, 58)
    depth = torch.stack([b["depth"] for b in batch], dim=0) # (B, T, 1, H, W)
    z_label = torch.stack([b["z_label"] for b in batch], dim=0) # (B,)
    sample_idx = torch.stack([b["sample_idx"] for b in batch], dim=0)
    objective_components = torch.stack([b["objective_components"] for b in batch], dim=0)

    return {
        "proprio": proprio,
        "depth": depth,
        "z_label": z_label,
        "sample_idx": sample_idx,
        "objective_components": objective_components,
    }


if __name__ == "__main__":
    data_dir = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2"
    tss_csv = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2/tss_labels/beta_analysis_with_z.csv"

    dataset = SpotTSSDataset(
        data_dir=data_dir,
        tss_csv=tss_csv,
        seq_len=20,
        stride=5,
        use_depth=True,
        use_better_only=True,
    )

    print("len(dataset) =", len(dataset))
    sample = dataset[0]
    print("proprio:", sample["proprio"].shape)
    print("depth:", sample["depth"].shape)
    print("z_label:", sample["z_label"])
    print("sample_idx:", sample["sample_idx"])
    print("objective_components:", sample["objective_components"])