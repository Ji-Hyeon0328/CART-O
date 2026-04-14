from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from isaaclab_carto.utils.spot_carto_dataset import SpotCartoDataset


def build_preference_pairs(
    scores_csv: str | Path,
    top_ratio: float = 0.2,
    bottom_ratio: float = 0.2,
    max_pairs: int = 2000,
    seed: int = 42,
):
    scores_csv = Path(scores_csv).expanduser()
    df = pd.read_csv(scores_csv)

    df = df.sort_values("score_total", ascending=False).reset_index(drop=True)

    n = len(df)
    top_k = max(1, int(n * top_ratio))
    bottom_k = max(1, int(n * bottom_ratio))

    top_df = df.iloc[:top_k].copy()
    bottom_df = df.iloc[-bottom_k:].copy()

    top_indices = top_df["index"].tolist()
    bottom_indices = bottom_df["index"].tolist()

    rng = random.Random(seed)

    pairs = []
    all_candidates = [(i, j) for i in top_indices for j in bottom_indices]
    rng.shuffle(all_candidates)

    for i, j in all_candidates[:max_pairs]:
        pairs.append({
            "better_idx": int(i),
            "worse_idx": int(j),
            "label": 1,
        })

    pair_df = pd.DataFrame(pairs)
    return pair_df, top_df, bottom_df


class SpotPreferenceDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        scores_csv: str | Path,
        seq_len: int = 20,
        stride: int = 5,
        use_depth: bool = True,
        top_ratio: float = 0.2,
        bottom_ratio: float = 0.2,
        max_pairs: int = 2000,
    ):
        self.base_dataset = SpotCartoDataset(
            data_dir=data_dir,
            seq_len=seq_len,
            stride=stride,
            use_rgb=False,
            use_depth=use_depth,
        )

        self.pair_df, self.top_df, self.bottom_df = build_preference_pairs(
            scores_csv=scores_csv,
            top_ratio=top_ratio,
            bottom_ratio=bottom_ratio,
            max_pairs=max_pairs,
        )

    def __len__(self):
        return len(self.pair_df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.pair_df.iloc[idx]

        better_idx = int(row["better_idx"])
        worse_idx = int(row["worse_idx"])

        better = self.base_dataset[better_idx]
        worse = self.base_dataset[worse_idx]

        # return {
        #     "better": better,
        #     "worse": worse,
        #     "label": torch.tensor(1.0, dtype=torch.float32),
        #     "better_idx": better_idx,
        #     "worse_idx": worse_idx,
        # }

        return {
            "better": {
                "proprio": better["proprio"],
                "depth": better["depth"],
                "objective_components": better["objective_components"],
            },
            "worse": {
                "proprio": worse["proprio"],
                "depth": worse["depth"],
                "objective_components": worse["objective_components"],
            },
            "label": torch.tensor(1.0, dtype=torch.float32),
            "better_idx": better_idx,
            "worse_idx": worse_idx,
        }


def collate_preference(batch):
    better_proprio = torch.stack([b["better"]["proprio"] for b in batch], dim=0)
    better_depth = torch.stack([b["better"]["depth"] for b in batch], dim=0)

    worse_proprio = torch.stack([b["worse"]["proprio"] for b in batch], dim=0)
    worse_depth = torch.stack([b["worse"]["depth"] for b in batch], dim=0)

    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)

    better_idx = torch.tensor([b["better_idx"] for b in batch], dtype=torch.long)
    worse_idx = torch.tensor([b["worse_idx"] for b in batch], dtype=torch.long)

    better_obj = torch.stack([b["better"]["objective_components"] for b in batch],dim=0)
    worse_obj = torch.stack([b["worse"]["objective_components"] for b in batch ],dim=0)

    return {
        "better": {
            "proprio": better_proprio,
            "depth": better_depth,
            "objective_components":better_obj,
        },
        "worse": {
            "proprio": worse_proprio,
            "depth": worse_depth,
            "objective_components":worse_obj,
        },
        "label": labels,
        "better_idx": better_idx,
        "worse_idx": worse_idx,
    }


if __name__ == "__main__":
    data_dir = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2"
    scores_csv = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2/pseudo_expert/window_scores.csv"

    dataset = SpotPreferenceDataset(
        data_dir=data_dir,
        scores_csv=scores_csv,
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
    print("worse proprio:", sample["worse"]["proprio"].shape)
    print("worse depth:", sample["worse"]["depth"].shape)
    print("label:", sample["label"])
    print("better_idx:", sample["better_idx"])
    print("worse_idx:", sample["worse_idx"])
    print("better objective shape:",sample["better"]["objective_components"].shape)
    print("worse objective shape:",sample["worse"]["objective_components"].shape)