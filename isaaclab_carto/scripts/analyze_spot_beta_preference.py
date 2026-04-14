from __future__ import annotations

from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from isaaclab_carto.utils.spot_preference_dataset import (
    SpotPreferenceDataset,
    collate_preference,
)
from isaaclab_carto.utils.spot_beta_preference_model import SpotBetaPreferenceModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    data_dir = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2"
    scores_csv = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2/pseudo_expert/window_scores.csv"
    ckpt_path = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2/preference_checkpoints/spot_preference_best.pt"

    out_dir = Path("~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2/beta_analysis").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

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

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_preference,
    )

    model = SpotBetaPreferenceModel().to(device)

    ckpt = torch.load(Path(ckpt_path).expanduser(), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = []

    with torch.no_grad():
        for batch in loader:
            better_proprio = batch["better"]["proprio"].to(device)
            better_depth = batch["better"]["depth"].to(device)
            better_obj = batch["better"]["objective_components"].to(device)

            worse_proprio = batch["worse"]["proprio"].to(device)
            worse_depth = batch["worse"]["depth"].to(device)
            worse_obj = batch["worse"]["objective_components"].to(device)

            out = model(
                better_proprio=better_proprio,
                better_depth=better_depth,
                better_obj=better_obj,
                worse_proprio=worse_proprio,
                worse_depth=worse_depth,
                worse_obj=worse_obj,
            )

            better_reward = out["better_reward"].cpu()
            worse_reward = out["worse_reward"].cpu()
            better_beta = out["better_beta"].cpu()
            worse_beta = out["worse_beta"].cpu()

            better_idx = batch["better_idx"].cpu()
            worse_idx = batch["worse_idx"].cpu()

            better_obj_cpu = batch["better"]["objective_components"].cpu()
            worse_obj_cpu = batch["worse"]["objective_components"].cpu()

            B = better_reward.shape[0]
            for i in range(B):
                rows.append({
                    "better_idx": int(better_idx[i].item()),
                    "worse_idx": int(worse_idx[i].item()),

                    "better_reward": float(better_reward[i].item()),
                    "worse_reward": float(worse_reward[i].item()),
                    "reward_margin": float((better_reward[i] - worse_reward[i]).item()),

                    "better_beta_v": float(better_beta[i, 0].item()),
                    "better_beta_h": float(better_beta[i, 1].item()),
                    "better_beta_e": float(better_beta[i, 2].item()),

                    "worse_beta_v": float(worse_beta[i, 0].item()),
                    "worse_beta_h": float(worse_beta[i, 1].item()),
                    "worse_beta_e": float(worse_beta[i, 2].item()),

                    "better_J_v": float(better_obj_cpu[i, 0].item()),
                    "better_J_h": float(better_obj_cpu[i, 1].item()),
                    "better_J_e": float(better_obj_cpu[i, 2].item()),

                    "worse_J_v": float(worse_obj_cpu[i, 0].item()),
                    "worse_J_h": float(worse_obj_cpu[i, 1].item()),
                    "worse_J_e": float(worse_obj_cpu[i, 2].item()),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "beta_analysis_pairs.csv", index=False)

    print("\n=== saved ===")
    print(out_dir / "beta_analysis_pairs.csv")

    print("\n=== overall describe ===")
    print(df.describe())

    print("\n=== beta mean (better) ===")
    print(df[["better_beta_v", "better_beta_h", "better_beta_e"]].mean())

    print("\n=== beta std (better) ===")
    print(df[["better_beta_v", "better_beta_h", "better_beta_e"]].std())

    print("\n=== beta mean (worse) ===")
    print(df[["worse_beta_v", "worse_beta_h", "worse_beta_e"]].mean())

    print("\n=== beta std (worse) ===")
    print(df[["worse_beta_v", "worse_beta_h", "worse_beta_e"]].std())

    print("\n=== correlations: better beta vs better objectives ===")
    corr_cols = [
        "better_beta_v", "better_beta_h", "better_beta_e",
        "better_J_v", "better_J_h", "better_J_e",
    ]
    print(df[corr_cols].corr())

    print("\n=== top 10 reward margins ===")
    print(df.sort_values("reward_margin", ascending=False).head(10).to_string(index=False))

    print("\n=== bottom 10 reward margins ===")
    print(df.sort_values("reward_margin", ascending=True).head(10).to_string(index=False))


if __name__ == "__main__":
    main()