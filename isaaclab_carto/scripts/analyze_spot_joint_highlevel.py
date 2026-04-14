from __future__ import annotations

from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from isaaclab_carto.utils.spot_joint_highlevel_dataset import (
    SpotJointHighLevelDataset,
    collate_joint_highlevel,
)
from isaaclab_carto.networks.spot_joint_highlevel_model import SpotJointHighLevelModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    data_dir = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2"
    scores_csv = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2/pseudo_expert/window_scores.csv"
    tss_csv = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2/tss_labels/beta_analysis_with_z.csv"

    # joint model checkpoint 경로 확인
    ckpt_path = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2/joint_highlevel_checkpoints/spot_joint_highlevel_best.pt"

    out_dir = Path(
        "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2/joint_highlevel_analysis"
    ).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

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

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_joint_highlevel,
    )

    model = SpotJointHighLevelModel(num_modes=2, freeze_encoder=False).to(device)

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

            z_label = batch["z_label"].to(device)

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

            better_z_logits = out["better_z_logits"].cpu()
            better_z_pred = torch.argmax(better_z_logits, dim=-1)

            better_idx = batch["better_idx"].cpu()
            worse_idx = batch["worse_idx"].cpu()

            better_obj_cpu = batch["better"]["objective_components"].cpu()
            worse_obj_cpu = batch["worse"]["objective_components"].cpu()
            z_label_cpu = z_label.cpu()

            B = better_reward.shape[0]
            for i in range(B):
                rows.append({
                    "better_idx": int(better_idx[i].item()),
                    "worse_idx": int(worse_idx[i].item()),

                    "z_true": int(z_label_cpu[i].item()),
                    "z_pred": int(better_z_pred[i].item()),

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
    out_csv = out_dir / "joint_highlevel_analysis.csv"
    df.to_csv(out_csv, index=False)

    # 기본 정확도
    acc = (df["z_true"] == df["z_pred"]).mean()

    print("\n=== saved ===")
    print(out_csv)

    print("\n=== overall summary ===")
    print(df.describe())

    print("\n=== z prediction accuracy ===")
    print(f"z_acc = {acc:.4f}")

    print("\n=== beta mean by predicted z ===")
    beta_by_z = df.groupby("z_pred")[["better_beta_v", "better_beta_h", "better_beta_e"]].mean()
    print(beta_by_z)

    print("\n=== objective mean by predicted z ===")
    obj_by_z = df.groupby("z_pred")[["better_J_v", "better_J_h", "better_J_e"]].mean()
    print(obj_by_z)

    print("\n=== reward mean by predicted z ===")
    reward_by_z = df.groupby("z_pred")[["better_reward", "reward_margin"]].mean()
    print(reward_by_z)

    print("\n=== correlations: beta vs objective ===")
    corr_cols = [
        "better_beta_v", "better_beta_h", "better_beta_e",
        "better_J_v", "better_J_h", "better_J_e",
        "reward_margin",
    ]
    print(df[corr_cols].corr())

    print("\n=== top 10 reward margins ===")
    print(df.sort_values("reward_margin", ascending=False).head(10).to_string(index=False))

    print("\n=== bottom 10 reward margins ===")
    print(df.sort_values("reward_margin", ascending=True).head(10).to_string(index=False))


if __name__ == "__main__":
    main()