from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def main():
    csv_path = Path(
        "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2/joint_highlevel_analysis/joint_highlevel_analysis.csv"
    ).expanduser()

    out_dir = csv_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    print("Loaded:", csv_path)

    # =========================================================
    # 1. beta mean per z
    # =========================================================
    beta_cols = ["better_beta_v", "better_beta_h", "better_beta_e"]

    beta_mean = df.groupby("z_pred")[beta_cols].mean().reset_index()

    plt.figure(figsize=(6, 4))
    beta_mean.set_index("z_pred").plot(kind="bar")
    plt.title("Beta mean per z")
    plt.ylabel("weight")
    plt.xlabel("z")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "beta_mean_per_z.png")
    plt.close()

    # =========================================================
    # 2. objective mean per z
    # =========================================================
    obj_cols = ["better_J_v", "better_J_h", "better_J_e"]

    obj_mean = df.groupby("z_pred")[obj_cols].mean().reset_index()

    plt.figure(figsize=(6, 4))
    obj_mean.set_index("z_pred").plot(kind="bar")
    plt.title("Objective mean per z")
    plt.ylabel("value")
    plt.xlabel("z")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "objective_mean_per_z.png")
    plt.close()

    # =========================================================
    # 3. beta scatter (v vs e)
    # =========================================================
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=df,
        x="better_beta_v",
        y="better_beta_e",
        hue="z_pred",
        palette="tab10",
        alpha=0.6,
    )
    plt.title("Beta scatter (v vs e)")
    plt.tight_layout()
    plt.savefig(out_dir / "beta_scatter_v_e.png")
    plt.close()

    # =========================================================
    # 4. beta scatter (v vs h)
    # =========================================================
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=df,
        x="better_beta_v",
        y="better_beta_h",
        hue="z_pred",
        palette="tab10",
        alpha=0.6,
    )
    plt.title("Beta scatter (v vs h)")
    plt.tight_layout()
    plt.savefig(out_dir / "beta_scatter_v_h.png")
    plt.close()

    # =========================================================
    # 5. reward margin distribution
    # =========================================================
    plt.figure(figsize=(6, 4))
    sns.histplot(
        data=df,
        x="reward_margin",
        hue="z_pred",
        bins=50,
        kde=True,
        palette="tab10",
    )
    plt.title("Reward margin distribution by z")
    plt.tight_layout()
    plt.savefig(out_dir / "reward_margin_hist.png")
    plt.close()

    print("\nSaved plots to:", out_dir)


if __name__ == "__main__":
    main()