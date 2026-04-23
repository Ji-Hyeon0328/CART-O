from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def kmeans_numpy(
    X: np.ndarray,
    n_clusters: int = 2,
    n_iter: int = 100,
    seed: int = 42,
    tol: float = 1e-6,
):
    """
    Simple NumPy-only K-means.

    Args:
        X: (N, D)
        n_clusters: number of clusters
        n_iter: max iterations
        seed: random seed
        tol: stopping tolerance on center movement

    Returns:
        labels: (N,)
        centers: (K, D)
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]

    if N < n_clusters:
        raise ValueError(f"Not enough samples: N={N}, n_clusters={n_clusters}")

    # random init
    init_idx = rng.choice(N, size=n_clusters, replace=False)
    centers = X[init_idx].copy()

    for _ in range(n_iter):
        # squared Euclidean distance
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2) # (N, K)
        labels = np.argmin(dists, axis=1)

        new_centers = []
        for k in range(n_clusters):
            pts = X[labels == k]
            if len(pts) == 0:
                # keep old center if cluster empty
                new_centers.append(centers[k])
            else:
                new_centers.append(pts.mean(axis=0))
        new_centers = np.stack(new_centers, axis=0)

        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift < tol:
            break

    # final assignment
    dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(dists, axis=1)

    return labels, centers


def reorder_clusters_by_energy(centers: np.ndarray, labels: np.ndarray):
    """
    Reorder cluster IDs so interpretation is stable.
    Rule: smaller beta_e cluster becomes 0, larger beta_e cluster becomes 1.

    centers: (K, 3) columns = [beta_v, beta_h, beta_e]
    """
    order = np.argsort(centers[:, 2]) # sort by beta_e
    new_centers = centers[order]

    remap = {old: new for new, old in enumerate(order)}
    new_labels = np.array([remap[int(l)] for l in labels], dtype=np.int64)

    return new_labels, new_centers


def main():
    analysis_csv = Path(
        "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2/beta_analysis/beta_analysis_pairs.csv"
    ).expanduser()

    out_dir = Path(
        "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2/tss_labels"
    ).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not analysis_csv.exists():
        raise FileNotFoundError(f"Cannot find analysis csv: {analysis_csv}")

    df = pd.read_csv(analysis_csv)

    # We start with better branch only
    beta_cols = ["better_beta_v", "better_beta_h", "better_beta_e"]
    X = df[beta_cols].to_numpy(dtype=np.float64)

    # K=2 first
    labels, centers = kmeans_numpy(
        X=X,
        n_clusters=2,
        n_iter=100,
        seed=42,
        tol=1e-6,
    )

    # reorder cluster ids for stable interpretation
    labels, centers = reorder_clusters_by_energy(centers, labels)

    df["z_label"] = labels

    centers_df = pd.DataFrame(
        centers,
        columns=["center_beta_v", "center_beta_h", "center_beta_e"],
    )
    centers_df["cluster_id"] = centers_df.index

    cluster_counts = df["z_label"].value_counts().sort_index()

    print("=== cluster centers ===")
    print(centers_df.to_string(index=False))

    print("\n=== cluster counts ===")
    print(cluster_counts)

    summary_rows = []
    for cluster_id in sorted(df["z_label"].unique()):
        sub = df[df["z_label"] == cluster_id]

        row = {
            "cluster_id": int(cluster_id),
            "count": int(len(sub)),

            "beta_v_mean": float(sub["better_beta_v"].mean()),
            "beta_h_mean": float(sub["better_beta_h"].mean()),
            "beta_e_mean": float(sub["better_beta_e"].mean()),

            "beta_v_std": float(sub["better_beta_v"].std()),
            "beta_h_std": float(sub["better_beta_h"].std()),
            "beta_e_std": float(sub["better_beta_e"].std()),

            "J_v_mean": float(sub["better_J_v"].mean()),
            "J_h_mean": float(sub["better_J_h"].mean()),
            "J_e_mean": float(sub["better_J_e"].mean()),

            "reward_mean": float(sub["better_reward"].mean()),
            "margin_mean": float(sub["reward_margin"].mean()),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    print("\n=== cluster summary ===")
    print(summary_df.to_string(index=False))

    # Save
    df.to_csv(out_dir / "beta_analysis_with_z.csv", index=False)
    centers_df.to_csv(out_dir / "tss_cluster_centers.csv", index=False)
    summary_df.to_csv(out_dir / "tss_cluster_summary.csv", index=False)

    # optional: save per-cluster subsets
    for cluster_id in sorted(df["z_label"].unique()):
        sub = df[df["z_label"] == cluster_id].copy()
        sub.to_csv(out_dir / f"cluster_{cluster_id}_samples.csv", index=False)

    print("\nSaved:")
    print(out_dir / "beta_analysis_with_z.csv")
    print(out_dir / "tss_cluster_centers.csv")
    print(out_dir / "tss_cluster_summary.csv")
    for cluster_id in sorted(df["z_label"].unique()):
        print(out_dir / f"cluster_{cluster_id}_samples.csv")


if __name__ == "__main__":
    main()