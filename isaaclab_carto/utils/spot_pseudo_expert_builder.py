from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from isaaclab_carto.utils.spot_carto_dataset import SpotCartoDataset


def compute_effort_norm_from_proprio(proprio_seq: torch.Tensor) -> torch.Tensor:
    """
    proprio layout:
      0:12   joint pos
      12:24  joint vel
      24:36  joint effort
      36:39  base lin vel
      39:42  base ang vel
      42:46  contact
      46:58  foot pos
    proprio_seq: (T, 58)
    return: (T,)
    """
    joint_eff = proprio_seq[:, 24:36]
    return torch.norm(joint_eff, dim=-1)


def compute_contact_score(contact_seq: torch.Tensor) -> float:
    """
    contact_seq: (T, 4)
    returns a scalar score where larger is better
    """
    if contact_seq.shape[0] == 0:
        return 0.0

    # 1) contact count preference
    # 현재는 간단히 2-contact 근처를 선호
    num_contact = contact_seq.sum(dim=-1)  # (T,)
    score_count = -torch.abs(num_contact - 2.0).mean()

    # 2) transition smoothness
    if contact_seq.shape[0] < 2:
        score_transition = torch.tensor(0.0)
    else:
        diff = torch.abs(contact_seq[1:] - contact_seq[:-1])
        score_transition = -diff.mean()

    # 가볍게 합침
    score = score_count + 0.5 * score_transition
    return float(score.item())

# def compute_contact_score(contact_seq: torch.Tensor) -> float:
#     if contact_seq.shape[0] < 2:
#         return 0.0

#     # contact 변화량
#     diff = torch.abs(contact_seq[1:] - contact_seq[:-1])  # (T-1, 4)

#     # 변화가 적을수록 좋음 (stable gait)
#     score = -diff.mean()

#     return float(score.item())

def score_window(sample: dict) -> dict:
    """
    sample keys:
      - proprio: (T, 58)
      - targets.forward_velocity: (T,)
      - targets.body_height: (T,)
      - targets.contact: (T, 4)
    """
    proprio = sample["proprio"].float()
    vx = sample["targets"]["forward_velocity"].float()
    pz = sample["targets"]["body_height"].float()
    contact = sample["targets"]["contact"].float()

    effort_norm = compute_effort_norm_from_proprio(proprio)

    # 1. progress: 높을수록 좋음
    s_progress = float(vx.mean().item())

    # 2. height stability: 기준 높이 근처 + 분산 작을수록 좋음
    h_ref = -0.15
    s_height = float((-(torch.abs(pz - h_ref).mean() + 0.5 * pz.std())).item())

    # 3. contact quality
    s_contact = compute_contact_score(contact)

    # 4. energy efficiency: effort 낮을수록 좋음
    s_energy = float((-(effort_norm.mean())).item())

    return {
        "score_progress_raw": s_progress,
        "score_height_raw": s_height,
        "score_contact_raw": s_contact,
        "score_energy_raw": s_energy,
    }

def zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if std < 1e-8:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std

def main():
    data_dir = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2"

    dataset = SpotCartoDataset(
        data_dir=data_dir,
        seq_len=20,
        stride=5,
        use_rgb=False,
        use_depth=True,
    )

    rows = []

    print(f"len(dataset) = {len(dataset)}")

    for i in range(len(dataset)):
        sample = dataset[i]
        score_dict = score_window(sample)

        row = {
            "index": i,
            "t_start": int(sample["timestamps_ns"][0].item()),
            "t_end": int(sample["timestamps_ns"][-1].item()),
            **score_dict,
        }
        rows.append(row)

        if i % 200 == 0:
            print(
                f"[{i:04d}/{len(dataset)}] "
                f"prog_raw={score_dict['score_progress_raw']:.4f}, "
                f"h_raw={score_dict['score_height_raw']:.4f}, "
                f"c_raw={score_dict['score_contact_raw']:.4f}, "
                f"e_raw={score_dict['score_energy_raw']:.4f}"
            )

    df = pd.DataFrame(rows)

    # normalize each raw score
    df["score_progress"] = zscore(df["score_progress_raw"])
    df["score_height"] = zscore(df["score_height_raw"])
    df["score_contact"] = zscore(df["score_contact_raw"])
    df["score_energy"] = zscore(df["score_energy_raw"])

    # final weighted score
    df["score_total"] = (
        2.0 * df["score_progress"] +
        0.8 * df["score_height"] +
        0.5 * df["score_contact"] +
        0.05 * df["score_energy"]
    )

    out_dir = Path(data_dir).expanduser() / "pseudo_expert"
    out_dir.mkdir(parents=True, exist_ok=True)

    # save all scores
    df.to_csv(out_dir / "window_scores.csv", index=False)

    # sort
    df_sorted = df.sort_values("score_total", ascending=False).reset_index(drop=True)

    # save top/bottom windows
    top_k = min(100, len(df_sorted))
    bottom_k = min(100, len(df_sorted))

    df_sorted.head(top_k).to_csv(out_dir / "top100_windows.csv", index=False)
    df_sorted.tail(bottom_k).to_csv(out_dir / "bottom100_windows.csv", index=False)

    print("\n=== score summary ===")
    print(df.describe())

    print("\n=== top 10 windows ===")
    print(df_sorted.head(10).to_string(index=False))

    print("\n=== bottom 10 windows ===")
    print(df_sorted.tail(10).to_string(index=False))

    print("\nSaved files:")
    print(out_dir / "window_scores.csv")
    print(out_dir / "top100_windows.csv")
    print(out_dir / "bottom100_windows.csv")


if __name__ == "__main__":
    main()