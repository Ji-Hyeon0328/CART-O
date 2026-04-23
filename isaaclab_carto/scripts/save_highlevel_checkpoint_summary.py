from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import torch


def safe_load_torch(path: Path):
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def main():
    root = Path("~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2").expanduser()

    pref_ckpt = root / "preference_checkpoints" / "spot_beta_preference_best.pt"
    tss_ckpt = root / "tss_checkpoints" / "spot_tss_best.pt"

    beta_pairs_csv = root / "beta_analysis" / "beta_analysis_pairs.csv"
    tss_centers_csv = root / "tss_labels" / "tss_cluster_centers.csv"
    tss_summary_csv = root / "tss_labels" / "tss_cluster_summary.csv"

    out_dir = root / "highlevel_checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "stage_name": "high_level_framework_checkpoint_before_option2",
        "dataset_root": str(root),
        "artifacts": {},
        "preference_training": {},
        "beta_analysis": {},
        "tss_training": {},
        "tss_clustering": {},
        "notes": [
            "This checkpoint was saved before moving from separated Objective Selector/TSS pipeline to integrated Option 2.",
            "Preference learning used beta-dependent reward: R = beta dot J.",
            "TSS labels were built from beta clustering with k=2.",
        ],
    }

    # -------- preference checkpoint --------
    pref = safe_load_torch(pref_ckpt)
    if pref is not None:
        summary["artifacts"]["preference_ckpt"] = str(pref_ckpt)
        summary["preference_training"] = {
            "epoch": pref.get("epoch", None),
            "train_stats": pref.get("train_stats", {}),
            "val_stats": pref.get("val_stats", {}),
            "beta_mean": pref.get("beta_mean", None),
            "beta_std": pref.get("beta_std", None),
        }

    # -------- beta analysis csv --------
    if beta_pairs_csv.exists():
        df_beta = pd.read_csv(beta_pairs_csv)

        better_beta_mean = df_beta[["better_beta_v", "better_beta_h", "better_beta_e"]].mean().to_dict()
        better_beta_std = df_beta[["better_beta_v", "better_beta_h", "better_beta_e"]].std().to_dict()
        worse_beta_mean = df_beta[["worse_beta_v", "worse_beta_h", "worse_beta_e"]].mean().to_dict()
        worse_beta_std = df_beta[["worse_beta_v", "worse_beta_h", "worse_beta_e"]].std().to_dict()

        corr_cols = [
            "better_beta_v", "better_beta_h", "better_beta_e",
            "better_J_v", "better_J_h", "better_J_e",
        ]
        corr = df_beta[corr_cols].corr().round(6).to_dict()

        summary["artifacts"]["beta_analysis_pairs_csv"] = str(beta_pairs_csv)
        summary["beta_analysis"] = {
            "num_pairs": int(len(df_beta)),
            "better_beta_mean": better_beta_mean,
            "better_beta_std": better_beta_std,
            "worse_beta_mean": worse_beta_mean,
            "worse_beta_std": worse_beta_std,
            "correlation_matrix": corr,
        }

    # -------- tss checkpoint --------
    tss = safe_load_torch(tss_ckpt)
    if tss is not None:
        summary["artifacts"]["tss_ckpt"] = str(tss_ckpt)
        summary["tss_training"] = {
            "epoch": tss.get("epoch", None),
            "train_stats": tss.get("train_stats", {}),
            "val_stats": tss.get("val_stats", {}),
        }

    # -------- tss clustering outputs --------
    if tss_centers_csv.exists():
        df_centers = pd.read_csv(tss_centers_csv)
        summary["artifacts"]["tss_cluster_centers_csv"] = str(tss_centers_csv)
        summary["tss_clustering"]["cluster_centers"] = df_centers.to_dict(orient="records")

    if tss_summary_csv.exists():
        df_summary = pd.read_csv(tss_summary_csv)
        summary["artifacts"]["tss_cluster_summary_csv"] = str(tss_summary_csv)
        summary["tss_clustering"]["cluster_summary"] = df_summary.to_dict(orient="records")

    # -------- save json --------
    json_path = out_dir / "highlevel_stage_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # -------- save markdown --------
    md_path = out_dir / "highlevel_stage_summary.md"

    lines = []
    lines.append("# High-Level Framework Checkpoint Summary")
    lines.append("")
    lines.append("## 목적")
    lines.append("- Option 2 통합 high-level planner로 넘어가기 전의 checkpoint를 남긴다.")
    lines.append("- Objective Selector(beta-dependent reward)와 TSS의 현재 성능을 기록한다.")
    lines.append("")

    lines.append("## 주요 아티팩트")
    for k, v in summary["artifacts"].items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")

    if summary["preference_training"]:
        p = summary["preference_training"]
        lines.append("## Preference Training")
        lines.append(f"- Best epoch: `{p.get('epoch', None)}`")
        lines.append(f"- Train stats: `{p.get('train_stats', {})}`")
        lines.append(f"- Val stats: `{p.get('val_stats', {})}`")
        lines.append(f"- Beta mean: `{p.get('beta_mean', None)}`")
        lines.append(f"- Beta std: `{p.get('beta_std', None)}`")
        lines.append("")

    if summary["beta_analysis"]:
        b = summary["beta_analysis"]
        lines.append("## Beta Analysis")
        lines.append(f"- Num pairs: `{b.get('num_pairs', None)}`")
        lines.append(f"- Better beta mean: `{b.get('better_beta_mean', {})}`")
        lines.append(f"- Better beta std: `{b.get('better_beta_std', {})}`")
        lines.append(f"- Worse beta mean: `{b.get('worse_beta_mean', {})}`")
        lines.append(f"- Worse beta std: `{b.get('worse_beta_std', {})}`")
        lines.append("")
        lines.append("### 해석")
        lines.append("- beta는 uniform에서 벗어나기 시작했다.")
        lines.append("- worse 샘플에서 energy 쪽 beta 비중이 더 커지는 경향이 관찰되었다.")
        lines.append("- Objective Selector가 reward ranking뿐 아니라 objective preference 분화도 일부 학습하기 시작한 상태로 해석할 수 있다.")
        lines.append("")

    if summary["tss_training"]:
        t = summary["tss_training"]
        lines.append("## TSS Training")
        lines.append(f"- Best epoch: `{t.get('epoch', None)}`")
        lines.append(f"- Train stats: `{t.get('train_stats', {})}`")
        lines.append(f"- Val stats: `{t.get('val_stats', {})}`")
        lines.append("")

    if summary["tss_clustering"]:
        lines.append("## TSS Clustering")
        centers = summary["tss_clustering"].get("cluster_centers", [])
        cluster_summary = summary["tss_clustering"].get("cluster_summary", [])
        lines.append(f"- Cluster centers: `{centers}`")
        lines.append(f"- Cluster summary: `{cluster_summary}`")
        lines.append("")
        lines.append("### 해석")
        lines.append("- k=2 clustering 기준으로 objective preference가 두 개의 pseudo mode로 분리되었다.")
        lines.append("- 현재 mode는 gait primitive라기보다는 aggressive/conservative behavior tendency에 가까운 해석이 적절하다.")
        lines.append("")

    lines.append("## 다음 단계 (Option 2)")
    lines.append("- shared encoder/context 위에 beta head와 z head를 함께 두는 통합 high-level planner로 이동")
    lines.append("- loss는 preference loss + z classification loss를 joint하게 사용")
    lines.append("- 초기화는 현재 preference/TSS checkpoint를 활용")
    lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Saved summary files:")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()