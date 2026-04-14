from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from isaaclab_carto.utils.spot_joint_highlevel_dataset import (
    SpotJointHighLevelDataset,
    collate_joint_highlevel,
)
from isaaclab_carto.networks.spot_joint_highlevel_model import SpotJointHighLevelModel


def preference_loss(better_reward: torch.Tensor, worse_reward: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(better_reward - worse_reward).mean()


def beta_entropy(beta: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    return -(beta * (beta + eps).log()).sum(dim=-1).mean()


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=-1)
    return (pred == labels).float().mean().item()


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    train: bool,
    pref_weight: float = 1.0,
    z_weight: float = 1.0,
    entropy_weight: float = 0.01,
):
    z_criterion = nn.CrossEntropyLoss()

    if train:
        model.train()
    else:
        model.eval()

    total_pref = 0.0
    total_z = 0.0
    total_entropy = 0.0
    total_loss = 0.0
    total_margin = 0.0
    total_acc = 0.0
    n_batches = 0

    for batch in loader:
        better_proprio = batch["better"]["proprio"].to(device)
        better_depth = batch["better"]["depth"].to(device)
        better_obj = batch["better"]["objective_components"].to(device)

        worse_proprio = batch["worse"]["proprio"].to(device)
        worse_depth = batch["worse"]["depth"].to(device)
        worse_obj = batch["worse"]["objective_components"].to(device)

        z_label = batch["z_label"].to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            out = model(
                better_proprio=better_proprio,
                better_depth=better_depth,
                better_obj=better_obj,
                worse_proprio=worse_proprio,
                worse_depth=worse_depth,
                worse_obj=worse_obj,
            )

            better_reward = out["better_reward"]
            worse_reward = out["worse_reward"]
            better_beta = out["better_beta"]
            worse_beta = out["worse_beta"]
            better_z_logits = out["better_z_logits"]

            loss_pref = preference_loss(better_reward, worse_reward)
            loss_z = z_criterion(better_z_logits, z_label)

            ent_better = beta_entropy(better_beta)
            ent_worse = beta_entropy(worse_beta)
            loss_entropy = -(ent_better + ent_worse) * 0.5

            loss = (
                pref_weight * loss_pref +
                z_weight * loss_z +
                entropy_weight * loss_entropy
            )

            if train:
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            margin = (better_reward - worse_reward).mean().item()
            acc = compute_accuracy(better_z_logits, z_label)

        total_pref += loss_pref.item()
        total_z += loss_z.item()
        total_entropy += loss_entropy.item()
        total_loss += loss.item()
        total_margin += margin
        total_acc += acc
        n_batches += 1

    total_pref /= max(n_batches, 1)
    total_z /= max(n_batches, 1)
    total_entropy /= max(n_batches, 1)
    total_loss /= max(n_batches, 1)
    total_margin /= max(n_batches, 1)
    total_acc /= max(n_batches, 1)

    return {
        "loss_pref": total_pref,
        "loss_z": total_z,
        "loss_entropy": total_entropy,
        "loss_total": total_loss,
        "reward_margin": total_margin,
        "z_acc": total_acc,
    }


def inspect_joint_beta(model, loader, device, max_batches: int = 5):
    model.eval()

    beta_list = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break

            better_proprio = batch["better"]["proprio"].to(device)
            better_depth = batch["better"]["depth"].to(device)
            better_obj = batch["better"]["objective_components"].to(device)

            better_ctx = model.encode(better_proprio, better_depth)
            _, beta = model.compute_reward(better_ctx, better_obj)

            beta_list.append(beta.cpu())

    if len(beta_list) == 0:
        return None

    beta_all = torch.cat(beta_list, dim=0)
    beta_mean = beta_all.mean(dim=0)
    beta_std = beta_all.std(dim=0)

    return {
        "beta_mean": beta_mean,
        "beta_std": beta_std,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    data_dir = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2"
    scores_csv = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2/pseudo_expert/window_scores.csv"
    tss_csv = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2/tss_labels/beta_analysis_with_z.csv"

    save_dir = Path("~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2/joint_highlevel_checkpoints").expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

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

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_joint_highlevel,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_joint_highlevel,
    )

    model = SpotJointHighLevelModel(num_modes=2, freeze_encoder=False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    best_val = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_stats = run_epoch(
            model, train_loader, optimizer, device,
            train=True,
            pref_weight=1.0,
            z_weight=1.0,
            entropy_weight=0.01,
        )
        val_stats = run_epoch(
            model, val_loader, optimizer, device,
            train=False,
            pref_weight=1.0,
            z_weight=1.0,
            entropy_weight=0.01,
        )

        beta_stats = inspect_joint_beta(model, val_loader, device, max_batches=5)

        if beta_stats is not None:
            beta_mean = beta_stats["beta_mean"].tolist()
            beta_std = beta_stats["beta_std"].tolist()
        else:
            beta_mean = [0.0, 0.0, 0.0]
            beta_std = [0.0, 0.0, 0.0]

        print(
            f"[Epoch {epoch:02d}] "
            f"train_total={train_stats['loss_total']:.4f} "
            f"train_pref={train_stats['loss_pref']:.4f} "
            f"train_z={train_stats['loss_z']:.4f} "
            f"train_margin={train_stats['reward_margin']:.4f} "
            f"train_zacc={train_stats['z_acc']:.4f} | "
            f"val_total={val_stats['loss_total']:.4f} "
            f"val_pref={val_stats['loss_pref']:.4f} "
            f"val_z={val_stats['loss_z']:.4f} "
            f"val_margin={val_stats['reward_margin']:.4f} "
            f"val_zacc={val_stats['z_acc']:.4f} | "
            f"beta_mean={beta_mean} beta_std={beta_std}"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_stats": train_stats,
            "val_stats": val_stats,
            "beta_mean": beta_mean,
            "beta_std": beta_std,
        }

        torch.save(ckpt, save_dir / "spot_joint_highlevel_last.pt")

        if val_stats["loss_total"] < best_val:
            best_val = val_stats["loss_total"]
            torch.save(ckpt, save_dir / "spot_joint_highlevel_best.pt")
            print(f" -> saved best checkpoint (val_total={best_val:.4f})")


if __name__ == "__main__":
    main()