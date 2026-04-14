from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from isaaclab_carto.utils.spot_tss_dataset import SpotTSSDataset, collate_tss
from isaaclab_carto.networks.spot_tss_model import SpotTSSModel


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=-1)
    acc = (pred == labels).float().mean().item()
    return acc


def run_epoch(model, loader, optimizer, device, train: bool):
    criterion = nn.CrossEntropyLoss()

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for batch in loader:
        proprio = batch["proprio"].to(device)
        depth = batch["depth"].to(device)
        z_label = batch["z_label"].to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            out = model(proprio, depth)
            logits = out["logits"]

            loss = criterion(logits, z_label)
            acc = compute_accuracy(logits, z_label)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

    total_loss /= max(n_batches, 1)
    total_acc /= max(n_batches, 1)

    return {
        "loss": total_loss,
        "acc": total_acc,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    data_dir = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2"
    tss_csv = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2/tss_labels/beta_analysis_with_z.csv"

    save_dir = Path("~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2/tss_checkpoints").expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = SpotTSSDataset(
        data_dir=data_dir,
        tss_csv=tss_csv,
        seq_len=20,
        stride=5,
        use_depth=True,
        use_better_only=True,
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
        batch_size=8,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_tss,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_tss,
    )

    model = SpotTSSModel(num_modes=2, freeze_encoder=True).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
    )

    num_epochs = 10
    best_val = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, device, train=True)
        val_stats = run_epoch(model, val_loader, optimizer, device, train=False)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_stats['loss']:.4f} "
            f"train_acc={train_stats['acc']:.4f} | "
            f"val_loss={val_stats['loss']:.4f} "
            f"val_acc={val_stats['acc']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_stats": train_stats,
            "val_stats": val_stats,
        }

        torch.save(ckpt, save_dir / "spot_tss_last.pt")

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(ckpt, save_dir / "spot_tss_best.pt")
            print(f" -> saved best checkpoint (val_loss={best_val:.4f})")


if __name__ == "__main__":
    main()