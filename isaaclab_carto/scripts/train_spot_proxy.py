from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from isaaclab_carto.utils.spot_carto_dataset import SpotCartoDataset
from isaaclab_carto.networks.spot_proxy_model import SpotProxyModel


def collate_fn(batch):
    proprio = torch.stack([b["proprio"] for b in batch], dim=0)   # (B, T, 58)
    depth = torch.stack([b["depth"] for b in batch], dim=0)       # (B, T, 1, H, W)

    forward_velocity = torch.stack(
        [b["targets"]["forward_velocity"] for b in batch], dim=0
    )  # (B, T)

    body_height = torch.stack(
        [b["targets"]["body_height"] for b in batch], dim=0
    )  # (B, T)

    contact = torch.stack(
        [b["targets"]["contact"] for b in batch], dim=0
    )  # (B, T, 4)

    return {
        "proprio": proprio,
        "depth": depth,
        "targets": {
            "forward_velocity": forward_velocity,
            "body_height": body_height,
            "contact": contact,
        }
    }


def compute_losses(out, batch, mse_loss, bce_loss):
    vx_pred = out["vx_pred"].squeeze(-1)             # (B, T)
    height_pred = out["height_pred"].squeeze(-1)     # (B, T)
    contact_logit = out["contact_logit"]             # (B, T, 4)

    vx_target = batch["targets"]["forward_velocity"] # (B, T)
    height_target = batch["targets"]["body_height"]  # (B, T)
    contact_target = batch["targets"]["contact"]     # (B, T, 4)

    loss_vx = mse_loss(vx_pred, vx_target)
    loss_height = mse_loss(height_pred, height_target)
    loss_contact = bce_loss(contact_logit, contact_target)

    total_loss = 1.0 * loss_vx + 1.0 * loss_height + 0.5 * loss_contact

    return total_loss, {
        "loss_vx": loss_vx.item(),
        "loss_height": loss_height.item(),
        "loss_contact": loss_contact.item(),
        "loss_total": total_loss.item(),
    }


def run_epoch(model, loader, optimizer, device, train: bool):
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    if train:
        model.train()
    else:
        model.eval()

    total = {
        "loss_vx": 0.0,
        "loss_height": 0.0,
        "loss_contact": 0.0,
        "loss_total": 0.0,
    }
    num_batches = 0

    for batch in loader:
        proprio = batch["proprio"].to(device)
        depth = batch["depth"].to(device)

        targets = {
            "forward_velocity": batch["targets"]["forward_velocity"].to(device),
            "body_height": batch["targets"]["body_height"].to(device),
            "contact": batch["targets"]["contact"].to(device),
        }
        batch["targets"] = targets

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            out = model(proprio, depth)
            loss, stats = compute_losses(out, batch, mse_loss, bce_loss)

            if train:
                loss.backward()
                optimizer.step()

        for k in total:
            total[k] += stats[k]
        num_batches += 1

    for k in total:
        total[k] /= max(num_batches, 1)

    return total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    data_dir = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2"
    save_dir = Path("~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2/checkpoints").expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = SpotCartoDataset(
        data_dir=data_dir,
        seq_len=20,
        stride=5,
        use_rgb=False,
        use_depth=True,
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
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = SpotProxyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 5
    best_val = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, device, train=True)
        val_stats = run_epoch(model, val_loader, optimizer, device, train=False)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_total={train_stats['loss_total']:.4f} "
            f"train_vx={train_stats['loss_vx']:.4f} "
            f"train_h={train_stats['loss_height']:.4f} "
            f"train_c={train_stats['loss_contact']:.4f} | "
            f"val_total={val_stats['loss_total']:.4f} "
            f"val_vx={val_stats['loss_vx']:.4f} "
            f"val_h={val_stats['loss_height']:.4f} "
            f"val_c={val_stats['loss_contact']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_stats": train_stats,
            "val_stats": val_stats,
        }

        torch.save(ckpt, save_dir / "spot_proxy_last.pt")

        if val_stats["loss_total"] < best_val:
            best_val = val_stats["loss_total"]
            torch.save(ckpt, save_dir / "spot_proxy_best.pt")
            print(f"  -> saved best checkpoint (val_total={best_val:.4f})")


if __name__ == "__main__":
    main()