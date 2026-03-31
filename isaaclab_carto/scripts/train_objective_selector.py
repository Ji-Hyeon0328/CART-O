import os
import json
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train Objective Selector from pseudo-expert dataset")
parser.add_argument("--dataset-path", type=str, required=True, help="Path to selector_dataset.json")
parser.add_argument("--save-dir", type=str, required=True, help="Directory to save checkpoints")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--val-ratio", type=float, default=0.2)
parser.add_argument("--hidden-dim", type=int, default=128)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
simulation_app = launcher.app

try:
    from isaaclab_carto.networks.objective_selector import ObjectiveSelector
except ImportError:
    from isaaclab_carto.isaaclab_carto.networks.objective_selector import ObjectiveSelector


class SelectorDataset(Dataset):
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        samples = []
        for item in raw:
            latent = item.get("latent", None)
            command = item.get("command", None)
            beta = item.get("beta", None)

            if latent is None or command is None or beta is None:
                continue

            samples.append(
                {
                    "latent": torch.tensor(latent, dtype=torch.float32),
                    "command": torch.tensor(command, dtype=torch.float32),
                    "beta": torch.tensor(beta, dtype=torch.float32),
                }
            )

        if len(samples) == 0:
            raise RuntimeError("No valid samples found in selector dataset.")

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return item["latent"], item["command"], item["beta"]


class SelectorWrapper(nn.Module):
    """
    Lightweight wrapper around ObjectiveSelector.

    We only use latent + command for now.
    proprio and aux are replaced by zeros to match existing forward signature.
    """
    def __init__(self, latent_dim: int, cmd_dim: int):
        super().__init__()

        self.selector = ObjectiveSelector(
            context_dim=latent_dim,
            cmd_dim=cmd_dim,
            state_dim=36,
            aux_dim=5,
            output_dim=3,
        )

    def forward(self, latent: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        batch = latent.shape[0]

        dummy_state = torch.zeros(batch, 36, device=latent.device)
        dummy_aux = torch.zeros(batch, 5, device=latent.device)

        beta = self.selector(latent, command, dummy_state, dummy_aux)
        return beta


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for latent, command, beta_target in loader:
            latent = latent.to(device)
            command = command.to(device)
            beta_target = beta_target.to(device)

            beta_pred = model(latent, command)
            loss = criterion(beta_pred, beta_target)

            total_loss += loss.item() * latent.size(0)
            count += latent.size(0)

    return total_loss / max(count, 1)


def main():
    os.makedirs(args.save_dir, exist_ok=True)

    dataset = SelectorDataset(args.dataset_path)

    latent_dim = dataset[0][0].numel()
    cmd_dim = dataset[0][1].numel()

    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SelectorWrapper(
        latent_dim=latent_dim,
        cmd_dim=cmd_dim,
        #hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_path = os.path.join(args.save_dir, "objective_selector_best.pt")

    print("-" * 60)
    print("[INFO] Objective Selector supervised training started")
    print(f"[INFO] dataset size = {len(dataset)}")
    print(f"[INFO] train size = {len(train_dataset)}")
    print(f"[INFO] val size = {len(val_dataset)}")
    print(f"[INFO] latent_dim = {latent_dim}, cmd_dim = {cmd_dim}")
    print("-" * 60)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        count = 0

        for latent, command, beta_target in train_loader:
            latent = latent.to(device)
            command = command.to(device)
            beta_target = beta_target.to(device)

            beta_pred = model(latent, command)
            loss = criterion(beta_pred, beta_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * latent.size(0)
            count += latent.size(0)

        train_loss = running_loss / max(count, 1)
        val_loss = evaluate(model, val_loader, device, criterion)

        print(f"[Epoch {epoch+1:03d}] train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        #latest_path = os.path.join(args.save_dir, "objective_selector_latest.pt")
        best_path = os.path.join(args.save_dir, "objective_selector_best.pt")
        best_selector_only_path = os.path.join(args.save_dir, "objective_selector_only_best.pt")
        latest_path = os.path.join(args.save_dir, "objective_selector_latest.pt")
        latest_selector_only_path = os.path.join(args.save_dir, "objective_selector_only_latest.pt")
        torch.save(model.state_dict(), latest_path)
        torch.save(model.selector.state_dict(), latest_selector_only_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            torch.save(model.selector.state_dict(), best_selector_only_path)
            print(f"[INFO] best model updated -> {best_path}")
            print(f"[INFO] best selector-only model updated -> {best_selector_only_path}")

    print(f"[INFO] training complete. best_val_loss={best_val_loss:.6f}")


if __name__ == "__main__":
    main()