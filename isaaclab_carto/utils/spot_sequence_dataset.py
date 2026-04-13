from __future__ import annotations

from pathlib import Path
import json
import ast
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _safe_literal(x: Any) -> Any:
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x


def load_csv(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path).expanduser()
    df = pd.read_csv(csv_path)

    for col in df.columns:
        df[col] = df[col].apply(_safe_literal)

    return df


def load_jsonl(jsonl_path: str | Path) -> list[dict[str, Any]]:
    jsonl_path = Path(jsonl_path).expanduser()
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def nearest_index(sorted_ts: np.ndarray, query_t: int) -> int:
    idx = int(np.searchsorted(sorted_ts, query_t))
    if idx <= 0:
        return 0
    if idx >= len(sorted_ts):
        return len(sorted_ts) - 1
    left = idx - 1
    right = idx
    if abs(sorted_ts[left] - query_t) <= abs(sorted_ts[right] - query_t):
        return left
    return right


def parse_feet_record(feet_record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """
    returns:
        contact: (4,)
        foot_pos_body: (4, 3)
    """
    states = feet_record["data"]["states"]

    contact = []
    foot_pos = []

    for s in states:
        contact.append(float(s.get("contact", 0)))

        pos = s.get("foot_position_rt_body", {})
        foot_pos.append([
            float(pos.get("x", 0.0)),
            float(pos.get("y", 0.0)),
            float(pos.get("z", 0.0)),
        ])

    contact = np.asarray(contact, dtype=np.float32)          # (4,)
    foot_pos = np.asarray(foot_pos, dtype=np.float32)        # (4,3)
    return contact, foot_pos


def build_proprio_vector(joint_row, odom_row, feet_record) -> np.ndarray:
    joint_pos = np.asarray(joint_row["position_json"], dtype=np.float32)   # (12,)
    joint_vel = np.asarray(joint_row["velocity_json"], dtype=np.float32)   # (12,)
    joint_eff = np.asarray(joint_row["effort_json"], dtype=np.float32)     # (12,)

    base_lin = np.asarray([
        odom_row["vx"], odom_row["vy"], odom_row["vz"]
    ], dtype=np.float32)

    base_ang = np.asarray([
        odom_row["wx"], odom_row["wy"], odom_row["wz"]
    ], dtype=np.float32)

    contact, foot_pos = parse_feet_record(feet_record)  # (4,), (4,3)

    proprio = np.concatenate([
        joint_pos,                    # 12
        joint_vel,                    # 12
        joint_eff,                    # 12
        base_lin,                     # 3
        base_ang,                     # 3
        contact,                      # 4
        foot_pos.reshape(-1),         # 12
    ], axis=0).astype(np.float32)

    # total = 58 dims
    return proprio


class SpotSequenceDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        seq_len: int = 20,
        stride: int = 5,
    ):
        self.data_dir = Path(data_dir).expanduser()
        self.seq_len = seq_len
        self.stride = stride

        self.joint_df = load_csv(self.data_dir / "tabular" / "joint_states.csv")
        self.odom_df = load_csv(self.data_dir / "tabular" / "odometry.csv")
        self.feet_list = load_jsonl(self.data_dir / "status" / "feet.jsonl")

        self.joint_ts = self.joint_df["t_ns"].to_numpy(dtype=np.int64)
        self.odom_ts = self.odom_df["t_ns"].to_numpy(dtype=np.int64)
        self.feet_ts = np.asarray([int(x["t_ns"]) for x in self.feet_list], dtype=np.int64)

        # 기준 시계열: odometry
        self.anchor_ts = self.odom_ts

        self.samples = []
        self._build_index()

    def _build_index(self):
        N = len(self.anchor_ts)
        for start in range(0, N - self.seq_len + 1, self.stride):
            end = start + self.seq_len
            self.samples.append((start, end))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        start, end = self.samples[idx]

        proprio_seq = []
        timestamp_seq = []

        for k in range(start, end):
            t = int(self.anchor_ts[k])

            j_idx = nearest_index(self.joint_ts, t)
            f_idx = nearest_index(self.feet_ts, t)

            joint_row = self.joint_df.iloc[j_idx]
            odom_row = self.odom_df.iloc[k]
            feet_record = self.feet_list[f_idx]

            proprio = build_proprio_vector(joint_row, odom_row, feet_record)

            proprio_seq.append(proprio)
            timestamp_seq.append(t)

        proprio_seq = np.stack(proprio_seq, axis=0)  # (T, 58)

        # 예비 target들: 나중 objective selector / reward reconstruction용
        target_dict = {
            "forward_velocity": np.asarray(
                [self.odom_df.iloc[k]["vx"] for k in range(start, end)],
                dtype=np.float32,
            ),
            "body_height": np.asarray(
                [self.odom_df.iloc[k]["pz"] for k in range(start, end)],
                dtype=np.float32,
            ),
        }

        return {
            "proprio": torch.from_numpy(proprio_seq),        # (T, 58)
            "timestamps_ns": torch.tensor(timestamp_seq, dtype=torch.long),
            "targets": {
                "forward_velocity": torch.from_numpy(target_dict["forward_velocity"]),
                "body_height": torch.from_numpy(target_dict["body_height"]),
            },
        }


if __name__ == "__main__":
    data_dir = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2"
    dataset = SpotSequenceDataset(data_dir=data_dir, seq_len=20, stride=5)

    print("len(dataset) =", len(dataset))

    sample = dataset[0]
    print("\nproprio shape =", sample["proprio"].shape)   # expected: (20, 58)
    print("timestamps shape =", sample["timestamps_ns"].shape)
    print("forward_velocity shape =", sample["targets"]["forward_velocity"].shape)
    print("body_height shape =", sample["targets"]["body_height"].shape)

    print("\nfirst proprio row:")
    print(sample["proprio"][0])