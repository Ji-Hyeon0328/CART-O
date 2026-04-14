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

def load_depth_npy(path):
    depth = np.load(path)  # (H, W)

    # NaN / inf 처리
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    # clipping (optional but recommended)
    depth = np.clip(depth, 0.0, 5.0)  # 5m max

    # normalize
    depth = depth / 5.0  # → [0, 1]

    return depth.astype(np.float32)

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

    contact = np.asarray(contact, dtype=np.float32)      # (4,)
    foot_pos = np.asarray(foot_pos, dtype=np.float32)    # (4,3)
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

    contact, foot_pos = parse_feet_record(feet_record)

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


def list_timestamped_files(folder: Path, suffix: str) -> tuple[np.ndarray, list[Path]]:
    files = sorted(folder.glob(f"*{suffix}"), key=lambda p: int(p.stem))
    ts = np.asarray([int(p.stem) for p in files], dtype=np.int64)
    return ts, files

def compute_effort_norm_from_proprio(proprio_seq: np.ndarray) -> np.ndarray:
    """
    proprio layout:
      0:12 joint pos
      12:24 joint vel
      24:36 joint effort
      36:39 base lin vel
      39:42 base ang vel
      42:46 contact
      46:58 foot pos

    proprio_seq: (T, 58)
    return: (T,)
    """
    joint_eff = proprio_seq[:, 24:36] # (T, 12)
    return np.linalg.norm(joint_eff, axis=-1)


def compute_objective_components(
    proprio_seq: np.ndarray,
    forward_velocity_seq: np.ndarray,
    body_height_seq: np.ndarray,
    h_ref: float = -0.15,
) -> np.ndarray:
    """
    returns:
        J = [J_velocity, J_height, J_energy] shape (3,)
    """
    # J_v: forward velocity objective
    J_v = float(np.mean(forward_velocity_seq))

    # J_h: height/stability objective
    J_h = float(-(np.mean(np.abs(body_height_seq - h_ref)) + 0.5 * np.std(body_height_seq)))

    # J_e: energy objective
    effort_norm = compute_effort_norm_from_proprio(proprio_seq)
    J_e = float(-np.mean(effort_norm))

    return np.asarray([J_v, J_h, J_e], dtype=np.float32)



class SpotCartoDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        seq_len: int = 20,
        stride: int = 5,
        use_rgb: bool = True,
        use_depth: bool = True,
    ):
        self.data_dir = Path(data_dir).expanduser()
        self.seq_len = seq_len
        self.stride = stride
        self.use_rgb = use_rgb
        self.use_depth = use_depth

        self.joint_df = load_csv(self.data_dir / "tabular" / "joint_states.csv")
        self.odom_df = load_csv(self.data_dir / "tabular" / "odometry.csv")
        self.feet_list = load_jsonl(self.data_dir / "status" / "feet.jsonl")

        self.joint_ts = self.joint_df["t_ns"].to_numpy(dtype=np.int64)
        self.odom_ts = self.odom_df["t_ns"].to_numpy(dtype=np.int64)
        self.feet_ts = np.asarray([int(x["t_ns"]) for x in self.feet_list], dtype=np.int64)

        # frontleft RGB / registered depth
        self.rgb_folder = self.data_dir / "images" / "frontleft"
        self.depth_folder = self.data_dir / "depth_registered" / "frontleft"

        if self.use_rgb:
            self.rgb_ts, self.rgb_files = list_timestamped_files(self.rgb_folder, ".png")
        else:
            self.rgb_ts, self.rgb_files = np.asarray([], dtype=np.int64), []

        if self.use_depth:
            self.depth_ts, self.depth_files = list_timestamped_files(self.depth_folder, ".npy")
        else:
            self.depth_ts, self.depth_files = np.asarray([], dtype=np.int64), []

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

        rgb_path_seq = []
        rgb_time_seq = []

        depth_path_seq = []
        depth_time_seq = []
        depth_tensor_seq = []

        contact_seq = []

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

            contact, foot_pos = parse_feet_record(feet_record)
            contact_seq.append(contact)

            if self.use_rgb and len(self.rgb_ts) > 0:
                r_idx = nearest_index(self.rgb_ts, t)
                rgb_path_seq.append(str(self.rgb_files[r_idx]))
                rgb_time_seq.append(int(self.rgb_ts[r_idx]))
            else:
                rgb_path_seq.append("")
                rgb_time_seq.append(-1)

            if self.use_depth and len(self.depth_ts) > 0:
                d_idx = nearest_index(self.depth_ts, t)
                depth_path= self.depth_files[d_idx]

                depth= load_depth_npy(depth_path)
                depth_tensor_seq.append(depth)

                depth_path_seq.append(str(self.depth_files[d_idx]))
                depth_time_seq.append(int(self.depth_ts[d_idx]))
            else:
                depth_path_seq.append("")
                depth_time_seq.append(-1)

        depth_tensor_seq = np.stack(depth_tensor_seq,axis=0) # (T,H,W)
        depth_tensor_seq = depth_tensor_seq[:,None,:,:] #(T 1 H W)

        proprio_seq = np.stack(proprio_seq, axis=0)  # (T, 58)

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

        forward_velocity_seq = np.asarray(
            [self.odom_df.iloc[k]["vx"] for k in range(start, end)],
            dtype=np.float32,
        )

        body_height_seq = np.asarray(
            [self.odom_df.iloc[k]["pz"] for k in range(start, end)],
            dtype=np.float32,
        )

        objective_components = compute_objective_components(
            proprio_seq=proprio_seq,
            forward_velocity_seq=forward_velocity_seq,
            body_height_seq=body_height_seq,
        )



        #assert len(depth_path_seq) == self.seq_len, f"depth_path_seq len={len(depth_path_seq)}"
        #assert len(depth_time_seq) == self.seq_len, f"depth_time_seq len={len(depth_time_seq)}"
        #assert depth_tensor_seq.shape[0] == self.seq_len, f"depth_tensor_seq shape={depth_tensor_seq.shape}"

        return {
            "proprio": torch.from_numpy(proprio_seq),              # (T, 58)
            "timestamps_ns": torch.tensor(timestamp_seq, dtype=torch.long),

            "rgb_paths": rgb_path_seq,
            "rgb_timestamps_ns": torch.tensor(rgb_time_seq, dtype=torch.long),

            "depth_paths": depth_path_seq,
            "depth_timestamps_ns": torch.tensor(depth_time_seq, dtype=torch.long),

            "depth": torch.from_numpy(depth_tensor_seq),

            "objective_components": torch.from_numpy(objective_components),

            "targets": {
                "forward_velocity": torch.from_numpy(target_dict["forward_velocity"]),
                "body_height": torch.from_numpy(target_dict["body_height"]),
                "contact": torch.from_numpy(np.stack(contact_seq, axis=0).astype(np.float32)),
            },

            
            
        }


if __name__ == "__main__":
    data_dir = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2"
    dataset = SpotCartoDataset(
        data_dir=data_dir,
        seq_len=20,
        stride=5,
        use_rgb=True,
        use_depth=True,
    )
    
    # for i in [0,100,500]:
    #     print(dataset[i]["objective_components"])

    print("len(dataset) =", len(dataset))

    sample = dataset[0]
    print("\nproprio shape =", sample["proprio"].shape)
    print("timestamps shape =", sample["timestamps_ns"].shape)

    print("\nrgb path[0] =", sample["rgb_paths"][0])
    print("rgb ts shape =", sample["rgb_timestamps_ns"].shape)
    if len(sample["rgb_timestamps_ns"]) > 0:
        print("rgb ts[0] =", sample["rgb_timestamps_ns"][0].item())

    print("depth shape =", sample["depth"].shape)
    print("depth ts shape =", sample["depth_timestamps_ns"].shape)
    if len(sample["depth_timestamps_ns"]) > 0:
        print("depth ts[0] =", sample["depth_timestamps_ns"][0].item())

    print("depth min/max =", sample["depth"][0].min().item(), sample["depth"][0].max().item())

    print("\nforward_velocity shape =", sample["targets"]["forward_velocity"].shape)
    print("body_height shape =", sample["targets"]["body_height"].shape)

    print("\nfirst proprio row:")
    print(sample["proprio"][0])

    print("\n--- time alignment check (first step) ---")
    t_anchor = sample["timestamps_ns"][0].item()

    if len(sample["rgb_timestamps_ns"]) > 0:
        t_rgb = sample["rgb_timestamps_ns"][0].item()
        print("anchor t_ns =", t_anchor)
        print("rgb    t_ns =", t_rgb, " | dt =", abs(t_anchor - t_rgb) / 1e6, "ms")

    if len(sample["depth_timestamps_ns"]) > 0:
        t_depth = sample["depth_timestamps_ns"][0].item()
        print("depth  t_ns =", t_depth, " | dt =", abs(t_anchor - t_depth) / 1e6, "ms")

    print("objective_components shape =", sample["objective_components"].shape)
    print("objective_components =", sample["objective_components"])
    print("J_velocity =", sample["objective_components"][0].item())
    print("J_height =", sample["objective_components"][1].item())
    print("J_energy =", sample["objective_components"][2].item())