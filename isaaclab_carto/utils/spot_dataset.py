from __future__ import annotations

from pathlib import Path
import json
import ast
from typing import Any

import pandas as pd


def _safe_literal(x: Any) -> Any:
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x


def load_csv(csv_path: str | Path) -> list[dict[str, Any]]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    records = df.to_dict(orient="records")

    # 문자열로 저장된 list/dict를 복원
    parsed_records = []
    for row in records:
        parsed_row = {}
        for k, v in row.items():
            parsed_row[k] = _safe_literal(v)
        parsed_records.append(parsed_row)

    return parsed_records


def load_jsonl(jsonl_path: str | Path) -> list[dict[str, Any]]:
    jsonl_path = Path(jsonl_path)
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


class SpotDataset:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir).expanduser() 

        self.joint = load_csv(self.data_dir / "tabular" / "joint_states.csv")
        self.odom = load_csv(self.data_dir / "tabular" / "odometry.csv")
        self.feet = load_jsonl(self.data_dir / "status" / "feet.jsonl")

        self.length = min(len(self.joint), len(self.odom), len(self.feet))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.length}")

        return {
            "joint": self.joint[idx],
            "odom": self.odom[idx],
            "feet": self.feet[idx],
        }


if __name__ == "__main__":
    data_dir = "~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2"
    dataset = SpotDataset(data_dir)

    print("len(dataset) =", len(dataset))
    sample = dataset[0]

    print("\n[joint sample]")
    print(sample["joint"])

    print("\n[odom sample]")
    print(sample["odom"])

    print("\n[feet sample]")
    print(sample["feet"])