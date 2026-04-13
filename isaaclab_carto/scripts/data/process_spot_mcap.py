from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import argparse
import ast
import json
from typing import Any

import numpy as np
import pandas as pd
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore


def ros_time_to_sec(t_ns: int) -> float:
    return t_ns / 1e9


def ensure_dirs(out_dir: Path) -> None:
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    (out_dir / "tabular").mkdir(parents=True, exist_ok=True)


def nearest_time_index(ref_t: float, candidate_ts: np.ndarray) -> int:
    idx = np.searchsorted(candidate_ts, ref_t)
    if idx == 0:
        return 0
    if idx >= len(candidate_ts):
        return len(candidate_ts) - 1
    left = idx - 1
    right = idx
    if abs(candidate_ts[left] - ref_t) <= abs(candidate_ts[right] - ref_t):
        return int(left)
    return int(right)


def safe_literal_list(x: Any) -> Any:
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x


def summarize_topics(topics_df: pd.DataFrame, counts: dict[str, int]) -> None:
    print("\n========== TOPIC SUMMARY ==========")
    print(f"num_topics = {len(topics_df)}")

    count_df = pd.DataFrame(
        [{"topic": k, "count": v} for k, v in counts.items()]
    ).sort_values("count", ascending=False)

    print("\n--- top 20 message counts ---")
    print(count_df.head(20).to_string(index=False))

    interesting_keywords = ["feet", "odom", "joint", "camera", "depth", "command", "cmd", "status"]

    print("\n--- interesting topics ---")
    for kw in interesting_keywords:
        sub = topics_df[topics_df["topic"].str.contains(kw, case=False, na=False)]
        if len(sub) > 0:
            print(f"\n[{kw}]")
            print(sub[["topic", "msgtype"]].to_string(index=False))


def summarize_training_df(df: pd.DataFrame) -> None:
    print("\n========== TRAINING FEATURE SUMMARY ==========")
    print(f"num_rows = {len(df)}")

    cols = [
        "forward_speed",
        "body_height",
        "joint_vel_norm",
        "joint_effort_norm",
        "energy_proxy",
        "reward_total_proxy",
    ]

    for c in cols:
        if c in df.columns:
            print(f"\n--- {c}.describe() ---")
            print(df[c].describe())

    print("\n--- head(training_features) ---")
    show_cols = [c for c in cols if c in df.columns]
    extra = [c for c in ["t", "rgb_time", "depth_time"] if c in df.columns]
    print(df[extra + show_cols].head(10).to_string(index=False))

    if len(df) > 0 and "joint_position" in df.columns:
        try:
            sample = safe_literal_list(df.iloc[0]["joint_position"])
            print("\n--- joint_position length ---")
            print(len(sample))
        except Exception as e:
            print("\n[WARN] failed to inspect joint_position length:", e)


def build_training_features(synced_df: pd.DataFrame) -> pd.DataFrame:
    joint_vel = synced_df["joint_velocity"].apply(safe_literal_list)
    joint_eff = synced_df["joint_effort"].apply(safe_literal_list)

    synced_df["forward_speed"] = synced_df["vx"]
    synced_df["body_height"] = synced_df["pz"]

    synced_df["joint_vel_norm"] = joint_vel.apply(
        lambda x: float(np.linalg.norm(np.array(x, dtype=float))) if x is not None else np.nan
    )
    synced_df["joint_effort_norm"] = joint_eff.apply(
        lambda x: float(np.linalg.norm(np.array(x, dtype=float))) if x is not None else np.nan
    )

    synced_df["energy_proxy"] = synced_df["joint_vel_norm"] * synced_df["joint_effort_norm"]

    # command topic이 아직 없으므로 actual forward speed를 임시 proxy로 사용
    synced_df["reward_velocity_proxy"] = synced_df["forward_speed"]

    # feet/contact custom parser 추가 전까지는 0으로 둠
    synced_df["reward_slip_proxy"] = 0.0

    # 매우 단순한 total proxy
    synced_df["reward_total_proxy"] = (
        1.0 * synced_df["reward_velocity_proxy"]
        - 0.01 * synced_df["energy_proxy"]
        - 0.1 * np.maximum(0.0, 0.20 - synced_df["body_height"])
    )
    return synced_df


def process_bag(
    bag_dir: Path,
    out_dir: Path,
    rgb_topic: str,
    depth_topic: str,
) -> None:
    ensure_dirs(out_dir)
    typestore = get_typestore(Stores.LATEST)

    topic_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)

    joint_rows: list[dict[str, Any]] = []
    odom_rows: list[dict[str, Any]] = []
    twist_rows: list[dict[str, Any]] = []

    image_timestamps: dict[str, list[int]] = defaultdict(list)

    print(f"[INFO] reading bag: {bag_dir}")

    with Reader(bag_dir) as reader:
        connections = list(reader.connections)

        for conn in connections:
            topic_rows.append(
                {
                    "topic": conn.topic,
                    "msgtype": conn.msgtype,
                    "serialization_format": getattr(conn.ext, "serialization_format", ""),
                }
            )

        for conn, timestamp_ns, rawdata in reader.messages(connections=connections):
            topic = conn.topic
            msgtype = conn.msgtype
            counts[topic] += 1

            # 이미지 topic은 지금은 timestamp만 저장
            if topic in [rgb_topic, depth_topic]:
                image_timestamps[topic].append(timestamp_ns)
                continue

            # 표준 메시지만 우선 파싱
            if msgtype not in typestore.types:
                continue

            try:
                msg = typestore.deserialize_cdr(rawdata, msgtype)
            except Exception:
                continue

            if topic == "/joint_states":
                joint_rows.append(
                    {
                        "t": ros_time_to_sec(timestamp_ns),
                        "name": list(msg.name),
                        "position": np.array(msg.position).tolist(),
                        "velocity": np.array(msg.velocity).tolist(),
                        "effort": np.array(msg.effort).tolist(),
                    }
                )

            elif topic == "/odometry":
                odom_rows.append(
                    {
                        "t": ros_time_to_sec(timestamp_ns),
                        "frame_id": msg.header.frame_id,
                        "child_frame_id": msg.child_frame_id,
                        "px": msg.pose.pose.position.x,
                        "py": msg.pose.pose.position.y,
                        "pz": msg.pose.pose.position.z,
                        "qx": msg.pose.pose.orientation.x,
                        "qy": msg.pose.pose.orientation.y,
                        "qz": msg.pose.pose.orientation.z,
                        "qw": msg.pose.pose.orientation.w,
                        "vx": msg.twist.twist.linear.x,
                        "vy": msg.twist.twist.linear.y,
                        "vz": msg.twist.twist.linear.z,
                        "wx": msg.twist.twist.angular.x,
                        "wy": msg.twist.twist.angular.y,
                        "wz": msg.twist.twist.angular.z,
                    }
                )

            elif topic == "/odometry/twist":
                twist_rows.append(
                    {
                        "t": ros_time_to_sec(timestamp_ns),
                        "frame_id": msg.header.frame_id,
                        "vx": msg.twist.twist.linear.x,
                        "vy": msg.twist.twist.linear.y,
                        "vz": msg.twist.twist.linear.z,
                        "wx": msg.twist.twist.angular.x,
                        "wy": msg.twist.twist.angular.y,
                        "wz": msg.twist.twist.angular.z,
                    }
                )

    topics_df = pd.DataFrame(topic_rows).drop_duplicates().sort_values("topic")
    topics_df.to_csv(out_dir / "meta" / "topics.csv", index=False)

    with open(out_dir / "meta" / "message_counts.json", "w") as f:
        json.dump(dict(counts), f, indent=2)

    summarize_topics(topics_df, counts)

    joints_df = pd.DataFrame(joint_rows)
    odom_df = pd.DataFrame(odom_rows)
    twist_df = pd.DataFrame(twist_rows)

    joints_df.to_csv(out_dir / "tabular" / "joint_states.csv", index=False)
    odom_df.to_csv(out_dir / "tabular" / "odometry.csv", index=False)
    twist_df.to_csv(out_dir / "tabular" / "odometry_twist.csv", index=False)

    print("\n[INFO] saved raw csv files")

    if len(odom_df) == 0 or len(joints_df) == 0:
        print("[WARN] odometry or joint_states is empty. stop here.")
        return

    rgb_ts = np.array(sorted([t / 1e9 for t in image_timestamps.get(rgb_topic, [])]), dtype=float)
    depth_ts = np.array(sorted([t / 1e9 for t in image_timestamps.get(depth_topic, [])]), dtype=float)
    joint_ts = joints_df["t"].values.astype(float)

    synced_rows: list[dict[str, Any]] = []
    for _, o in odom_df.iterrows():
        t = float(o["t"])
        j_idx = nearest_time_index(t, joint_ts)

        rgb_t = rgb_ts[nearest_time_index(t, rgb_ts)] if len(rgb_ts) > 0 else np.nan
        depth_t = depth_ts[nearest_time_index(t, depth_ts)] if len(depth_ts) > 0 else np.nan

        synced_rows.append(
            {
                "t": t,
                "px": o["px"],
                "py": o["py"],
                "pz": o["pz"],
                "qx": o["qx"],
                "qy": o["qy"],
                "qz": o["qz"],
                "qw": o["qw"],
                "vx": o["vx"],
                "vy": o["vy"],
                "vz": o["vz"],
                "wx": o["wx"],
                "wy": o["wy"],
                "wz": o["wz"],
                "joint_t": joints_df.iloc[j_idx]["t"],
                "joint_position": joints_df.iloc[j_idx]["position"],
                "joint_velocity": joints_df.iloc[j_idx]["velocity"],
                "joint_effort": joints_df.iloc[j_idx]["effort"],
                "rgb_topic": rgb_topic,
                "rgb_time": rgb_t,
                "depth_topic": depth_topic,
                "depth_time": depth_t,
            }
        )

    synced_df = pd.DataFrame(synced_rows)
    synced_df.to_csv(out_dir / "tabular" / "synced_frames.csv", index=False)

    print("[INFO] saved synced_frames.csv")

    train_df = build_training_features(synced_df.copy())
    train_df.to_csv(out_dir / "tabular" / "training_features.csv", index=False)

    print("[INFO] saved training_features.csv")
    summarize_training_df(train_df)

    print("\n[INFO] done.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bag-dir",
        type=str,
        required=True,
        help="Path to rosbag2/mcap directory",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--rgb-topic",
        type=str,
        default="/camera/frontleft/image",
    )
    parser.add_argument(
        "--depth-topic",
        type=str,
        default="/depth_registered/frontleft/image",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_bag(
        bag_dir=Path(args.bag_dir).expanduser(),
        out_dir=Path(args.out_dir).expanduser(),
        rgb_topic=args.rgb_topic,
        depth_topic=args.depth_topic,
    )