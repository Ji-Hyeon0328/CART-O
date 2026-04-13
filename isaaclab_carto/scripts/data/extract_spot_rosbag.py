from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistWithCovarianceStamped

from rosidl_runtime_py.convert import message_to_ordereddict

# Spot custom msgs
from spot_msgs.msg import FootStateArray, Feedback


def time_msg_to_ns(time_msg) -> int:
    return int(time_msg.sec) * 10**9 + int(time_msg.nanosec)


def try_get_stamp_ns(msg) -> int | None:
    # case 1: standard ROS header
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        return time_msg_to_ns(msg.header.stamp)

    # case 2: direct stamp field
    if hasattr(msg, "stamp"):
        stamp = getattr(msg, "stamp")
        if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
            return time_msg_to_ns(stamp)

    # case 3: acquisition timestamp style names
    candidate_names = [
        "acquisition_timestamp",
        "timestamp",
        "robot_timestamp",
        "local_timestamp",
    ]
    for name in candidate_names:
        if hasattr(msg, name):
            stamp = getattr(msg, name)
            if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
                return time_msg_to_ns(stamp)

    return None


def ensure_dirs(out_dir: Path) -> None:
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    (out_dir / "tabular").mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / "frontleft").mkdir(parents=True, exist_ok=True)
    (out_dir / "depth_registered" / "frontleft").mkdir(parents=True, exist_ok=True)
    (out_dir / "status").mkdir(parents=True, exist_ok=True)


def image_msg_to_numpy(msg: Image) -> np.ndarray:
    h, w = msg.height, msg.width
    enc = msg.encoding.lower()

    if enc in ("rgb8", "bgr8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
        if enc == "rgb8":
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr

    if enc == "mono8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)

    if enc in ("16uc1", "mono16"):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)

    if enc == "32fc1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)

    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


class SpotBagExtractor(Node):

    def __init__(self, out_dir: Path):
        super().__init__("spot_bag_extractor")
        self.out_dir = out_dir
        ensure_dirs(self.out_dir)

        self.counts = {
            "joint_states": 0,
            "odometry": 0,
            "odometry_twist": 0,
            "feet": 0,
            "feedback": 0,
            "rgb_frontleft": 0,
            "depth_registered_frontleft": 0,
        }

        self.last_ns = {}

        # CSV writers
        self.joint_csv_file = open(self.out_dir / "tabular" / "joint_states.csv", "w", newline="")
        self.odom_csv_file = open(self.out_dir / "tabular" / "odometry.csv", "w", newline="")
        self.twist_csv_file = open(self.out_dir / "tabular" / "odometry_twist.csv", "w", newline="")

        self.joint_writer = csv.writer(self.joint_csv_file)
        self.odom_writer = csv.writer(self.odom_csv_file)
        self.twist_writer = csv.writer(self.twist_csv_file)

        self.joint_writer.writerow(["t_ns", "name_json", "position_json", "velocity_json", "effort_json"])
        self.odom_writer.writerow([
            "t_ns",
            "frame_id",
            "child_frame_id",
            "px", "py", "pz",
            "qx", "qy", "qz", "qw",
            "vx", "vy", "vz",
            "wx", "wy", "wz",
        ])
        self.twist_writer.writerow([
            "t_ns",
            "frame_id",
            "vx", "vy", "vz",
            "wx", "wy", "wz",
        ])

        # JSONL files for custom msgs
        self.feet_file = open(self.out_dir / "status" / "feet.jsonl", "w")
        self.feedback_file = open(self.out_dir / "status" / "feedback.jsonl", "w")

        qos = 10

        # Subscriptions
        self.create_subscription(JointState, "/joint_states", self.cb_joint_states, qos)
        self.create_subscription(Odometry, "/odometry", self.cb_odometry, qos)
        self.create_subscription(TwistWithCovarianceStamped, "/odometry/twist", self.cb_odometry_twist, qos)

        self.create_subscription(FootStateArray, "/status/feet", self.cb_feet, qos)
        self.create_subscription(Feedback, "/status/feedback", self.cb_feedback, qos)

        self.create_subscription(Image, "/camera/frontleft/image", self.cb_rgb_frontleft, qos)
        self.create_subscription(Image, "/depth_registered/frontleft/image", self.cb_depth_registered_frontleft, qos)

        self.get_logger().info("SpotBagExtractor initialized.")

    def get_msg_timestamp_ns(self, msg) -> int:
        t_ns = try_get_stamp_ns(msg)
        if t_ns is not None:
            return t_ns

        # fallback: message receive time
        return self.get_clock().now().nanoseconds

    def cb_joint_states(self, msg: JointState):
        t_ns = self.get_msg_timestamp_ns(msg)
        self.joint_writer.writerow([
            t_ns,
            json.dumps(list(msg.name)),
            json.dumps(list(msg.position)),
            json.dumps(list(msg.velocity)),
            json.dumps(list(msg.effort)),
        ])
        self.counts["joint_states"] += 1
        self.last_ns["joint_states"] = t_ns

    def cb_odometry(self, msg: Odometry):
        t_ns = self.get_msg_timestamp_ns(msg)
        self.odom_writer.writerow([
            t_ns,
            msg.header.frame_id,
            msg.child_frame_id,
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ])
        self.counts["odometry"] += 1
        self.last_ns["odometry"] = t_ns

    def cb_odometry_twist(self, msg: TwistWithCovarianceStamped):
        t_ns = self.get_msg_timestamp_ns(msg)
        self.twist_writer.writerow([
            t_ns,
            msg.header.frame_id,
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ])
        self.counts["odometry_twist"] += 1
        self.last_ns["odometry_twist"] = t_ns

    def cb_feet(self, msg: FootStateArray):
        try:
            t_ns = self.get_msg_timestamp_ns(msg)
            payload = {
                "t_ns": t_ns,
                "data": message_to_ordereddict(msg),
            }
            self.feet_file.write(json.dumps(payload) + "\n")
            self.counts["feet"] += 1
            self.last_ns["feet"] = t_ns
        except Exception as e:
            self.get_logger().error(f"Feet callback failed: {e}")

    def cb_feedback(self, msg: Feedback):
        try:
            t_ns = self.get_msg_timestamp_ns(msg)
            payload = {
                "t_ns": t_ns,
                "data": message_to_ordereddict(msg),
            }
            self.feedback_file.write(json.dumps(payload) + "\n")
            self.counts["feedback"] += 1
            self.last_ns["feedback"] = t_ns
            
        except Exception as e:
            self.get_logger().error(f"Feedback callback failed: {e}")

    def cb_rgb_frontleft(self, msg: Image):
        t_ns = self.get_msg_timestamp_ns(msg)
        try:
            arr = image_msg_to_numpy(msg)
            cv2.imwrite(str(self.out_dir / "images" / "frontleft" / f"{t_ns}.png"), arr)
            self.counts["rgb_frontleft"] += 1
            self.last_ns["rgb_frontleft"] = t_ns
        except Exception as e:
            self.get_logger().error(f"RGB save failed at {t_ns}: {e}")

    def cb_depth_registered_frontleft(self, msg: Image):
        t_ns = self.get_msg_timestamp_ns(msg)
        try:
            arr = image_msg_to_numpy(msg)
            np.save(self.out_dir / "depth_registered" / "frontleft" / f"{t_ns}.npy", arr)
            self.counts["depth_registered_frontleft"] += 1
            self.last_ns["depth_registered_frontleft"] = t_ns
        except Exception as e:
            self.get_logger().error(f"Depth save failed at {t_ns}: {e}")

    def write_summary(self):
        summary = {
            "counts": self.counts,
            "last_timestamps_ns": self.last_ns,
        }
        with open(self.out_dir / "meta" / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("\n========== EXTRACTION SUMMARY ==========")
        for k, v in self.counts.items():
            print(f"{k}: {v}")
        print("\nlast timestamps:")
        for k, v in self.last_ns.items():
            print(f"{k}: {v}")

    def close_all(self):
        self.joint_csv_file.close()
        self.odom_csv_file.close()
        self.twist_csv_file.close()
        self.feet_file.close()
        self.feedback_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory, e.g. ~/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()

    rclpy.init()
    node = SpotBagExtractor(out_dir)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received, shutting down.")
    finally:
        node.write_summary()
        node.close_all()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()