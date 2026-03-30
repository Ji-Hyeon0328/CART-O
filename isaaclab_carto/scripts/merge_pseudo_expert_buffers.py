import os
import json
import argparse
from glob import glob


def main():
    parser = argparse.ArgumentParser(description="Merge multiple pseudo_expert_buffer.json files")
    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Root directory containing multiple run folders, e.g. logs/carto_spot",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save merged pseudo expert buffer JSON",
    )
    args = parser.parse_args()

    pattern = os.path.join(args.input_root, "*", "pseudo_expert_buffer.json")
    files = sorted(glob(pattern))

    if len(files) == 0:
        raise RuntimeError(f"No pseudo_expert_buffer.json files found under: {args.input_root}")

    merged = []
    total_files = 0

    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"[WARN] skipping non-list file: {fpath}")
            continue

        merged.extend(data)
        total_files += 1
        print(f"[INFO] loaded {len(data)} episodes from {fpath}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print("============================================================")
    print(f"[DONE] merged files: {total_files}")
    print(f"[DONE] total merged episodes: {len(merged)}")
    print(f"[DONE] saved merged buffer -> {args.output_path}")
    print("============================================================")


if __name__ == "__main__":
    main()
