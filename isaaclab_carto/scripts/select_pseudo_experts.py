import os
import json
import argparse

from isaaclab_carto.utils.pseudo_expert_buffer import PseudoExpertBuffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="per_preset", choices=["global", "per_preset"])
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--k-per-preset", type=int, default=50)
    parser.add_argument("--min-episode-length", type=int, default=50)
    args = parser.parse_args()

    buffer = PseudoExpertBuffer.load_json(args.buffer_path)
    print(f"[INFO] loaded buffer with {len(buffer)} episodes")

    buffer.annotate_scores(
        w_success=2.0,
        w_velocity=1.0,
        w_slip=1.0,
        w_energy=1.0,
    )

    if args.mode == "global":
        selected = buffer.select_top_k(
            k=args.top_k,
            ensure_success=False, #False <-> True
            diversify_by_terrain=False,
            min_episode_length=args.min_episode_length,
            min_mean_velocity=-1e9,
            min_mean_slip=-1e9,
            min_mean_energy=-1e9,
        )
    else:
        selected = buffer.select_top_k_per_preset(
            k_per_preset=args.k_per_preset,
            ensure_success=False, #False <-> True
            min_episode_length=args.min_episode_length,
            min_mean_velocity=-1e9,
            min_mean_slip=-1e9,
            min_mean_energy=-1e9,
        )

    print(f"[INFO] selected {len(selected)} pseudo-expert episodes")

    dataset = buffer.build_selector_dataset(only_selected=True)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"[INFO] saved selector dataset -> {args.output_path}")


if __name__ == "__main__":
    main()