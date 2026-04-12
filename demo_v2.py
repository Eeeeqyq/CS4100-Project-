"""
Presentation-friendly demo wrapper for the rebuilt V2.2 pipeline.

Shows the same held-out context under the four explicit goals so the user-facing
behavior is easy to compare without exposing the intermediate training stages.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent

from src.v2.inference.recommend import AmbientRecommenderV2


DEFAULT_GOALS = ["focus", "wind_down", "uplift", "movement"]
PREFERRED_ROW_IDS = [128, 129]


def pick_demo_row(recommender: AmbientRecommenderV2, split: str) -> int:
    split_rows = recommender.decision_df.index[recommender.decision_df["split"] == split].to_numpy(dtype=np.int64)
    split_set = set(int(x) for x in split_rows.tolist())
    for idx in PREFERRED_ROW_IDS:
        if idx in split_set:
            return int(idx)
    return int(split_rows[0])


def print_context_summary(recommender: AmbientRecommenderV2, row_idx: int) -> None:
    row = recommender.decision_df.iloc[int(row_idx)]
    print("=" * 96)
    print("V2.2 DEMO")
    print("=" * 96)
    print(f"Decision row: {int(row_idx)}")
    print(f"Decision id: {row['decision_id']}")
    print(f"User id: {row['user_id']}")
    print(f"Split: {row['split']}")
    print(
        f"Pre-state: valence={float(row['pre_valence']):+.2f}  "
        f"arousal={float(row['pre_arousal']):+.2f}"
    )
    print(
        f"Context: time_bucket={int(row['time_bucket'])}  weather_bucket={int(row['weather_bucket'])}  "
        f"speed_norm={float(row['speed_norm']):.2f}"
    )
    print()


def print_goal_result(recommender: AmbientRecommenderV2, row_idx: int, goal: str, top_k: int, candidate_k: int) -> None:
    result = recommender.score_rows(
        row_indices=np.asarray([row_idx], dtype=np.int64),
        explicit_goal=goal,
        top_k=top_k,
        candidate_k=candidate_k,
    )
    top1 = result["final_recommendations"][0][0]

    print("-" * 96)
    print(f"Goal: {goal}")
    print(
        f"Top-1: {top1['title']} / {top1['artist']}  "
        f"[{top1['source']}]  score={float(top1['display_score']):.4f}"
    )
    print(
        f"Kind: {top1['kind']}  supported={bool(top1.get('supported', False))}  "
        f"benefit_hat={float(top1.get('benefit_hat', 0.0)):.4f}  "
        f"accept_hat={float(top1.get('accept_hat', 0.0)):.4f}"
    )
    print(f"Reason: {top1.get('reason', '')}")
    print("Top recommendations:")
    for rank, item in enumerate(result["final_recommendations"][0], start=1):
        print(
            f"  {rank}. {item['title']} / {item['artist']} [{item['source']}] "
            f"score={float(item['display_score']):.4f}  reason={item.get('reason', '')}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a clean presentation demo for V2.2.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--row-idx", type=int, default=-1, help="Use a specific decision-table row if desired.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    recommender = AmbientRecommenderV2(device=args.device)
    row_idx = int(args.row_idx) if int(args.row_idx) >= 0 else pick_demo_row(recommender, args.split)

    print_context_summary(recommender, row_idx)
    for goal in DEFAULT_GOALS:
        print_goal_result(
            recommender=recommender,
            row_idx=row_idx,
            goal=goal,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
        )


if __name__ == "__main__":
    main()
