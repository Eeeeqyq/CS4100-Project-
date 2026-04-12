"""
Presentation-friendly evaluation wrapper for the rebuilt V2.2 pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models" / "rebuild"


def run_command(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def print_primary_metrics(primary: dict[str, float]) -> None:
    print("Primary Metrics")
    print("-" * 96)
    print(f"{'Anchor query recall@20':<38} {primary['anchor_query_recall_at_20']:.4f}")
    print(f"{'Anchor query recall@50':<38} {primary['anchor_query_recall_at_50']:.4f}")
    print(f"{'Weighted query recall@20':<38} {primary['anchor_query_weighted_recall_at_20']:.4f}")
    print(f"{'Anchor rerank hit@10':<38} {primary['anchor_rerank_hit_at_10']:.4f}")
    print(f"{'Anchor rerank mean rank':<38} {primary['anchor_rerank_conditional_mean_rank']:.4f}")
    print(f"{'Weighted rerank NDCG@10':<38} {primary['anchor_rerank_weighted_ndcg_at_10']:.4f}")
    print(f"{'Benefit MAE':<38} {primary['benefit_mae']:.4f}")
    print(f"{'Blended acceptance MAE':<38} {primary['blended_accept_mae']:.4f}")
    print(f"{'Top-1 predicted acceptance mean':<38} {primary['top1_predicted_accept_mean']:.4f}")
    print(f"{'Public-transfer-supported share':<38} {primary['public_transfer_supported_share']:.4f}")
    print(f"{'Top-1 source max share':<38} {primary['top1_source_max_share']:.4f}")
    print()


def print_goal_breakdown(goal_breakdown: dict[str, dict[str, float]]) -> None:
    print("Per-Goal Summary")
    print("-" * 96)
    print(f"{'Goal':<14} {'Rows':>6} {'WQ@20':>10} {'Hit@10':>10} {'MeanRank':>10}")
    for goal in ["focus", "wind_down", "uplift", "movement"]:
        if goal not in goal_breakdown:
            continue
        payload = goal_breakdown[goal]
        print(
            f"{goal:<14} {int(payload['rows']):>6} "
            f"{float(payload['anchor_query_weighted_recall_at_20']):>10.4f} "
            f"{float(payload['anchor_rerank_hit_at_10']):>10.4f} "
            f"{float(payload['anchor_rerank_conditional_mean_rank']):>10.4f}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or print the rebuilt V2.2 evaluation summary.")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--no-rerun", action="store_true", help="Read the latest saved artifacts without rerunning eval.")
    args = parser.parse_args()

    if not args.no_rerun:
        run_command([sys.executable, "src/v2/eval/offline_eval.py", "--split", "test", "--candidate-k", str(args.candidate_k)])
        run_command([sys.executable, "src/v2/eval/check_readiness.py"])

    offline_eval = load_json(MODELS_DIR / "offline_eval_v2.json")
    readiness = load_json(MODELS_DIR / "v2_readiness.json")

    print("=" * 96)
    print("V2.2 OFFLINE EVALUATION")
    print("=" * 96)
    print(f"Readiness: {'READY' if bool(readiness['ready']) else 'NOT READY'}")
    print(f"Rows evaluated: {int(offline_eval['rows'])}")
    print(f"Candidate set size: {int(offline_eval['candidate_k'])}")
    print()

    print_primary_metrics(offline_eval["primary_metrics"])
    print_goal_breakdown(offline_eval["goal_breakdown"])

    print("Top-1 Source Distribution")
    print("-" * 96)
    for source, share in offline_eval["top1_source_distribution"].items():
        print(f"{source:<14} {float(share):.4f}")
    print()

    print("Legacy Diagnostics")
    print("-" * 96)
    legacy = offline_eval["legacy_diagnostics"]
    print(f"{'Exact-song query recall@50':<38} {float(legacy['exact_song_query_recall_at_50']):.4f}")
    print(f"{'Exact-song rerank hit@10':<38} {float(legacy['exact_song_rerank_hit_at_10']):.4f}")
    print(f"{'Exact-song conditional rank':<38} {float(legacy['exact_song_conditional_rank']):.4f}")
    print()

    if readiness["blockers"]:
        print("Blockers")
        print("-" * 96)
        for blocker in readiness["blockers"]:
            print(f"- {blocker}")
        print()

    print("Artifacts")
    print("-" * 96)
    print(f"- {MODELS_DIR / 'offline_eval_v2.json'}")
    print(f"- {MODELS_DIR / 'v2_readiness.json'}")


if __name__ == "__main__":
    main()
