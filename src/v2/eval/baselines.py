"""
Baselines for the V2.2 anchor-retrieval and reranking stages.

Answers: how much better than simple alternatives is the learned query tower /
reranker? All metrics reuse the definitions in offline_eval.py so the numbers
are directly comparable to offline_eval_v2.json.

Baselines (all use only information available before listening):
- random over all train anchors
- random within the router goal
- best historical anchors for the goal (sort by train anchor success, no context)
- nearest neighbours within the goal in the 15-D anchor context space

Caveat: tier-3 positives are *defined* as same-goal, high-success anchors that
are close in that same 15-D space (see src/v2/data/anchors.py), so the kNN and
success baselines partly replicate the labelling rule. They are reported as
reference points that expose how circular the labels are, not as fair rivals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v2.data.anchor_features import load_supervision_sets, tier_gain_array
from src.v2.eval.offline_eval import (
    best_rank_in_order,
    map_global_sets_to_train,
    weighted_stats_for_order,
)

REBUILD_DIR = PROJECT_ROOT / "data" / "processed" / "rebuild"
MODELS_DIR = PROJECT_ROOT / "models" / "rebuild"
HEADLINE_METRICS = (
    "recall_at_20",
    "weighted_recall_at_20",
    "weighted_ndcg_at_10",
    "hit_at_10",
)


def rank_with_goal_priority(
    scores: np.ndarray, anchor_goal: np.ndarray, query_goal: int | None
) -> np.ndarray:
    """Same-goal anchors first, each block sorted by score desc, ties broken by index."""
    scores = np.asarray(scores, dtype=np.float64)
    other_goal = (
        np.zeros(len(scores), dtype=np.int8)
        if query_goal is None
        else (anchor_goal != query_goal).astype(np.int8)
    )
    # np.lexsort sorts by the last key first.
    return np.lexsort((np.arange(len(scores)), -scores, other_goal))


def knn_scores(query_features: np.ndarray, anchor_features: np.ndarray) -> np.ndarray:
    """Negative squared Euclidean distance: higher means closer."""
    diff = np.asarray(anchor_features, dtype=np.float64) - np.asarray(
        query_features, dtype=np.float64
    )
    return -np.sum(diff * diff, axis=1)


def random_order(
    rng: np.random.Generator, anchor_goal: np.ndarray, query_goal: int | None
) -> np.ndarray:
    perm = rng.permutation(len(anchor_goal))
    if query_goal is None:
        return perm
    same = anchor_goal[perm] == query_goal
    return np.concatenate([perm[same], perm[~same]])


def order_metrics(
    order: np.ndarray, positives: set[int], gains: dict[int, float]
) -> dict[str, float]:
    rank = best_rank_in_order(order, positives)
    weighted = weighted_stats_for_order(
        order, positives, gains, factual_positives=set()
    )
    return {
        "best_rank": float(rank),
        "hit_at_10": float(rank <= 10),
        "recall_at_20": float(rank <= 20),
        "recall_at_50": float(rank <= 50),
        "weighted_recall_at_20": weighted["weighted_recall_at_20"],
        "weighted_recall_at_50": weighted["weighted_recall_at_50"],
        "weighted_ndcg_at_10": weighted["weighted_ndcg_at_10"],
    }


def user_bootstrap_ci(
    values: np.ndarray,
    users: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 2000,
) -> tuple[float, float]:
    """95% CI from resampling whole users (rows from one user are not independent)."""
    values = np.asarray(values, dtype=np.float64)
    unique_users = np.unique(users)
    per_user = [values[users == user] for user in unique_users]
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        picked = rng.integers(0, len(per_user), size=len(per_user))
        means[b] = np.concatenate([per_user[i] for i in picked]).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mean_random_metrics(
    rng: np.random.Generator,
    n_reps: int,
    anchor_goal: np.ndarray,
    query_goal: int | None,
    positives: set[int],
    gains: dict[int, float],
) -> dict[str, float]:
    runs = [
        order_metrics(random_order(rng, anchor_goal, query_goal), positives, gains)
        for _ in range(n_reps)
    ]
    return {key: float(np.mean([run[key] for run in runs])) for key in runs[0]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare V2.2 retrieval/reranking against simple baselines."
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--random-reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    from src.v2.inference.recommend import AmbientRecommenderV2

    rng = np.random.default_rng(args.seed)
    recommender = AmbientRecommenderV2(device=args.device)
    decision_df = recommender.decision_df
    anchor_df = recommender.anchor_df
    if not (
        anchor_df["decision_id"].to_numpy() == decision_df["decision_id"].to_numpy()
    ).all():
        raise RuntimeError("anchor_table and decision_table rows are not aligned.")

    row_indices = decision_df.index[decision_df["split"] == args.split].to_numpy(
        dtype=np.int64
    )
    train_idx = np.asarray(recommender.train_anchor_idx, dtype=np.int64)
    context = np.load(REBUILD_DIR / "anchor_context_features.npy").astype(np.float32)
    anchor_goal = anchor_df["goal_idx"].to_numpy(dtype=np.int64)[train_idx]
    anchor_success = anchor_df["anchor_success_obs"].to_numpy(dtype=np.float64)[
        train_idx
    ]
    anchor_context = context[train_idx]

    bundle = load_supervision_sets(str(REBUILD_DIR / "anchor_positive_sets.npz"))
    global_to_local = np.full(len(anchor_df), -1, dtype=np.int32)
    global_to_local[train_idx] = np.arange(len(train_idx), dtype=np.int32)
    pos_local, pos_count, pos_tier = map_global_sets_to_train(
        bundle["indices"], bundle["counts"], global_to_local, meta=bundle["tiers"]
    )

    result = recommender.score_rows(
        row_indices=row_indices,
        explicit_goal=None,
        top_k=max(50, args.candidate_k),
        candidate_k=args.candidate_k,
    )

    records = []
    for local_row, global_idx in enumerate(row_indices.tolist()):
        n_pos = int(pos_count[global_idx])
        positives = {int(i) for i in pos_local[global_idx, :n_pos].tolist() if i >= 0}
        tiers = pos_tier[global_idx, :n_pos]
        gains = {
            int(idx): float(gain)
            for idx, gain in zip(
                pos_local[global_idx, :n_pos].tolist(), tier_gain_array(tiers).tolist()
            )
            if idx >= 0
        }
        goal = int(result["goal_idx"][local_row])
        query_order = np.asarray(
            result["candidate_local_idx"][local_row], dtype=np.int64
        )
        rerank_order = query_order[
            np.asarray(result["anchor_order"][local_row], dtype=np.int64)
        ]
        knn = knn_scores(context[global_idx], anchor_context)

        retrieval = {
            "learned_query_tower": order_metrics(query_order, positives, gains),
            "random_all_anchors": mean_random_metrics(
                rng, args.random_reps, anchor_goal, None, positives, gains
            ),
            "random_within_goal": mean_random_metrics(
                rng, args.random_reps, anchor_goal, goal, positives, gains
            ),
            "best_success_within_goal": order_metrics(
                rank_with_goal_priority(anchor_success, anchor_goal, goal),
                positives,
                gains,
            ),
            "knn_within_goal": order_metrics(
                rank_with_goal_priority(knn, anchor_goal, goal), positives, gains
            ),
        }
        # Rerankers all reorder the SAME learned top-k candidate set.
        cand_success = anchor_success[query_order]
        cand_knn = knn[query_order]
        cand_goal = anchor_goal[query_order]
        rerank = {
            "query_order": order_metrics(query_order, positives, gains),
            "learned_reranker": order_metrics(rerank_order, positives, gains),
            "sort_by_success": order_metrics(
                query_order[rank_with_goal_priority(cand_success, cand_goal, goal)],
                positives,
                gains,
            ),
            "sort_by_knn": order_metrics(
                query_order[rank_with_goal_priority(cand_knn, cand_goal, goal)],
                positives,
                gains,
            ),
        }
        record = {
            "row_idx": int(global_idx),
            "user_id": int(decision_df.iloc[global_idx]["user_id"]),
        }
        for stage, methods in (("retrieval", retrieval), ("rerank", rerank)):
            for method, metrics in methods.items():
                for key, value in metrics.items():
                    record[f"{stage}|{method}|{key}"] = value
        records.append(record)

    df = pd.DataFrame(records)
    users = df["user_id"].to_numpy()
    summary: dict = {
        "split": args.split,
        "rows": len(df),
        "users": len(np.unique(users)),
        "candidate_k": int(args.candidate_k),
        "random_reps": int(args.random_reps),
        "seed": int(args.seed),
        "ci": "95% bootstrap over users (2000 resamples)",
        "retrieval": {},
        "rerank": {},
    }
    for column in df.columns:
        if "|" not in column:
            continue
        stage, method, key = column.split("|")
        entry = summary[stage].setdefault(method, {})
        entry[key] = float(df[column].mean())
        if key in HEADLINE_METRICS:
            entry[f"{key}_ci95"] = list(
                user_bootstrap_ci(df[column].to_numpy(), users, rng)
            )

    (MODELS_DIR / "baselines_v2.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    def fmt(entry: dict, key: str) -> str:
        lo, hi = entry[f"{key}_ci95"]
        return f"{entry[key]:.3f} [{lo:.2f}, {hi:.2f}]"

    print(
        f"Split={args.split}  rows={summary['rows']}  users={summary['users']}  (95% CI: bootstrap over users)"
    )
    print("\nRetrieval over all train anchors")
    print(
        f"{'method':28s} {'recall@20':>20s} {'weighted recall@20':>20s} {'weighted NDCG@10':>20s}"
    )
    for method, entry in summary["retrieval"].items():
        print(
            f"{method:28s} {fmt(entry, 'recall_at_20'):>20s} "
            f"{fmt(entry, 'weighted_recall_at_20'):>20s} {fmt(entry, 'weighted_ndcg_at_10'):>20s}"
        )
    print(f"\nReranking the learned top-{args.candidate_k}")
    print(f"{'method':28s} {'hit@10':>20s} {'weighted NDCG@10':>20s}")
    for method, entry in summary["rerank"].items():
        print(
            f"{method:28s} {fmt(entry, 'hit_at_10'):>20s} {fmt(entry, 'weighted_ndcg_at_10'):>20s}"
        )


if __name__ == "__main__":
    main()
