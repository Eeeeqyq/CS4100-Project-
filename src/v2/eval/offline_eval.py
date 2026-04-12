"""
Offline evaluation for the rebuilt V2.2 pipeline.

Primary contract:
- anchor retrieval
- anchor reranking
- anchor-conditioned public transfer

Legacy exact-song recovery is retained as diagnostics only.
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
from src.v2.inference.recommend import AmbientRecommenderV2


REBUILD_DIR = PROJECT_ROOT / "data" / "processed" / "rebuild"
MODELS_DIR = PROJECT_ROOT / "models" / "rebuild"


def map_global_sets_to_train(
    indices: np.ndarray,
    counts: np.ndarray,
    global_to_local: np.ndarray,
    meta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    mapped = np.full_like(indices, -1, dtype=np.int32)
    out_counts = np.zeros(len(counts), dtype=np.int32)
    mapped_meta = np.zeros_like(indices, dtype=np.int32) if meta is not None else None
    for row in range(len(indices)):
        valid: list[int] = []
        valid_meta: list[int] = []
        seen: set[int] = set()
        for col, idx in enumerate(indices[row, : counts[row]].tolist()):
            if idx < 0:
                continue
            local = int(global_to_local[int(idx)])
            if local < 0 or local in seen:
                continue
            valid.append(local)
            if mapped_meta is not None and meta is not None:
                valid_meta.append(int(meta[row, col]))
            seen.add(local)
        out_counts[row] = len(valid)
        if valid:
            mapped[row, : len(valid)] = np.asarray(valid, dtype=np.int32)
            if mapped_meta is not None:
                mapped_meta[row, : len(valid_meta)] = np.asarray(valid_meta, dtype=np.int32)
    return mapped, out_counts, mapped_meta


def best_rank_in_order(order: np.ndarray, positives: set[int]) -> int:
    if not positives:
        return len(order) + 1
    for rank, idx in enumerate(order.tolist(), start=1):
        if int(idx) in positives:
            return rank
    return len(order) + 1


def weighted_stats_for_order(
    order: np.ndarray,
    positives: set[int],
    gains: dict[int, float],
    factual_positives: set[int],
    top_ks: tuple[int, int] = (20, 50),
) -> dict[str, float]:
    total_gain = float(sum(gains.values()))
    weighted_recall = {}
    for k in top_ks:
        top_idx = order[: min(k, len(order))]
        hit_gain = float(sum(gains.get(int(idx), 0.0) for idx in top_idx.tolist()))
        weighted_recall[k] = hit_gain / total_gain if total_gain > 0.0 else 0.0

    top10 = order[: min(10, len(order))]
    dcg = 0.0
    for rank, idx in enumerate(top10.tolist(), start=1):
        gain = float(gains.get(int(idx), 0.0))
        if gain > 0.0:
            dcg += gain / float(np.log2(rank + 1.0))
    ideal_gains = sorted(gains.values(), reverse=True)[: len(top10)]
    idcg = sum(gain / float(np.log2(rank + 1.0)) for rank, gain in enumerate(ideal_gains, start=1))
    weighted_ndcg_at_10 = dcg / idcg if idcg > 0.0 else 0.0

    weighted_mrr = 0.0
    first_positive_is_factual = 0.0
    for rank, idx in enumerate(order.tolist(), start=1):
        local_idx = int(idx)
        if local_idx in positives:
            weighted_mrr = float(gains.get(local_idx, 0.0)) / float(rank)
            first_positive_is_factual = float(local_idx in factual_positives)
            break

    factual_rank = len(order) + 1
    if factual_positives:
        for rank, idx in enumerate(order.tolist(), start=1):
            if int(idx) in factual_positives:
                factual_rank = rank
                break

    return {
        "weighted_recall_at_20": float(weighted_recall[20]),
        "weighted_recall_at_50": float(weighted_recall[50]),
        "weighted_mrr": float(weighted_mrr),
        "weighted_ndcg_at_10": float(weighted_ndcg_at_10),
        "factual_positive_available": float(bool(factual_positives)),
        "factual_positive_rank": float(factual_rank),
        "first_positive_is_factual": float(first_positive_is_factual),
    }


def goal_breakdown(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for goal_name, group in df.groupby("goal"):
        factual = group[group["factual_positive_available"] == 1]
        query_contained = group[group["anchor_query_hit_50"] == 1]
        rerank_contained = group[group["anchor_rerank_rank"] <= 50]
        factual_query_contained = factual[factual["anchor_query_factual_rank"] <= 50]
        factual_rerank_contained = factual[factual["anchor_rerank_factual_rank"] <= 50]
        out[str(goal_name)] = {
            "rows": int(len(group)),
            "anchor_query_recall_at_20": float(group["anchor_query_hit_20"].mean()),
            "anchor_query_recall_at_50": float(group["anchor_query_hit_50"].mean()),
            "anchor_query_weighted_recall_at_20": float(group["anchor_query_weighted_recall_20"].mean()),
            "anchor_query_weighted_recall_at_50": float(group["anchor_query_weighted_recall_50"].mean()),
            "anchor_query_conditional_mean_rank": float(query_contained["anchor_query_rank"].mean()) if len(query_contained) else 51.0,
            "anchor_query_weighted_mrr": float(group["anchor_query_weighted_mrr"].mean()),
            "anchor_query_weighted_ndcg_at_10": float(group["anchor_query_weighted_ndcg_at_10"].mean()),
            "anchor_rerank_hit_at_10": float(group["anchor_rerank_hit_10"].mean()),
            "anchor_rerank_conditional_mean_rank": float(rerank_contained["anchor_rerank_rank"].mean()) if len(rerank_contained) else 51.0,
            "anchor_rerank_weighted_ndcg_at_10": float(group["anchor_rerank_weighted_ndcg_at_10"].mean()),
            "factual_positive_rate": float(group["factual_positive_available"].mean()),
            "factual_query_conditional_mean_rank": float(factual_query_contained["anchor_query_factual_rank"].mean()) if len(factual_query_contained) else 51.0,
            "factual_rerank_conditional_mean_rank": float(factual_rerank_contained["anchor_rerank_factual_rank"].mean()) if len(factual_rerank_contained) else 51.0,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the rebuilt V2.2 pipeline end to end.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--explicit-goal", type=str, default="")
    args = parser.parse_args()

    recommender = AmbientRecommenderV2(device=args.device)
    row_indices = recommender.decision_df.index[recommender.decision_df["split"] == args.split].to_numpy(dtype=np.int64)
    if len(row_indices) == 0:
        raise RuntimeError(f"No rows found for split={args.split}")

    positive_bundle = load_supervision_sets(str(REBUILD_DIR / "anchor_positive_sets.npz"))
    positive_idx = positive_bundle["indices"]
    positive_count = positive_bundle["counts"]
    positive_tier = positive_bundle["tiers"]
    global_to_local = np.full(len(recommender.anchor_df), -1, dtype=np.int32)
    global_to_local[recommender.train_anchor_idx] = np.arange(len(recommender.train_anchor_idx), dtype=np.int32)
    pos_local, pos_count_local, pos_tier_local = map_global_sets_to_train(positive_idx, positive_count, global_to_local, meta=positive_tier)
    if pos_tier_local is None:
        raise RuntimeError("Positive supervision tiers are required for V2.2 offline evaluation.")

    explicit_goal = str(args.explicit_goal).strip().lower() or None
    result = recommender.score_rows(
        row_indices=row_indices,
        explicit_goal=explicit_goal,
        top_k=max(50, args.candidate_k),
        candidate_k=args.candidate_k,
    )

    train_anchor_rows = recommender.anchor_df.iloc[recommender.train_anchor_idx].reset_index(drop=True)
    train_anchor_song_id = train_anchor_rows["song_id"].astype(str).to_numpy()
    records = []
    benefit_err = []
    accept_err = []
    for local_row, global_idx in enumerate(row_indices.tolist()):
        positives = set(int(idx) for idx in pos_local[int(global_idx), : pos_count_local[int(global_idx)]].tolist() if idx >= 0)
        tiers = {
            int(pos_local[int(global_idx), col]): int(pos_tier_local[int(global_idx), col])
            for col in range(int(pos_count_local[int(global_idx)]))
            if int(pos_local[int(global_idx), col]) >= 0
        }
        gains = {idx: float(tier_gain_array(np.asarray([tier], dtype=np.int32))[0]) for idx, tier in tiers.items()}
        factual_anchor_pos = {
            idx for idx in positives if str(train_anchor_song_id[int(idx)]) == str(recommender.decision_df.iloc[int(global_idx)]["song_id"])
        }
        query_order = result["candidate_local_idx"][local_row]
        rerank_order = result["candidate_local_idx"][local_row][result["anchor_order"][local_row]]
        query_rank = best_rank_in_order(query_order, positives)
        rerank_rank = best_rank_in_order(rerank_order, positives)
        query_weighted = weighted_stats_for_order(query_order, positives, gains, factual_anchor_pos)
        rerank_weighted = weighted_stats_for_order(rerank_order, positives, gains, factual_anchor_pos)

        benefit_pred = result["benefit_hat"][local_row]
        accept_pred = result["latent_accept_hat"][local_row]
        cand_local = result["candidate_local_idx"][local_row].clip(min=0)
        benefit_true = train_anchor_rows["benefit_target"].to_numpy(dtype=np.float32)[cand_local]
        accept_pref = train_anchor_rows["accept_pref_target"].to_numpy(dtype=np.float32)[cand_local]
        accept_rating = train_anchor_rows["accept_rating_target"].to_numpy(dtype=np.float32)[cand_local]
        accept_pref_mask = train_anchor_rows["accept_pref_mask"].to_numpy(dtype=np.float32)[cand_local]
        accept_rating_mask = train_anchor_rows["accept_rating_mask"].to_numpy(dtype=np.float32)[cand_local]
        accept_true = np.where(accept_pref_mask > 0.5, accept_pref, accept_rating)
        accept_mask = (accept_pref_mask > 0.5) | (accept_rating_mask > 0.5)
        benefit_err.append(np.abs(benefit_pred - benefit_true))
        if np.any(accept_mask):
            accept_err.append(np.abs(accept_pred[accept_mask] - accept_true[accept_mask]))

        row = recommender.decision_df.iloc[int(global_idx)]
        final_song_ids = [str(item["song_id"]) for item in result["final_recommendations"][local_row]]
        factual_song_id = str(row["song_id"])
        legacy_query_song_ids = list(dict.fromkeys([str(recommender.song_ids[int(recommender.train_anchor_views["anchor_song_idx"][int(local)])]) for local in query_order.tolist() if int(local) >= 0]))
        legacy_query_rank = legacy_query_song_ids.index(factual_song_id) + 1 if factual_song_id in legacy_query_song_ids else args.candidate_k + 1
        legacy_final_rank = final_song_ids.index(factual_song_id) + 1 if factual_song_id in final_song_ids else max(50, args.candidate_k) + 1

        records.append(
            {
                "row_idx": int(global_idx),
                "decision_id": str(row["decision_id"]),
                "goal": recommender.goal_name(int(result["goal_idx"][local_row])),
                "goal_source": str(result["goal_source"][local_row]),
                "anchor_query_hit_20": int(query_rank <= 20),
                "anchor_query_hit_50": int(query_rank <= 50),
                "anchor_query_rank": int(query_rank),
                "anchor_query_weighted_recall_20": float(query_weighted["weighted_recall_at_20"]),
                "anchor_query_weighted_recall_50": float(query_weighted["weighted_recall_at_50"]),
                "anchor_query_weighted_mrr": float(query_weighted["weighted_mrr"]),
                "anchor_query_weighted_ndcg_at_10": float(query_weighted["weighted_ndcg_at_10"]),
                "anchor_rerank_hit_10": int(rerank_rank <= 10),
                "anchor_rerank_rank": int(rerank_rank),
                "anchor_rerank_weighted_recall_20": float(rerank_weighted["weighted_recall_at_20"]),
                "anchor_rerank_weighted_recall_50": float(rerank_weighted["weighted_recall_at_50"]),
                "anchor_rerank_weighted_mrr": float(rerank_weighted["weighted_mrr"]),
                "anchor_rerank_weighted_ndcg_at_10": float(rerank_weighted["weighted_ndcg_at_10"]),
                "factual_positive_available": int(query_weighted["factual_positive_available"]),
                "anchor_query_factual_rank": int(query_weighted["factual_positive_rank"]),
                "anchor_rerank_factual_rank": int(rerank_weighted["factual_positive_rank"]),
                "anchor_query_first_positive_is_factual": float(query_weighted["first_positive_is_factual"]),
                "anchor_rerank_first_positive_is_factual": float(rerank_weighted["first_positive_is_factual"]),
                "top1_source": str(result["top1_source"][local_row]),
                "top1_song_id": str(result["top1_song_id"][local_row]),
                "top1_score": float(result["top1_score"][local_row]),
                "top1_accept": float(result["top1_accept"][local_row]),
                "top1_kind": str(result["top1_kind"][local_row]),
                "top1_transfer_supported": int(result["top1_transfer_supported"][local_row]),
                "max_public_support": float(result["max_public_support"][local_row]),
                "anchor_support_strength": float(result["anchor_support_strength"][local_row]),
                "legacy_query_hit_50": int(legacy_query_rank <= 50),
                "legacy_query_rank": int(legacy_query_rank),
                "legacy_final_hit_10": int(legacy_final_rank <= 10),
                "legacy_final_rank": int(legacy_final_rank),
            }
        )

    pred_df = pd.DataFrame(records)
    pred_df.to_parquet(REBUILD_DIR / "offline_eval_v2_predictions.parquet", index=False)

    benefit_mae = float(np.mean(np.concatenate(benefit_err))) if benefit_err else 0.0
    blended_accept_mae = float(np.mean(np.concatenate(accept_err))) if accept_err else 0.0
    top1_source_share = float(pred_df["top1_source"].value_counts(normalize=True).max()) if len(pred_df) else 1.0
    factual_df = pred_df[pred_df["factual_positive_available"] == 1]
    query_contained = pred_df[pred_df["anchor_query_hit_50"] == 1]
    rerank_contained = pred_df[pred_df["anchor_rerank_rank"] <= args.candidate_k]
    factual_query_contained = factual_df[factual_df["anchor_query_factual_rank"] <= args.candidate_k]
    factual_rerank_contained = factual_df[factual_df["anchor_rerank_factual_rank"] <= args.candidate_k]

    summary = {
        "split": args.split,
        "rows": int(len(pred_df)),
        "candidate_k": int(args.candidate_k),
        "primary_metrics": {
            "anchor_query_recall_at_20": float(pred_df["anchor_query_hit_20"].mean()),
            "anchor_query_recall_at_50": float(pred_df["anchor_query_hit_50"].mean()),
            "anchor_query_weighted_recall_at_20": float(pred_df["anchor_query_weighted_recall_20"].mean()),
            "anchor_query_weighted_recall_at_50": float(pred_df["anchor_query_weighted_recall_50"].mean()),
            "anchor_query_conditional_mean_rank": float(query_contained["anchor_query_rank"].mean()) if len(query_contained) else float(args.candidate_k + 1),
            "anchor_query_weighted_mrr": float(pred_df["anchor_query_weighted_mrr"].mean()),
            "anchor_query_weighted_ndcg_at_10": float(pred_df["anchor_query_weighted_ndcg_at_10"].mean()),
            "anchor_rerank_hit_at_10": float(pred_df["anchor_rerank_hit_10"].mean()),
            "anchor_rerank_conditional_mean_rank": float(rerank_contained["anchor_rerank_rank"].mean()) if len(rerank_contained) else float(args.candidate_k + 1),
            "anchor_rerank_weighted_recall_at_20": float(pred_df["anchor_rerank_weighted_recall_20"].mean()),
            "anchor_rerank_weighted_recall_at_50": float(pred_df["anchor_rerank_weighted_recall_50"].mean()),
            "anchor_rerank_weighted_mrr": float(pred_df["anchor_rerank_weighted_mrr"].mean()),
            "anchor_rerank_weighted_ndcg_at_10": float(pred_df["anchor_rerank_weighted_ndcg_at_10"].mean()),
            "factual_positive_rate": float(pred_df["factual_positive_available"].mean()),
            "factual_query_conditional_mean_rank": float(factual_query_contained["anchor_query_factual_rank"].mean()) if len(factual_query_contained) else float(args.candidate_k + 1),
            "factual_rerank_conditional_mean_rank": float(factual_rerank_contained["anchor_rerank_factual_rank"].mean()) if len(factual_rerank_contained) else float(args.candidate_k + 1),
            "benefit_mae": benefit_mae,
            "blended_accept_mae": blended_accept_mae,
            "top1_predicted_accept_mean": float(pred_df["top1_accept"].mean()),
            "public_transfer_supported_share": float(pred_df["top1_transfer_supported"].mean()),
            "top1_source_max_share": top1_source_share,
        },
        "goal_breakdown": goal_breakdown(pred_df),
        "top1_source_distribution": {
            str(k): float(v)
            for k, v in pred_df["top1_source"].value_counts(normalize=True).sort_index().to_dict().items()
        },
        "legacy_diagnostics": {
            "exact_song_query_recall_at_50": float(pred_df["legacy_query_hit_50"].mean()),
            "exact_song_rerank_hit_at_10": float(pred_df["legacy_final_hit_10"].mean()),
            "exact_song_conditional_rank": float(
                pred_df.loc[pred_df["legacy_query_hit_50"] == 1, "legacy_final_rank"].mean()
            )
            if (pred_df["legacy_query_hit_50"] == 1).any()
            else float(max(50, args.candidate_k) + 1),
        },
    }

    (MODELS_DIR / "offline_eval_v2.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
