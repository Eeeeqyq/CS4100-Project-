"""
Build split-safe SiTunes anchor supervision artifacts for V2.2.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import ANCHOR_COLUMNS, GOAL_NAMES, Goal, validate_anchor_table


POSITIVE_CAP = 5
NEGATIVE_CAP = 32
SUCCESS_QUANTILE_BY_GOAL = {
    int(Goal.FOCUS): 0.75,
    int(Goal.WIND_DOWN): 0.75,
    int(Goal.UPLIFT): 0.75,
    int(Goal.MOVEMENT): 0.70,
}
CONTEXT_SIM_QUANTILE_BY_GOAL = {
    int(Goal.FOCUS): 0.85,
    int(Goal.WIND_DOWN): 0.85,
    int(Goal.UPLIFT): 0.85,
    int(Goal.MOVEMENT): 0.80,
}
RELAXED_CONTEXT_SIM_QUANTILE_BY_GOAL = {
    int(Goal.FOCUS): 0.72,
    int(Goal.WIND_DOWN): 0.72,
    int(Goal.UPLIFT): 0.72,
    int(Goal.MOVEMENT): 0.65,
}

NEG_HARD_FAILURE = 1
NEG_SAME_SONG_MISMATCH = 2
NEG_GOAL_CONFOUNDER = 3
NEG_FALLBACK = 4


def _zscore_from_train(values: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if not np.any(train_mask):
        mean = float(arr.mean())
        std = float(arr.std())
    else:
        train_values = arr[train_mask]
        mean = float(train_values.mean())
        std = float(train_values.std())
    std = max(std, 1e-6)
    return ((arr - mean) / std).astype(np.float32)


def _context_feature_matrix(anchor_df: pd.DataFrame) -> np.ndarray:
    train_mask = anchor_df["split"].eq("train").to_numpy(dtype=bool)
    continuous = np.column_stack(
        [
            _zscore_from_train(anchor_df["pre_valence"].to_numpy(dtype=np.float32), train_mask),
            _zscore_from_train(anchor_df["pre_arousal"].to_numpy(dtype=np.float32), train_mask),
            _zscore_from_train(anchor_df["tau_valence"].to_numpy(dtype=np.float32), train_mask),
            _zscore_from_train(anchor_df["tau_arousal"].to_numpy(dtype=np.float32), train_mask),
            _zscore_from_train(anchor_df["speed_norm"].to_numpy(dtype=np.float32), train_mask),
            _zscore_from_train(anchor_df["temp_z"].to_numpy(dtype=np.float32), train_mask),
            _zscore_from_train(anchor_df["humidity_z"].to_numpy(dtype=np.float32), train_mask),
            anchor_df["weekend_flag"].to_numpy(dtype=np.float32),
            anchor_df["checkin_mask"].to_numpy(dtype=np.float32),
        ]
    ).astype(np.float32)
    time_oh = np.eye(3, dtype=np.float32)[anchor_df["time_bucket"].clip(0, 2).to_numpy(dtype=np.int64)]
    weather_oh = np.eye(3, dtype=np.float32)[anchor_df["weather_bucket"].clip(0, 2).to_numpy(dtype=np.int64)]
    return np.concatenate([continuous, time_oh, weather_oh], axis=1).astype(np.float32)


def _pairwise_sqdist(features: np.ndarray) -> np.ndarray:
    feats = np.asarray(features, dtype=np.float32)
    gram = feats @ feats.T
    sq = np.sum(feats * feats, axis=1, keepdims=True)
    dist = np.maximum(sq + sq.T - 2.0 * gram, 0.0)
    return dist.astype(np.float32)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(arr, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-6, None)
    return (arr / denom).astype(np.float32)


def _context_similarity_row(dist_row: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.asarray(dist_row, dtype=np.float32))).astype(np.float32)


def _take_best(indices: np.ndarray, scores: np.ndarray, cap: int, exclude: set[int]) -> list[int]:
    if cap <= 0 or len(indices) == 0:
        return []
    order = np.argsort(-np.asarray(scores, dtype=np.float32))
    out: list[int] = []
    for local in order.tolist():
        idx = int(indices[local])
        if idx in exclude:
            continue
        out.append(idx)
        exclude.add(idx)
        if len(out) >= cap:
            break
    return out


def _pad_sets(
    sets: list[list[int]],
    cap: int,
    meta: list[list[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    out = np.full((len(sets), cap), -1, dtype=np.int32)
    counts = np.zeros(len(sets), dtype=np.int32)
    meta_out = np.zeros((len(sets), cap), dtype=np.int32) if meta is not None else None
    for row, items in enumerate(sets):
        trimmed = [int(idx) for idx in items[:cap]]
        counts[row] = len(trimmed)
        if trimmed:
            out[row, : len(trimmed)] = np.asarray(trimmed, dtype=np.int32)
            if meta_out is not None and meta is not None:
                trimmed_meta = [int(value) for value in meta[row][: len(trimmed)]]
                meta_out[row, : len(trimmed_meta)] = np.asarray(trimmed_meta, dtype=np.int32)
    return out, counts, meta_out


def _goal_thresholds(anchor_success: np.ndarray, train_mask: np.ndarray, goal_idx: np.ndarray) -> dict[int, float]:
    thresholds: dict[int, float] = {}
    for goal in [int(Goal.FOCUS), int(Goal.WIND_DOWN), int(Goal.UPLIFT), int(Goal.MOVEMENT)]:
        mask = train_mask & (goal_idx == goal)
        if not np.any(mask):
            thresholds[goal] = 0.0
            continue
        quantile = float(SUCCESS_QUANTILE_BY_GOAL.get(goal, 0.75))
        thresholds[goal] = float(max(0.0, np.quantile(anchor_success[mask], quantile)))
    return thresholds


def _score_summary_dict(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if not np.any(mask):
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float32)[mask]
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def build_anchor_supervision(
    decision_df: pd.DataFrame,
    song_static: np.ndarray,
    out_dir: Path,
) -> dict[str, object]:
    anchor_df = decision_df.copy().reset_index(drop=True)
    anchor_df["anchor_idx"] = np.arange(len(anchor_df), dtype=np.int32)
    anchor_df["available_for_index"] = anchor_df["split"].eq("train").astype(np.int32)

    pref_mask = anchor_df["accept_pref_mask"].to_numpy(dtype=np.float32)
    rating_mask = anchor_df["accept_rating_mask"].to_numpy(dtype=np.float32)
    pref_target = anchor_df["accept_pref_target"].to_numpy(dtype=np.float32)
    rating_target = anchor_df["accept_rating_target"].to_numpy(dtype=np.float32)
    acceptance_obs = np.where(pref_mask > 0.5, pref_target, np.where(rating_mask > 0.5, rating_target, 0.0)).astype(np.float32)
    acceptance_source = np.where(pref_mask > 0.5, "preference", np.where(rating_mask > 0.5, "rating", "missing"))
    anchor_success = (0.7 * anchor_df["benefit_target"].to_numpy(dtype=np.float32) + 0.3 * acceptance_obs).astype(np.float32)

    anchor_df["acceptance_obs"] = acceptance_obs
    anchor_df["acceptance_source"] = acceptance_source
    anchor_df["anchor_success_obs"] = anchor_success
    anchor_df["local_support_count"] = 0.0
    anchor_df["support_count"] = 0.0
    anchor_df["support_norm"] = 0.0
    anchor_df["positive_tier1_count"] = 0
    anchor_df["positive_tier2_count"] = 0
    anchor_df["positive_tier3_count"] = 0
    anchor_df["factual_positive_available"] = anchor_df["split"].eq("train").astype(np.int32)

    context_features = _context_feature_matrix(anchor_df)
    context_dist = _pairwise_sqdist(context_features)

    song_vectors = _normalize_rows(song_static[anchor_df["factual_song_idx"].to_numpy(dtype=np.int64)])
    song_sim = np.clip(song_vectors @ song_vectors.T, -1.0, 1.0).astype(np.float32)

    train_mask = anchor_df["available_for_index"].to_numpy(dtype=bool)
    goal_idx = anchor_df["goal_idx"].to_numpy(dtype=np.int64)
    song_id = anchor_df["song_id"].astype(str).to_numpy()
    split = anchor_df["split"].astype(str).to_numpy()
    benefit_target = anchor_df["benefit_target"].to_numpy(dtype=np.float32)

    goal_good_threshold = _goal_thresholds(anchor_success, train_mask, goal_idx)

    positive_sets: list[list[int]] = []
    positive_tiers: list[list[int]] = []
    negative_sets: list[list[int]] = []
    negative_types: list[list[int]] = []
    local_support_count = np.zeros(len(anchor_df), dtype=np.float32)

    for row in range(len(anchor_df)):
        eligible = train_mask.copy()
        row_goal = int(goal_idx[row])
        same_goal = goal_idx == row_goal
        same_song = song_id == str(song_id[row])
        row_ctx_sim = _context_similarity_row(context_dist[row])
        good_thresh = float(goal_good_threshold.get(row_goal, 0.0))

        strong_quantile = float(CONTEXT_SIM_QUANTILE_BY_GOAL.get(row_goal, 0.85))
        relaxed_quantile = float(RELAXED_CONTEXT_SIM_QUANTILE_BY_GOAL.get(row_goal, 0.72))
        goal_success_pool = np.where(eligible & same_goal & (anchor_success >= good_thresh))[0]
        if len(goal_success_pool) > 0:
            strong_ctx_thresh = float(np.quantile(row_ctx_sim[goal_success_pool], strong_quantile))
            relaxed_ctx_thresh = float(np.quantile(row_ctx_sim[goal_success_pool], relaxed_quantile))
        else:
            strong_ctx_thresh = 1.0
            relaxed_ctx_thresh = 0.0

        strong_local_support = np.where(
            eligible & same_goal & (anchor_success >= good_thresh) & (row_ctx_sim >= strong_ctx_thresh)
        )[0]
        local_support_count[row] = float(max(len([idx for idx in strong_local_support.tolist() if idx != row]), 0))

        positives: list[int] = []
        tiers: list[int] = []
        negatives: list[int] = []
        neg_types_row: list[int] = []
        seen_pos: set[int] = set()
        seen_neg: set[int] = set()

        def add_positive(candidates: np.ndarray, scores: np.ndarray, cap: int, tier: int) -> None:
            chosen = _take_best(candidates, scores, cap, seen_pos)
            positives.extend(chosen)
            tiers.extend([int(tier)] * len(chosen))

        def add_negative(candidates: np.ndarray, scores: np.ndarray, cap: int, neg_type: int) -> None:
            chosen = _take_best(candidates, scores, cap, seen_neg)
            negatives.extend(chosen)
            neg_types_row.extend([int(neg_type)] * len(chosen))

        if split[row] == "train" and eligible[row]:
            positives.append(row)
            tiers.append(1)
            seen_pos.add(row)

        same_song_good = np.where(
            eligible
            & same_song
            & same_goal
            & (anchor_success >= good_thresh)
            & (np.arange(len(anchor_df)) != row)
        )[0]
        if len(same_song_good) > 0 and len(positives) < POSITIVE_CAP:
            same_song_scores = (
                0.78 * song_sim[row, same_song_good]
                + 0.22 * row_ctx_sim[same_song_good]
            )
            add_positive(same_song_good, same_song_scores, POSITIVE_CAP - len(positives), 2)

        goal_good_strong = np.where(
            eligible
            & same_goal
            & (anchor_success >= good_thresh)
            & (row_ctx_sim >= strong_ctx_thresh)
            & (~same_song)
        )[0]
        if len(goal_good_strong) > 0 and len(positives) < POSITIVE_CAP:
            strong_scores = (
                0.52 * anchor_success[goal_good_strong]
                + 0.30 * row_ctx_sim[goal_good_strong]
                + 0.18 * song_sim[row, goal_good_strong]
            )
            add_positive(goal_good_strong, strong_scores, POSITIVE_CAP - len(positives), 3)

        goal_good_relaxed = np.where(
            eligible
            & same_goal
            & (anchor_success >= good_thresh)
            & (row_ctx_sim >= relaxed_ctx_thresh)
            & (~same_song)
        )[0]
        if len(goal_good_relaxed) > 0 and len(positives) < POSITIVE_CAP:
            relaxed_scores = (
                0.48 * anchor_success[goal_good_relaxed]
                + 0.34 * row_ctx_sim[goal_good_relaxed]
                + 0.18 * song_sim[row, goal_good_relaxed]
            )
            add_positive(goal_good_relaxed, relaxed_scores, POSITIVE_CAP - len(positives), 3)

        goal_nonnegative = np.where(
            eligible
            & same_goal
            & (benefit_target >= 0.0)
            & (row_ctx_sim >= relaxed_ctx_thresh)
            & (~same_song)
        )[0]
        if len(goal_nonnegative) > 0 and len(positives) < POSITIVE_CAP:
            fallback_scores = (
                0.40 * np.maximum(anchor_success[goal_nonnegative], 0.0)
                + 0.40 * row_ctx_sim[goal_nonnegative]
                + 0.20 * song_sim[row, goal_nonnegative]
            )
            add_positive(goal_nonnegative, fallback_scores, POSITIVE_CAP - len(positives), 3)

        if not positives:
            closest_same_goal = np.where(eligible & same_goal)[0]
            if len(closest_same_goal) > 0:
                closest_scores = 0.65 * row_ctx_sim[closest_same_goal] + 0.35 * song_sim[row, closest_same_goal]
                add_positive(closest_same_goal, closest_scores, min(POSITIVE_CAP, 2), 3)

        if not positives:
            global_fallback = np.where(eligible)[0]
            if len(global_fallback) > 0:
                fallback_scores = 0.60 * row_ctx_sim[global_fallback] + 0.40 * song_sim[row, global_fallback]
                add_positive(global_fallback, fallback_scores, 1, 3)

        goal_poor = np.where(
            eligible & same_goal & (anchor_success < 0.0) & (np.arange(len(anchor_df)) != row)
        )[0]
        if len(goal_poor) > 0:
            poor_scores = (
                0.60 * row_ctx_sim[goal_poor]
                + 0.20 * song_sim[row, goal_poor]
                + 0.20 * np.clip(-anchor_success[goal_poor], 0.0, None)
            )
            add_negative(goal_poor, poor_scores, 10, NEG_HARD_FAILURE)

        song_bad = np.where(
            eligible
            & same_song
            & (((goal_idx != row_goal) | (anchor_success < 0.0)))
            & (np.arange(len(anchor_df)) != row)
        )[0]
        if len(song_bad) > 0:
            song_bad_scores = 0.72 * song_sim[row, song_bad] + 0.28 * row_ctx_sim[song_bad]
            add_negative(song_bad, song_bad_scores, 8, NEG_SAME_SONG_MISMATCH)

        goal_confounders = np.where(
            eligible
            & same_goal
            & (anchor_success >= 0.0)
            & (~same_song)
            & (np.arange(len(anchor_df)) != row)
        )[0]
        if len(goal_confounders) > 0:
            goal_confounders = np.asarray(
                [idx for idx in goal_confounders.tolist() if idx not in seen_pos],
                dtype=np.int64,
            )
            if len(goal_confounders) > 0:
                confounder_scores = (
                    0.42 * row_ctx_sim[goal_confounders]
                    + 0.38 * song_sim[row, goal_confounders]
                    + 0.20 * np.maximum(anchor_success[goal_confounders], 0.0)
                )
                add_negative(goal_confounders, confounder_scores, 10, NEG_GOAL_CONFOUNDER)

        fallback_neg = np.asarray(
            [idx for idx in np.argsort(-row_ctx_sim).tolist() if eligible[idx] and idx not in seen_pos and idx != row],
            dtype=np.int64,
        )
        if len(fallback_neg) > 0 and len(negatives) < NEGATIVE_CAP:
            fallback_scores = row_ctx_sim[fallback_neg]
            add_negative(fallback_neg, fallback_scores, NEGATIVE_CAP - len(negatives), NEG_FALLBACK)

        tier_arr = np.asarray(tiers, dtype=np.int32) if tiers else np.zeros(0, dtype=np.int32)
        anchor_df.at[row, "positive_tier1_count"] = int(np.sum(tier_arr == 1))
        anchor_df.at[row, "positive_tier2_count"] = int(np.sum(tier_arr == 2))
        anchor_df.at[row, "positive_tier3_count"] = int(np.sum(tier_arr == 3))

        positive_sets.append(positives[:POSITIVE_CAP])
        positive_tiers.append(tiers[:POSITIVE_CAP])
        negative_sets.append(negatives[:NEGATIVE_CAP])
        negative_types.append(neg_types_row[:NEGATIVE_CAP])

    support_scale = float(
        max(
            np.quantile(local_support_count[train_mask], 0.90) if np.any(train_mask) else np.quantile(local_support_count, 0.90),
            1.0,
        )
    )
    support_norm = np.clip(local_support_count / support_scale, 0.0, 1.0).astype(np.float32)
    anchor_df["local_support_count"] = local_support_count
    anchor_df["support_count"] = local_support_count
    anchor_df["support_norm"] = support_norm

    positive_idx, positive_count, positive_tier_arr = _pad_sets(positive_sets, POSITIVE_CAP, meta=positive_tiers)
    negative_idx, negative_count, negative_type_arr = _pad_sets(negative_sets, NEGATIVE_CAP, meta=negative_types)
    if positive_tier_arr is None or negative_type_arr is None:
        raise RuntimeError("Failed to materialize positive/negative supervision metadata.")

    export_anchor_df = anchor_df[ANCHOR_COLUMNS].copy()
    validate_anchor_table(export_anchor_df)
    export_anchor_df.to_parquet(out_dir / "anchor_table.parquet", index=False)
    np.savez(
        out_dir / "anchor_positive_sets.npz",
        indices=positive_idx,
        counts=positive_count,
        tiers=positive_tier_arr,
    )
    np.savez(
        out_dir / "anchor_negative_pools.npz",
        indices=negative_idx,
        counts=negative_count,
        types=negative_type_arr,
    )
    np.save(out_dir / "anchor_context_features.npy", context_features.astype(np.float32))

    tier1_count = anchor_df["positive_tier1_count"].to_numpy(dtype=np.float32)
    tier2_count = anchor_df["positive_tier2_count"].to_numpy(dtype=np.float32)
    tier3_count = anchor_df["positive_tier3_count"].to_numpy(dtype=np.float32)
    per_split_positive = {
        split_name: float(positive_count[anchor_df["split"].eq(split_name).to_numpy(dtype=bool)].mean())
        for split_name in ["train", "val", "test"]
        if np.any(anchor_df["split"].eq(split_name))
    }
    per_goal_positive = {
        GOAL_NAMES.get(int(goal), str(goal)): float(positive_count[goal_idx == goal].mean())
        for goal in sorted(np.unique(goal_idx).tolist())
    }
    deployable_positive = {
        split_name: float(np.mean(positive_count[anchor_df["split"].eq(split_name).to_numpy(dtype=bool)] > 0))
        for split_name in ["train", "val", "test"]
        if np.any(anchor_df["split"].eq(split_name))
    }
    local_support_by_split = {
        split_name: round(float(local_support_count[anchor_df["split"].eq(split_name).to_numpy(dtype=bool)].mean()), 4)
        for split_name in ["train", "val", "test"]
        if np.any(anchor_df["split"].eq(split_name))
    }
    tier_fraction = {
        "tier1": float(tier1_count.sum() / max(float(positive_count.sum()), 1.0)),
        "tier2": float(tier2_count.sum() / max(float(positive_count.sum()), 1.0)),
        "tier3": float(tier3_count.sum() / max(float(positive_count.sum()), 1.0)),
    }
    valid_negative_types = negative_type_arr[negative_idx >= 0]
    negative_type_fraction = {
        "hard_failure": float(np.mean(valid_negative_types == NEG_HARD_FAILURE)) if len(valid_negative_types) else 0.0,
        "same_song_mismatch": float(np.mean(valid_negative_types == NEG_SAME_SONG_MISMATCH)) if len(valid_negative_types) else 0.0,
        "goal_confounder": float(np.mean(valid_negative_types == NEG_GOAL_CONFOUNDER)) if len(valid_negative_types) else 0.0,
    }
    stats = {
        "anchors": int(len(export_anchor_df)),
        "train_anchors": int(int(export_anchor_df["available_for_index"].sum())),
        "positive_cap": int(POSITIVE_CAP),
        "negative_cap": int(NEGATIVE_CAP),
        "mean_positive_count": float(np.mean(positive_count)),
        "mean_negative_count": float(np.mean(negative_count)),
        "mean_positive_count_by_split": {k: round(v, 4) for k, v in per_split_positive.items()},
        "mean_positive_count_by_goal": {k: round(v, 4) for k, v in per_goal_positive.items()},
        "nonempty_positive_rate_by_split": {k: round(v, 4) for k, v in deployable_positive.items()},
        "factual_only_rate": round(float(np.mean((tier1_count == positive_count.astype(np.float32)) & (positive_count > 0))), 4),
        "tier_composition": {k: round(v, 4) for k, v in tier_fraction.items()},
        "local_support": {
            "overall_mean": round(float(np.mean(local_support_count)), 4),
            "overall_max": round(float(np.max(local_support_count)), 4),
            "by_split_mean": local_support_by_split,
        },
        "negative_type_fraction": {k: round(v, 4) for k, v in negative_type_fraction.items()},
        "goal_good_threshold": {str(k): round(v, 4) for k, v in goal_good_threshold.items()},
        "positive_tier_counts_summary": {
            "tier1": _score_summary_dict(tier1_count, np.ones(len(tier1_count), dtype=bool)),
            "tier2": _score_summary_dict(tier2_count, np.ones(len(tier2_count), dtype=bool)),
            "tier3": _score_summary_dict(tier3_count, np.ones(len(tier3_count), dtype=bool)),
        },
    }
    (out_dir / "anchor_supervision_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
