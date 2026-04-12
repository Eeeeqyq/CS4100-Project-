"""
Shared anchor-retrieval and anchor-rerank helpers for V2.2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.v2.data.anchor_features import normalize_rows


ANCHOR_PAIR_FEATURE_DIM = 346


def cosine_feature(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = normalize_rows(a)
    b_norm = normalize_rows(b)
    return np.sum(a_norm * b_norm, axis=1, keepdims=True).astype(np.float32)


def build_train_anchor_views(
    anchor_df: pd.DataFrame,
    train_anchor_idx: np.ndarray,
    context_embeddings: np.ndarray,
    row_user_embeddings: np.ndarray,
    song_embeddings: np.ndarray,
    song_affect: np.ndarray,
    song_dyn_summary: np.ndarray,
) -> dict[str, np.ndarray]:
    anchor_rows = anchor_df.iloc[train_anchor_idx].reset_index(drop=True)
    song_idx = anchor_rows["factual_song_idx"].to_numpy(dtype=np.int64)
    return {
        "train_anchor_idx": np.asarray(train_anchor_idx, dtype=np.int64),
        "anchor_song_idx": song_idx.astype(np.int64),
        "anchor_song_id": anchor_rows["song_id"].astype(str).to_numpy(),
        "anchor_context_emb": context_embeddings[train_anchor_idx].astype(np.float32),
        "anchor_user_emb": row_user_embeddings[train_anchor_idx].astype(np.float32),
        "anchor_song_emb": song_embeddings[song_idx].astype(np.float32),
        "anchor_song_affect": song_affect[song_idx].astype(np.float32),
        "anchor_song_dyn": song_dyn_summary[song_idx].astype(np.float32),
        "anchor_success": anchor_rows["anchor_success_obs"].to_numpy(dtype=np.float32),
        "anchor_support_norm": anchor_rows["support_norm"].to_numpy(dtype=np.float32),
        "anchor_goal_idx": anchor_rows["goal_idx"].to_numpy(dtype=np.int64),
        "anchor_decision_id": anchor_rows["decision_id"].astype(str).to_numpy(),
    }


def build_anchor_pair_features(
    row_indices: np.ndarray,
    query_embeddings: np.ndarray,
    candidate_local_idx: np.ndarray,
    candidate_score: np.ndarray,
    decision_df: pd.DataFrame,
    train_anchor_views: dict[str, np.ndarray],
    row_user_embeddings: np.ndarray,
    stage1_exact_rating: dict[tuple[int, str], float],
    stage1_user_prior: dict[int, float],
    stage1_global_prior: float,
    anchor_embeddings: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    row_indices = np.asarray(row_indices, dtype=np.int64)
    candidate_local_idx = np.asarray(candidate_local_idx, dtype=np.int64)
    candidate_score = np.asarray(candidate_score, dtype=np.float32)
    anchor_embeddings = np.asarray(anchor_embeddings, dtype=np.float32)

    num_rows, cand = candidate_local_idx.shape
    goal_onehot = np.eye(4, dtype=np.float32)[decision_df.iloc[row_indices]["goal_idx"].to_numpy(dtype=np.int64)]
    tau = decision_df.iloc[row_indices][["tau_valence", "tau_arousal"]].to_numpy(dtype=np.float32)
    out = np.zeros((num_rows, cand, ANCHOR_PAIR_FEATURE_DIM), dtype=np.float32)
    uncertainty = np.zeros((num_rows, cand), dtype=np.float32)

    anchor_context_norm = normalize_rows(train_anchor_views["anchor_context_emb"])
    anchor_user_norm = normalize_rows(train_anchor_views["anchor_user_emb"])
    anchor_song_user_norm = normalize_rows(train_anchor_views["anchor_song_emb"][:, : row_user_embeddings.shape[1]])

    for row_pos, global_idx in enumerate(row_indices.tolist()):
        user_id = int(decision_df.iloc[int(global_idx)]["user_id"])
        user_prior_value = float(stage1_user_prior.get(user_id, stage1_global_prior))
        local_idx = candidate_local_idx[row_pos]
        valid_local = local_idx.clip(min=0)

        query = np.repeat(query_embeddings[row_pos][None, :], cand, axis=0)
        user = np.repeat(row_user_embeddings[global_idx][None, :], cand, axis=0)
        goal = np.repeat(goal_onehot[row_pos][None, :], cand, axis=0)
        tau_row = np.repeat(tau[row_pos][None, :], cand, axis=0)
        anchor_emb = anchor_embeddings[valid_local]
        affect = train_anchor_views["anchor_song_affect"][valid_local]
        dyn = train_anchor_views["anchor_song_dyn"][valid_local]
        success = train_anchor_views["anchor_success"][valid_local][:, None]
        support_norm = train_anchor_views["anchor_support_norm"][valid_local][:, None]

        ctx_cos = cosine_feature(
            np.repeat(normalize_rows(query_embeddings[row_pos][None, :]), cand, axis=0),
            anchor_context_norm[valid_local],
        )
        user_anchor_cos = cosine_feature(
            np.repeat(normalize_rows(row_user_embeddings[global_idx][None, :]), cand, axis=0),
            anchor_user_norm[valid_local],
        )
        user_song_cos = cosine_feature(
            np.repeat(normalize_rows(row_user_embeddings[global_idx][None, :]), cand, axis=0),
            anchor_song_user_norm[valid_local],
        )

        ctx_weak = 1.0 - np.clip((ctx_cos + 1.0) / 2.0, 0.0, 1.0)
        user_weak = 1.0 - np.clip((user_anchor_cos + 1.0) / 2.0, 0.0, 1.0)
        support_weak = 1.0 - np.clip(support_norm, 0.0, 1.0)
        uncertainty_triplet = np.concatenate([ctx_weak, user_weak, support_weak], axis=1).astype(np.float32)
        uncertainty[row_pos] = (
            0.45 * ctx_weak[:, 0] + 0.30 * user_weak[:, 0] + 0.25 * support_weak[:, 0]
        ).astype(np.float32)

        user_prior = np.full((cand, 1), user_prior_value, dtype=np.float32)
        exact_rating = np.full((cand, 1), user_prior_value, dtype=np.float32)
        exact_mask = np.zeros((cand, 1), dtype=np.float32)
        for cand_pos, anchor_local in enumerate(valid_local.tolist()):
            key = (user_id, str(train_anchor_views["anchor_song_id"][int(anchor_local)]))
            if key in stage1_exact_rating:
                exact_rating[cand_pos, 0] = float(stage1_exact_rating[key])
                exact_mask[cand_pos, 0] = 1.0

        feat = np.concatenate(
            [
                query,
                user,
                goal,
                tau_row,
                anchor_emb,
                affect,
                dyn,
                candidate_score[row_pos][:, None],
                success,
                support_norm,
                ctx_cos,
                user_anchor_cos,
                user_song_cos,
                uncertainty_triplet,
                user_prior,
                exact_rating,
                exact_mask,
            ],
            axis=1,
        ).astype(np.float32)

        invalid_mask = local_idx < 0
        if np.any(invalid_mask):
            feat[invalid_mask] = 0.0
            uncertainty[row_pos, invalid_mask] = 1.0
        out[row_pos] = feat

    return out, {"uncertainty": uncertainty.astype(np.float32)}
