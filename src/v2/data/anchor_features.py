"""
Shared anchor-side feature builders for V2.2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import CTX_EMB_DIM, GOAL_DIM, SONG_EMB_DIM, TAU_DIM, USER_EMB_DIM


ANCHOR_EXTRA_DIM = 4
ANCHOR_FEATURE_DIM = CTX_EMB_DIM + USER_EMB_DIM + SONG_EMB_DIM + GOAL_DIM + TAU_DIM + ANCHOR_EXTRA_DIM
POSITIVE_TIER_GAIN = {
    0: 0.0,
    1: 1.0,
    2: 0.65,
    3: 0.35,
}
NEGATIVE_TYPE_NAMES = {
    0: "none",
    1: "hard_failure",
    2: "same_song_mismatch",
    3: "goal_confounder",
    4: "fallback",
}


def normalize_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(arr, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-6, None)
    return (arr / denom).astype(np.float32)


def load_positive_negative_sets(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    obj = np.load(npz_path)
    return obj["indices"].astype(np.int32), obj["counts"].astype(np.int32)


def load_supervision_sets(npz_path: str) -> dict[str, np.ndarray]:
    obj = np.load(npz_path)
    out = {
        "indices": obj["indices"].astype(np.int32),
        "counts": obj["counts"].astype(np.int32),
    }
    if "tiers" in obj:
        out["tiers"] = obj["tiers"].astype(np.int32)
    if "types" in obj:
        out["types"] = obj["types"].astype(np.int32)
    return out


def tier_gain_array(tiers: np.ndarray) -> np.ndarray:
    arr = np.asarray(tiers, dtype=np.int32)
    gains = np.zeros(arr.shape, dtype=np.float32)
    for tier, gain in POSITIVE_TIER_GAIN.items():
        gains[arr == int(tier)] = float(gain)
    return gains.astype(np.float32)


def build_row_user_views(
    decision_df: pd.DataFrame,
    user_out: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    user_map = {
        int(user_id): idx
        for idx, user_id in enumerate(user_out["user_ids"].astype(np.int32).tolist())
    }
    row_user_embeddings = np.stack(
        [
            user_out["user_embeddings"][user_map[int(user_id)]]
            for user_id in decision_df["user_id"]
        ],
        axis=0,
    ).astype(np.float32)
    row_user_conf = np.stack(
        [
            user_out["user_conf"][user_map[int(user_id)]][0:1]
            for user_id in decision_df["user_id"]
        ],
        axis=0,
    ).astype(np.float32)
    return row_user_embeddings, row_user_conf, user_map


def build_stage1_acceptance_features(
    stage1_df: pd.DataFrame,
) -> tuple[dict[tuple[int, str], float], dict[int, float], float]:
    stage1 = stage1_df.copy()
    stage1["rating_norm"] = ((stage1["rating"].astype(float) - 3.0) / 2.0).clip(-1.0, 1.0)
    exact = {
        (int(row.user_id), str(row.song_id)): float(row.rating_norm)
        for row in stage1[["user_id", "song_id", "rating_norm"]].itertuples(index=False)
    }
    user_prior = {
        int(user_id): float(value)
        for user_id, value in stage1.groupby("user_id")["rating_norm"].mean().items()
    }
    global_prior = float(stage1["rating_norm"].mean()) if len(stage1) else 0.0
    return exact, user_prior, global_prior


def observed_acceptance_from_table(df: pd.DataFrame) -> np.ndarray:
    pref = df["accept_pref_target"].to_numpy(dtype=np.float32)
    rating = df["accept_rating_target"].to_numpy(dtype=np.float32)
    pref_mask = df["accept_pref_mask"].to_numpy(dtype=np.float32)
    rating_mask = df["accept_rating_mask"].to_numpy(dtype=np.float32)
    return np.where(pref_mask > 0.5, pref, np.where(rating_mask > 0.5, rating, 0.0)).astype(np.float32)


def build_anchor_encoder_features(
    anchor_df: pd.DataFrame,
    context_embeddings: np.ndarray,
    row_user_embeddings: np.ndarray,
    song_embeddings: np.ndarray,
) -> np.ndarray:
    anchor_row_idx = anchor_df["anchor_idx"].to_numpy(dtype=np.int64)
    song_idx = anchor_df["factual_song_idx"].to_numpy(dtype=np.int64)
    goal_onehot = np.eye(4, dtype=np.float32)[anchor_df["goal_idx"].to_numpy(dtype=np.int64)]
    tau = anchor_df[["tau_valence", "tau_arousal"]].to_numpy(dtype=np.float32)
    extras = anchor_df[["anchor_success_obs", "benefit_target", "acceptance_obs", "support_norm"]].to_numpy(dtype=np.float32)
    return np.concatenate(
        [
            context_embeddings[anchor_row_idx],
            row_user_embeddings[anchor_row_idx],
            song_embeddings[song_idx],
            goal_onehot,
            tau,
            extras,
        ],
        axis=1,
    ).astype(np.float32)


def latent_acceptance(
    accept_pref_hat: np.ndarray,
    accept_rating_hat: np.ndarray,
    user_prior: np.ndarray | float,
    exact_rating: np.ndarray | None = None,
    exact_mask: np.ndarray | None = None,
) -> np.ndarray:
    pref = np.asarray(accept_pref_hat, dtype=np.float32)
    rating = np.asarray(accept_rating_hat, dtype=np.float32)
    base = 0.40 * pref + 0.40 * rating + 0.20 * np.asarray(user_prior, dtype=np.float32)
    if exact_rating is not None and exact_mask is not None:
        exact = np.asarray(exact_rating, dtype=np.float32)
        mask = np.asarray(exact_mask, dtype=np.float32)
        base = mask * exact + (1.0 - mask) * base
    return np.clip(base, -1.0, 1.0).astype(np.float32)
