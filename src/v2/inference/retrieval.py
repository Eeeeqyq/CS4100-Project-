"""
Hybrid candidate retrieval helpers for the V2 rebuild.

This layer sits between the learned query tower and the utility reranker.
It is intentionally more structured than raw cosine search:

- raw query similarity still matters most
- user-song similarity helps recover taste
- goal-conditioned target-affect fit helps recover situational relevance
- source-aware quotas prevent the 89k-track Spotify catalog from swamping the
  much smaller supervised SiTunes domain
"""

from __future__ import annotations

from typing import Final

import numpy as np


SOURCE_ORDER: Final[tuple[str, ...]] = ("situnes", "spotify", "pmemo")


def normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-8, None)
    return x / denom


def goal_retrieval_profile(goal_idx: int) -> dict[str, object]:
    goal = int(goal_idx)
    if goal == 0:  # focus
        return {
            "weights": {"query": 0.52, "user": 0.16, "affect": 0.22, "quality": 0.05},
            "source_bias": {"situnes": 0.12, "spotify": 0.00, "pmemo": 0.08},
            "source_quota": {"situnes": 0.60, "spotify": 0.20, "pmemo": 0.20},
        }
    if goal == 1:  # wind_down
        return {
            "weights": {"query": 0.50, "user": 0.15, "affect": 0.25, "quality": 0.05},
            "source_bias": {"situnes": 0.10, "spotify": 0.00, "pmemo": 0.10},
            "source_quota": {"situnes": 0.55, "spotify": 0.15, "pmemo": 0.30},
        }
    if goal == 2:  # uplift
        return {
            "weights": {"query": 0.55, "user": 0.20, "affect": 0.20, "quality": 0.05},
            "source_bias": {"situnes": 0.10, "spotify": 0.00, "pmemo": 0.04},
            "source_quota": {"situnes": 0.50, "spotify": 0.35, "pmemo": 0.15},
        }
    return {  # movement
        "weights": {"query": 0.58, "user": 0.18, "affect": 0.19, "quality": 0.05},
        "source_bias": {"situnes": 0.10, "spotify": 0.02, "pmemo": 0.03},
        "source_quota": {"situnes": 0.55, "spotify": 0.35, "pmemo": 0.10},
    }


def _allocate_source_counts(
    source_quota: dict[str, float],
    source_sizes: dict[str, int],
    candidate_k: int,
) -> dict[str, int]:
    counts = {source: 0 for source in SOURCE_ORDER}
    active = [source for source in SOURCE_ORDER if source_sizes.get(source, 0) > 0]
    if not active or candidate_k <= 0:
        return counts

    for source in active:
        counts[source] = min(
            int(np.floor(candidate_k * float(source_quota.get(source, 0.0)))),
            int(source_sizes[source]),
        )

    remaining = candidate_k - sum(counts.values())
    ranked_sources = sorted(active, key=lambda item: float(source_quota.get(item, 0.0)), reverse=True)
    while remaining > 0:
        progressed = False
        for source in ranked_sources:
            if counts[source] >= int(source_sizes[source]):
                continue
            counts[source] += 1
            remaining -= 1
            progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break

    return counts


def _topk_from_scores(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    k = min(int(k), int(scores.shape[0]))
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    ord_idx = np.argsort(-scores[idx])
    idx = idx[ord_idx]
    return idx.astype(np.int32), scores[idx].astype(np.float32)


def hybrid_candidate_search(
    query_embeddings: np.ndarray,
    user_embeddings: np.ndarray,
    goal_idx: np.ndarray,
    tau: np.ndarray,
    norm_song_emb: np.ndarray,
    user_song_emb: np.ndarray,
    song_affect: np.ndarray,
    song_quality: np.ndarray,
    song_sources: np.ndarray,
    candidate_k: int,
) -> dict[str, np.ndarray]:
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    user_embeddings = np.asarray(user_embeddings, dtype=np.float32)
    goal_idx = np.asarray(goal_idx, dtype=np.int64)
    tau = np.asarray(tau, dtype=np.float32)
    song_affect = np.asarray(song_affect, dtype=np.float32)
    song_quality = np.asarray(song_quality, dtype=np.float32)
    song_sources = np.asarray(song_sources)

    norm_user_embeddings = normalize_rows(user_embeddings)
    raw_scores = query_embeddings @ norm_song_emb.T
    user_scores = norm_user_embeddings @ user_song_emb.T

    song_affect_emo = (2.0 * song_affect - 1.0).astype(np.float32)
    num_rows = query_embeddings.shape[0]
    raw_candidate_idx = np.zeros((num_rows, candidate_k), dtype=np.int32)
    raw_candidate_score = np.zeros((num_rows, candidate_k), dtype=np.float32)
    hybrid_candidate_idx = np.zeros((num_rows, candidate_k), dtype=np.int32)
    hybrid_candidate_score = np.zeros((num_rows, candidate_k), dtype=np.float32)

    source_indices = {
        source: np.where(song_sources == source)[0].astype(np.int32)
        for source in SOURCE_ORDER
    }
    source_sizes = {source: int(len(indices)) for source, indices in source_indices.items()}

    for row in range(num_rows):
        raw_top_idx, raw_top_score = _topk_from_scores(raw_scores[row], candidate_k)
        raw_candidate_idx[row, : len(raw_top_idx)] = raw_top_idx
        raw_candidate_score[row, : len(raw_top_score)] = raw_top_score

        profile = goal_retrieval_profile(int(goal_idx[row]))
        weights = profile["weights"]
        source_bias = profile["source_bias"]
        quotas = _allocate_source_counts(profile["source_quota"], source_sizes, candidate_k)

        tau_v = float(tau[row, 0])
        tau_a = float(tau[row, 1])
        affect_dist = (
            0.5 * (song_affect_emo[:, 0] - tau_v) ** 2
            + 0.5 * (song_affect_emo[:, 1] - tau_a) ** 2
        )
        affect_fit = 1.0 - np.clip(affect_dist / 3.0, 0.0, 1.0)

        hybrid_score = (
            float(weights["query"]) * raw_scores[row]
            + float(weights["user"]) * user_scores[row]
            + float(weights["affect"]) * affect_fit
            + float(weights["quality"]) * song_quality
        ).astype(np.float32)
        for source in SOURCE_ORDER:
            hybrid_score[song_sources == source] += float(source_bias.get(source, 0.0))

        chosen_idx_parts = []
        chosen_score_parts = []
        for source in SOURCE_ORDER:
            take = int(quotas.get(source, 0))
            if take <= 0:
                continue
            pool = source_indices[source]
            src_scores = hybrid_score[pool]
            local_idx, local_score = _topk_from_scores(src_scores, take)
            chosen_idx_parts.append(pool[local_idx])
            chosen_score_parts.append(local_score)

        if chosen_idx_parts:
            chosen_idx = np.concatenate(chosen_idx_parts, axis=0)
            chosen_score = np.concatenate(chosen_score_parts, axis=0)
            order = np.argsort(-chosen_score)
            chosen_idx = chosen_idx[order][:candidate_k]
            chosen_score = chosen_score[order][:candidate_k]
        else:
            chosen_idx, chosen_score = raw_top_idx, raw_top_score

        hybrid_candidate_idx[row, : len(chosen_idx)] = chosen_idx
        hybrid_candidate_score[row, : len(chosen_score)] = chosen_score

    return {
        "raw_candidate_idx": raw_candidate_idx,
        "raw_candidate_score": raw_candidate_score,
        "candidate_idx": hybrid_candidate_idx,
        "candidate_score": hybrid_candidate_score,
        "raw_score_matrix": raw_scores.astype(np.float32),
        "user_score_matrix": user_scores.astype(np.float32),
    }
