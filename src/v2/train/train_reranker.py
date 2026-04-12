"""
Train the V2.2 reranker on SiTunes anchor candidate sets.

The reranker predicts:
- benefit_hat
- accept_pref_hat
- accept_rating_hat
- anchor_relevance_logit

Public songs are not treated as supervised negatives here.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v2.data.anchor_features import (
    build_row_user_views,
    build_stage1_acceptance_features,
    latent_acceptance,
    load_positive_negative_sets,
    load_supervision_sets,
    tier_gain_array,
)
from src.v2.inference.anchor_pipeline import (
    ANCHOR_PAIR_FEATURE_DIM,
    build_anchor_pair_features,
    build_train_anchor_views,
)
from src.v2.models.reranker import UtilityReranker


REBUILD_DIR = PROJECT_ROOT / "data" / "processed" / "rebuild"
MODELS_DIR = PROJECT_ROOT / "models" / "rebuild"
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def row_normalize(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((arr - mean) / std).astype(np.float32)


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


def build_candidate_sets(
    query_embeddings: np.ndarray,
    anchor_embeddings: np.ndarray,
    pos_local: np.ndarray,
    pos_count: np.ndarray,
    pos_tier: np.ndarray | None,
    neg_local: np.ndarray | None,
    neg_count: np.ndarray | None,
    neg_type: np.ndarray | None,
    candidate_size: int,
    inject_positive: bool,
) -> tuple[np.ndarray, np.ndarray]:
    sims = np.asarray(query_embeddings, dtype=np.float32) @ np.asarray(anchor_embeddings, dtype=np.float32).T
    k = min(int(candidate_size), sims.shape[1])
    top_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    top_score = np.take_along_axis(sims, top_idx, axis=1)
    order = np.argsort(-top_score, axis=1)
    top_idx = np.take_along_axis(top_idx, order, axis=1)
    top_score = np.take_along_axis(top_score, order, axis=1)

    if inject_positive:
        for row in range(len(top_idx)):
            present = set(int(idx) for idx in top_idx[row].tolist() if int(idx) >= 0)
            replace_positions = list(range(top_idx.shape[1] - 1, -1, -1))

            def inject(local_idx: int) -> None:
                if local_idx < 0 or local_idx in present or not replace_positions:
                    return
                replace = replace_positions.pop(0)
                top_idx[row, replace] = int(local_idx)
                top_score[row, replace] = float(sims[row, int(local_idx)])
                present.add(int(local_idx))

            positives = [int(idx) for idx in pos_local[row, : pos_count[row]].tolist() if idx >= 0]
            if positives and pos_tier is not None:
                factual = [
                    int(pos_local[row, col])
                    for col in range(int(pos_count[row]))
                    if int(pos_local[row, col]) >= 0 and int(pos_tier[row, col]) == 1
                ]
                fallback_same_song = [
                    int(pos_local[row, col])
                    for col in range(int(pos_count[row]))
                    if int(pos_local[row, col]) >= 0 and int(pos_tier[row, col]) == 2
                ]
                chosen_pool = factual if factual else fallback_same_song
                if chosen_pool and not any(int(idx) in chosen_pool for idx in top_idx[row].tolist()):
                    pos_scores = sims[row, chosen_pool]
                    best_local = int(chosen_pool[int(np.argmax(pos_scores))])
                    inject(best_local)

            if neg_local is not None and neg_count is not None and neg_type is not None:
                for required_type in (1, 3):
                    typed = [
                        int(neg_local[row, col])
                        for col in range(int(neg_count[row]))
                        if int(neg_local[row, col]) >= 0 and int(neg_type[row, col]) == required_type
                    ]
                    if typed and not any(int(idx) in typed for idx in top_idx[row].tolist()):
                        typed_scores = sims[row, typed]
                        inject(int(typed[int(np.argmax(typed_scores))]))
            order = np.argsort(-top_score[row])
            top_idx[row] = top_idx[row, order]
            top_score[row] = top_score[row, order]
    return top_idx.astype(np.int32), top_score.astype(np.float32)


def build_candidate_positive_payload(
    candidate_local_idx: np.ndarray,
    pos_local: np.ndarray,
    pos_count: np.ndarray,
    pos_tier: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.zeros(candidate_local_idx.shape, dtype=np.float32)
    gain = np.zeros(candidate_local_idx.shape, dtype=np.float32)
    factual = np.zeros(candidate_local_idx.shape, dtype=np.float32)
    for row in range(len(candidate_local_idx)):
        positive_map = {
            int(pos_local[row, col]): int(pos_tier[row, col])
            for col in range(int(pos_count[row]))
            if int(pos_local[row, col]) >= 0
        }
        if not positive_map:
            continue
        for col, idx in enumerate(candidate_local_idx[row].tolist()):
            tier = int(positive_map.get(int(idx), 0))
            if tier > 0:
                mask[row, col] = 1.0
                gain[row, col] = float(tier_gain_array(np.asarray([tier], dtype=np.int32))[0])
                factual[row, col] = float(tier in {1, 2})
    return mask, gain, factual


def masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    weight = torch.clamp(mask, 0.0, 1.0)
    denom = torch.clamp(weight.sum(), min=1.0)
    loss = F.smooth_l1_loss(pred * weight, target * weight, reduction="sum")
    return loss / denom


def relevance_set_loss(relevance_logit: torch.Tensor, positive_mask: torch.Tensor, positive_gain: torch.Tensor) -> torch.Tensor:
    losses = []
    for row in range(relevance_logit.shape[0]):
        pos = positive_mask[row] > 0.5
        if not torch.any(pos):
            continue
        gain_bias = torch.log(torch.clamp(positive_gain[row, pos], min=1e-6))
        pos_term = torch.logsumexp(relevance_logit[row, pos] + gain_bias, dim=0)
        all_term = torch.logsumexp(relevance_logit[row], dim=0)
        losses.append(-(pos_term - all_term))
    if not losses:
        return torch.zeros((), dtype=relevance_logit.dtype, device=relevance_logit.device)
    return torch.stack(losses).mean()


def relevance_pair_loss(
    relevance_logit: torch.Tensor,
    positive_mask: torch.Tensor,
    positive_gain: torch.Tensor,
    margin: float = 0.10,
) -> torch.Tensor:
    losses = []
    for row in range(relevance_logit.shape[0]):
        pos = positive_mask[row] > 0.5
        neg = ~pos
        if not torch.any(pos) or not torch.any(neg):
            continue
        pos_score = torch.max(relevance_logit[row, pos] + torch.log(torch.clamp(positive_gain[row, pos], min=1e-6)))
        neg_score = torch.max(relevance_logit[row, neg])
        losses.append(F.softplus(neg_score - pos_score + float(margin)))
    if not losses:
        return torch.zeros((), dtype=relevance_logit.dtype, device=relevance_logit.device)
    return torch.stack(losses).mean()


def factual_priority_loss(
    relevance_logit: torch.Tensor,
    positive_mask: torch.Tensor,
    factual_mask: torch.Tensor,
    margin: float = 0.12,
) -> torch.Tensor:
    losses = []
    for row in range(relevance_logit.shape[0]):
        factual = factual_mask[row] > 0.5
        pos = positive_mask[row] > 0.5
        neg = ~pos
        if not torch.any(factual):
            continue
        factual_score = torch.max(relevance_logit[row, factual])
        other_pos = pos & (~factual)
        other_pos_score = torch.max(relevance_logit[row, other_pos]) if torch.any(other_pos) else factual_score - 0.5
        neg_score = torch.max(relevance_logit[row, neg]) if torch.any(neg) else factual_score - 0.5
        losses.append(
            0.5 * F.softplus(other_pos_score - factual_score + float(margin) * 0.5)
            + 0.5 * F.softplus(neg_score - factual_score + float(margin))
        )
    if not losses:
        return torch.zeros((), dtype=relevance_logit.dtype, device=relevance_logit.device)
    return torch.stack(losses).mean()


def fit_affine(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    pred_arr = np.asarray(pred, dtype=np.float32).reshape(-1)
    target_arr = np.asarray(target, dtype=np.float32).reshape(-1)
    if mask is not None:
        valid = np.asarray(mask, dtype=np.float32).reshape(-1) > 0.5
        pred_arr = pred_arr[valid]
        target_arr = target_arr[valid]
    if len(pred_arr) == 0:
        return 1.0, 0.0
    X = np.column_stack([pred_arr, np.ones(len(pred_arr), dtype=np.float32)])
    coef, _, _, _ = np.linalg.lstsq(X, target_arr, rcond=None)
    return float(coef[0]), float(coef[1])


def final_anchor_score(
    relevance_logit: np.ndarray,
    benefit_hat: np.ndarray,
    latent_accept_hat: np.ndarray,
    uncertainty: np.ndarray,
) -> np.ndarray:
    rel = row_normalize(relevance_logit)
    ben = row_normalize(benefit_hat)
    acc = row_normalize(latent_accept_hat)
    unc = np.asarray(uncertainty, dtype=np.float32)
    return (0.52 * rel + 0.28 * ben + 0.20 * acc - 0.35 * unc).astype(np.float32)


def rank_metrics(
    final_score: np.ndarray,
    positive_mask: np.ndarray,
    positive_gain: np.ndarray,
    factual_mask: np.ndarray,
) -> dict[str, float]:
    hit_at_10 = []
    ranks = []
    weighted_ndcg = []
    factual_ranks = []
    factual_hit_at_10 = []
    for row in range(len(final_score)):
        order = np.argsort(-final_score[row])
        pos = np.where(positive_mask[row] > 0.5)[0]
        if len(pos) == 0:
            ranks.append(final_score.shape[1] + 1)
            hit_at_10.append(0.0)
            weighted_ndcg.append(0.0)
            if np.any(factual_mask[row] > 0.5):
                factual_ranks.append(final_score.shape[1] + 1)
                factual_hit_at_10.append(0.0)
            continue
        best_rank = final_score.shape[1] + 1
        for rank, idx in enumerate(order.tolist(), start=1):
            if int(idx) in pos.tolist():
                best_rank = rank
                break
        ranks.append(float(best_rank))
        hit_at_10.append(float(best_rank <= 10))
        dcg = 0.0
        top10 = order[: min(10, len(order))]
        for rank, idx in enumerate(top10.tolist(), start=1):
            gain = float(positive_gain[row, int(idx)])
            if gain > 0.0:
                dcg += gain / float(np.log2(rank + 1.0))
        ideal_gains = sorted([float(positive_gain[row, int(idx)]) for idx in pos.tolist()], reverse=True)[: len(top10)]
        idcg = sum(gain / float(np.log2(rank + 1.0)) for rank, gain in enumerate(ideal_gains, start=1))
        weighted_ndcg.append(dcg / idcg if idcg > 0.0 else 0.0)
        factual = np.where(factual_mask[row] > 0.5)[0]
        if len(factual) > 0:
            factual_rank = final_score.shape[1] + 1
            factual_set = set(int(idx) for idx in factual.tolist())
            for rank, idx in enumerate(order.tolist(), start=1):
                if int(idx) in factual_set:
                    factual_rank = rank
                    break
            factual_ranks.append(float(factual_rank))
            factual_hit_at_10.append(float(factual_rank <= 10))
    ranks_arr = np.asarray(ranks, dtype=np.float32)
    valid = ranks_arr <= float(final_score.shape[1])
    factual_ranks_arr = np.asarray(factual_ranks, dtype=np.float32) if factual_ranks else np.zeros(0, dtype=np.float32)
    factual_valid = factual_ranks_arr <= float(final_score.shape[1]) if len(factual_ranks_arr) else np.zeros(0, dtype=bool)
    return {
        "hit_at_10": float(np.mean(hit_at_10)),
        "conditional_mean_rank": float(ranks_arr[valid].mean()) if np.any(valid) else float(final_score.shape[1] + 1),
        "contained_rows": int(np.sum(valid)),
        "weighted_ndcg_at_10": float(np.mean(weighted_ndcg)) if weighted_ndcg else 0.0,
        "factual_positive_rate": float(np.mean(np.any(factual_mask > 0.5, axis=1))),
        "factual_hit_at_10": float(np.mean(factual_hit_at_10)) if factual_hit_at_10 else 0.0,
        "factual_conditional_mean_rank": float(factual_ranks_arr[factual_valid].mean()) if np.any(factual_valid) else float(final_score.shape[1] + 1),
    }


def collect_candidate_targets(
    anchor_df: pd.DataFrame,
    train_anchor_idx: np.ndarray,
    candidate_local_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    train_anchor_df = anchor_df.iloc[train_anchor_idx].reset_index(drop=True)
    valid_local = np.asarray(candidate_local_idx, dtype=np.int64).clip(min=0)
    out = {
        "benefit_target": train_anchor_df["benefit_target"].to_numpy(dtype=np.float32)[valid_local],
        "accept_pref_target": train_anchor_df["accept_pref_target"].to_numpy(dtype=np.float32)[valid_local],
        "accept_rating_target": train_anchor_df["accept_rating_target"].to_numpy(dtype=np.float32)[valid_local],
        "accept_pref_mask": train_anchor_df["accept_pref_mask"].to_numpy(dtype=np.float32)[valid_local],
        "accept_rating_mask": train_anchor_df["accept_rating_mask"].to_numpy(dtype=np.float32)[valid_local],
        "acceptance_obs": train_anchor_df["acceptance_obs"].to_numpy(dtype=np.float32)[valid_local],
        "candidate_song_id": train_anchor_df["song_id"].astype(str).to_numpy()[valid_local],
    }
    return out


@dataclass
class RerankSample:
    pair_features: torch.Tensor
    positive_mask: torch.Tensor
    positive_gain: torch.Tensor
    factual_mask: torch.Tensor
    benefit_target: torch.Tensor
    accept_pref_target: torch.Tensor
    accept_rating_target: torch.Tensor
    accept_pref_mask: torch.Tensor
    accept_rating_mask: torch.Tensor


class AnchorCandidateDataset(Dataset):
    def __init__(
        self,
        pair_features: np.ndarray,
        positive_mask: np.ndarray,
        positive_gain: np.ndarray,
        factual_mask: np.ndarray,
        targets: dict[str, np.ndarray],
    ) -> None:
        self.pair_features = np.asarray(pair_features, dtype=np.float32)
        self.positive_mask = np.asarray(positive_mask, dtype=np.float32)
        self.positive_gain = np.asarray(positive_gain, dtype=np.float32)
        self.factual_mask = np.asarray(factual_mask, dtype=np.float32)
        self.targets = {key: np.asarray(value, dtype=np.float32) for key, value in targets.items() if key != "candidate_song_id"}

    def __len__(self) -> int:
        return int(len(self.pair_features))

    def __getitem__(self, idx: int) -> RerankSample:
        return RerankSample(
            pair_features=torch.from_numpy(self.pair_features[idx]),
            positive_mask=torch.from_numpy(self.positive_mask[idx]),
            positive_gain=torch.from_numpy(self.positive_gain[idx]),
            factual_mask=torch.from_numpy(self.factual_mask[idx]),
            benefit_target=torch.from_numpy(self.targets["benefit_target"][idx]),
            accept_pref_target=torch.from_numpy(self.targets["accept_pref_target"][idx]),
            accept_rating_target=torch.from_numpy(self.targets["accept_rating_target"][idx]),
            accept_pref_mask=torch.from_numpy(self.targets["accept_pref_mask"][idx]),
            accept_rating_mask=torch.from_numpy(self.targets["accept_rating_mask"][idx]),
        )


def collate_rerank_batch(batch: list[RerankSample]) -> dict[str, torch.Tensor]:
    return {
        "pair_features": torch.stack([item.pair_features for item in batch]),
        "positive_mask": torch.stack([item.positive_mask for item in batch]),
        "positive_gain": torch.stack([item.positive_gain for item in batch]),
        "factual_mask": torch.stack([item.factual_mask for item in batch]),
        "benefit_target": torch.stack([item.benefit_target for item in batch]),
        "accept_pref_target": torch.stack([item.accept_pref_target for item in batch]),
        "accept_rating_target": torch.stack([item.accept_rating_target for item in batch]),
        "accept_pref_mask": torch.stack([item.accept_pref_mask for item in batch]),
        "accept_rating_mask": torch.stack([item.accept_rating_mask for item in batch]),
    }


def evaluate_split(
    model: UtilityReranker,
    pair_features: np.ndarray,
    positive_mask: np.ndarray,
    positive_gain: np.ndarray,
    factual_mask: np.ndarray,
    targets: dict[str, np.ndarray],
    diagnostics: dict[str, np.ndarray],
    user_prior: np.ndarray,
    exact_rating: np.ndarray,
    exact_mask: np.ndarray,
    device: torch.device,
    batch_size: int,
    calibration: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    benefit_hat = []
    pref_hat = []
    rating_hat = []
    relevance = []
    with torch.no_grad():
        pair_tensor = torch.from_numpy(pair_features)
        for start in range(0, len(pair_tensor), batch_size):
            end = min(start + batch_size, len(pair_tensor))
            outputs = model(pair_tensor[start:end].to(device))
            benefit_hat.append(outputs["benefit_hat"].cpu().numpy())
            pref_hat.append(outputs["accept_pref_hat"].cpu().numpy())
            rating_hat.append(outputs["accept_rating_hat"].cpu().numpy())
            relevance.append(outputs["relevance_logit"].cpu().numpy())

    pred = {
        "benefit_hat": np.concatenate(benefit_hat, axis=0),
        "accept_pref_hat": np.concatenate(pref_hat, axis=0),
        "accept_rating_hat": np.concatenate(rating_hat, axis=0),
        "relevance_logit": np.concatenate(relevance, axis=0),
    }

    if calibration is None:
        benefit_scale, benefit_bias = fit_affine(pred["benefit_hat"], targets["benefit_target"])
        pref_scale, pref_bias = fit_affine(pred["accept_pref_hat"], targets["accept_pref_target"], targets["accept_pref_mask"])
        rating_scale, rating_bias = fit_affine(pred["accept_rating_hat"], targets["accept_rating_target"], targets["accept_rating_mask"])
    else:
        benefit_scale = float(calibration["benefit_scale"])
        benefit_bias = float(calibration["benefit_bias"])
        pref_scale = float(calibration["accept_pref_scale"])
        pref_bias = float(calibration["accept_pref_bias"])
        rating_scale = float(calibration["accept_rating_scale"])
        rating_bias = float(calibration["accept_rating_bias"])

    pred["benefit_hat"] = np.clip(pred["benefit_hat"] * benefit_scale + benefit_bias, -1.5, 1.5)
    pred["accept_pref_hat"] = np.clip(pred["accept_pref_hat"] * pref_scale + pref_bias, -1.0, 1.0)
    pred["accept_rating_hat"] = np.clip(pred["accept_rating_hat"] * rating_scale + rating_bias, -1.0, 1.0)
    pred["latent_accept_hat"] = latent_acceptance(
        pred["accept_pref_hat"],
        pred["accept_rating_hat"],
        user_prior=user_prior,
        exact_rating=exact_rating,
        exact_mask=exact_mask,
    )
    pred["final_anchor_score"] = final_anchor_score(
        relevance_logit=pred["relevance_logit"],
        benefit_hat=pred["benefit_hat"],
        latent_accept_hat=pred["latent_accept_hat"],
        uncertainty=diagnostics["uncertainty"],
    )

    benefit_mae = float(np.mean(np.abs(pred["benefit_hat"] - targets["benefit_target"])))
    pref_mask = targets["accept_pref_mask"] > 0.5
    rating_mask = targets["accept_rating_mask"] > 0.5
    accept_obs_mask = pref_mask | rating_mask
    accept_obs = np.where(pref_mask, targets["accept_pref_target"], targets["accept_rating_target"])
    metrics = {
        "benefit_mae": benefit_mae,
        "accept_pref_mae": float(np.mean(np.abs(pred["accept_pref_hat"][pref_mask] - targets["accept_pref_target"][pref_mask]))) if np.any(pref_mask) else 0.0,
        "accept_rating_mae": float(np.mean(np.abs(pred["accept_rating_hat"][rating_mask] - targets["accept_rating_target"][rating_mask]))) if np.any(rating_mask) else 0.0,
        "blended_accept_mae": float(np.mean(np.abs(pred["latent_accept_hat"][accept_obs_mask] - accept_obs[accept_obs_mask]))) if np.any(accept_obs_mask) else 0.0,
    }
    metrics.update(rank_metrics(pred["final_anchor_score"], positive_mask, positive_gain, factual_mask))
    calib = {
        "benefit_scale": benefit_scale,
        "benefit_bias": benefit_bias,
        "accept_pref_scale": pref_scale,
        "accept_pref_bias": pref_bias,
        "accept_rating_scale": rating_scale,
        "accept_rating_bias": rating_bias,
    }
    return metrics, calib, pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the V2.2 anchor reranker.")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=7.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--candidate-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    decision_df = pd.read_parquet(REBUILD_DIR / "decision_table.parquet")
    anchor_df = pd.read_parquet(REBUILD_DIR / "anchor_table.parquet")
    stage1_df = pd.read_parquet(REBUILD_DIR / "stage1_history_table.parquet")
    song_catalog = pd.read_parquet(REBUILD_DIR / "song_catalog.parquet")
    context_embeddings = np.load(REBUILD_DIR / "context_embeddings.npy").astype(np.float32)
    song_embeddings = np.load(REBUILD_DIR / "song_embeddings.npy").astype(np.float32)
    query_embeddings = np.load(REBUILD_DIR / "query_embeddings.npy").astype(np.float32)
    anchor_embeddings = np.load(REBUILD_DIR / "anchor_query_embeddings.npy").astype(np.float32)
    user_out = np.load(REBUILD_DIR / "user_encoder_outputs.npz")
    positive_bundle = load_supervision_sets(str(REBUILD_DIR / "anchor_positive_sets.npz"))
    positive_idx = positive_bundle["indices"]
    positive_count = positive_bundle["counts"]
    positive_tier = positive_bundle["tiers"]
    negative_bundle = load_supervision_sets(str(REBUILD_DIR / "anchor_negative_pools.npz"))
    negative_idx = negative_bundle["indices"]
    negative_count = negative_bundle["counts"]
    negative_type = negative_bundle["types"]
    song_pred = pd.read_parquet(REBUILD_DIR / "song_encoder_predictions.parquet")

    row_user_embeddings, _, _ = build_row_user_views(decision_df, user_out)
    stage1_exact_rating, stage1_user_prior, stage1_global_prior = build_stage1_acceptance_features(stage1_df)
    train_anchor_idx = anchor_df.index[anchor_df["available_for_index"] == 1].to_numpy(dtype=np.int64)
    global_to_local = np.full(len(anchor_df), -1, dtype=np.int32)
    global_to_local[train_anchor_idx] = np.arange(len(train_anchor_idx), dtype=np.int32)
    pos_local, pos_count_local, pos_tier_local = map_global_sets_to_train(positive_idx, positive_count, global_to_local, meta=positive_tier)
    neg_local, neg_count_local, neg_type_local = map_global_sets_to_train(negative_idx, negative_count, global_to_local, meta=negative_type)
    if pos_tier_local is None or neg_type_local is None:
        raise RuntimeError("V2.2 reranker requires positive tiers and negative types.")

    song_affect = song_pred[["pred_valence", "pred_arousal"]].to_numpy(dtype=np.float32)
    song_dyn_summary = song_catalog[
        [
            "dyn_valence_delta",
            "dyn_arousal_delta",
            "dyn_valence_vol",
            "dyn_arousal_vol",
            "dyn_arousal_peak",
            "eda_impact_norm",
        ]
    ].to_numpy(dtype=np.float32)
    train_anchor_views = build_train_anchor_views(
        anchor_df=anchor_df,
        train_anchor_idx=train_anchor_idx,
        context_embeddings=context_embeddings,
        row_user_embeddings=row_user_embeddings,
        song_embeddings=song_embeddings,
        song_affect=song_affect,
        song_dyn_summary=song_dyn_summary,
    )

    train_rows = decision_df.index[decision_df["split"] == "train"].to_numpy(dtype=np.int64)
    val_rows = decision_df.index[decision_df["split"] == "val"].to_numpy(dtype=np.int64)
    test_rows = decision_df.index[decision_df["split"] == "test"].to_numpy(dtype=np.int64)

    train_candidate_idx, train_candidate_score = build_candidate_sets(
        query_embeddings=query_embeddings[train_rows],
        anchor_embeddings=anchor_embeddings,
        pos_local=pos_local[train_rows],
        pos_count=pos_count_local[train_rows],
        pos_tier=pos_tier_local[train_rows],
        neg_local=neg_local[train_rows],
        neg_count=neg_count_local[train_rows],
        neg_type=neg_type_local[train_rows],
        candidate_size=args.candidate_size,
        inject_positive=True,
    )
    val_candidate_idx, val_candidate_score = build_candidate_sets(
        query_embeddings=query_embeddings[val_rows],
        anchor_embeddings=anchor_embeddings,
        pos_local=pos_local[val_rows],
        pos_count=pos_count_local[val_rows],
        pos_tier=pos_tier_local[val_rows],
        neg_local=None,
        neg_count=None,
        neg_type=None,
        candidate_size=args.candidate_size,
        inject_positive=False,
    )
    test_candidate_idx, test_candidate_score = build_candidate_sets(
        query_embeddings=query_embeddings[test_rows],
        anchor_embeddings=anchor_embeddings,
        pos_local=pos_local[test_rows],
        pos_count=pos_count_local[test_rows],
        pos_tier=pos_tier_local[test_rows],
        neg_local=None,
        neg_count=None,
        neg_type=None,
        candidate_size=args.candidate_size,
        inject_positive=False,
    )

    train_pair_features, train_diag = build_anchor_pair_features(
        row_indices=train_rows,
        query_embeddings=query_embeddings[train_rows],
        candidate_local_idx=train_candidate_idx,
        candidate_score=train_candidate_score,
        decision_df=decision_df,
        train_anchor_views=train_anchor_views,
        row_user_embeddings=row_user_embeddings,
        stage1_exact_rating=stage1_exact_rating,
        stage1_user_prior=stage1_user_prior,
        stage1_global_prior=stage1_global_prior,
        anchor_embeddings=anchor_embeddings,
    )
    val_pair_features, val_diag = build_anchor_pair_features(
        row_indices=val_rows,
        query_embeddings=query_embeddings[val_rows],
        candidate_local_idx=val_candidate_idx,
        candidate_score=val_candidate_score,
        decision_df=decision_df,
        train_anchor_views=train_anchor_views,
        row_user_embeddings=row_user_embeddings,
        stage1_exact_rating=stage1_exact_rating,
        stage1_user_prior=stage1_user_prior,
        stage1_global_prior=stage1_global_prior,
        anchor_embeddings=anchor_embeddings,
    )
    test_pair_features, test_diag = build_anchor_pair_features(
        row_indices=test_rows,
        query_embeddings=query_embeddings[test_rows],
        candidate_local_idx=test_candidate_idx,
        candidate_score=test_candidate_score,
        decision_df=decision_df,
        train_anchor_views=train_anchor_views,
        row_user_embeddings=row_user_embeddings,
        stage1_exact_rating=stage1_exact_rating,
        stage1_user_prior=stage1_user_prior,
        stage1_global_prior=stage1_global_prior,
        anchor_embeddings=anchor_embeddings,
    )

    train_positive_mask, train_positive_gain, train_factual_mask = build_candidate_positive_payload(
        train_candidate_idx,
        pos_local[train_rows],
        pos_count_local[train_rows],
        pos_tier_local[train_rows],
    )
    val_positive_mask, val_positive_gain, val_factual_mask = build_candidate_positive_payload(
        val_candidate_idx,
        pos_local[val_rows],
        pos_count_local[val_rows],
        pos_tier_local[val_rows],
    )
    test_positive_mask, test_positive_gain, test_factual_mask = build_candidate_positive_payload(
        test_candidate_idx,
        pos_local[test_rows],
        pos_count_local[test_rows],
        pos_tier_local[test_rows],
    )

    train_targets = collect_candidate_targets(anchor_df, train_anchor_idx, train_candidate_idx)
    val_targets = collect_candidate_targets(anchor_df, train_anchor_idx, val_candidate_idx)
    test_targets = collect_candidate_targets(anchor_df, train_anchor_idx, test_candidate_idx)

    train_dataset = AnchorCandidateDataset(
        train_pair_features,
        train_positive_mask,
        train_positive_gain,
        train_factual_mask,
        train_targets,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_rerank_batch)

    model = UtilityReranker(feature_dim=ANCHOR_PAIR_FEATURE_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_tuple = (float("inf"), float("inf"), float("inf"), float("inf"))
    best_state = None
    history = []

    val_user_prior = val_pair_features[:, :, -3]
    val_exact_rating = val_pair_features[:, :, -2]
    val_exact_mask = val_pair_features[:, :, -1]
    test_user_prior = test_pair_features[:, :, -3]
    test_exact_rating = test_pair_features[:, :, -2]
    test_exact_mask = test_pair_features[:, :, -1]

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rel = 0.0
        total_pair = 0.0
        total_fact = 0.0
        total_benefit = 0.0
        total_pref = 0.0
        total_rating = 0.0
        steps = 0

        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            outputs = model(batch["pair_features"])
            loss_rel = relevance_set_loss(outputs["relevance_logit"], batch["positive_mask"], batch["positive_gain"])
            loss_pair = relevance_pair_loss(outputs["relevance_logit"], batch["positive_mask"], batch["positive_gain"])
            loss_fact = factual_priority_loss(outputs["relevance_logit"], batch["positive_mask"], batch["factual_mask"])
            loss_benefit = F.smooth_l1_loss(outputs["benefit_hat"], batch["benefit_target"])
            loss_pref = masked_smooth_l1(outputs["accept_pref_hat"], batch["accept_pref_target"], batch["accept_pref_mask"])
            loss_rating = masked_smooth_l1(outputs["accept_rating_hat"], batch["accept_rating_target"], batch["accept_rating_mask"])
            loss = loss_rel + 0.35 * loss_pair + 0.50 * loss_fact + 0.35 * loss_benefit + 0.25 * loss_pref + 0.25 * loss_rating
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_rel += float(loss_rel.item())
            total_pair += float(loss_pair.item())
            total_fact += float(loss_fact.item())
            total_benefit += float(loss_benefit.item())
            total_pref += float(loss_pref.item())
            total_rating += float(loss_rating.item())
            steps += 1

        val_metrics, val_calib, _ = evaluate_split(
            model=model,
            pair_features=val_pair_features,
            positive_mask=val_positive_mask,
            positive_gain=val_positive_gain,
            factual_mask=val_factual_mask,
            targets=val_targets,
            diagnostics=val_diag,
            user_prior=val_user_prior,
            exact_rating=val_exact_rating,
            exact_mask=val_exact_mask,
            device=device,
            batch_size=args.batch_size,
        )
        scheduler.step(val_metrics["blended_accept_mae"])
        candidate_tuple = (
            max(float(val_metrics["blended_accept_mae"]) - 0.35, 0.0),
            float(val_metrics["conditional_mean_rank"]),
            float(val_metrics["factual_conditional_mean_rank"]),
            float(val_metrics["benefit_mae"]),
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(steps, 1),
                "train_relevance_loss": total_rel / max(steps, 1),
                "train_pair_loss": total_pair / max(steps, 1),
                "train_factual_loss": total_fact / max(steps, 1),
                "train_benefit_loss": total_benefit / max(steps, 1),
                "train_pref_loss": total_pref / max(steps, 1),
                "train_rating_loss": total_rating / max(steps, 1),
                "val_hit_at_10": float(val_metrics["hit_at_10"]),
                "val_conditional_mean_rank": float(val_metrics["conditional_mean_rank"]),
                "val_weighted_ndcg_at_10": float(val_metrics["weighted_ndcg_at_10"]),
                "val_factual_hit_at_10": float(val_metrics["factual_hit_at_10"]),
                "val_factual_conditional_mean_rank": float(val_metrics["factual_conditional_mean_rank"]),
                "val_benefit_mae": float(val_metrics["benefit_mae"]),
                "val_blended_accept_mae": float(val_metrics["blended_accept_mae"]),
            }
        )
        if candidate_tuple < best_tuple:
            best_tuple = candidate_tuple
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "calibration": val_calib,
            }

    if best_state is None:
        raise RuntimeError("Reranker training did not produce a checkpoint.")

    model.load_state_dict(best_state["model"])
    calib = best_state["calibration"]
    model.set_calibration(
        benefit_scale=float(calib["benefit_scale"]),
        benefit_bias=float(calib["benefit_bias"]),
        accept_pref_scale=float(calib["accept_pref_scale"]),
        accept_pref_bias=float(calib["accept_pref_bias"]),
        accept_rating_scale=float(calib["accept_rating_scale"]),
        accept_rating_bias=float(calib["accept_rating_bias"]),
    )

    val_metrics, _, val_pred = evaluate_split(
        model=model,
        pair_features=val_pair_features,
        positive_mask=val_positive_mask,
        positive_gain=val_positive_gain,
        factual_mask=val_factual_mask,
        targets=val_targets,
        diagnostics=val_diag,
        user_prior=val_user_prior,
        exact_rating=val_exact_rating,
        exact_mask=val_exact_mask,
        device=device,
        batch_size=args.batch_size,
        calibration=calib,
    )
    test_metrics, _, test_pred = evaluate_split(
        model=model,
        pair_features=test_pair_features,
        positive_mask=test_positive_mask,
        positive_gain=test_positive_gain,
        factual_mask=test_factual_mask,
        targets=test_targets,
        diagnostics=test_diag,
        user_prior=test_user_prior,
        exact_rating=test_exact_rating,
        exact_mask=test_exact_mask,
        device=device,
        batch_size=args.batch_size,
        calibration=calib,
    )

    torch.save(
        {
            "model": model.state_dict(),
            "feature_dim": ANCHOR_PAIR_FEATURE_DIM,
            "config": {
                "candidate_size": int(args.candidate_size),
                "best_epoch": int(best_state["epoch"]),
                "calibration": calib,
                "alpha": 0.7,
                "beta": 0.3,
                "benefit_range": 1.5,
            },
        },
        MODELS_DIR / "reranker.pt",
    )

    metrics = {
        "best_epoch": int(best_state["epoch"]),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
    }
    (MODELS_DIR / "reranker_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    test_df = pd.DataFrame(
        {
            "row_idx": test_rows.astype(np.int32),
            "contained_positive_count": test_positive_mask.sum(axis=1).astype(np.int32),
            "contained_factual_count": test_factual_mask.sum(axis=1).astype(np.int32),
            "top_score": test_pred["final_anchor_score"][:, 0],
            "top_relevance": test_pred["relevance_logit"][:, 0],
            "top_benefit_hat": test_pred["benefit_hat"][:, 0],
            "top_accept_pref_hat": test_pred["accept_pref_hat"][:, 0],
            "top_accept_rating_hat": test_pred["accept_rating_hat"][:, 0],
            "top_latent_accept_hat": test_pred["latent_accept_hat"][:, 0],
        }
    )
    test_df.to_parquet(REBUILD_DIR / "anchor_rerank_test_predictions.parquet", index=False)

    with (MODELS_DIR / "reranker_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
