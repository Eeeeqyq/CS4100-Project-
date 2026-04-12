"""
Train the V2.2 query tower for split-safe SiTunes anchor retrieval.

Primary task:
- retrieve good train-anchor interventions for a context/user/goal query

Legacy exact-song recovery is deliberately left to downstream diagnostics.
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
    build_anchor_encoder_features,
    build_row_user_views,
    load_positive_negative_sets,
    load_supervision_sets,
    normalize_rows,
    tier_gain_array,
)
from src.v2.data.schema import GOAL_NAMES
from src.v2.models.anchor_encoder import AnchorEncoder
from src.v2.models.query_tower import QueryTower


REBUILD_DIR = PROJECT_ROOT / "data" / "processed" / "rebuild"
MODELS_DIR = PROJECT_ROOT / "models" / "rebuild"
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class QuerySample:
    row_idx: torch.Tensor
    z_ctx: torch.Tensor
    u_user: torch.Tensor
    goal_onehot: torch.Tensor
    tau: torch.Tensor


class QueryDataset(Dataset):
    def __init__(
        self,
        row_indices: np.ndarray,
        z_ctx: np.ndarray,
        u_user: np.ndarray,
        goal_idx: np.ndarray,
        tau: np.ndarray,
    ) -> None:
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.z_ctx = np.asarray(z_ctx, dtype=np.float32)
        self.u_user = np.asarray(u_user, dtype=np.float32)
        self.goal_onehot = np.eye(4, dtype=np.float32)[np.asarray(goal_idx, dtype=np.int64)]
        self.tau = np.asarray(tau, dtype=np.float32)

    def __len__(self) -> int:
        return int(len(self.row_indices))

    def __getitem__(self, item: int) -> QuerySample:
        idx = int(self.row_indices[item])
        return QuerySample(
            row_idx=torch.tensor(idx, dtype=torch.int64),
            z_ctx=torch.from_numpy(self.z_ctx[idx]),
            u_user=torch.from_numpy(self.u_user[idx]),
            goal_onehot=torch.from_numpy(self.goal_onehot[idx]),
            tau=torch.from_numpy(self.tau[idx]),
        )


def collate_query_batch(batch: list[QuerySample]) -> dict[str, torch.Tensor]:
    return {
        "row_idx": torch.stack([item.row_idx for item in batch]),
        "z_ctx": torch.stack([item.z_ctx for item in batch]),
        "u_user": torch.stack([item.u_user for item in batch]),
        "goal_onehot": torch.stack([item.goal_onehot for item in batch]),
        "tau": torch.stack([item.tau for item in batch]),
    }


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


def multi_positive_loss(
    logits: torch.Tensor,
    pos_local: np.ndarray,
    pos_count: np.ndarray,
    pos_tier: np.ndarray,
    temperature: float,
) -> torch.Tensor:
    scaled = logits / float(temperature)
    losses = []
    for row in range(logits.shape[0]):
        count = int(pos_count[row])
        if count <= 0:
            continue
        pos_idx = torch.as_tensor(pos_local[row, :count], dtype=torch.long, device=logits.device)
        gains = torch.from_numpy(tier_gain_array(pos_tier[row, :count])).to(logits.device)
        gain_bias = torch.log(torch.clamp(gains, min=1e-6))
        pos_term = torch.logsumexp(scaled[row, pos_idx] + gain_bias, dim=0)
        all_term = torch.logsumexp(scaled[row], dim=0)
        losses.append(-(pos_term - all_term))
    if not losses:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    return torch.stack(losses).mean()


def hard_negative_margin_loss(
    logits: torch.Tensor,
    pos_local: np.ndarray,
    pos_count: np.ndarray,
    pos_tier: np.ndarray,
    neg_local: np.ndarray,
    neg_count: np.ndarray,
    margin: float,
) -> torch.Tensor:
    losses = []
    for row in range(logits.shape[0]):
        pcount = int(pos_count[row])
        ncount = int(neg_count[row])
        if pcount <= 0:
            continue
        pos_idx = torch.as_tensor(pos_local[row, :pcount], dtype=torch.long, device=logits.device)
        gains = torch.from_numpy(tier_gain_array(pos_tier[row, :pcount])).to(logits.device)
        pos_score = torch.max(logits[row, pos_idx] + torch.log(torch.clamp(gains, min=1e-6)))
        if ncount > 0:
            neg_idx = torch.as_tensor(neg_local[row, :ncount], dtype=torch.long, device=logits.device)
            neg_score = torch.max(logits[row, neg_idx])
        else:
            mask = torch.ones(logits.shape[1], dtype=torch.bool, device=logits.device)
            mask[pos_idx] = False
            neg_score = torch.max(logits[row, mask]) if mask.any() else pos_score - 1.0
        losses.append(F.softplus(neg_score - pos_score + float(margin)))
    if not losses:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    return torch.stack(losses).mean()


def rank_metrics(
    query_embeddings: np.ndarray,
    anchor_embeddings: np.ndarray,
    pos_local: np.ndarray,
    pos_count: np.ndarray,
    pos_tier: np.ndarray,
    row_goal_idx: np.ndarray,
    row_song_id: np.ndarray,
    anchor_song_id: np.ndarray,
    top_ks: tuple[int, int] = (20, 50),
) -> tuple[dict[str, float], pd.DataFrame]:
    sims = np.asarray(query_embeddings, dtype=np.float32) @ np.asarray(anchor_embeddings, dtype=np.float32).T
    max_k = min(max(top_ks), sims.shape[1])
    top_idx = np.argpartition(-sims, kth=max_k - 1, axis=1)[:, :max_k]
    top_score = np.take_along_axis(sims, top_idx, axis=1)
    order = np.argsort(-top_score, axis=1)
    top_idx = np.take_along_axis(top_idx, order, axis=1)
    top_score = np.take_along_axis(top_score, order, axis=1)

    records = []
    for row in range(len(query_embeddings)):
        positives = set(int(idx) for idx in pos_local[row, : pos_count[row]].tolist() if idx >= 0)
        tiers = {
            int(pos_local[row, col]): int(pos_tier[row, col])
            for col in range(int(pos_count[row]))
            if int(pos_local[row, col]) >= 0
        }
        gains = {idx: float(tier_gain_array(np.asarray([tier], dtype=np.int32))[0]) for idx, tier in tiers.items()}
        factual_anchor_pos = {
            idx for idx in positives if str(anchor_song_id[int(idx)]) == str(row_song_id[row])
        }
        hit_flags = {}
        best_rank = sims.shape[1] + 1
        reciprocal = 0.0
        factual_rank = sims.shape[1] + 1
        weighted_rr = 0.0
        weighted_recall = {k: 0.0 for k in top_ks}
        weighted_ndcg_10 = 0.0
        first_positive_is_factual = 0.0
        if positives:
            full_order = np.argsort(-sims[row])
            for rank, idx in enumerate(full_order.tolist(), start=1):
                if int(idx) in positives:
                    best_rank = rank
                    reciprocal = 1.0 / float(rank)
                    weighted_rr = float(gains.get(int(idx), 0.0)) / float(rank)
                    first_positive_is_factual = float(int(idx) in factual_anchor_pos)
                    break
            if factual_anchor_pos:
                for rank, idx in enumerate(full_order.tolist(), start=1):
                    if int(idx) in factual_anchor_pos:
                        factual_rank = rank
                        break
        for k in top_ks:
            k_idx = top_idx[row, : min(k, top_idx.shape[1])]
            hit_flags[f"recall_at_{k}"] = float(any(int(idx) in positives for idx in k_idx.tolist()))
            total_gain = float(sum(gains.values()))
            retrieved_gain = float(sum(gains.get(int(idx), 0.0) for idx in k_idx.tolist()))
            weighted_recall[k] = retrieved_gain / total_gain if total_gain > 0.0 else 0.0
        top10 = top_idx[row, : min(10, top_idx.shape[1])]
        dcg = 0.0
        ideal_gains = sorted(gains.values(), reverse=True)[: len(top10)]
        for rank, idx in enumerate(top10.tolist(), start=1):
            gain = gains.get(int(idx), 0.0)
            if gain > 0.0:
                dcg += gain / float(np.log2(rank + 1.0))
        idcg = sum(gain / float(np.log2(rank + 1.0)) for rank, gain in enumerate(ideal_gains, start=1))
        weighted_ndcg_10 = dcg / idcg if idcg > 0.0 else 0.0
        records.append(
            {
                "row_local": row,
                "goal_idx": int(row_goal_idx[row]),
                "positive_count": int(pos_count[row]),
                "factual_positive_available": int(len(factual_anchor_pos) > 0),
                "best_positive_rank": int(best_rank),
                "mrr": float(reciprocal),
                "weighted_mrr": float(weighted_rr),
                "weighted_ndcg_at_10": float(weighted_ndcg_10),
                "factual_positive_rank": int(factual_rank),
                "first_positive_is_factual": float(first_positive_is_factual),
                **hit_flags,
                **{f"weighted_recall_at_{k}": float(weighted_recall[k]) for k in top_ks},
            }
        )

    pred_df = pd.DataFrame(records)
    summary = {
        f"recall_at_{k}": float(pred_df[f"recall_at_{k}"].mean()) for k in top_ks
    }
    summary.update({
        f"weighted_recall_at_{k}": float(pred_df[f"weighted_recall_at_{k}"].mean()) for k in top_ks
    })
    contained = pred_df[pred_df["positive_count"] > 0]
    summary["conditional_mean_rank"] = float(contained["best_positive_rank"].mean()) if len(contained) else float(anchor_embeddings.shape[0] + 1)
    summary["mrr"] = float(contained["mrr"].mean()) if len(contained) else 0.0
    summary["weighted_mrr"] = float(contained["weighted_mrr"].mean()) if len(contained) else 0.0
    summary["weighted_ndcg_at_10"] = float(contained["weighted_ndcg_at_10"].mean()) if len(contained) else 0.0
    factual = pred_df[pred_df["factual_positive_available"] == 1]
    summary["factual_positive_rate"] = float(pred_df["factual_positive_available"].mean()) if len(pred_df) else 0.0
    summary["factual_positive_conditional_mean_rank"] = float(factual["factual_positive_rank"].mean()) if len(factual) else float(anchor_embeddings.shape[0] + 1)
    summary["first_positive_is_factual_rate"] = float(factual["first_positive_is_factual"].mean()) if len(factual) else 0.0
    goal_breakdown = {}
    for goal, group in pred_df.groupby("goal_idx"):
        goal_breakdown[GOAL_NAMES.get(int(goal), str(goal))] = {
            "rows": int(len(group)),
            "recall_at_20": float(group["recall_at_20"].mean()) if "recall_at_20" in group else 0.0,
            "recall_at_50": float(group["recall_at_50"].mean()) if "recall_at_50" in group else 0.0,
            "weighted_recall_at_20": float(group["weighted_recall_at_20"].mean()),
            "weighted_recall_at_50": float(group["weighted_recall_at_50"].mean()),
            "conditional_mean_rank": float(group["best_positive_rank"].mean()),
            "mrr": float(group["mrr"].mean()),
            "weighted_mrr": float(group["weighted_mrr"].mean()),
            "weighted_ndcg_at_10": float(group["weighted_ndcg_at_10"].mean()),
            "factual_positive_rate": float(group["factual_positive_available"].mean()),
        }
    summary["goal_breakdown"] = goal_breakdown
    return summary, pred_df


def export_query_embeddings(
    model: QueryTower,
    z_ctx: np.ndarray,
    u_user: np.ndarray,
    goal_idx: np.ndarray,
    tau: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outs = []
    goal_onehot = np.eye(4, dtype=np.float32)[goal_idx.astype(np.int64)]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(z_ctx), batch_size):
            end = min(start + batch_size, len(z_ctx))
            query = model(
                torch.from_numpy(z_ctx[start:end]).to(device),
                torch.from_numpy(u_user[start:end]).to(device),
                torch.from_numpy(goal_onehot[start:end]).to(device),
                torch.from_numpy(tau[start:end]).to(device),
            )
            outs.append(F.normalize(query, dim=-1).cpu().numpy())
    return np.concatenate(outs, axis=0)


def load_inputs() -> dict[str, object]:
    decision_df = pd.read_parquet(REBUILD_DIR / "decision_table.parquet")
    anchor_df = pd.read_parquet(REBUILD_DIR / "anchor_table.parquet")
    context_embeddings = np.load(REBUILD_DIR / "context_embeddings.npy").astype(np.float32)
    song_embeddings = np.load(REBUILD_DIR / "song_embeddings.npy").astype(np.float32)
    user_out = np.load(REBUILD_DIR / "user_encoder_outputs.npz")
    positive_bundle = load_supervision_sets(str(REBUILD_DIR / "anchor_positive_sets.npz"))
    positive_idx = positive_bundle["indices"]
    positive_count = positive_bundle["counts"]
    positive_tier = positive_bundle["tiers"]
    negative_idx, negative_count = load_positive_negative_sets(str(REBUILD_DIR / "anchor_negative_pools.npz"))
    row_user_embeddings, _, _ = build_row_user_views(decision_df, user_out)
    anchor_features = build_anchor_encoder_features(anchor_df, context_embeddings, row_user_embeddings, song_embeddings)

    train_anchor_idx = anchor_df.index[anchor_df["available_for_index"] == 1].to_numpy(dtype=np.int64)
    global_to_local = np.full(len(anchor_df), -1, dtype=np.int32)
    global_to_local[train_anchor_idx] = np.arange(len(train_anchor_idx), dtype=np.int32)
    pos_local, pos_count_local, pos_tier_local = map_global_sets_to_train(positive_idx, positive_count, global_to_local, meta=positive_tier)
    neg_local, neg_count_local, _ = map_global_sets_to_train(negative_idx, negative_count, global_to_local)
    if pos_tier_local is None:
        raise RuntimeError("Positive supervision tiers are required for V2.2 query training.")

    return {
        "decision_df": decision_df,
        "anchor_df": anchor_df,
        "context_embeddings": context_embeddings,
        "row_user_embeddings": row_user_embeddings,
        "anchor_features": anchor_features,
        "train_anchor_idx": train_anchor_idx,
        "pos_local": pos_local,
        "pos_count_local": pos_count_local,
        "pos_tier_local": pos_tier_local,
        "neg_local": neg_local,
        "neg_count_local": neg_count_local,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the V2.2 anchor query tower.")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--pairwise-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    bundle = load_inputs()
    decision_df: pd.DataFrame = bundle["decision_df"]  # type: ignore[assignment]
    anchor_df: pd.DataFrame = bundle["anchor_df"]  # type: ignore[assignment]
    context_embeddings: np.ndarray = bundle["context_embeddings"]  # type: ignore[assignment]
    row_user_embeddings: np.ndarray = bundle["row_user_embeddings"]  # type: ignore[assignment]
    anchor_features: np.ndarray = bundle["anchor_features"]  # type: ignore[assignment]
    train_anchor_idx: np.ndarray = bundle["train_anchor_idx"]  # type: ignore[assignment]
    pos_local: np.ndarray = bundle["pos_local"]  # type: ignore[assignment]
    pos_count_local: np.ndarray = bundle["pos_count_local"]  # type: ignore[assignment]
    pos_tier_local: np.ndarray = bundle["pos_tier_local"]  # type: ignore[assignment]
    neg_local: np.ndarray = bundle["neg_local"]  # type: ignore[assignment]
    neg_count_local: np.ndarray = bundle["neg_count_local"]  # type: ignore[assignment]

    tau = decision_df[["tau_valence", "tau_arousal"]].to_numpy(dtype=np.float32)
    goal_idx = decision_df["goal_idx"].to_numpy(dtype=np.int64)
    train_rows = decision_df.index[decision_df["split"] == "train"].to_numpy(dtype=np.int64)
    val_rows = decision_df.index[decision_df["split"] == "val"].to_numpy(dtype=np.int64)
    test_rows = decision_df.index[decision_df["split"] == "test"].to_numpy(dtype=np.int64)

    train_dataset = QueryDataset(train_rows, context_embeddings, row_user_embeddings, goal_idx, tau)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_query_batch)

    device = torch.device(args.device)
    query_model = QueryTower().to(device)
    anchor_model = AnchorEncoder().to(device)
    optimizer = torch.optim.AdamW(
        list(query_model.parameters()) + list(anchor_model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    train_anchor_features = torch.from_numpy(anchor_features[train_anchor_idx]).to(device)
    best_score = (float("inf"), float("inf"), float("inf"))
    best_state = None
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        query_model.train()
        anchor_model.train()
        epoch_loss = 0.0
        epoch_set = 0.0
        epoch_pair = 0.0
        steps = 0

        for batch in train_loader:
            optimizer.zero_grad()
            batch = {key: value.to(device) for key, value in batch.items()}
            row_idx_np = batch["row_idx"].cpu().numpy().astype(np.int64)

            query = F.normalize(
                query_model(batch["z_ctx"], batch["u_user"], batch["goal_onehot"], batch["tau"]),
                dim=-1,
            )
            anchor_emb = F.normalize(anchor_model(train_anchor_features), dim=-1)
            logits = query @ anchor_emb.T

            loss_set = multi_positive_loss(
                logits,
                pos_local[row_idx_np],
                pos_count_local[row_idx_np],
                pos_tier_local[row_idx_np],
                args.temperature,
            )
            loss_pair = hard_negative_margin_loss(
                logits,
                pos_local[row_idx_np],
                pos_count_local[row_idx_np],
                pos_tier_local[row_idx_np],
                neg_local[row_idx_np],
                neg_count_local[row_idx_np],
                args.margin,
            )
            loss = loss_set + float(args.pairwise_weight) * loss_pair
            loss.backward()
            nn.utils.clip_grad_norm_(list(query_model.parameters()) + list(anchor_model.parameters()), max_norm=5.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_set += float(loss_set.item())
            epoch_pair += float(loss_pair.item())
            steps += 1

        query_model.eval()
        anchor_model.eval()
        with torch.no_grad():
            train_anchor_embeddings = F.normalize(anchor_model(train_anchor_features), dim=-1).cpu().numpy()
            val_query_embeddings = export_query_embeddings(
                query_model,
                z_ctx=context_embeddings[val_rows],
                u_user=row_user_embeddings[val_rows],
                goal_idx=goal_idx[val_rows],
                tau=tau[val_rows],
                device=device,
                batch_size=args.batch_size,
            )
        val_summary, _ = rank_metrics(
            query_embeddings=val_query_embeddings,
            anchor_embeddings=train_anchor_embeddings,
            pos_local=pos_local[val_rows],
            pos_count=pos_count_local[val_rows],
            pos_tier=pos_tier_local[val_rows],
            row_goal_idx=goal_idx[val_rows],
            row_song_id=decision_df.loc[val_rows, "song_id"].astype(str).to_numpy(),
            anchor_song_id=anchor_df.loc[train_anchor_idx, "song_id"].astype(str).to_numpy(),
        )
        scheduler.step(val_summary["weighted_mrr"])
        score = (
            float(val_summary["conditional_mean_rank"]),
            -float(val_summary["weighted_mrr"]),
            -float(val_summary["weighted_recall_at_20"]),
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(steps, 1),
                "train_set_loss": epoch_set / max(steps, 1),
                "train_pair_loss": epoch_pair / max(steps, 1),
                "val_recall_at_20": float(val_summary["recall_at_20"]),
                "val_recall_at_50": float(val_summary["recall_at_50"]),
                "val_weighted_recall_at_20": float(val_summary["weighted_recall_at_20"]),
                "val_weighted_recall_at_50": float(val_summary["weighted_recall_at_50"]),
                "val_conditional_mean_rank": float(val_summary["conditional_mean_rank"]),
                "val_mrr": float(val_summary["mrr"]),
                "val_weighted_mrr": float(val_summary["weighted_mrr"]),
                "val_weighted_ndcg_at_10": float(val_summary["weighted_ndcg_at_10"]),
                "val_factual_positive_rate": float(val_summary["factual_positive_rate"]),
                "val_factual_positive_conditional_mean_rank": float(val_summary["factual_positive_conditional_mean_rank"]),
                "val_first_positive_is_factual_rate": float(val_summary["first_positive_is_factual_rate"]),
            }
        )
        if score < best_score:
            best_score = score
            best_state = {
                "query_model": query_model.state_dict(),
                "anchor_model": anchor_model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_summary,
            }

    if best_state is None:
        raise RuntimeError("Query tower training did not produce a checkpoint.")

    query_model.load_state_dict(best_state["query_model"])
    anchor_model.load_state_dict(best_state["anchor_model"])
    query_model.eval()
    anchor_model.eval()

    with torch.no_grad():
        train_anchor_embeddings = F.normalize(anchor_model(train_anchor_features), dim=-1).cpu().numpy()
    all_query_embeddings = export_query_embeddings(
        query_model,
        z_ctx=context_embeddings,
        u_user=row_user_embeddings,
        goal_idx=goal_idx,
        tau=tau,
        device=device,
        batch_size=args.batch_size,
    )

    val_summary, val_diag = rank_metrics(
        query_embeddings=all_query_embeddings[val_rows],
        anchor_embeddings=train_anchor_embeddings,
        pos_local=pos_local[val_rows],
        pos_count=pos_count_local[val_rows],
        pos_tier=pos_tier_local[val_rows],
        row_goal_idx=goal_idx[val_rows],
        row_song_id=decision_df.loc[val_rows, "song_id"].astype(str).to_numpy(),
        anchor_song_id=anchor_df.loc[train_anchor_idx, "song_id"].astype(str).to_numpy(),
    )
    test_summary, test_diag = rank_metrics(
        query_embeddings=all_query_embeddings[test_rows],
        anchor_embeddings=train_anchor_embeddings,
        pos_local=pos_local[test_rows],
        pos_count=pos_count_local[test_rows],
        pos_tier=pos_tier_local[test_rows],
        row_goal_idx=goal_idx[test_rows],
        row_song_id=decision_df.loc[test_rows, "song_id"].astype(str).to_numpy(),
        anchor_song_id=anchor_df.loc[train_anchor_idx, "song_id"].astype(str).to_numpy(),
    )

    anchor_ids = anchor_df.loc[train_anchor_idx, "anchor_idx"].astype(str).tolist()
    np.save(REBUILD_DIR / "query_embeddings.npy", all_query_embeddings.astype(np.float32))
    np.save(REBUILD_DIR / "anchor_query_embeddings.npy", train_anchor_embeddings.astype(np.float32))
    np.savez(
        REBUILD_DIR / "situnes_anchor_index.npz",
        song_ids=np.asarray(anchor_ids),
        norm_song_emb=normalize_rows(train_anchor_embeddings),
        anchor_global_idx=train_anchor_idx.astype(np.int32),
        anchor_decision_id=anchor_df.loc[train_anchor_idx, "decision_id"].astype(str).to_numpy(),
    )

    torch.save(
        {
            "query_model": query_model.state_dict(),
            "anchor_model": anchor_model.state_dict(),
            "config": {
                "temperature": float(args.temperature),
                "margin": float(args.margin),
                "pairwise_weight": float(args.pairwise_weight),
            },
            "best_epoch": int(best_state["epoch"]),
            "best_val_metrics": best_state["val_metrics"],
        },
        MODELS_DIR / "query_tower.pt",
    )

    metrics = {
        "best_epoch": int(best_state["epoch"]),
        "val_metrics": val_summary,
        "test_metrics": test_summary,
        "history": history,
    }
    (MODELS_DIR / "query_tower_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    val_diag.to_parquet(REBUILD_DIR / "query_anchor_diag_val.parquet", index=False)
    test_diag.to_parquet(REBUILD_DIR / "query_anchor_diag_test.parquet", index=False)

    with (MODELS_DIR / "query_tower_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
