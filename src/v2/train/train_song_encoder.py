"""
Train the V2 song encoder on the rebuilt music catalog.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v2.data.schema import SONG_EMB_DIM
from src.v2.models.song_encoder import SongEncoder


REBUILD_DIR = PROJECT_ROOT / "data" / "processed" / "rebuild"
MODELS_DIR = PROJECT_ROOT / "models" / "rebuild"
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class SongBatch:
    static: torch.Tensor
    dyn: torch.Tensor
    dyn_mask: torch.Tensor
    static_target: torch.Tensor
    quality_target: torch.Tensor
    source_id: torch.Tensor
    index: torch.Tensor


class SongDataset(Dataset):
    def __init__(
        self,
        x_static: np.ndarray,
        x_dyn: np.ndarray,
        x_dyn_mask: np.ndarray,
        static_target: np.ndarray,
        quality_target: np.ndarray,
        source_id: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.x_static = x_static
        self.x_dyn = x_dyn
        self.x_dyn_mask = x_dyn_mask
        self.static_target = static_target
        self.quality_target = quality_target
        self.source_id = source_id
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> SongBatch:
        idx = int(self.indices[item])
        return SongBatch(
            static=torch.from_numpy(self.x_static[idx]),
            dyn=torch.from_numpy(self.x_dyn[idx]),
            dyn_mask=torch.from_numpy(self.x_dyn_mask[idx]),
            static_target=torch.from_numpy(self.static_target[idx]),
            quality_target=torch.tensor([self.quality_target[idx]], dtype=torch.float32),
            source_id=torch.tensor(int(self.source_id[idx]), dtype=torch.int64),
            index=torch.tensor(idx, dtype=torch.int64),
        )


def collate_song_batch(batch: list[SongBatch]) -> dict[str, torch.Tensor]:
    return {
        "static": torch.stack([item.static for item in batch]),
        "dyn": torch.stack([item.dyn for item in batch]),
        "dyn_mask": torch.stack([item.dyn_mask for item in batch]),
        "static_target": torch.stack([item.static_target for item in batch]),
        "quality_target": torch.stack([item.quality_target for item in batch]),
        "source_id": torch.stack([item.source_id for item in batch]),
        "index": torch.stack([item.index for item in batch]),
    }


def stratified_source_split(source_series: pd.Series, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    for source, idx in source_series.groupby(source_series).groups.items():
        group_idx = np.asarray(sorted(int(i) for i in idx), dtype=np.int64)
        rng.shuffle(group_idx)
        n_val = max(1, int(round(len(group_idx) * val_frac)))
        val_idx.extend(group_idx[:n_val].tolist())
        train_idx.extend(group_idx[n_val:].tolist())
    return np.asarray(sorted(train_idx), dtype=np.int64), np.asarray(sorted(val_idx), dtype=np.int64)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sq = (pred - target).pow(2) * mask
    denom = mask.sum().clamp(min=1.0)
    return sq.sum() / denom


def source_balanced_weights(source_series: pd.Series, train_idx: np.ndarray) -> np.ndarray:
    train_sources = source_series.iloc[train_idx]
    counts = train_sources.value_counts().to_dict()
    return np.asarray([1.0 / counts[train_sources.iloc[i]] for i in range(len(train_idx))], dtype=np.float64)


def compute_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    static_mae = torch.mean(torch.abs(outputs["song_affect_hat"] - batch["static_target"])).item()
    dyn_mask = batch["dyn_mask"]
    if float(dyn_mask.sum().item()) > 0:
        dyn_mae = (torch.abs(outputs["song_dyn_hat"] - batch["dyn"]) * dyn_mask).sum() / dyn_mask.sum().clamp(min=1.0)
        dyn_mae_value = float(dyn_mae.item())
    else:
        dyn_mae_value = 0.0
    quality_mae = torch.mean(torch.abs(outputs["song_quality"] - batch["quality_target"])).item()
    return {
        "static_mae": float(static_mae),
        "dyn_mae": dyn_mae_value,
        "quality_mae": float(quality_mae),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total = {"loss": 0.0, "static_loss": 0.0, "dyn_loss": 0.0, "quality_loss": 0.0, "static_mae": 0.0, "dyn_mae": 0.0, "quality_mae": 0.0}
    steps = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["static"], batch["dyn"], batch["dyn_mask"])
            static_loss = F.mse_loss(outputs["song_affect_hat"], batch["static_target"])
            dyn_loss = masked_mse(outputs["song_dyn_hat"], batch["dyn"], batch["dyn_mask"])
            quality_loss = F.mse_loss(outputs["song_quality"], batch["quality_target"])
            loss = static_loss + 0.5 * dyn_loss + 0.1 * quality_loss
            metrics = compute_metrics(outputs, batch)

            total["loss"] += float(loss.item())
            total["static_loss"] += float(static_loss.item())
            total["dyn_loss"] += float(dyn_loss.item())
            total["quality_loss"] += float(quality_loss.item())
            for key in ["static_mae", "dyn_mae", "quality_mae"]:
                total[key] += metrics[key]
            steps += 1

    if steps == 0:
        return total
    return {key: value / steps for key, value in total.items()}


def export_embeddings(
    model: nn.Module,
    x_static: np.ndarray,
    x_dyn: np.ndarray,
    x_dyn_mask: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    embeddings = []
    static_pred = []
    quality_pred = []
    with torch.no_grad():
        for start in range(0, len(x_static), batch_size):
            end = min(start + batch_size, len(x_static))
            static_batch = torch.from_numpy(x_static[start:end]).to(device)
            dyn_batch = torch.from_numpy(x_dyn[start:end]).to(device)
            mask_batch = torch.from_numpy(x_dyn_mask[start:end]).to(device)
            outputs = model(static_batch, dyn_batch, mask_batch)
            embeddings.append(outputs["embedding"].cpu().numpy())
            static_pred.append(outputs["song_affect_hat"].cpu().numpy())
            quality_pred.append(outputs["song_quality"].cpu().numpy())
    return (
        np.concatenate(embeddings, axis=0),
        np.concatenate(static_pred, axis=0),
        np.concatenate(quality_pred, axis=0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the V2 song encoder.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    x_static = np.load(REBUILD_DIR / "song_static.npy").astype(np.float32)
    x_dyn = np.load(REBUILD_DIR / "song_dynamic.npy").astype(np.float32)
    x_dyn_mask = np.load(REBUILD_DIR / "song_dynamic_mask.npy").astype(np.float32)
    catalog = pd.read_parquet(REBUILD_DIR / "song_catalog.parquet")

    static_target = catalog[["valence_static", "arousal_static"]].to_numpy(dtype=np.float32)
    quality_target = catalog["song_quality"].to_numpy(dtype=np.float32)
    source_map = {source: idx for idx, source in enumerate(sorted(catalog["source"].unique().tolist()))}
    source_id = catalog["source"].map(source_map).to_numpy(dtype=np.int64)

    train_idx, val_idx = stratified_source_split(catalog["source"], args.val_frac, args.seed)
    train_dataset = SongDataset(x_static, x_dyn, x_dyn_mask, static_target, quality_target, source_id, train_idx)
    val_dataset = SongDataset(x_static, x_dyn, x_dyn_mask, static_target, quality_target, source_id, val_idx)

    weights = source_balanced_weights(catalog["source"], train_idx)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_idx), replacement=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_song_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_song_batch,
    )

    device = torch.device(args.device)
    model = SongEncoder(emb_dim=SONG_EMB_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = math.inf
    best_state = None
    history: list[dict[str, float]] = []

    max_train_steps = args.max_train_steps if args.max_train_steps > 0 else None
    max_eval_batches = args.max_eval_batches if args.max_eval_batches > 0 else None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"loss": 0.0, "static_loss": 0.0, "dyn_loss": 0.0, "quality_loss": 0.0, "static_mae": 0.0, "dyn_mae": 0.0, "quality_mae": 0.0}
        steps = 0

        for step_idx, batch in enumerate(train_loader, start=1):
            if max_train_steps is not None and step_idx > max_train_steps:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["static"], batch["dyn"], batch["dyn_mask"])
            static_loss = F.mse_loss(outputs["song_affect_hat"], batch["static_target"])
            dyn_loss = masked_mse(outputs["song_dyn_hat"], batch["dyn"], batch["dyn_mask"])
            quality_loss = F.mse_loss(outputs["song_quality"], batch["quality_target"])
            loss = static_loss + 0.5 * dyn_loss + 0.1 * quality_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            metrics = compute_metrics(outputs, batch)
            running["loss"] += float(loss.item())
            running["static_loss"] += float(static_loss.item())
            running["dyn_loss"] += float(dyn_loss.item())
            running["quality_loss"] += float(quality_loss.item())
            for key in ["static_mae", "dyn_mae", "quality_mae"]:
                running[key] += metrics[key]
            steps += 1

        train_metrics = {key: value / max(steps, 1) for key, value in running.items()}
        val_metrics = evaluate(model, val_loader, device, max_batches=max_eval_batches)
        scheduler.step(val_metrics["loss"])

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "config": vars(args),
                "source_map": source_map,
            }

    if best_state is None:
        raise RuntimeError("Song encoder training produced no checkpoint.")

    model.load_state_dict(best_state["model"])
    checkpoint_path = MODELS_DIR / "song_encoder.pt"
    torch.save(best_state, checkpoint_path)

    embeddings, static_pred, quality_pred = export_embeddings(
        model, x_static, x_dyn, x_dyn_mask, batch_size=args.batch_size, device=device
    )
    np.save(REBUILD_DIR / "song_embeddings.npy", embeddings.astype(np.float32))

    pred_df = catalog[["song_id", "source", "title", "artist"]].copy()
    pred_df["pred_valence"] = static_pred[:, 0].astype(np.float32)
    pred_df["pred_arousal"] = static_pred[:, 1].astype(np.float32)
    pred_df["pred_quality"] = quality_pred[:, 0].astype(np.float32)
    pred_df.to_parquet(REBUILD_DIR / "song_encoder_predictions.parquet", index=False)

    log_path = MODELS_DIR / "song_encoder_train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "best_epoch": int(best_state["epoch"]),
        "best_val_loss": float(best_val_loss),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "embedding_dim": SONG_EMB_DIM,
        "sources": catalog["source"].value_counts().sort_index().to_dict(),
        "source_map": source_map,
        "best_val_metrics": best_state["val_metrics"],
    }
    (MODELS_DIR / "song_encoder_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
