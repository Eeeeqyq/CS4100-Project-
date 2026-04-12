"""
Train the V2 user encoder on Stage 1 leave-one-out preference prediction.
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

from src.v2.models.user_encoder import UserPreferenceModel


REBUILD_DIR = PROJECT_ROOT / "data" / "processed" / "rebuild"
MODELS_DIR = PROJECT_ROOT / "models" / "rebuild"
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class UserSample:
    hist_song_emb: torch.Tensor
    hist_rating: torch.Tensor
    hist_mask: torch.Tensor
    candidate_song_emb: torch.Tensor
    candidate_song_affect: torch.Tensor
    target_rating: torch.Tensor
    taste_target: torch.Tensor
    conf_target: torch.Tensor
    user_id: torch.Tensor
    split_id: torch.Tensor


class LeaveOneOutStage1Dataset(Dataset):
    def __init__(
        self,
        hist_song_idx: np.ndarray,
        hist_rating: np.ndarray,
        hist_mask: np.ndarray,
        user_ids: np.ndarray,
        split_by_user: dict[int, str],
        song_embeddings: np.ndarray,
        song_affect: np.ndarray,
        split_name: str,
    ) -> None:
        self.samples = []
        for user_row, user_id in enumerate(user_ids.tolist()):
            if split_by_user.get(int(user_id)) != split_name:
                continue
            valid_positions = np.where(hist_mask[user_row] > 0.5)[0]
            if len(valid_positions) < 2:
                continue
            for holdout_pos in valid_positions:
                keep_mask = hist_mask[user_row].copy()
                keep_mask[holdout_pos] = 0.0

                context_indices = hist_song_idx[user_row].copy()
                context_ratings = hist_rating[user_row].copy()
                context_indices[holdout_pos] = -1
                context_ratings[holdout_pos] = 0.0

                valid_context = keep_mask > 0.5
                context_song_idx = context_indices[valid_context].astype(np.int64)
                context_song_emb = song_embeddings[context_indices.clip(min=0)].copy()
                context_song_emb[~valid_context] = 0.0

                candidate_idx = int(hist_song_idx[user_row, holdout_pos])
                candidate_emb = song_embeddings[candidate_idx]
                candidate_affect = song_affect[candidate_idx]
                target_rating = float(hist_rating[user_row, holdout_pos])

                context_affect = song_affect[context_song_idx]
                context_rating = context_ratings[valid_context]
                positive_weights = np.clip(context_rating + 1.0, 0.0, None) + 1e-3
                if positive_weights.sum() <= 1e-6:
                    positive_weights = np.ones_like(context_rating, dtype=np.float32)
                taste_target = np.average(context_affect, axis=0, weights=positive_weights).astype(np.float32)
                conf_target = np.asarray([float(np.mean(np.abs(context_rating)))], dtype=np.float32)

                self.samples.append(
                    {
                        "hist_song_emb": context_song_emb.astype(np.float32),
                        "hist_rating": context_ratings[:, None].astype(np.float32),
                        "hist_mask": keep_mask[:, None].astype(np.float32),
                        "candidate_song_emb": candidate_emb.astype(np.float32),
                        "candidate_song_affect": candidate_affect.astype(np.float32),
                        "target_rating": np.asarray([target_rating], dtype=np.float32),
                        "taste_target": taste_target.astype(np.float32),
                        "conf_target": conf_target,
                        "user_id": int(user_id),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> UserSample:
        sample = self.samples[idx]
        return UserSample(
            hist_song_emb=torch.from_numpy(sample["hist_song_emb"]),
            hist_rating=torch.from_numpy(sample["hist_rating"]),
            hist_mask=torch.from_numpy(sample["hist_mask"]),
            candidate_song_emb=torch.from_numpy(sample["candidate_song_emb"]),
            candidate_song_affect=torch.from_numpy(sample["candidate_song_affect"]),
            target_rating=torch.from_numpy(sample["target_rating"]),
            taste_target=torch.from_numpy(sample["taste_target"]),
            conf_target=torch.from_numpy(sample["conf_target"]),
            user_id=torch.tensor(sample["user_id"], dtype=torch.int64),
            split_id=torch.tensor(0, dtype=torch.int64),
        )


def collate_user_batch(batch: list[UserSample]) -> dict[str, torch.Tensor]:
    return {
        "hist_song_emb": torch.stack([item.hist_song_emb for item in batch]),
        "hist_rating": torch.stack([item.hist_rating for item in batch]),
        "hist_mask": torch.stack([item.hist_mask for item in batch]),
        "candidate_song_emb": torch.stack([item.candidate_song_emb for item in batch]),
        "candidate_song_affect": torch.stack([item.candidate_song_affect for item in batch]),
        "target_rating": torch.stack([item.target_rating for item in batch]),
        "taste_target": torch.stack([item.taste_target for item in batch]),
        "conf_target": torch.stack([item.conf_target for item in batch]),
        "user_id": torch.stack([item.user_id for item in batch]),
    }


def compute_metrics(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, float]:
    pred = outputs["pred_rating"]
    target = batch["target_rating"]
    rating_mae = torch.mean(torch.abs(pred - target)).item()
    sign_acc = torch.mean(((pred >= 0.0) == (target >= 0.0)).float()).item()
    taste_mae = torch.mean(torch.abs(outputs["taste_affect"] - batch["taste_target"])).item()
    conf_mae = torch.mean(torch.abs(outputs["user_conf"] - batch["conf_target"])).item()
    return {
        "rating_mae": float(rating_mae),
        "sign_acc": float(sign_acc),
        "taste_mae": float(taste_mae),
        "conf_mae": float(conf_mae),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "rating_loss": 0.0, "taste_loss": 0.0, "conf_loss": 0.0, "rating_mae": 0.0, "sign_acc": 0.0, "taste_mae": 0.0, "conf_mae": 0.0}
    steps = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                batch["hist_song_emb"],
                batch["hist_rating"],
                batch["hist_mask"],
                batch["candidate_song_emb"],
                batch["candidate_song_affect"],
            )
            rating_loss = F.smooth_l1_loss(outputs["pred_rating"], batch["target_rating"])
            taste_loss = F.mse_loss(outputs["taste_affect"], batch["taste_target"])
            conf_loss = F.mse_loss(outputs["user_conf"], batch["conf_target"])
            loss = rating_loss + 0.5 * taste_loss + 0.1 * conf_loss
            metrics = compute_metrics(outputs, batch)

            totals["loss"] += float(loss.item())
            totals["rating_loss"] += float(rating_loss.item())
            totals["taste_loss"] += float(taste_loss.item())
            totals["conf_loss"] += float(conf_loss.item())
            for key in ["rating_mae", "sign_acc", "taste_mae", "conf_mae"]:
                totals[key] += metrics[key]
            steps += 1

    if steps == 0:
        raise RuntimeError("Evaluation loader produced zero batches.")
    return {key: value / steps for key, value in totals.items()}


def export_user_embeddings(
    model: UserPreferenceModel,
    hist_song_idx: np.ndarray,
    hist_rating: np.ndarray,
    hist_mask: np.ndarray,
    user_ids: np.ndarray,
    song_embeddings: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    all_user_emb = []
    all_taste = []
    all_conf = []
    with torch.no_grad():
        for row in range(len(user_ids)):
            song_idx = hist_song_idx[row].clip(min=0)
            song_emb = torch.from_numpy(song_embeddings[song_idx][None, ...]).to(device)
            rating = torch.from_numpy(hist_rating[row][:, None][None, ...]).to(device)
            mask = torch.from_numpy(hist_mask[row][:, None][None, ...]).to(device)
            outputs = model.encoder(song_emb, rating, mask)
            all_user_emb.append(outputs["u_user"].cpu().numpy())
            all_taste.append(outputs["taste_affect"].cpu().numpy())
            all_conf.append(outputs["user_conf"].cpu().numpy())
    return {
        "user_ids": user_ids.astype(np.int32),
        "user_embeddings": np.concatenate(all_user_emb, axis=0).astype(np.float32),
        "taste_affect": np.concatenate(all_taste, axis=0).astype(np.float32),
        "user_conf": np.concatenate(all_conf, axis=0).astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the V2 user encoder.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    hist = np.load(REBUILD_DIR / "stage1_histories.npz")
    user_ids = hist["user_ids"].astype(np.int32)
    hist_song_idx = hist["hist_song_idx"].astype(np.int64)
    hist_rating = hist["hist_rating"].astype(np.float32)
    hist_mask = hist["hist_mask"].astype(np.float32)
    song_embeddings = np.load(REBUILD_DIR / "song_embeddings.npy").astype(np.float32)
    song_catalog = pd.read_parquet(REBUILD_DIR / "song_catalog.parquet")
    song_affect = song_catalog[["valence_static", "arousal_static"]].to_numpy(dtype=np.float32)
    split_manifest = json.loads((PROJECT_ROOT / "data" / "processed" / "split_manifest.json").read_text(encoding="utf-8"))
    split_by_user = {
        int(user_id): split_name
        for split_name, payload in split_manifest.items()
        for user_id in payload["users"]
    }

    train_dataset = LeaveOneOutStage1Dataset(
        hist_song_idx, hist_rating, hist_mask, user_ids, split_by_user, song_embeddings, song_affect, "train"
    )
    val_dataset = LeaveOneOutStage1Dataset(
        hist_song_idx, hist_rating, hist_mask, user_ids, split_by_user, song_embeddings, song_affect, "val"
    )
    test_dataset = LeaveOneOutStage1Dataset(
        hist_song_idx, hist_rating, hist_mask, user_ids, split_by_user, song_embeddings, song_affect, "test"
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_user_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_user_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_user_batch)

    device = torch.device(args.device)
    model = UserPreferenceModel(song_emb_dim=song_embeddings.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_state = None
    history = []
    max_train_steps = args.max_train_steps if args.max_train_steps > 0 else None
    max_eval_batches = args.max_eval_batches if args.max_eval_batches > 0 else None

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "rating_loss": 0.0, "taste_loss": 0.0, "conf_loss": 0.0, "rating_mae": 0.0, "sign_acc": 0.0, "taste_mae": 0.0, "conf_mae": 0.0}
        steps = 0

        for step_idx, batch in enumerate(train_loader, start=1):
            if max_train_steps is not None and step_idx > max_train_steps:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                batch["hist_song_emb"],
                batch["hist_rating"],
                batch["hist_mask"],
                batch["candidate_song_emb"],
                batch["candidate_song_affect"],
            )
            rating_loss = F.smooth_l1_loss(outputs["pred_rating"], batch["target_rating"])
            taste_loss = F.mse_loss(outputs["taste_affect"], batch["taste_target"])
            conf_loss = F.mse_loss(outputs["user_conf"], batch["conf_target"])
            loss = rating_loss + 0.5 * taste_loss + 0.1 * conf_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            metrics = compute_metrics(outputs, batch)
            totals["loss"] += float(loss.item())
            totals["rating_loss"] += float(rating_loss.item())
            totals["taste_loss"] += float(taste_loss.item())
            totals["conf_loss"] += float(conf_loss.item())
            for key in ["rating_mae", "sign_acc", "taste_mae", "conf_mae"]:
                totals[key] += metrics[key]
            steps += 1

        train_metrics = {key: value / max(steps, 1) for key, value in totals.items()}
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
            }

    if best_state is None:
        raise RuntimeError("User encoder training produced no checkpoint.")

    model.load_state_dict(best_state["model"])
    checkpoint_path = MODELS_DIR / "user_encoder.pt"
    torch.save(best_state, checkpoint_path)

    test_metrics = evaluate(model, test_loader, device, max_batches=max_eval_batches)
    exports = export_user_embeddings(model, hist_song_idx, hist_rating, hist_mask, user_ids, song_embeddings, device)
    np.savez(REBUILD_DIR / "user_encoder_outputs.npz", **exports)

    log_path = MODELS_DIR / "user_encoder_train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "best_epoch": int(best_state["epoch"]),
        "best_val_loss": float(best_val_loss),
        "train_samples": int(len(train_dataset)),
        "val_samples": int(len(val_dataset)),
        "test_samples": int(len(test_dataset)),
        "best_val_metrics": best_state["val_metrics"],
        "test_metrics": test_metrics,
    }
    (MODELS_DIR / "user_encoder_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
