"""
Train the V2 context encoder on rebuilt SiTunes decision tensors.
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

from src.v2.models.context_encoder import ContextEncoder


REBUILD_DIR = PROJECT_ROOT / "data" / "processed" / "rebuild"
MODELS_DIR = PROJECT_ROOT / "models" / "rebuild"
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def derive_movement_label(x_wrist: np.ndarray, x_env: np.ndarray) -> np.ndarray:
    step_frac = x_wrist[:, :, 3].mean(axis=1)
    walk_frac = x_wrist[:, :, 6].mean(axis=1)
    run_frac = x_wrist[:, :, 8].mean(axis=1)
    speed_norm = x_env[:, 7]
    score = 0.45 * step_frac + 0.25 * walk_frac + 0.15 * run_frac + 0.15 * speed_norm
    label = np.zeros(len(score), dtype=np.int64)
    label[score >= 0.15] = 1
    label[score >= 0.40] = 2
    return label


def proxy_uncertainty_target(x_self: np.ndarray) -> np.ndarray:
    mask = x_self[:, 2]
    return (1.0 - mask).astype(np.float32)[:, None]


@dataclass
class ContextSample:
    wrist: torch.Tensor
    env: torch.Tensor
    self_vec: torch.Tensor
    pre_affect: torch.Tensor
    movement_label: torch.Tensor
    unc_target: torch.Tensor


class ContextDataset(Dataset):
    def __init__(
        self,
        x_wrist: np.ndarray,
        x_env: np.ndarray,
        x_self: np.ndarray,
        pre_affect: np.ndarray,
        movement_label: np.ndarray,
        unc_target: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.x_wrist = x_wrist
        self.x_env = x_env
        self.x_self = x_self
        self.pre_affect = pre_affect
        self.movement_label = movement_label
        self.unc_target = unc_target
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> ContextSample:
        idx = int(self.indices[item])
        return ContextSample(
            wrist=torch.from_numpy(self.x_wrist[idx]),
            env=torch.from_numpy(self.x_env[idx]),
            self_vec=torch.from_numpy(self.x_self[idx]),
            pre_affect=torch.from_numpy(self.pre_affect[idx]),
            movement_label=torch.tensor(int(self.movement_label[idx]), dtype=torch.int64),
            unc_target=torch.from_numpy(self.unc_target[idx]),
        )


def collate_context_batch(batch: list[ContextSample]) -> dict[str, torch.Tensor]:
    return {
        "wrist": torch.stack([item.wrist for item in batch]),
        "env": torch.stack([item.env for item in batch]),
        "self_vec": torch.stack([item.self_vec for item in batch]),
        "pre_affect": torch.stack([item.pre_affect for item in batch]),
        "movement_label": torch.stack([item.movement_label for item in batch]),
        "unc_target": torch.stack([item.unc_target for item in batch]),
    }


def compute_metrics(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, float]:
    affect_mae = torch.mean(torch.abs(outputs["pre_affect_hat"] - batch["pre_affect"])).item()
    move_acc = torch.mean((outputs["movement_logits"].argmax(dim=1) == batch["movement_label"]).float()).item()
    unc_mae = torch.mean(torch.abs(outputs["ctx_unc"] - batch["unc_target"])).item()
    return {
        "affect_mae": float(affect_mae),
        "movement_acc": float(move_acc),
        "unc_mae": float(unc_mae),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "affect_loss": 0.0, "movement_loss": 0.0, "unc_loss": 0.0, "affect_mae": 0.0, "movement_acc": 0.0, "unc_mae": 0.0}
    steps = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["wrist"], batch["env"], batch["self_vec"])
            affect_loss = F.mse_loss(outputs["pre_affect_hat"], batch["pre_affect"])
            movement_loss = F.cross_entropy(outputs["movement_logits"], batch["movement_label"])
            unc_loss = F.mse_loss(outputs["ctx_unc"], batch["unc_target"])
            loss = affect_loss + 0.5 * movement_loss + 0.1 * unc_loss
            metrics = compute_metrics(outputs, batch)

            totals["loss"] += float(loss.item())
            totals["affect_loss"] += float(affect_loss.item())
            totals["movement_loss"] += float(movement_loss.item())
            totals["unc_loss"] += float(unc_loss.item())
            for key in ["affect_mae", "movement_acc", "unc_mae"]:
                totals[key] += metrics[key]
            steps += 1

    if steps == 0:
        raise RuntimeError("Evaluation loader produced zero batches.")
    return {key: value / steps for key, value in totals.items()}


def export_context_embeddings(
    model: ContextEncoder,
    x_wrist: np.ndarray,
    x_env: np.ndarray,
    x_self: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    embeddings = []
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x_wrist), batch_size):
            end = min(start + batch_size, len(x_wrist))
            wrist = torch.from_numpy(x_wrist[start:end]).to(device)
            env = torch.from_numpy(x_env[start:end]).to(device)
            self_vec = torch.from_numpy(x_self[start:end]).to(device)
            outputs = model(wrist, env, self_vec)
            embeddings.append(outputs["z_ctx"].cpu().numpy())
            pred = torch.cat(
                [
                    outputs["pre_affect_hat"],
                    torch.softmax(outputs["movement_logits"], dim=1),
                    outputs["ctx_unc"],
                ],
                dim=1,
            )
            predictions.append(pred.cpu().numpy())
    return np.concatenate(embeddings, axis=0), np.concatenate(predictions, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the V2 context encoder.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    decision_df = pd.read_parquet(REBUILD_DIR / "decision_table.parquet")
    x_wrist = np.load(REBUILD_DIR / "wrist_windows.npy").astype(np.float32)
    x_env = np.load(REBUILD_DIR / "env_features.npy").astype(np.float32)
    x_self = np.load(REBUILD_DIR / "self_report.npy").astype(np.float32)

    pre_affect = decision_df[["pre_valence", "pre_arousal"]].to_numpy(dtype=np.float32)
    movement_label = derive_movement_label(x_wrist, x_env)
    unc_target = proxy_uncertainty_target(x_self)

    train_idx = decision_df.index[decision_df["split"] == "train"].to_numpy(dtype=np.int64)
    val_idx = decision_df.index[decision_df["split"] == "val"].to_numpy(dtype=np.int64)
    test_idx = decision_df.index[decision_df["split"] == "test"].to_numpy(dtype=np.int64)

    train_dataset = ContextDataset(x_wrist, x_env, x_self, pre_affect, movement_label, unc_target, train_idx)
    val_dataset = ContextDataset(x_wrist, x_env, x_self, pre_affect, movement_label, unc_target, val_idx)
    test_dataset = ContextDataset(x_wrist, x_env, x_self, pre_affect, movement_label, unc_target, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_context_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_context_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_context_batch)

    device = torch.device(args.device)
    model = ContextEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_state = None
    history = []
    max_train_steps = args.max_train_steps if args.max_train_steps > 0 else None
    max_eval_batches = args.max_eval_batches if args.max_eval_batches > 0 else None

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "affect_loss": 0.0, "movement_loss": 0.0, "unc_loss": 0.0, "affect_mae": 0.0, "movement_acc": 0.0, "unc_mae": 0.0}
        steps = 0

        for step_idx, batch in enumerate(train_loader, start=1):
            if max_train_steps is not None and step_idx > max_train_steps:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["wrist"], batch["env"], batch["self_vec"])
            affect_loss = F.mse_loss(outputs["pre_affect_hat"], batch["pre_affect"])
            movement_loss = F.cross_entropy(outputs["movement_logits"], batch["movement_label"])
            unc_loss = F.mse_loss(outputs["ctx_unc"], batch["unc_target"])
            loss = affect_loss + 0.5 * movement_loss + 0.1 * unc_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            metrics = compute_metrics(outputs, batch)
            totals["loss"] += float(loss.item())
            totals["affect_loss"] += float(affect_loss.item())
            totals["movement_loss"] += float(movement_loss.item())
            totals["unc_loss"] += float(unc_loss.item())
            for key in ["affect_mae", "movement_acc", "unc_mae"]:
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
        raise RuntimeError("Context encoder training produced no checkpoint.")

    model.load_state_dict(best_state["model"])
    torch.save(best_state, MODELS_DIR / "context_encoder.pt")
    test_metrics = evaluate(model, test_loader, device, max_batches=max_eval_batches)

    ctx_embeddings, predictions = export_context_embeddings(
        model, x_wrist, x_env, x_self, batch_size=args.batch_size, device=device
    )
    np.save(REBUILD_DIR / "context_embeddings.npy", ctx_embeddings.astype(np.float32))
    pred_df = decision_df[["decision_id", "split"]].copy()
    pred_df["pred_pre_valence"] = predictions[:, 0].astype(np.float32)
    pred_df["pred_pre_arousal"] = predictions[:, 1].astype(np.float32)
    pred_df["pred_move_rest"] = predictions[:, 2].astype(np.float32)
    pred_df["pred_move_mixed"] = predictions[:, 3].astype(np.float32)
    pred_df["pred_move_active"] = predictions[:, 4].astype(np.float32)
    pred_df["pred_uncertainty"] = predictions[:, 5].astype(np.float32)
    pred_df.to_parquet(REBUILD_DIR / "context_encoder_predictions.parquet", index=False)

    with (MODELS_DIR / "context_encoder_train_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "best_epoch": int(best_state["epoch"]),
        "best_val_loss": float(best_val_loss),
        "train_rows": int(len(train_dataset)),
        "val_rows": int(len(val_dataset)),
        "test_rows": int(len(test_dataset)),
        "best_val_metrics": best_state["val_metrics"],
        "test_metrics": test_metrics,
    }
    (MODELS_DIR / "context_encoder_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
