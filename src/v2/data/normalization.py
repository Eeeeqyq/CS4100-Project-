"""
Shared normalization helpers for the V2 rebuild.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def fit_scalar_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr, ddof=0))
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return {"mean": mean, "std": std}


def apply_zscore(values: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return ((arr - float(stats["mean"])) / max(float(stats["std"]), 1e-8)).astype(np.float32)


def fit_train_only_env_stats(decision_df: pd.DataFrame) -> dict:
    train = decision_df[decision_df["split"] == "train"].copy()
    return {
        "temp": fit_scalar_stats(train["temp_raw"].to_numpy(dtype=np.float64)),
        "humidity": fit_scalar_stats(train["humidity_raw"].to_numpy(dtype=np.float64)),
    }


def fit_song_stats(song_df: pd.DataFrame) -> dict:
    return {
        "tempo": fit_scalar_stats(song_df["tempo_raw"].to_numpy(dtype=np.float64)),
        "loudness": fit_scalar_stats(song_df["loudness_raw"].to_numpy(dtype=np.float64)),
        "popularity": fit_scalar_stats(song_df["popularity_raw"].to_numpy(dtype=np.float64)),
    }


def save_stats(stats: dict, path: Path) -> None:
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_stats(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
