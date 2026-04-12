"""
Final anchor and public-transfer scoring helpers for V2.2.
"""

from __future__ import annotations

import numpy as np

from src.v2.data.schema import GOAL_NAMES


def row_normalize(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((arr - mean) / std).astype(np.float32)


def build_anchor_final_score(
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


def pmemo_dynamic_bonus(
    goal_idx: int,
    dyn_valence_delta: np.ndarray,
    dyn_arousal_delta: np.ndarray,
    dyn_arousal_vol: np.ndarray,
    dyn_arousal_peak: np.ndarray,
    dyn_quality: np.ndarray,
) -> np.ndarray:
    q = np.clip(np.asarray(dyn_quality, dtype=np.float32), 0.0, 1.0)
    valence_rise = np.clip(np.asarray(dyn_valence_delta, dtype=np.float32) / 0.10, 0.0, 1.0)
    arousal_rise = np.clip(np.asarray(dyn_arousal_delta, dtype=np.float32) / 0.10, 0.0, 1.0)
    arousal_fall = np.clip(-np.asarray(dyn_arousal_delta, dtype=np.float32) / 0.10, 0.0, 1.0)
    arousal_peak_norm = np.clip((np.asarray(dyn_arousal_peak, dtype=np.float32) - 0.20) / 0.80, 0.0, 1.0)
    arousal_stability = 1.0 - np.clip(np.asarray(dyn_arousal_vol, dtype=np.float32) / 0.05, 0.0, 1.0)

    goal_name = GOAL_NAMES.get(int(goal_idx), "focus")
    if goal_name == "focus":
        score = q * (0.10 * arousal_stability + 0.06 * (1.0 - arousal_peak_norm) + 0.04 * arousal_fall)
    elif goal_name == "wind_down":
        score = q * (0.12 * arousal_fall + 0.06 * (1.0 - arousal_peak_norm) + 0.06 * arousal_stability)
    elif goal_name == "movement":
        score = q * (0.10 * arousal_rise + 0.08 * arousal_peak_norm + 0.04 * valence_rise)
    else:
        score = q * (0.10 * valence_rise + 0.05 * arousal_rise + 0.03 * arousal_stability)
    return np.clip(score, 0.0, 0.18).astype(np.float32)


def pmemo_dynamic_reason(
    goal_idx: int,
    dyn_valence_delta: float,
    dyn_arousal_delta: float,
    dyn_arousal_vol: float,
    dyn_arousal_peak: float,
    dyn_quality: float,
) -> str:
    bonus = float(
        pmemo_dynamic_bonus(
            goal_idx=goal_idx,
            dyn_valence_delta=np.asarray([dyn_valence_delta], dtype=np.float32),
            dyn_arousal_delta=np.asarray([dyn_arousal_delta], dtype=np.float32),
            dyn_arousal_vol=np.asarray([dyn_arousal_vol], dtype=np.float32),
            dyn_arousal_peak=np.asarray([dyn_arousal_peak], dtype=np.float32),
            dyn_quality=np.asarray([dyn_quality], dtype=np.float32),
        )[0]
    )
    if bonus <= 0.0:
        return ""
    goal_name = GOAL_NAMES.get(int(goal_idx), "focus")
    if goal_name in {"focus", "wind_down"} and dyn_arousal_delta < -0.03:
        return "gentle calming trajectory"
    if goal_name in {"focus", "wind_down"} and dyn_arousal_vol <= 0.04 and dyn_arousal_peak <= 0.45:
        return "stable low-volatility contour"
    if goal_name in {"movement", "uplift"} and dyn_arousal_delta > 0.03:
        return "rising arousal trajectory"
    if dyn_arousal_vol <= 0.04:
        return "stable low-volatility contour"
    return ""
