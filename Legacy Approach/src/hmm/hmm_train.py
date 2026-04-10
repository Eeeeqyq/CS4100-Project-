"""
Train the wrist-only 3-state HMM on the train split and calibrate belief fusion
on the validation split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.common import MODELS_DIR, PROCESSED_DIR, STATE_NAMES, hr_bucket
from src.hmm.hmm_inference import corrected_belief, physical_target_state
from src.hmm.hmm_model import HMM


N_STATES = 3
N_OBS = 60
N_ITER = 40
TOL = 1e-3
N_RESTARTS = 5
SEED = 42


def load_data() -> tuple[pd.DataFrame, np.ndarray, dict]:
    df = pd.read_csv(PROCESSED_DIR / "interactions_clean.csv")
    wrist_obs = np.load(PROCESSED_DIR / "wrist_obs_all.npy")
    with (PROCESSED_DIR / "split_manifest.json").open("r", encoding="utf-8") as handle:
        split_manifest = json.load(handle)
    return df, wrist_obs, split_manifest


def diagonal_transition_init(diagonal: float = 0.90) -> np.ndarray:
    off_diag = (1.0 - diagonal) / (N_STATES - 1)
    a = np.full((N_STATES, N_STATES), off_diag, dtype=np.float64)
    np.fill_diagonal(a, diagonal)
    return a


def informed_emission_init(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    b = np.full((N_STATES, N_OBS), 0.02, dtype=np.float64)

    for obs in range(N_OBS):
        hr_band = obs // 20
        intensity = (obs % 20) // 5
        activity = obs % 5

        low = 0.20
        moderate = 0.20
        high = 0.20

        if activity in {0, 3}:
            low += 2.0
        if activity in {1, 2}:
            moderate += 1.6
        if activity == 4:
            high += 2.6

        if intensity == 0:
            low += 1.5
        elif intensity == 1:
            low += 0.4
            moderate += 0.8
        elif intensity == 2:
            moderate += 1.1
            high += 0.6
        else:
            high += 2.0

        if hr_band == 0:
            low += 0.8
        elif hr_band == 1:
            moderate += 0.3
        else:
            moderate += 1.0
            high += 0.5

        weights = np.asarray([low, moderate, high], dtype=np.float64)
        weights *= rng.lognormal(mean=0.0, sigma=0.10, size=N_STATES)
        b[:, obs] = weights

    b /= b.sum(axis=1, keepdims=True)
    return b


def initialize_model(seed: int) -> HMM:
    model = HMM(n_states=N_STATES, n_obs=N_OBS, seed=seed)
    model.set_params(
        diagonal_transition_init(diagonal=0.90),
        informed_emission_init(seed=seed),
        np.asarray([0.40, 0.30, 0.30], dtype=np.float64),
    )
    return model


def reorder_states(model: HMM) -> tuple[HMM, list[int]]:
    activity_energy = {0: 0.0, 1: 0.8, 2: 1.2, 3: 0.0, 4: 2.0}
    hr_energy = {0: -0.4, 1: 0.2, 2: 1.0}
    obs_scores = []
    for obs in range(model.n_obs):
        hr_band = obs // 20
        intensity = (obs % 20) // 5
        activity = obs % 5
        obs_scores.append(float(intensity) + activity_energy[activity] + hr_energy[hr_band])
    obs_scores = np.asarray(obs_scores, dtype=np.float64)

    state_scores = model.B @ obs_scores
    order = np.argsort(state_scores)
    model.set_params(
        model.A[np.ix_(order, order)],
        model.B[order],
        model.pi[order],
    )
    return model, order.astype(int).tolist()


def evaluate_physical_alignment(
    model: HMM,
    df: pd.DataFrame,
    sequences: np.ndarray,
    temperature: float,
    prior_strength: float,
) -> dict:
    beliefs = []
    targets = []
    for idx, row in df.iterrows():
        belief = corrected_belief(
            model,
            sequences[idx],
            int(row["activity_majority"]),
            temperature=temperature,
            prior_strength=prior_strength,
        )
        beliefs.append(belief)
        targets.append(
            physical_target_state(
                int(row["activity_majority"]),
                int(row["intensity_bucket_mean"]),
                hr_bucket(float(row["hr_mean"])),
            )
        )

    beliefs = np.asarray(beliefs, dtype=np.float32)
    preds = beliefs.argmax(axis=1)
    targets = np.asarray(targets, dtype=np.int32)
    entropy = -np.sum(beliefs * np.log(beliefs + 1e-12), axis=1)

    return {
        "accuracy": float((preds == targets).mean()),
        "mean_entropy": float(entropy.mean()),
        "state_usage": {int(k): int(v) for k, v in zip(*np.unique(preds, return_counts=True))},
    }


def calibrate_belief(model: HMM, val_df: pd.DataFrame, val_sequences: np.ndarray) -> dict:
    best = None
    for temperature in [1.0, 1.1, 1.25, 1.4, 1.6, 1.8]:
        for prior_strength in [0.50, 0.75, 1.0, 1.25, 1.5]:
            metrics = evaluate_physical_alignment(
                model,
                val_df,
                val_sequences,
                temperature=temperature,
                prior_strength=prior_strength,
            )
            used_states = len(metrics["state_usage"])
            score = metrics["accuracy"] + 0.15 * min(metrics["mean_entropy"], 0.65) + 0.03 * used_states
            candidate = {
                "temperature": temperature,
                "prior_strength": prior_strength,
                "score": round(float(score), 6),
                **metrics,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
    return best


def diagnostic_summary(
    model: HMM,
    df: pd.DataFrame,
    sequences: np.ndarray,
    calibration: dict,
) -> dict:
    beliefs = []
    decoded = []
    for idx, row in df.iterrows():
        beliefs.append(
            corrected_belief(
                model,
                sequences[idx],
                int(row["activity_majority"]),
                temperature=calibration["temperature"],
                prior_strength=calibration["prior_strength"],
            )
        )
        decoded.append(int(model.viterbi(sequences[idx])[0][-1]))

    beliefs = np.asarray(beliefs, dtype=np.float32)
    decoded = np.asarray(decoded, dtype=np.int32)
    corrected = beliefs.argmax(axis=1)
    entropy = -np.sum(beliefs * np.log(beliefs + 1e-12), axis=1)

    summary = {
        "decoded_usage": {int(k): int(v) for k, v in zip(*np.unique(decoded, return_counts=True))},
        "corrected_usage": {int(k): int(v) for k, v in zip(*np.unique(corrected, return_counts=True))},
        "belief_entropy_mean": float(entropy.mean()),
        "belief_unique_rows_rounded": int(np.unique(np.round(beliefs, 4), axis=0).shape[0]),
        "spearman_pre_valence": float(spearmanr(corrected, df["emo_pre_valence"]).statistic),
        "spearman_pre_arousal": float(spearmanr(corrected, df["emo_pre_arousal"]).statistic),
    }
    return summary


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    df, wrist_obs, split_manifest = load_data()

    train_idx = np.asarray(split_manifest["train"]["row_indices"], dtype=np.int32)
    val_idx = np.asarray(split_manifest["val"]["row_indices"], dtype=np.int32)
    test_idx = np.asarray(split_manifest["test"]["row_indices"], dtype=np.int32)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_sequences = wrist_obs[train_idx]
    val_sequences = wrist_obs[val_idx]
    test_sequences = wrist_obs[test_idx]

    best_model = None
    best_curve = []
    best_ll = -np.inf

    for restart in range(N_RESTARTS):
        seed = SEED + restart * 101
        print(f"\nRestart {restart + 1}/{N_RESTARTS} (seed={seed})")
        model = initialize_model(seed)
        curve = model.baum_welch(train_sequences, n_iter=N_ITER, tol=TOL, verbose=True)
        if curve[-1] > best_ll:
            best_model = model
            best_curve = curve
            best_ll = curve[-1]
            print(f"  New best log-likelihood: {best_ll:.2f}")

    assert best_model is not None
    best_model, reorder = reorder_states(best_model)
    calibration = calibrate_belief(best_model, val_df, val_sequences)
    best_model.metadata.update(
        {
            "state_names": STATE_NAMES,
            "belief_temperature": calibration["temperature"],
            "belief_prior_strength": calibration["prior_strength"],
            "state_reorder": reorder,
            "observation_space": N_OBS,
            "hr_buckets": "0:<-8, 1:-8..12, 2:>12",
        }
    )

    metrics = {
        "train": diagnostic_summary(best_model, train_df, train_sequences, calibration),
        "val": diagnostic_summary(best_model, val_df, val_sequences, calibration),
        "test": diagnostic_summary(best_model, test_df, test_sequences, calibration),
        "calibration": calibration,
        "train_log_likelihood": float(best_ll),
    }

    print("\nValidation calibration:")
    print(
        f"  temperature={calibration['temperature']:.2f}  "
        f"prior_strength={calibration['prior_strength']:.2f}  "
        f"accuracy={calibration['accuracy']:.3f}  "
        f"entropy={calibration['mean_entropy']:.3f}"
    )
    print("\nBelief-space diagnostics:")
    for split_name in ["train", "val", "test"]:
        split_metrics = metrics[split_name]
        print(
            f"  {split_name:<5} entropy={split_metrics['belief_entropy_mean']:.3f}  "
            f"unique={split_metrics['belief_unique_rows_rounded']:>3d}  "
            f"usage={split_metrics['corrected_usage']}"
        )

    best_model.save(str(MODELS_DIR / "hmm"))
    pd.DataFrame(
        {
            "iteration": range(1, len(best_curve) + 1),
            "log_likelihood": best_curve,
        }
    ).to_csv(MODELS_DIR / "hmm_convergence.csv", index=False)
    with (MODELS_DIR / "hmm_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("\nSaved:")
    print("  models/hmm.npz")
    print("  models/hmm_convergence.csv")
    print("  models/hmm_metrics.json")


if __name__ == "__main__":
    main()
