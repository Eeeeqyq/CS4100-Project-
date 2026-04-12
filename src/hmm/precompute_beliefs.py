"""
Precompute corrected HMM beliefs and 16D state vectors for all splits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.common import PROCESSED_DIR, STATE_DIM, state_vector_from_components
from src.hmm.hmm_inference import corrected_belief
from src.hmm.hmm_model import HMM


def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "interactions_clean.csv")
    wrist_obs = np.load(PROCESSED_DIR / "wrist_obs_all.npy")
    with (PROCESSED_DIR / "split_manifest.json").open("r", encoding="utf-8") as handle:
        split_manifest = json.load(handle)

    hmm = HMM.load("models/hmm.npz")

    beliefs = np.zeros((len(df), hmm.n_states), dtype=np.float32)
    state_vectors = np.zeros((len(df), STATE_DIM), dtype=np.float32)

    for idx, row in df.iterrows():
        belief = corrected_belief(hmm, wrist_obs[idx], int(row["activity_majority"]))
        beliefs[idx] = belief
        state_vectors[idx] = state_vector_from_components(
            belief,
            int(row["time_bucket"]),
            int(row["activity_majority"]),
            weather_bucket=int(row.get("weather_bucket", 1)),
            gps_speed=float(row.get("gps_speed", 0.0)),
            hr_mean_rel_user=float(row.get("hr_mean_rel_user", 0.0)),
            hr_std=float(row.get("hr_std", 0.0)),
            pre_valence=float(row["emo_pre_valence"]) if pd.notna(row.get("emo_pre_valence")) else None,
            pre_arousal=float(row["emo_pre_arousal"]) if pd.notna(row.get("emo_pre_arousal")) else None,
            pre_emotion_mask=float(row.get("pre_emotion_mask", 1.0)),
            user_valence_pref=float(row.get("user_valence_pref", 0.0)),
            user_energy_pref=float(row.get("user_energy_pref", 0.0)),
            step_mean=float(row.get("step_mean", 0.0)),
            step_nonzero_frac=float(row.get("step_nonzero_frac", 0.0)),
        )

    np.save(PROCESSED_DIR / "belief_states.npy", beliefs)
    np.save(PROCESSED_DIR / "state_vectors.npy", state_vectors)

    for split_name, payload in split_manifest.items():
        indices = np.asarray(payload["row_indices"], dtype=np.int32)
        np.save(PROCESSED_DIR / f"belief_states_{split_name}.npy", beliefs[indices])
        np.save(PROCESSED_DIR / f"state_vectors_{split_name}.npy", state_vectors[indices])

    entropy = -np.sum(beliefs * np.log(beliefs + 1e-12), axis=1)
    print("Belief-state summary:")
    print(f"  rows={len(df)}  entropy_mean={entropy.mean():.4f}")
    print(f"  unique_rounded={np.unique(np.round(beliefs, 4), axis=0).shape[0]}")
    print(f"  usage={dict(zip(*np.unique(beliefs.argmax(axis=1), return_counts=True)))}")
    print("\nSaved split-aware belief/state artifacts to data/processed")


if __name__ == "__main__":
    main()
