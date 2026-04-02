"""
Precompute corrected HMM beliefs and 5D state vectors for all splits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.common import PROCESSED_DIR, state_vector_from_components
from src.hmm.hmm_inference import corrected_belief
from src.hmm.hmm_model import HMM


def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "interactions_clean.csv")
    wrist_obs = np.load(PROCESSED_DIR / "wrist_obs_all.npy")
    with (PROCESSED_DIR / "split_manifest.json").open("r", encoding="utf-8") as handle:
        split_manifest = json.load(handle)

    hmm = HMM.load("models/hmm.npz")

    beliefs = np.zeros((len(df), hmm.n_states), dtype=np.float32)
    state_vectors = np.zeros((len(df), 5), dtype=np.float32)

    for idx, row in df.iterrows():
        belief = corrected_belief(hmm, wrist_obs[idx], int(row["activity_majority"]))
        beliefs[idx] = belief
        state_vectors[idx] = state_vector_from_components(
            belief,
            int(row["time_bucket"]),
            int(row["activity_majority"]),
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
