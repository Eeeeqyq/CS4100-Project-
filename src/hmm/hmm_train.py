"""
src/hmm/hmm_train.py
Train the HMM on SiTunes wrist observation sequences.

Run from project root:
    python src/hmm/hmm_train.py

Reads:  data/processed/wrist2_encoded.npy
        data/processed/wrist3_encoded.npy
        data/processed/stage2_clean.csv
        data/processed/stage3_clean.csv

Writes: models/hmm.npz
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

# Add project root to path so `from src.hmm.hmm_model import HMM` works
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.hmm.hmm_model import HMM

PROCESSED  = Path("data/processed")
MODELS     = Path("models")
MODELS.mkdir(exist_ok=True)

N_STATES   = 6
N_OBS      = 180
N_ITER     = 80
TOL        = 1e-3
N_RESTARTS = 3
SEED       = 42


def load_data():
    wrist2 = np.load(PROCESSED / "wrist2_encoded.npy")   # (897, 30)
    wrist3 = np.load(PROCESSED / "wrist3_encoded.npy")   # (509, 30)
    wrist  = np.vstack([wrist2, wrist3])                  # (1406, 30)

    df2 = pd.read_csv(PROCESSED / "stage2_clean.csv")
    df3 = pd.read_csv(PROCESSED / "stage3_clean.csv")
    df  = pd.concat([df2, df3], ignore_index=True)

    sequences   = [wrist[i] for i in range(len(wrist))]
    pre_valence = df["emo_pre_valence"].values
    pre_arousal = df["emo_pre_arousal"].values

    print(f"Sequences: {len(sequences)}  length: {wrist.shape[1]}")
    print(f"Obs range: {wrist.min()} – {wrist.max()}  (should be 0–179)")
    return sequences, pre_valence, pre_arousal


def validate(model, sequences, pre_valence, pre_arousal):
    decoded = np.array([model.viterbi(s)[0][-1] for s in sequences])

    print("\nMean pre-valence per decoded state:")
    for s in range(model.n_states):
        mask = decoded == s
        if mask.sum() == 0:
            print(f"  S{s} {HMM.STATE_NAMES[s]:<20}: UNUSED")
            continue
        mv = pre_valence[mask].mean()
        ma = pre_arousal[mask].mean()
        print(f"  S{s} {HMM.STATE_NAMES[s]:<20}: n={mask.sum():3d}  "
              f"valence={mv:+.3f}  arousal={ma:+.3f}")

    r_v, _ = spearmanr(decoded, pre_valence)
    r_a, _ = spearmanr(decoded, pre_arousal)
    print(f"\nSpearman r vs pre_valence: {r_v:.3f}")
    print(f"Spearman r vs pre_arousal: {r_a:.3f}")


def main():
    sequences, pre_valence, pre_arousal = load_data()

    best_model = None
    best_ll    = -np.inf

    for restart in range(N_RESTARTS):
        print(f"\n{'='*50}")
        print(f"Restart {restart+1}/{N_RESTARTS}  (seed={SEED + restart*100})")
        print(f"{'='*50}")

        model    = HMM(n_states=N_STATES, n_obs=N_OBS, seed=SEED + restart*100)
        ll_curve = model.baum_welch(sequences, n_iter=N_ITER, tol=TOL, verbose=True)

        if ll_curve[-1] > best_ll:
            best_ll    = ll_curve[-1]
            best_model = model
            print(f"  ★ New best  (ll={best_ll:.2f})")

    print(f"\nBest log-likelihood: {best_ll:.2f}")
    validate(best_model, sequences, pre_valence, pre_arousal)
    best_model.save(str(MODELS / "hmm"))
    print("\n✓ Done. Next: python train_agent.py")


if __name__ == "__main__":
    main()