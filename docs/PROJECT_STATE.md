# Project State

Last updated: 2026-04-02

Rule: after any meaningful implementation change, update this file in the same work session. If the public workflow or user-facing commands changed, update `README.md` too.

## Current Architecture

The current implemented pipeline is:

1. `python -m src.data.preprocess`
2. `python src/hmm/hmm_train.py`
3. `python src/hmm/precompute_beliefs.py`
4. `python src/data/generate_synthetic.py`
5. `python train_agent.py --synthetic-weight 0.25`
6. `python eval_agent.py`
7. `python demo.py`
8. `python simulate_user.py`

Core design:

- Preprocessing is script-backed under `src/data/`.
- The HMM is wrist-only with 3 hidden states and 60 observations.
- Corrected beliefs are converted into 14D DQN state vectors.
- The DQN state includes biometrics, context, optional emotion check-in, and Stage 1 preference features.
- The RL environment is one-step and action-dependent.
- Rewards are split into emotional benefit and acceptance, then recombined for policy learning.
- The hierarchical reward model conditions on HMM state, pre-emotion bins, time, activity, and action, with preference alignment blended into acceptance.
- Synthetic data is optional lower-weight augmentation.
- Retrieval is deterministic, preference-aware, and treats PMEmo as a soft signal.
- PMEmo soft bucket hints now use PMEmo-internal tempo percentiles instead of the transferred absolute tempo scale.

## Raw Data Expectations

Expected raw-data roots:

- SiTunes: `data/raw/situnes/SiTunes/`
- PMEmo: `data/raw/pmemo/`
- Spotify Kaggle: `data/raw/spotify_kaggle/dataset.csv`

SiTunes is required. PMEmo and Spotify are optional but recommended for the current retrieval path.

## Canonical Source Files

Main implementation files:

- `src/data/common.py`
- `src/data/preprocess.py`
- `src/data/generate_synthetic.py`
- `src/hmm/hmm_model.py`
- `src/hmm/hmm_train.py`
- `src/hmm/hmm_inference.py`
- `src/hmm/precompute_beliefs.py`
- `src/rl_agent/reward_model.py`
- `src/rl_agent/environment.py`
- `src/rl_agent/dqn_agent.py`
- `src/music/music_library.py`
- `train_agent.py`
- `eval_agent.py`
- `demo.py`
- `simulate_user.py`
- `README.md`
- `docs/TA_DEFENSE_GUIDE.md`
- `docs/TA_QA_SHEET.md`

## Current Split Discipline

User-level split:

- train users: 20
- validation users: 5
- test users: 5

The split manifest is written to `data/processed/split_manifest.json` and reused throughout the pipeline.

## Current HMM Truth

- Hidden states: 3
- Observation count: 60
- Observation encoding:
  - `obs = hr_bucket * 20 + intensity_bucket * 5 + activity_remapped`
- HR buckets:
  - `0`: `< -8`
  - `1`: `-8 .. 12`
  - `2`: `> 12`
- Training script: `src/hmm/hmm_train.py`
- Default training constants in code:
  - `N_ITER = 40`
  - `N_RESTARTS = 5`
  - `TOL = 1e-3`
  - `SEED = 42`
- State reorder happens after fitting.
- Belief calibration is selected on the validation split and stored in HMM metadata.

Generated HMM artifacts:

- `models/hmm.npz`
- `models/hmm_convergence.csv`
- `models/hmm_metrics.json`

## Current DQN State Truth

State dimensionality: `14`

State layout:

1. `belief_0`
2. `belief_1`
3. `belief_2`
4. `time_norm`
5. `activity_norm`
6. `weather_norm`
7. `speed_norm`
8. `hr_mean_rel_norm`
9. `hr_std_norm`
10. `pre_valence_norm`
11. `pre_arousal_norm`
12. `pre_emotion_mask`
13. `user_valence_pref_norm`
14. `user_energy_pref_norm`

## Current RL Truth

`train_agent.py` currently trains a Double DQN using:

- one-step `MusicEnv`
- expected combined reward by default
- optional synthetic context mixing
- validation combined reward for checkpoint selection

Key training constants in code:

- `N_EPISODES = 4500`
- `BURN_IN = 400`
- `LR = 1e-3`
- `GAMMA = 0.90`
- `EPS_START = 1.0`
- `EPS_DECAY = 0.997`
- `EPS_MIN = 0.05`
- `BATCH_SIZE = 128`
- `BUFFER_CAP = 20000`
- `TARGET_SYNC = 20`
- `HIDDEN = 128`
- `EVAL_EVERY = 150`
- `SEED = 42`

Generated RL artifacts:

- `models/reward_model.json`
- `models/agent.pt`
- `models/training_log.csv`
- `models/training_summary.json`

## Current Reward Modeling Truth

Reward components:

- `emotion_benefit = clip(0.7 * delta_valence + 0.3 * delta_arousal, -1, 1)`
- `acceptance_score`:
  - Stage 3 `preference` if present
  - otherwise rating-based fallback
- `combined_reward = 0.7 * emotion_benefit + 0.3 * acceptance_score`

Hierarchical conditioning:

- global action prior
- action
- `(hmm_state, pre_valence_bin, pre_arousal_bin, action)`
- `(hmm_state, time_bucket, activity_majority, pre_valence_bin, pre_arousal_bin, action)`

Acceptance is additionally blended with Stage 1 preference alignment against the chosen bucket.

## Current Synthetic Data Truth

`src/data/generate_synthetic.py` currently:

- fits and saves the reward model
- generates synthetic contexts from train-split templates
- uses train-derived time transitions
- samples reward signs and mood deltas from the reward model
- emits 14D synthetic state vectors with `pre_emotion_mask = 0`

Current code constants:

- `SEED = 99`
- `SYNTHETIC_ROWS = 900`
- `REBALANCE_TEMPERATURE = 0.78`
- `TIME_SMOOTH = 0.15`

Important caveat:

- `REBALANCE_TEMPERATURE` mainly rebalances synthetic context frequency, not true rare-action support.

Generated synthetic artifacts:

- `data/processed/synthetic_clean.csv`
- `data/processed/synthetic_state_vectors.npy`
- `data/processed/synthetic_report.json`

## Current Retrieval Truth

`src/music/music_library.py` currently:

- builds a unified catalog from SiTunes, PMEmo, and Spotify
- uses PMEmo as a soft retrieval signal, not strong hard-bucket supervision
- uses PMEmo soft bucket hints generated from PMEmo-internal tempo percentiles
- ranks tracks using:
  - bucket fit
  - mode/scenario fit
  - genre cues
  - acousticness / instrumentalness
  - popularity / familiarity
  - Stage 1 user preference profile
  - optional PMEmo EDA impact

## Current Evaluation Story

`eval_agent.py` is the main offline evaluation entrypoint.

It reports held-out test metrics for:

- `dqn`
- `state_prior`
- `always7`
- uniform random expected reward

It now reports:

- combined reward
- emotional benefit
- acceptance
- regret
- support
- action entropy

It also prints a scenario gallery and supports:

- `python eval_agent.py --interactive`

Generated evaluation artifact:

- `models/eval_report.json`

## Current Demo And Simulation Truth

`demo.py` now shows:

- same physical state, different emotion -> different bucket
- same context, different preference -> different bucket / track ranking
- same user, different scenarios -> different explanations and rankings

`simulate_user.py`:

- uses fixed user profiles
- mixes mask=1 and mask=0 sessions
- samples reward signs from the shared reward model
- updates mood sequentially across sessions
- prints ASCII-safe output

## Last Verified Results

Last full local pipeline run completed successfully for:

- `python -m src.data.preprocess`
- `python src/hmm/hmm_train.py`
- `python src/hmm/precompute_beliefs.py`
- `python src/data/generate_synthetic.py`
- `python train_agent.py`
- `python eval_agent.py`
- `python demo.py`
- `python simulate_user.py`

Observed from current generated artifacts:

- mean belief entropy: `0.1592`
- rounded unique belief vectors: `473`
- HMM corrected state usage:
  - state 0: `656`
  - state 1: `481`
  - state 2: `269`
- held-out mean combined reward:
  - DQN: `+0.1630`
  - state-prior: `+0.1645`
  - always-7: `+0.0805`
  - random uniform expected reward: `+0.0778`
- PMEmo soft bucket distribution after the percentile-tempo fix:
  - bucket 0: `8`
  - bucket 2: `37`
  - bucket 3: `8`
  - bucket 4: `15`
  - bucket 6: `307`
  - bucket 7: `361`

Current qualitative demo behavior:

- same physical state + different mood can produce different actions (`bucket 2` vs `bucket 5`)
- same context + different preferences can produce different actions (`bucket 2` vs `bucket 5`)

These numbers are only valid until the next retrain.

## Known Caveats

- The HMM still reflects physical-context energy more than deep internal psychology.
- DQN currently remains slightly below the state-prior baseline on held-out combined reward, even though it clearly beats trivial baselines.
- Action usage is still concentrated in a small subset of buckets because the underlying data is highly imbalanced.
- Exercise-specific claims should stay modest because running-like contexts are rare in SiTunes.
- PMEmo helps retrieval, not the core intervention-outcome claim.
- The no-check-in path is supported in the state interface, but it has not been separately benchmarked as a standalone deployment mode.

## Current Documentation Policy

After any meaningful change:

1. update this file
2. update `README.md` if setup, commands, outputs, or public behavior changed
3. mention if metrics are stale because retraining was not rerun

Current study/defense docs:

- `docs/TA_DEFENSE_GUIDE.md` is the full technical walkthrough for TA prep
- `docs/TA_QA_SHEET.md` is the short speaking-answer sheet for likely challenge questions

## Next High-Value Improvements

- improve the reward model so DQN clearly beats the state-prior baseline on held-out combined reward
- reduce semantically awkward demo outputs for recovery-like scenarios
- evaluate confidence-aware action filtering for sparse buckets
- consider a post-class session-level latent-state redesign
