# Project State

Last updated: 2026-04-12

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

Experimental parallel rebuild scaffold:

9. `python scripts/build_v2_data.py`

The `v2` rebuild now also has an explicit execution contract in:

- `docs/V2_EXECUTION_PLAN.md`
- `docs/PRESENTATION_REPORT.md` provides the presentation-oriented explanation of the rebuilt path and its current verified results.
- top-level wrapper entrypoints now exist:
  - `train_v2.py`
  - `eval_v2.py`
  - `demo_v2.py`

Core design:

- Preprocessing is script-backed under `src/data/`.
- The HMM is wrist-only with 3 hidden states and 60 observations.
- Corrected beliefs are converted into 16D DQN state vectors.
- The DQN state includes biometrics, context, optional emotion check-in, and Stage 1 preference features.
- The RL environment is one-step and action-dependent.
- Rewards are split into emotional benefit and acceptance, then recombined for policy learning.
- The hierarchical reward model conditions on HMM state, pre-emotion bins, time, activity, step-active context, and action, with preference alignment blended into acceptance.
- Synthetic data is optional lower-weight augmentation.
- Retrieval is deterministic, preference-aware, and treats PMEmo as a soft signal.
- PMEmo soft bucket hints now use PMEmo-internal tempo percentiles instead of the transferred absolute tempo scale.
- PMEmo dynamic affect curves are loaded for retrieval-only scoring.

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

State dimensionality: `16`

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
15. `step_mean_norm`
16. `step_nonzero_frac`

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
- `(hmm_state, time_bucket, activity_majority, step_active, pre_valence_bin, pre_arousal_bin, action)`

Acceptance is additionally blended with Stage 1 preference alignment against the chosen bucket.

## Current Synthetic Data Truth

`src/data/generate_synthetic.py` currently:

- fits and saves the reward model
- generates synthetic contexts from train-split templates
- uses train-derived time transitions
- samples reward signs and mood deltas from the reward model
- carries through real step summaries from the template rows
- emits 16D synthetic state vectors with `pre_emotion_mask = 0`

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
- loads PMEmo dynamic affect features:
  - `dyn_valence_delta`
  - `dyn_arousal_delta`
  - `dyn_arousal_volatility`
  - `dyn_arousal_peak`
  - `dyn_quality`
- ranks tracks using:
  - bucket fit
  - mode/scenario fit
  - genre cues
  - acousticness / instrumentalness
  - popularity / familiarity
  - Stage 1 user preference profile
  - PMEmo dynamic trajectory bonuses
  - optional PMEmo EDA impact
- deduplicates ranked outputs by `track_name` + `artist` before returning the top `n`

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

## Experimental V2.2 Rebuild

The rebuilt `v2.2` path exists in parallel under `src/v2/`.

Current scientific contract:

- explicit goal is the main interface
- SiTunes anchors supervise intervention quality
- Spotify and PMEmo are transfer catalogs, not primary intervention labels
- exact-song recovery is a legacy diagnostic, not the main success criterion

Current supported goals:

- `focus`
- `wind_down`
- `uplift`
- `movement`

### Current V2.2 data contract

`python scripts/build_v2_data.py` writes:

- `data/processed/rebuild/decision_table.parquet`
- `data/processed/rebuild/stage1_history_table.parquet`
- `data/processed/rebuild/stage1_histories.npz`
- `data/processed/rebuild/wrist_windows.npy`
- `data/processed/rebuild/env_features.npy`
- `data/processed/rebuild/self_report.npy`
- `data/processed/rebuild/song_catalog.parquet`
- `data/processed/rebuild/song_static.npy`
- `data/processed/rebuild/song_dynamic.npy`
- `data/processed/rebuild/song_dynamic_mask.npy`
- `data/processed/rebuild/factual_song_idx.npy`
- `data/processed/rebuild/anchor_table.parquet`
- `data/processed/rebuild/anchor_positive_sets.npz`
- `data/processed/rebuild/anchor_negative_pools.npz`
- `data/processed/rebuild/anchor_context_features.npy`
- `data/processed/rebuild/anchor_supervision_stats.json`

Important `decision_table.parquet` truth:

- `goal_idx` is the primary goal field
- `goal_source` explicitly stores:
  - `explicit`
  - `router_fallback`
- acceptance supervision is split into:
  - `accept_pref_target`
  - `accept_rating_target`
  - `accept_pref_mask`
  - `accept_rating_mask`

Important `anchor_table.parquet` truth:

- one supervised SiTunes intervention row exists per decision
- validation and test inference are split-safe:
  - only train anchors are allowed into the deployment index
- anchor-side audit fields now include:
  - `local_support_count`
  - `positive_tier1_count`
  - `positive_tier2_count`
  - `positive_tier3_count`
  - `factual_positive_available`

Important supervision artifact truth:

- `anchor_positive_sets.npz` stores:
  - positive indices
  - per-row positive counts
  - positive tiers
- `anchor_negative_pools.npz` stores:
  - negative indices
  - per-row negative counts
  - negative types

Last verified `v2.2` build output:

- decision rows: `1406`
- anchors: `1406`
- train anchors available for retrieval: `815`
- users: `30`
- split rows:
  - train: `815`
  - val: `280`
  - test: `311`
- tensor shapes:
  - `wrist_windows.npy`: `(1406, 30, 9)`
  - `env_features.npy`: `(1406, 9)`
  - `self_report.npy`: `(1406, 3)`
  - `song_static.npy`: `(91264, 20)`
  - `song_dynamic.npy`: `(91264, 20, 2)`
  - `song_dynamic_mask.npy`: `(91264, 20, 1)`

Anchor supervision summary from current generated artifacts:

- positive cap: `5`
- negative cap: `32`
- mean positive count: `4.9957`
- mean negative count: `32.0000`
- mean positive count by split:
  - train: `4.9951`
  - val: `5.0000`
  - test: `4.9936`
- mean positive count by goal:
  - focus: `5.0000`
  - wind_down: `5.0000`
  - uplift: `5.0000`
  - movement: `4.8800`
- nonempty positive coverage:
  - train: `1.0000`
  - val: `1.0000`
  - test: `1.0000`
- factual-only rate: `0.0000`
- tier composition:
  - tier1 factual: `0.1160`
  - tier2 same-song successful: `0.1438`
  - tier3 nearby successful neighbors: `0.7402`
- local support:
  - overall mean: `11.7902`
  - overall max: `14`

### Current V2.2 query tower truth

Current behavior:

- retrieves only against the train-only SiTunes anchor index
- uses:
  - context embeddings
  - user embeddings
  - goal one-hot
  - target affect state `tau`
- trains with:
  - multi-positive set-softmax
  - tier-weighted positives:
    - factual > same-song > nearby neighbor
  - hard negatives from the rebuilt negative pools
  - hardest-negative pairwise margin pressure
- selects checkpoints by:
  - conditional mean positive rank
  - weighted MRR
  - recall@20

Last verified run:

- `python src/v2/train/train_query_tower.py`

Current verified query result from `models/rebuild/query_tower_metrics.json`:

- best epoch: `1`
- held-out test:
  - anchor `recall@20 = 0.7460`
  - anchor `recall@50 = 0.8810`
  - weighted `recall@20 = 0.3011`
  - weighted `recall@50 = 0.4358`
  - weighted MRR: `0.1123`
  - weighted NDCG@10: `0.1688`
  - factual-positive rate: `0.3441`
  - first-positive-is-factual rate: `0.1402`

Important caveat:

- `query_tower_metrics.json` still logs very large raw factual-rank numbers because they are computed before the stricter contained-row correction
- use `models/rebuild/offline_eval_v2.json` as the source of truth for contained factual rank

### Current V2.2 reranker truth

Current behavior:

- reranks SiTunes anchor candidate sets, not mixed public-song sets
- candidate sets are built from the hardened query stage and explicitly try to include:
  - a factual positive when available
  - a hard failure negative
  - a same-goal confounder when available
- predicts:
  - `benefit_hat`
  - `accept_pref_hat`
  - `accept_rating_hat`
  - `anchor_relevance_logit`
- uses:
  - tier-aware set relevance loss
  - tier-aware pairwise relevance loss
  - explicit factual-priority loss
  - separate benefit / preference / rating regression heads
  - deterministic uncertainty rather than a learned uncertainty head

Last verified run:

- `python src/v2/train/train_reranker.py`

Current verified reranker result from `models/rebuild/reranker_metrics.json`:

- best epoch: `10`
- held-out test:
  - benefit MAE: `0.1367`
  - preference-head MAE: `0.2774`
  - rating-head MAE: `0.2569`
  - blended acceptance MAE: `0.3279`
  - anchor hit@10: `0.7138`
  - anchor conditional mean positive rank: `6.0511`
  - weighted NDCG@10: `0.3575`
  - factual hit@10 where a same-song factual anchor exists: `0.6667`
  - factual conditional mean rank: `10.0417`

Important caveat:

- `reranker_metrics.json` is a reranker-stage report over contained anchor candidate sets
- use `models/rebuild/offline_eval_v2.json` as the source of truth for the full end-to-end path, because it includes the live transfer stage and the corrected contained-rank computation

### Current V2.2 inference and offline eval truth

Current behavior:

- `recommend.py` now runs:
  - explicit-goal query embedding
  - train-anchor retrieval
  - anchor reranking
  - anchor-conditioned public transfer
- public songs are only allowed to outrank anchors when transfer support clears the active threshold
- active public-transfer threshold after the full rebuilt rerun: `0.72`
- PMEmo dynamic cues are only used inside the public-transfer scoring path
- if public transfer is not strong enough, explanations explicitly fall back to the anchor story
- exact-song recovery is still emitted as a legacy diagnostic only

Current commands:

- `python train_v2.py`
- `python eval_v2.py`
- `python demo_v2.py`
- `python src/v2/inference/recommend.py --split test --limit 3 --top-k 5 --candidate-k 50 --explicit-goal focus`
- `python src/v2/eval/offline_eval.py --split test --candidate-k 50`
- `python src/v2/eval/check_readiness.py`
- `python src/v2/eval/tune_final_ranker.py`

Generated artifacts:

- `models/rebuild/offline_eval_v2.json`
- `models/rebuild/v2_readiness.json`
- `models/rebuild/transfer_threshold_tuning.json`
- `data/processed/rebuild/offline_eval_v2_predictions.parquet`

Last verified end-to-end test result from `models/rebuild/offline_eval_v2.json`:

- rows evaluated: `311`
- candidate set size: `50`
- primary metrics:
  - anchor query `recall@20 = 0.7460`
  - anchor query `recall@50 = 0.8810`
  - anchor query weighted `recall@20 = 0.3011`
  - anchor query weighted `recall@50 = 0.4358`
  - anchor query conditional mean positive rank: `9.9161`
  - anchor query weighted MRR: `0.1118`
  - anchor query weighted NDCG@10: `0.1688`
  - anchor rerank `hit@10 = 0.7170`
  - anchor rerank conditional mean positive rank: `5.9270`
  - anchor rerank weighted `recall@20 = 0.3771`
  - anchor rerank weighted `recall@50 = 0.4358`
  - anchor rerank weighted MRR: `0.1465`
  - anchor rerank weighted NDCG@10: `0.2306`
  - benefit MAE: `0.1262`
  - blended acceptance MAE: `0.2947`
  - top-1 predicted acceptance mean: `0.5101`
  - public-transfer-supported share: `0.6399`
  - top-1 source max share: `0.6399`
- top-1 source distribution:
  - `situnes = 0.3601`
  - `spotify = 0.6399`

Goal-level weighted query `recall@20`:

- `focus = 0.3058`
- `movement = 0.5412`
- `uplift = 0.2139`
- `wind_down = 0.3955`

Legacy diagnostics:

- exact-song query `recall@50 = 0.3698`
- exact-song rerank `hit@10 = 0.0418`
- exact-song conditional rank: `37.58`

Important caveat:

- the primary `v2.2` contract now passes after rerunning the full rebuilt Block B and recalibrating the public-transfer threshold
- the legacy exact-song rerank `hit@10` diagnostic is still below the old `0.05` sanity threshold, but it is now much closer at `0.0418`

### Supporting encoder truth

Supporting encoder artifacts still exist under `src/v2/models/` and `src/v2/train/`:

- `song_encoder.py`
- `context_encoder.py`
- `user_encoder.py`

Current important caveat:

- the supporting encoders were rerun as part of the full rebuilt Block B validation
- they are now part of the current live artifact set used by `v2.2`
- they still are not the main headline evidence; the main scientific claim remains the end-to-end anchor retrieval, reranking, and transfer behavior

## Current Demo And Simulation Truth

`demo.py` now shows:

- same physical state, different emotion -> different reward profile / track ranking
- same context, different preference -> different track ranking
- same user, different scenarios -> different explanations and rankings
- same HR / mood signal, different movement evidence -> different bucket (`bucket 5` vs `bucket 7`)

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
- state vector artifact shapes:
  - `state_vectors.npy`: `(1406, 16)`
  - `synthetic_state_vectors.npy`: `(900, 16)`
- step summary coverage:
  - `step_active` rate in cleaned interactions: `0.2838`
- held-out mean combined reward:
  - DQN: `+0.1639`
  - state-prior: `+0.1641`
  - always-7: `+0.0821`
  - random uniform expected reward: `+0.0781`
- held-out mean component rewards:
  - DQN emotion benefit: `+0.0465`
  - DQN acceptance: `+0.4377`
- PMEmo soft bucket distribution after the percentile-tempo fix:
  - bucket 0: `8`
  - bucket 2: `37`
  - bucket 3: `8`
  - bucket 4: `15`
  - bucket 6: `307`
  - bucket 7: `361`
- PMEmo dynamic feature coverage:
  - all `736 / 736` cleaned PMEmo rows have non-null dynamic fields after preprocessing

Current qualitative demo behavior:

- same physical state + different mood changes the reward profile and ranked tracks
- same context + different preferences changes ranked tracks
- same HR / mood signal + different movement evidence changes the chosen bucket (`bucket 5` vs `bucket 7`)

These numbers are only valid until the next retrain.

## Known Caveats

- The HMM still reflects physical-context energy more than deep internal psychology.
- DQN currently remains slightly below the state-prior baseline on held-out combined reward, even though it clearly beats trivial baselines.
- Action usage is still concentrated in a small subset of buckets because the underlying data is highly imbalanced.
- Exercise-specific claims should stay modest because running-like contexts are rare in SiTunes.
- PMEmo helps retrieval, not the core intervention-outcome claim.
- PMEmo dynamic features currently improve PMEmo internal ordering more than overall top-of-list visibility because Spotify remains dominant in the merged catalog.
- The no-check-in path is supported in the state interface, but it has not been separately benchmarked as a standalone deployment mode.

## Current Documentation Policy

After any meaningful change:

1. update this file
2. update `README.md` if setup, commands, outputs, or public behavior changed
3. mention if metrics are stale because retraining was not rerun

## Next High-Value Improvements

- improve the reward model so DQN clearly beats the state-prior baseline on held-out combined reward
- reduce semantically awkward demo outputs for recovery-like scenarios
- evaluate confidence-aware action filtering for sparse buckets
- improve public-transfer quality and PMEmo visibility without regressing the new source-balance gate
- decide whether DEAM should be added as a public-song affect representation upgrade after the current `v2.2` contract is stable
- only revisit `v2.2` anchor positive-set tuning if a future pass shows one goal losing support under the current cap-5 supervision
- if legacy exact-song diagnostics matter again, improve them by better song-level transfer support rather than by returning to mixed-catalog exact-song supervision
- consider a post-class session-level latent-state redesign
