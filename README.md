# CS4100-Project-

# Ambient Music Recommendation Agent
### CS4100 Artificial Intelligence - Northeastern University

An AI system that recommends music based on the user's current biometrics, emotional state, environment, and baseline taste instead of only past listening history. A Hidden Markov Model (HMM) infers a coarse latent context-energy belief state from wrist sensor data, and a Deep Q-Network (DQN) chooses the music bucket most likely to help in the current situation. A deterministic retrieval layer then ranks concrete songs using the user's baseline preferences and track-level music-affect cues.

This project is intentionally built with from-scratch NumPy and PyTorch implementations for the core AI components. It does not use `hmmlearn`, `stable-baselines3`, or transformer-based models.

A parallel experimental `v2.2` rebuild path also exists under `src/v2/`. It is separate from the graded HMM + DQN pipeline above. The rebuilt path is now explicit-goal, anchor-first, and transfer-aware:

- retrieve strong SiTunes intervention anchors
- rerank anchors by predicted benefit and acceptance
- expand to Spotify / PMEmo only when public transfer support is strong enough

Its execution contract and finish criteria are tracked in `docs/V2_EXECUTION_PLAN.md`.
For presentation prep and an end-to-end explanation of the rebuilt path, see `docs/PRESENTATION_REPORT.md`.
For the short 5-6 minute presentation talk track, see `docs/PRESENTATION_SCRIPT.md`.

---

## Quick Start For `v2.2`

If you are new to the repo and only want to run the rebuilt `v2.2` system, use this section.

### Fastest path if rebuilt artifacts already exist

```bash
python eval_v2.py --no-rerun
python demo_v2.py
```

Use this when you want to inspect the latest verified `v2.2` results and presentation demo without retraining anything.

### Full fresh `v2.2` rebuild

```bash
python train_v2.py
python eval_v2.py
python demo_v2.py
```

What each command does:

- `train_v2.py`: runs the full rebuilt training and evaluation stack
- `eval_v2.py`: prints the current offline summary and readiness result
- `demo_v2.py`: shows a presentation-friendly goal-conditioned recommendation demo

Important notes:

- `train_v2.py` expects the raw datasets to exist under the paths listed in the Setup section below
- `train_v2.py` is the expensive path; use `eval_v2.py --no-rerun` and `demo_v2.py` if you only need the latest saved outputs
- the legacy HMM + DQN pipeline is still in the repo, but it is separate from the `v2.2` commands above

Current verified `v2.2` status from the latest saved artifacts:

- `ready = true`
- held-out test rows: `311`
- anchor query `recall@20 = 0.7460`
- anchor rerank `hit@10 = 0.7170`
- benefit MAE `= 0.1262`
- blended acceptance MAE `= 0.2947`

---

## The Problem

Spotify and Apple Music know what you liked last week. They do not know that you are tense at 2pm, low-energy in the evening, or trying to stay focused during a study block. Most recommenders ignore the most relevant signal: how you feel right now.

This project asks a harder question:

**Given your current biometric and environmental context and your baseline taste profile, what kind of music intervention is most likely to help right now while still being acceptable enough to follow?**

---

## Our Solution

A four-part ambient recommendation pipeline:

1. **Sense the current situation**  
   A 3-state HMM reads a 30-step wrist window and produces a belief distribution over coarse latent context-energy states.

2. **Remember the person**  
   Stage 1 of SiTunes is used to build a compact user preference profile from baseline ratings.

3. **Choose the right intervention**  
   A Double DQN receives a 16-dimensional state vector and chooses one of 8 music mood buckets.

4. **Retrieve real tracks**  
   A deterministic music library ranks concrete songs from SiTunes, Spotify, and PMEmo using bucket fit, scenario fit, and user preference fit.

The key idea is still POMDP-style: the user's internal state is partially hidden, so the system reasons over a belief state rather than pretending wrist data directly reveals mood.

---

## Experimental V2.2 Rebuild

The rebuilt `v2.2` path solves a different problem than the legacy bucket-first pipeline:

**Given an explicit goal (`focus`, `wind_down`, `uplift`, `movement`), current context, and baseline taste profile, retrieve strong historical SiTunes intervention anchors and only transfer to public songs when that transfer is well-supported.**

### What changed in `v2.2`

1. **Explicit goal is the main interface**  
   `goal_idx` is treated as the primary decision input for `v2.2`. The old goal router still exists as a fallback baseline, but the rebuilt path is evaluated mainly in explicit-goal mode.

2. **SiTunes anchors supervise intervention quality**  
   `v2.2` builds a split-safe train-anchor index from SiTunes decisions and learns to retrieve those anchors instead of directly treating the mixed public catalog as if it had intervention labels.

3. **Acceptance is modeled as two observed channels**  
   Preference and rating are stored separately:
   - `accept_pref_target`
   - `accept_rating_target`
   - `accept_pref_mask`
   - `accept_rating_mask`

4. **Public catalogs are transfer candidates, not primary labels**  
   Spotify and PMEmo are only allowed to outrank anchors when their transfer support is strong enough relative to the current anchor set.

5. **Anchor supervision is now tiered and harder**  
   Positive anchors are no longer treated as one flat set. `v2.2` now distinguishes:
   - factual train-row anchors
   - same-song successful anchors
   - nearby successful context neighbors

   Query training, reranker training, and readiness now use those tiers directly.

### Main `v2.2` artifacts

- `data/processed/rebuild/decision_table.parquet`
- `data/processed/rebuild/anchor_table.parquet`
- `data/processed/rebuild/anchor_positive_sets.npz`
- `data/processed/rebuild/anchor_negative_pools.npz`
- `data/processed/rebuild/situnes_anchor_index.npz`
- `data/processed/rebuild/query_embeddings.npy`
- `data/processed/rebuild/anchor_query_embeddings.npy`

Current anchor-supervision truth:

- positive cap: `5`
- negative cap: `32`
- anchor positive tiers:
  - tier 1: factual train-row anchors
  - tier 2: same-song successful anchors
  - tier 3: nearby successful context neighbors

### Main `v2.2` commands

Simplest top-level entrypoints:

```bash
python train_v2.py
python eval_v2.py
python demo_v2.py
```

Equivalent underlying module sequence:

```bash
python scripts/build_v2_data.py
python src/v2/train/train_song_encoder.py
python src/v2/train/train_user_encoder.py
python src/v2/train/train_context_encoder.py
python src/v2/train/train_query_tower.py
python src/v2/train/train_reranker.py
python src/v2/eval/offline_eval.py --split test --candidate-k 50
python src/v2/eval/check_readiness.py
python src/v2/inference/recommend.py --split test --limit 3 --explicit-goal focus
```

### What `v2.2` is evaluated on

The rebuilt path is **not** judged mainly by exact-song recovery anymore. Its primary offline metrics are:

- anchor retrieval recall@20 / recall@50
- anchor retrieval weighted recall@20 / weighted recall@50
- anchor retrieval weighted MRR / weighted NDCG@10
- anchor rerank hit@10 and conditional positive rank
- anchor rerank weighted NDCG@10
- factual-positive conditional rank where same-song anchors exist in the train index
- benefit MAE
- blended acceptance MAE
- top-1 predicted acceptance mean
- public-transfer-supported share
- top-1 source concentration

Exact factual-song recovery is still reported as a legacy diagnostic, not the main success criterion.

Latest verified `v2.2` readiness status:

- primary rebuilt contract: `ready = true`
- legacy exact-song rerank sanity metric still soft-fails

---

## Architecture

```text
Biometrics + Context + Optional Check-In
 [HR, intensity, activity, steps, weather, speed, pre-emotion]
                      |
                      v
    +----------------------------------+
    | HMM (3 states, 60 observations)  |
    | Baum-Welch + corrected belief     |
    +----------------+------------------+
                     |
                     | belief state (3-dim)
                     v
    +----------------------------------+
    | DQN Agent (16D state, 8 actions) |
    | Double DQN + replay + target net |
    +----------------+------------------+
                     |
                     | music bucket
                     v
    +----------------------------------+
    | Retrieval / Ranking Layer        |
    | preference-aware deterministic   |
    | ranking across 3 music catalogs  |
    +----------------------------------+
```

**Hidden States (HMM)**

| ID | State | Interpretation |
|----|-------|----------------|
| 0 | low-energy | sedentary / recovery / lying-still contexts |
| 1 | moderate | transitioning / walking / mixed activity contexts |
| 2 | high-energy | high-intensity / running-like contexts |

These states are intentionally coarse. The system does not claim that wrist data alone can recover detailed psychological moods.

**Action Buckets (DQN)**

| Bucket | Label | Valence | Energy | Tempo |
|--------|-------|---------|--------|-------|
| 0 | dark-slow | low | low | slow |
| 1 | dark-fast | low | low | fast |
| 2 | intense-slow | low | high | slow |
| 3 | aggressive | low | high | fast |
| 4 | chill-study | medium/high | low | slow |
| 5 | indie | medium/high | low | fast |
| 6 | soulful | medium/high | high | slow |
| 7 | energetic | medium/high | high | fast |

---

## Datasets

| Dataset | Role | Size |
|---------|------|------|
| [SiTunes](https://github.com/JiayuLi-997/SiTunes_dataset) | Core training data: biometrics, context, pre/post emotion labels, and baseline preference survey | ~2,000 raw interactions -> 1,406 cleaned Stage 2/3 interactions, 30 users |
| [PMEmo](https://github.com/HuiZhangDB/PMEmo) | Retrieval-side music-affect support and optional EDA impact signal | 736 labeled tracks |
| [Spotify Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) | Large retrieval catalog | ~89,500 tracks |

### How each dataset is actually used

- **SiTunes Stage 1**: baseline preference modeling
- **SiTunes Stage 2 and Stage 3**: intervention-outcome learning
- **PMEmo**: soft retrieval support only, not a strong policy-training bucket source
- **Spotify Kaggle**: large candidate pool for final song ranking

For `v2.2`, the same three datasets are still the only wired-in datasets:

- SiTunes: intervention anchors and user preference history
- PMEmo: transfer-side affect / dynamic trajectory support
- Spotify Kaggle: public transfer catalog

DEAM, WESAD, and other larger public datasets are not integrated into the current repo yet.

PMEmo is not treated as reliable hard bucket supervision for the policy. In the current system it is used more cautiously: valence/arousal guidance, dynamic affect contours, EDA impact hints, and soft bucket hints built from PMEmo-internal tempo percentiles.

---

## Observation Encoding

### HMM Observations (0-59)

```text
obs = hr_bucket * 20 + intensity_bucket * 5 + activity_remapped
```

**Heart-rate bucket**
- `0`: below-normal (`< -8`)
- `1`: normal (`-8` to `12`)
- `2`: elevated (`> 12`)

These thresholds are on the normalized SiTunes wrist HR channel, not raw BPM.

**Intensity bucket**
- `0`: `< 10`
- `1`: `10 - 30`
- `2`: `30 - 80`
- `3`: `>= 80`

**Activity remap**
- raw `0 -> 0` still
- raw `1 -> 1` transition
- raw `2 -> 2` walking
- raw `3 -> 0` missing treated as still
- raw `4 -> 3` lying
- raw `5 -> 4` running

This is a 60-value wrist-only observation space. Time, weather, speed, self-report, and preference signals remain downstream as explicit DQN features instead of being folded into the HMM emissions.

---

## DQN State Vector

The DQN receives a 16-dimensional state vector:

```text
[
  belief_0,
  belief_1,
  belief_2,
  time_norm,
  activity_norm,
  weather_norm,
  speed_norm,
  hr_mean_rel_norm,
  hr_std_norm,
  pre_valence_norm,
  pre_arousal_norm,
  pre_emotion_mask,
  user_valence_pref_norm,
  user_energy_pref_norm,
  step_mean_norm,
  step_nonzero_frac,
]
```

### What the added features do

- `weather_norm`, `speed_norm`, and `hr_*`: explicit context and biometric features
- `pre_valence_norm`, `pre_arousal_norm`: the user's current reported emotion when available
- `pre_emotion_mask`: tells the model whether those emotion values are real self-report or passive / fallback inputs
- `user_valence_pref_norm`, `user_energy_pref_norm`: compact baseline taste profile from Stage 1
- `step_mean_norm`, `step_nonzero_frac`: movement summaries that help distinguish elevated activation during motion from elevated activation while sedentary

The step channel is intentionally kept out of the HMM observation vocabulary. It is treated as an explicit downstream policy feature instead of inflating the 60-observation HMM space.

This is what makes the system ambient-first but optionally calibrated: it can run on passive signals alone, but it can also incorporate a quick check-in.

---

## Reward Model

The current project does not train the policy against only a single ternary label. It separates emotional improvement from user acceptance and then recombines them for offline decision-making.

### 1. Emotional Benefit

```text
emotion_score = 0.7 * (post_valence - pre_valence)
              + 0.3 * (post_arousal - pre_arousal)
emotion_benefit = clip(emotion_score, -1, 1)
```

### 2. Acceptance / Liking

- if Stage 3 `preference` exists: use that
- otherwise fall back to the session `rating`

```text
acceptance = (preference - 50) / 50      if preference exists
           = (rating - 3) / 2            otherwise
```

### 3. Combined Offline Reward

```text
combined_reward = 0.7 * emotion_benefit + 0.3 * acceptance
```

The hierarchical reward model is conditioned on:

- HMM state
- pre-valence bin
- pre-arousal bin
- time bucket
- activity bucket
- step-active flag
- action bucket

Rare contexts back off to broader averages through hierarchical shrinkage instead of pretending every fine-grained state-action combination has enough data. Acceptance is also blended with user preference alignment, so the policy has a reason to care about baseline taste rather than only global averages.

---

## Repository Structure

```text
CS4100-Project-/
|-- data/
|   |-- raw/
|   |   |-- situnes/
|   |   |-- pmemo/
|   |   `-- spotify_kaggle/
|   `-- processed/
|       |-- stage1_clean.csv
|       |-- stage2_clean.csv
|       |-- stage3_clean.csv
|       |-- interactions_clean.csv
|       |-- user_preferences.json
|       |-- wrist_obs_all.npy
|       |-- belief_states.npy
|       |-- state_vectors.npy
|       |-- synthetic_clean.csv
|       `-- synthetic_state_vectors.npy
|-- docs/
|   |-- PROJECT_STATE.md
|   |-- PRESENTATION_REPORT.md
|   |-- PRESENTATION_SCRIPT.md
|   `-- V2_EXECUTION_PLAN.md
|-- models/
|   |-- hmm.npz
|   |-- reward_model.json
|   |-- agent.pt
|   |-- eval_report.json
|   `-- rebuild/
|       |-- offline_eval_v2.json
|       `-- v2_readiness.json
|-- scripts/
|   `-- build_v2_data.py
|-- src/
|   |-- data/
|   |   |-- common.py
|   |   |-- preprocess.py
|   |   `-- generate_synthetic.py
|   |-- hmm/
|   |   |-- hmm_model.py
|   |   |-- hmm_train.py
|   |   |-- hmm_inference.py
|   |   `-- precompute_beliefs.py
|   |-- rl_agent/
|   |   |-- environment.py
|   |   |-- dqn_agent.py
|   |   `-- reward_model.py
|   |-- music/
|   |   `-- music_library.py
|   `-- v2/
|       |-- data/
|       |-- eval/
|       |-- inference/
|       |-- models/
|       `-- train/
|-- train_agent.py
|-- eval_agent.py
|-- demo.py
|-- train_v2.py
|-- eval_v2.py
|-- demo_v2.py
|-- simulate_user.py
`-- requirements.txt
```

Experimental rebuild outputs are written under:

```text
data/processed/rebuild/
```

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd CS4100-Project-
```

### 2. Create a Python environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the data

**SiTunes (required)**

The code expects the SiTunes root here:

```text
data/raw/situnes/SiTunes/
```

A direct clone that matches the code path is:

```bash
git clone https://github.com/JiayuLi-997/SiTunes_dataset data/raw/situnes/SiTunes
```

If you download it manually, make sure the final folder name is exactly `SiTunes`.

**PMEmo (optional, recommended for retrieval)**

Place the PMEmo files under:

```text
data/raw/pmemo/
```

Required files for the current pipeline:

- `annotations/static_annotations.csv`
- `annotations/static_annotations_std.csv`
- `annotations/dynamic_annotations.csv`
- `annotations/dynamic_annotations_std.csv`
- `features/static_features.csv`
- `metadata.csv`
- optional but supported: `EDA/*.csv`

**Spotify Kaggle (optional, recommended for retrieval breadth)**

Download `dataset.csv` from Kaggle and place it here:

```text
data/raw/spotify_kaggle/dataset.csv
```

If PMEmo or Spotify are missing, the core SiTunes pipeline still works. You can either let preprocessing auto-skip missing optional data, or be explicit with flags.

---

## Running the Project

> Always run commands from the project root.

### Which path should you run?

Use the rebuilt `v2.2` path if you want the current anchor-first system:

```bash
python train_v2.py
python eval_v2.py
python demo_v2.py
```

Use the legacy HMM + DQN path if you want the original graded pipeline:

```bash
python -m src.data.preprocess
python src/hmm/hmm_train.py
python src/hmm/precompute_beliefs.py
python src/data/generate_synthetic.py
python train_agent.py --synthetic-weight 0.25
python eval_agent.py
python demo.py
python simulate_user.py
```

### Step 1 - Preprocess the datasets

Full preprocessing:

```bash
python -m src.data.preprocess
```

Minimum SiTunes-only preprocessing:

```bash
python -m src.data.preprocess --skip-pmemo --skip-spotify
```

This builds:

- cleaned SiTunes interaction tables
- user preference profiles from Stage 1
- wrist observation sequences
- PMEmo and Spotify cleaned catalogs when available
- split manifests and dataset audit files

### Experimental Step - Build the `v2` rebuild data contract

```bash
python scripts/build_v2_data.py
```

This is a parallel experimental path. It does not replace the current HMM + DQN workflow yet.

It builds:

- `decision_table.parquet`
- `stage1_history_table.parquet`
- `stage1_histories.npz`
- `wrist_windows.npy`
- `env_features.npy`
- `self_report.npy`
- `song_catalog.parquet`
- `song_static.npy`
- `song_dynamic.npy`
- `song_dynamic_mask.npy`
- `factual_song_idx.npy`

all under `data/processed/rebuild/`.

### Experimental Step - Train the `v2` song encoder

```bash
python src/v2/train/train_song_encoder.py
```

This trains the first learned component of the rebuild:

- static affect supervision across the merged catalog
- dynamic contour reconstruction where PMEmo curves exist
- source-balanced sampling so Spotify does not dominate training

It writes:

- `models/rebuild/song_encoder.pt`
- `models/rebuild/song_encoder_metrics.json`
- `models/rebuild/song_encoder_train_log.csv`
- `data/processed/rebuild/song_embeddings.npy`
- `data/processed/rebuild/song_encoder_predictions.parquet`

This step was rerun in the latest full rebuilt Block B validation. The end-to-end `v2.2` claim should still be grounded in the downstream offline evaluation rather than this encoder metric alone.

### Experimental Step - Train the `v2` user encoder

```bash
python src/v2/train/train_user_encoder.py
```

This trains a leave-one-out Stage 1 taste model on top of the exported song embeddings:

- input: 20-song Stage 1 histories
- target: held-out normalized rating for one masked song
- auxiliary targets:
  - taste-affect summary
  - user confidence
- split discipline: train / val / test users from the main split manifest

It writes:

- `models/rebuild/user_encoder.pt`
- `models/rebuild/user_encoder_metrics.json`
- `models/rebuild/user_encoder_train_log.csv`
- `data/processed/rebuild/user_encoder_outputs.npz`

This step was rerun in the latest full rebuilt Block B validation. Its standalone metrics are useful, but the real `v2.2` claim comes from the downstream anchor retrieval, reranking, and transfer results.

### Experimental Step - Train the `v2` context encoder

```bash
python src/v2/train/train_context_encoder.py
```

This trains a learned context representation over the rebuilt decision tensors:

- input:
  - `wrist_windows.npy`
  - `env_features.npy`
  - `self_report.npy`
- auxiliary targets:
  - pre-affect `(valence, arousal)`
  - derived movement class
  - proxy uncertainty from the self-report mask

It writes:

- `models/rebuild/context_encoder.pt`
- `models/rebuild/context_encoder_metrics.json`
- `models/rebuild/context_encoder_train_log.csv`
- `data/processed/rebuild/context_embeddings.npy`
- `data/processed/rebuild/context_encoder_predictions.parquet`

This step was rerun in the latest full rebuilt Block B validation. Treat its standalone metrics as supporting evidence for the rebuilt stack, not the main headline result.

### Experimental Step - Train the `v2` query tower

```bash
python src/v2/train/train_query_tower.py
```

This trains the anchor-first query stage on top of:

- `context_embeddings.npy`
- `user_encoder_outputs.npz`
- `anchor_table.parquet`
- `anchor_positive_sets.npz`
- `anchor_negative_pools.npz`

Current query training now:

- retrieves only from the train-only SiTunes anchor index
- uses tier-weighted multi-positive supervision:
  - tier 1: factual train-row anchors
  - tier 2: same-song successful anchors
  - tier 3: nearby successful context neighbors
- keeps explicit hard-negative margin pressure
- selects checkpoints by:
  - conditional positive rank
  - weighted MRR
  - weighted recall@20

It writes:

- `models/rebuild/query_tower.pt`
- `models/rebuild/query_tower_metrics.json`
- `models/rebuild/query_tower_history.csv`
- `data/processed/rebuild/query_embeddings.npy`
- `data/processed/rebuild/query_anchor_diag_val.parquet`
- `data/processed/rebuild/query_anchor_diag_test.parquet`
- `data/processed/rebuild/situnes_anchor_index.npz`

Latest verified held-out test result:

- anchor `recall@20 = 0.7460`
- anchor `recall@50 = 0.8810`
- weighted `recall@20 = 0.3011`
- weighted `recall@50 = 0.4358`
- weighted NDCG@10 `= 0.1688`

Important caveat:

- the harder anchor task is now non-saturated, but exact same-song anchors are still sparse
- factual-positive conditional rank is still much weaker than any-positive rank because many rows only have tier-3 support

### Experimental Step - Train the `v2` reranker

```bash
python src/v2/train/train_reranker.py
```

This trains the anchor reranker over SiTunes anchor candidate sets built from the query tower.

Current reranker training now:

- stays anchor-only for primary supervision
- preserves stronger candidate composition on train rows:
  - a factual positive when available
  - a hard-failure negative when available
  - a same-goal confounder when available
- predicts:
  - `benefit_hat`
  - `accept_pref_hat`
  - `accept_rating_hat`
  - `anchor_relevance_logit`
- keeps preference and rating as separate supervised heads
- applies a factual-priority relevance loss so factual or same-song positives are pushed above weaker positives on average
- uses deterministic uncertainty from:
  - context mismatch
  - user-anchor mismatch
  - local support weakness

Checkpoint selection order:

- acceptance guard first
- then anchor conditional rank
- then factual conditional rank
- then benefit MAE

It writes:

- `models/rebuild/reranker.pt`
- `models/rebuild/reranker_metrics.json`
- `models/rebuild/reranker_history.csv`
- `data/processed/rebuild/anchor_rerank_test_predictions.parquet`

Latest verified held-out test result inside natural 50-way anchor candidate sets:

- benefit MAE `= 0.1367`
- blended acceptance MAE in reranker-internal evaluation `= 0.3279`
- anchor `hit@10 = 0.7138`
- anchor conditional mean rank `= 6.05`
- weighted NDCG@10 `= 0.3575`

Important caveat:

- the final `v2.2` story should still be read from the offline end-to-end evaluation below
- the reranker-internal acceptance MAE is slightly weaker than the final top-ranked blended acceptance MAE reported by the end-to-end evaluator

The current rebuilt path does **not** depend on the older source-biased mixed-catalog final scorer. The live `v2.2` recommender instead uses:

- anchor retrieval
- anchor reranking
- anchor-conditioned public transfer scoring

Important caveat:

- do not treat `final_ranker_config.json` as the source of truth for the hardened `v2.2` deployment path

### Experimental Step - Preview the `v2` recommender

```bash
python src/v2/inference/recommend.py --split test --limit 3 --top-k 5 --candidate-k 50 --explicit-goal focus
```

This loads the current experimental `v2` stack and prints end-to-end recommendations for rebuilt decision rows using:

- precomputed context embeddings
- user embeddings
- the query tower
- anchor retrieval
- the anchor reranker
- support-aware public transfer

It does not write new training artifacts. It is mainly a debugging and sanity-check entrypoint for the rebuilt explicit-goal path.

### Experimental Step - Evaluate the `v2` recommender end to end

```bash
python src/v2/eval/offline_eval.py --split test --candidate-k 50
```

This writes:

- `models/rebuild/offline_eval_v2.json`
- `data/processed/rebuild/offline_eval_v2_predictions.parquet`

It reports the rebuilt contract end to end:

- anchor retrieval quality
- anchor reranking quality
- tier-aware weighted retrieval metrics
- factual-positive diagnostics where same-song anchors exist
- support-aware public transfer and source balance

Current verified experimental result on the held-out test split:

- query `recall@20 = 0.7460`
- query `recall@50 = 0.8810`
- query weighted `recall@20 = 0.3011`
- query weighted NDCG@10 `= 0.1688`
- rerank `hit@10 = 0.7170`
- rerank conditional mean rank `= 5.93`
- rerank weighted NDCG@10 `= 0.2306`
- factual-positive conditional ranks:
  - query `= 19.00`
  - rerank `= 10.13`
- benefit MAE `= 0.1262`
- blended acceptance MAE `= 0.2947`
- top-1 source distribution:
  - `situnes = 0.3601`
  - `spotify = 0.6399`
- public-transfer-supported share `= 0.6399`

Important caveat:

- the primary rebuilt contract now passes after calibrating the public-transfer threshold to `0.72`
- exact-song rerank remains a legacy diagnostic and is still weak
- use this path for the rebuilt research story, not as a replacement headline for the graded HMM + DQN pipeline

### Experimental Step - Check whether `v2` is actually ready

```bash
python src/v2/eval/check_readiness.py
```

Optional strict mode:

```bash
python src/v2/eval/check_readiness.py --fail-on-not-ready
```

This writes:

- `models/rebuild/v2_readiness.json`

It is the explicit pass/fail gate for the hardened anchor-first rebuild.

Current verified readiness status:

- primary rebuilt contract: `ready = true`
- hard gates currently passing:
  - tier-aware anchor retrieval
  - anchor rerank quality
  - benefit MAE
  - blended acceptance MAE
  - source balance
  - public transfer activation
- soft legacy caveat:
  - exact-song rerank `hit@10` is still below the old sanity threshold

### Step 2 - Train the HMM

```bash
python src/hmm/hmm_train.py
```

This trains the 3-state, 60-observation HMM and saves:

- `models/hmm.npz`
- `models/hmm_metrics.json`
- `models/hmm_convergence.csv`

### Step 3 - Precompute beliefs and 16D state vectors

```bash
python src/hmm/precompute_beliefs.py
```

This saves belief-state and state-vector artifacts used by RL training and evaluation.

### Step 4 - Build synthetic augmentation

```bash
python src/data/generate_synthetic.py
```

This step is optional, but the default training workflow expects synthetic artifacts to exist if you want mixed real + synthetic training.

### Step 5 - Train the DQN

Recommended command:

```bash
python train_agent.py --synthetic-weight 0.25
```

Other supported flags:

```bash
python train_agent.py --synthetic-weight 0.25 --reward-mode expected --alpha 0.7 --beta 0.3
```

### Step 6 - Evaluate the system

```bash
python eval_agent.py
```

Interactive mode:

```bash
python eval_agent.py --interactive
```

This writes `models/eval_report.json`.

### Step 7 - Run the presentation demo

```bash
python demo.py
```

This prints the main comparison scenarios used in the presentation story:

- same physical state, different emotion
- same context, different preference
- same user, different scenarios
- same HR / mood signal, different movement evidence

### Step 8 - Run the multi-session simulation

```bash
python simulate_user.py
```

---

## Key Design Decisions

**Why only 3 hidden states?**  
SiTunes is heavily imbalanced toward sedentary contexts. Finer latent-state granularity looked attractive on paper, but it was not stable or defensible on this dataset.

**Why keep time, weather, speed, and preference outside the HMM?**  
Putting every variable into the HMM observation encoding creates unnecessary sparsity. The HMM works better as a wrist-signal belief model, while explicit context and taste features remain visible to the DQN.

**Why a 16D state instead of the original 5D state?**  
The old state ignored weather, speed, HR summaries, pre-emotions, Stage 1 preferences, and step summaries. The 16D state lets the policy separate cases that looked identical before, such as a stressed still person versus a balanced still person, or elevated activation while sedentary versus elevated activation during movement.

**Why keep HMM + DQN instead of switching to a contextual bandit now?**  
The course expects from-scratch HMM and DQN components. The current implementation keeps that backbone while making the offline decision problem more honest and better conditioned.

**Why a two-part reward?**  
A song can be helpful but disliked, or liked but not helpful. Keeping emotional benefit and acceptance separate makes the system easier to reason about and easier to evaluate honestly.

**Why is PMEmo not a main policy-training source?**  
PMEmo helps retrieval and music-affect scoring, but it is not trusted as strong action-bucket supervision. The current pipeline only uses it as a soft retrieval signal.

**Why not claim synthetic data solves sparse action buckets?**  
Synthetic augmentation helps context coverage and regularization. It does not create real evidence for buckets that barely appear in SiTunes.

---

## Results

Current end-to-end results on the held-out test set (5 users, 311 interactions):

| Metric | DQN | State-Prior | Always-7 | Random |
|--------|-----|-------------|----------|--------|
| Combined reward | +0.1639 | **+0.1641** | +0.0821 | +0.0781 |
| Emotion benefit | +0.0465 | +0.0439 | +0.0104 | - |
| Acceptance | +0.4377 | **+0.4446** | +0.2494 | - |
| Regret (vs oracle) | 0.0021 | **0.0019** | 0.0839 | - |

| HMM Metric | Value |
|------------|-------|
| States used on test | 3 / 3 |
| Mean belief entropy | 0.159 |
| Unique rounded belief vectors | 473 |

What these numbers mean:

- The rebuilt system clearly beats trivial baselines such as `always7` and uniform random.
- The current DQN is presentation-safe and behaviorally richer than the trivial baselines, but it is still slightly below the `state_prior` baseline on held-out combined reward.
- The HMM uses all 3 states, but belief entropy is still modest. This is a coarse latent-state model, not deep mood decoding.
- The two-part reward shows that the system evaluates both emotional improvement and user acceptance.
- The main qualitative win is that the policy now reacts to emotion, taste, and movement signals that the old 5D pipeline ignored.

The strongest presentation results are qualitative:

- Same HR / mood signal + different step evidence -> different bucket (`bucket 5` vs `bucket 7` in the current demo)
- Same context + different taste profile -> different track ranking
- Same user + different scenario -> different belief state, bucket, and explanation

---

## Limitations

1. **Still an offline system**  
   The policy is trained from historical data rather than live online feedback.

2. **Small intervention dataset**  
   Some action buckets are extremely sparse in SiTunes, so low usage of those buckets is often rational rather than a bug.

3. **Exercise learning is weak**  
   Running-like contexts are rare in the cleaned data, so exercise-specific claims should be kept modest.

4. **The HMM captures coarse latent context-energy structure, not deep psychology**  
   The HMM is useful, but it should not be oversold as direct mood recognition from wrist data.

5. **Ambient-first support exists, but full passive inference is not separately benchmarked yet**  
   The state interface supports no-check-in usage through `pre_emotion_mask`, but most real training rows still include self-report.

6. **PMEmo remains auxiliary**  
   It improves retrieval and music-affect guidance, not the core causal story of "this intervention helped this user in this situation."

7. **PMEmo dynamic features are retrieval-only and currently subtle at the whole-catalog level**  
   The new dynamic affect curves change PMEmo-to-PMEmo ordering, but top-ranked public outputs are still often dominated by Spotify because that catalog is far larger and has richer metadata coverage.

---

## References

- Li, J. et al. (2024). *SiTunes: A Situational Music Recommendation Dataset with Physiological and Psychological Signals.*
- Zhang, K. et al. (2018). *The PMEmo Dataset for Music Emotion Recognition.*
- van Hasselt, H. et al. (2016). *Deep Reinforcement Learning with Double Q-learning.*
- Russell, J. A. (1980). *A circumplex model of affect.*
