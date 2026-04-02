# Ambient Music Recommendation Agent
### CS4100 Artificial Intelligence | Northeastern University

This project is a from-scratch ambient music recommendation system built around a POMDP-style pipeline:

1. a 3-state HMM infers a latent belief state from wrist-sensor context
2. a Double DQN chooses a music mood bucket from that belief state
3. a deterministic retrieval layer returns concrete tracks from the cleaned music catalog

The main objective is not generic taste prediction. The objective is to recommend music that is appropriate for the user's current physical context and is more likely to improve short-term emotional response, using the pre/post valence and arousal labels available in SiTunes.

The repository now uses a script-backed, split-aware pipeline. The notebooks are exploratory only.

## Project Goal

The rebuilt project is optimized around three practical outcomes:

- `demo.py` should show believable, presentation-ready recommendations
- `simulate_user.py` should produce deterministic multi-session behavior using the same reward model as training
- `eval_agent.py` should report held-out policy quality using action-dependent offline metrics rather than historical reward replay alone

## Core Design

### Stage 1: HMM

The HMM operates on wrist-only observations. Each interaction contains a 30-step wrist window, and each timestep is encoded as:

```text
obs = intensity_bucket * 5 + activity_remapped
```

This gives a 20-value observation space:

- intensity bucket:
  - `0` if intensity `< 10`
  - `1` if intensity `< 30`
  - `2` if intensity `< 80`
  - `3` otherwise
- activity remap:
  - raw `0 -> 0` still
  - raw `1 -> 1` transitioning
  - raw `2 -> 2` walking
  - raw `3 -> 0` missing treated as still
  - raw `4 -> 3` lying
  - raw `5 -> 4` running

The hidden state space is intentionally small:

| State | Meaning |
|---|---|
| `S0` | low-energy |
| `S1` | moderate |
| `S2` | high-energy |

Time of day is not part of the HMM emission space anymore. It is kept as an explicit downstream feature so the HMM is less likely to collapse into a time-of-day classifier.

### Stage 2: DQN

The DQN receives a 5D state vector:

```text
[belief_0, belief_1, belief_2, time_norm, activity_norm]
```

where:

- `belief_*` is the corrected HMM posterior
- `time_norm = time_bucket / 2.0`
- `activity_norm = activity_remapped / 4.0`

The action space is 8 music buckets shared across SiTunes, PMEmo, and Spotify:

| Bucket | Label |
|---|---|
| `0` | `dark-slow` |
| `1` | `dark-fast` |
| `2` | `intense-slow` |
| `3` | `aggressive` |
| `4` | `chill-study` |
| `5` | `indie` |
| `6` | `soulful` |
| `7` | `energetic` |

The bucketing rule is:

```text
valence_level = 0 if valence < 0.33 else 1
energy_level  = 0 if energy  < 0.40 else 1
tempo_level   = 0 if tempo   < 100  else 1

bucket = valence_level * 4 + energy_level * 2 + tempo_level
```

### Reward Signal

The supervised emotional signal comes from SiTunes pre/post self-reports:

```python
score = 0.7 * (emo_post_valence - emo_pre_valence) \
      + 0.3 * (emo_post_arousal - emo_pre_arousal)

reward =  1   if score >  0.10
reward =  0   if abs(score) <= 0.10
reward = -1   if score < -0.10
```

This same reward definition is used during preprocessing and for training targets.

## What Changed In The Rebuild

### Canonical preprocessing moved into `src/data/`

The cleaning logic is no longer notebook-only. The canonical preprocessing entrypoint is:

```bash
python -m src.data.preprocess
```

That script:

- parses and cleans SiTunes Stage 1, Stage 2, and Stage 3
- aligns Stage 2/3 interactions with wrist and env context by `inter_id`
- recomputes reward labels from emotion deltas
- writes a fixed user-level split manifest
- emits a machine-readable dataset audit report
- cleans PMEmo and Spotify if those raw datasets are present

### Split discipline is user-level

All modeling uses a fixed user split:

- train: 20 users
- validation: 5 users
- test: 5 users

The split is stored in `data/processed/split_manifest.json` and reused throughout the pipeline.

### HMM training is train-only and calibrated on validation

`src/hmm/hmm_train.py`:

- trains only on train-split wrist sequences
- uses informed emission initialization
- uses diagonal-biased transition initialization
- reorders states after training by empirical physical-energy profile
- calibrates posterior correction on the validation split
- saves metadata needed for downstream corrected beliefs

### RL training uses an action-dependent offline objective

The environment in `src/rl_agent/environment.py` is one-step and offline.

It does not replay historical transitions as if the chosen action had happened. Instead, it scores actions with a hierarchical reward model built from real train data. The reward model shrinks across:

- global reward counts
- action-level counts
- `(state, action)`
- `(state, time, activity, action)`

This is what makes `eval_agent.py`, `demo.py`, and `simulate_user.py` share the same action-dependent scoring logic.

### Synthetic augmentation is optional and lower-weight

`src/data/generate_synthetic.py` creates synthetic contexts anchored to the train split:

- it samples from real train user/session structure
- it preserves train-like time transitions
- it reuses real wrist templates
- it samples rewards and mood deltas from the shared reward model

Synthetic rows are available for training augmentation, but they are not used for headline evaluation.

### Retrieval is deterministic

`src/music/music_library.py` no longer does popularity-first random sampling. It:

- filters by chosen action bucket
- reranks by bucket fit
- applies context-specific boosts for modes like `focus`, `wind_down`, `exercise`, and `uplift`
- uses acousticness, instrumentalness, speechiness, danceability, genre, popularity, and source-specific boosts

## Repository Layout

```text
.
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|-- notebooks/
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
|   |-- music/
|   |   `-- music_library.py
|   `-- rl_agent/
|       |-- dqn_agent.py
|       |-- environment.py
|       `-- reward_model.py
|-- train_agent.py
|-- eval_agent.py
|-- demo.py
`-- simulate_user.py
```

## Raw Data Placement

The code expects the raw datasets in these locations:

| Dataset | Expected path |
|---|---|
| SiTunes | `data/raw/situnes/SiTunes/` |
| PMEmo | `data/raw/pmemo/` |
| Spotify Kaggle | `data/raw/spotify_kaggle/dataset.csv` |

SiTunes is required. PMEmo and Spotify are optional auxiliary retrieval catalogs.

## Canonical Pipeline

Run everything from the project root.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess data

Full preprocessing:

```bash
python -m src.data.preprocess
```

If PMEmo or Spotify are not available:

```bash
python -m src.data.preprocess --skip-pmemo --skip-spotify
```

This writes:

- `data/processed/stage1_clean.csv`
- `data/processed/stage2_clean.csv`
- `data/processed/stage3_clean.csv`
- `data/processed/interactions_clean.csv`
- `data/processed/wrist_obs_all.npy`
- `data/processed/wrist_stage2_obs.npy`
- `data/processed/wrist_stage3_obs.npy`
- `data/processed/music_situnes_clean.csv`
- `data/processed/music_pmemo_clean.csv` if PMEmo is available and not skipped
- `data/processed/music_spotify_clean.csv` if Spotify is available and not skipped
- `data/processed/split_manifest.json`
- `data/processed/dataset_audit.json`

### 3. Train the HMM

```bash
python src/hmm/hmm_train.py
```

This writes:

- `models/hmm.npz`
- `models/hmm_convergence.csv`
- `models/hmm_metrics.json`

### 4. Precompute corrected beliefs and state vectors

```bash
python src/hmm/precompute_beliefs.py
```

This writes:

- `data/processed/belief_states.npy`
- `data/processed/state_vectors.npy`
- `data/processed/belief_states_train.npy`
- `data/processed/belief_states_val.npy`
- `data/processed/belief_states_test.npy`
- `data/processed/state_vectors_train.npy`
- `data/processed/state_vectors_val.npy`
- `data/processed/state_vectors_test.npy`

### 5. Generate synthetic augmentation

```bash
python src/data/generate_synthetic.py
```

This writes:

- `models/reward_model.json`
- `data/processed/synthetic_clean.csv`
- `data/processed/synthetic_state_vectors.npy`
- `data/processed/synthetic_report.json`

Note:

- `train_agent.py` can create `models/reward_model.json` on its own if it is missing
- synthetic files are only needed if you want training augmentation

### 6. Train the DQN

Train with the default synthetic weighting:

```bash
python train_agent.py --synthetic-weight 0.35
```

Train on real data only:

```bash
python train_agent.py --synthetic-weight 0
```

This writes:

- `models/agent.pt`
- `models/training_log.csv`
- `models/training_summary.json`

### 7. Evaluate and present

Held-out evaluation report:

```bash
python eval_agent.py
```

Optional interactive evaluation mode:

```bash
python eval_agent.py --interactive
```

Presentation demo:

```bash
python demo.py
```

Deterministic multi-session simulation:

```bash
python simulate_user.py
```

`eval_agent.py` writes:

- `models/eval_report.json`

## What Each Top-Level Script Does

### `train_agent.py`

- loads `interactions_clean.csv` and `state_vectors.npy`
- uses train users only for fitting the reward model
- optionally mixes in synthetic contexts with a lower sampling weight
- trains a Double DQN on one-step contexts
- uses validation expected reward for model selection
- reports held-out test metrics after training

### `eval_agent.py`

- loads the trained HMM, DQN, reward model, and music library
- evaluates the policy on held-out test users
- compares DQN against:
  - `state_prior`
  - `always7`
  - uniform random expected reward
- prints a scenario gallery using the same model artifacts

### `demo.py`

- runs three fixed presentation scenarios
- shows the corrected belief vector
- shows the selected bucket
- prints top-ranked example tracks

### `simulate_user.py`

- runs fixed user profiles across multiple sessions
- samples rewards and mood deltas from the shared reward model
- updates session mood sequentially
- prints deterministic, ASCII-safe output for presentation/demo use

## Auxiliary Catalog Handling

### PMEmo

PMEmo does not provide Spotify-style `energy` and `tempo` fields directly. The preprocessing script estimates transferable energy and tempo proxies from overlapping acoustic descriptors shared with SiTunes, then derives PMEmo action buckets from those inferred values plus PMEmo valence/arousal information.

PMEmo is used only to expand retrieval diversity. It is not part of the core reward-learning signal.

### Spotify

Spotify is used only as a retrieval catalog. The project does not call the live Spotify API, and the DQN is not trained on Spotify interaction behavior.

## Reproducibility Notes

- All major scripts use fixed seeds.
- The split manifest is deterministic.
- Generated artifacts live in `data/processed/` and `models/`.
- Raw data, generated artifacts, and local instruction files are ignored by git.

If you want to rebuild from scratch, delete the generated contents of `data/processed/` and `models/`, then rerun the canonical pipeline.

## Limitations

1. Wrist data is still a weak proxy for internal affect.
   The HMM is better interpreted as a latent physical-context model than as a direct psychological mood classifier.

2. Evaluation is still offline.
   The reward model is action-dependent and split-aware, but it is still a proxy learned from historical data rather than live interactive feedback.

3. Synthetic data is augmentation, not truth.
   It should improve coverage, not define headline performance.

4. PMEmo and Spotify improve retrieval breadth, not the core training signal.

## Out Of Scope

- live external APIs during inference
- LLMs or transformers
- external HMM or RL frameworks such as `hmmlearn` or `stable-baselines3`
- raw audio feature extraction inside the learning loop
- production auth, storage, or deployment infrastructure
