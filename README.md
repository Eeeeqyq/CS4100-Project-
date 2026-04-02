# Ambient Music Recommendation Agent
<<<<<<< Updated upstream
### CS4100 Artificial Intelligence | Northeastern University

This project is a from-scratch ambient music recommendation system built around a POMDP-style pipeline:
=======
### CS4100 Artificial Intelligence - Northeastern University

An AI system that recommends music based on your **physiological state and environment** - not just your listening history. A Hidden Markov Model infers a latent belief state from wrist sensor data, and a Deep Q-Network learns which music bucket is most likely to improve the user's short-term emotional response.
>>>>>>> Stashed changes

1. a 3-state HMM infers a latent belief state from wrist-sensor context
2. a Double DQN chooses a music mood bucket from that belief state
3. a deterministic retrieval layer returns concrete tracks from the cleaned music catalog

The main objective is not generic taste prediction. The objective is to recommend music that is appropriate for the user's current physical context and is more likely to improve short-term emotional response, using the pre/post valence and arousal labels available in SiTunes.

<<<<<<< Updated upstream
The repository now uses a script-backed, split-aware pipeline. The notebooks are exploratory only.
=======
Spotify and Apple Music know what you liked last week. They do not know that you are stressed at 11pm, running at 7am, or stuck in an afternoon focus session. Most recommendation systems ignore the most relevant signal: **how you feel right now**.
>>>>>>> Stashed changes

## Project Goal

The rebuilt project is optimized around three practical outcomes:

<<<<<<< Updated upstream
- `demo.py` should show believable, presentation-ready recommendations
- `simulate_user.py` should produce deterministic multi-session behavior using the same reward model as training
- `eval_agent.py` should report held-out policy quality using action-dependent offline metrics rather than historical reward replay alone

## Core Design
=======
1. **HMM (Hidden Markov Model)** - reads a 30-step wrist sensor window and infers a hidden latent state across 3 coarse context-energy categories
2. **DQN (Deep Q-Network)** - takes that corrected belief state plus explicit context features and learns which music mood bucket to recommend to maximize expected emotional lift, trained against real pre/post emotion outcomes through an offline reward model

The key insight: mood is hidden. You cannot directly observe it from wrist data alone. The POMDP-style framing (belief state -> action policy) is the principled way to handle uncertainty in the user's internal state.
>>>>>>> Stashed changes

### Stage 1: HMM

The HMM operates on wrist-only observations. Each interaction contains a 30-step wrist window, and each timestep is encoded as:

```text
<<<<<<< Updated upstream
obs = intensity_bucket * 5 + activity_remapped
=======
Wrist Signals + Context
  [30-step intensity/activity window, time of day]
              |
              v
    +------------------+
    |   HMM (3 states) |  <- trained with Baum-Welch EM
    | wrist-only obs   |
    +--------+---------+
             |
             | corrected belief state (3-dim probability vector)
             v
    +------------------+
    |   DQN Agent      |  <- trained with Double DQN
    |  5-dim state     |     on one-step offline contexts
    |  8 action buckets|
    +--------+---------+
             |
             | recommended music bucket
             v
    +------------------+
    |  Music Library   |  <- SiTunes + PMEmo + Spotify Kaggle
    | deterministic    |
    | context-aware    |
    +------------------+
>>>>>>> Stashed changes
```

This gives a 20-value observation space:

<<<<<<< Updated upstream
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
=======
| ID | State | Interpretation |
|----|-------|----------------|
| 0 | low-energy | sedentary / recovery / lying-still contexts |
| 1 | moderate | transitioning / walking / mixed activity contexts |
| 2 | high-energy | high-intensity / running contexts |

These are deliberately coarse. The rebuilt project does **not** claim that the HMM cleanly recovers six psychological moods from wrist data alone.
>>>>>>> Stashed changes

The hidden state space is intentionally small:

| State | Meaning |
|---|---|
| `S0` | low-energy |
| `S1` | moderate |
| `S2` | high-energy |

Time of day is not part of the HMM emission space anymore. It is kept as an explicit downstream feature so the HMM is less likely to collapse into a time-of-day classifier.

### Stage 2: DQN

<<<<<<< Updated upstream
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
=======
| Dataset | Role | Size |
|---------|------|------|
| [SiTunes](https://github.com/JiayuLi-997/SiTunes_dataset) | Primary training + reward signal | 2,006 total interactions, 30 users; 1,406 usable Stage 2/3 reward interactions after cleaning |
| [PMEmo](https://github.com/HuiZhangDB/PMEmo) | Auxiliary music emotion catalog | 736 tracks |
| [Spotify Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) | Extended music library | ~89,500 tracks |

SiTunes is the core dataset. It provides wrist physiological signals, environmental context, and most importantly, **pre/post emotional ratings** (valence + arousal) for each listening session. Those ratings are the basis of the reward signal.

**Reward function:**
```python
score = 0.7 * (emo_post_valence - emo_pre_valence) \
      + 0.3 * (emo_post_arousal - emo_pre_arousal)

reward =  1 if score >  0.1
reward =  0 if abs(score) <= 0.1
reward = -1 if score < -0.1
```

**Observation encoding (0-19):**
```text
obs = intensity_bucket * 5 + activity_remapped

intensity_bucket:
  0 = <10
  1 = 10-30
  2 = 30-80
  3 = >=80

activity_remapped:
  raw 0 -> 0  still
  raw 1 -> 1  transitioning
  raw 2 -> 2  walking
  raw 3 -> 0  missing -> still
  raw 4 -> 3  lying
  raw 5 -> 4  running
```

Time of day is no longer part of the HMM observation space. It is kept as an explicit downstream feature for the DQN.

**DQN state vector (5D):**
```text
state = [belief_0, belief_1, belief_2, time_norm, activity_norm]
>>>>>>> Stashed changes
```

### Reward Signal

The supervised emotional signal comes from SiTunes pre/post self-reports:

<<<<<<< Updated upstream
```python
score = 0.7 * (emo_post_valence - emo_pre_valence) \
      + 0.3 * (emo_post_arousal - emo_pre_arousal)

reward =  1   if score >  0.10
reward =  0   if abs(score) <= 0.10
reward = -1   if score < -0.10
=======
```text
ambient-music-agent/
|-- data/
|   |-- raw/
|   |   |-- situnes/          <- SiTunes must end up under data/raw/situnes/SiTunes/
|   |   |-- pmemo/            <- PMEmo files go here
|   |   `-- spotify_kaggle/   <- dataset.csv goes here
|   `-- processed/            <- auto-generated by preprocessing scripts
|-- models/                   <- auto-generated by training / evaluation scripts
|-- notebooks/
|   |-- 01_clean_situnes.ipynb
|   |-- 02_clean_pmemo.ipynb
|   `-- 03_clean_spotify.ipynb
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
|   `-- music/
|       `-- music_library.py
|-- train_agent.py
|-- eval_agent.py
|-- demo.py
|-- simulate_user.py
`-- requirements.txt
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
### 2. Preprocess data
=======
**requirements.txt:**
```text
numpy>=1.24
pandas>=2.0
scipy>=1.10
torch>=2.0
```
>>>>>>> Stashed changes

Full preprocessing:

<<<<<<< Updated upstream
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

=======
**SiTunes (required):**
Clone or copy the dataset so that the repo root ends up here:

```bash
git clone https://github.com/JiayuLi-997/SiTunes_dataset data/raw/situnes/SiTunes
```

The code expects:

- `data/raw/situnes/SiTunes/Stage1/`
- `data/raw/situnes/SiTunes/Stage2/`
- `data/raw/situnes/SiTunes/Stage3/`
- `data/raw/situnes/SiTunes/music_metadata/`

**PMEmo (optional - improves library diversity):**
Place the PMEmo files so these paths exist:

- `data/raw/pmemo/metadata.csv`
- `data/raw/pmemo/annotations/static_annotations.csv`
- `data/raw/pmemo/annotations/static_annotations_std.csv`
- `data/raw/pmemo/features/static_features.csv`

**Spotify Kaggle (optional - greatly expands the library):**
Download `dataset.csv` from: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

Place it here:

- `data/raw/spotify_kaggle/dataset.csv`

---

## Running the Project

> Always run all commands from the **project root** directory.

### Step 1 - Clean the data
The canonical preprocessing pipeline is now script-backed:

```bash
python -m src.data.preprocess
```

If PMEmo or Spotify are not available:

```bash
python -m src.data.preprocess --skip-pmemo --skip-spotify
```

Outputs land in `data/processed/`.

### Step 2 - Train the HMM
>>>>>>> Stashed changes
```bash
python src/hmm/hmm_train.py
```

<<<<<<< Updated upstream
This writes:

- `models/hmm.npz`
- `models/hmm_convergence.csv`
- `models/hmm_metrics.json`

### 4. Precompute corrected beliefs and state vectors

=======
- trains a 3-state wrist-only HMM using Baum-Welch EM
- uses informed emission initialization and diagonal-biased transitions
- runs 3 restarts and keeps the best model
- calibrates corrected belief behavior on the validation split
- outputs: `models/hmm.npz`, `models/hmm_convergence.csv`, `models/hmm_metrics.json`

### Step 3 - Precompute corrected belief states
>>>>>>> Stashed changes
```bash
python src/hmm/precompute_beliefs.py
```

<<<<<<< Updated upstream
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

=======
- converts each cleaned interaction into a corrected belief vector and 5D DQN state
- writes full-dataset and split-specific `.npy` artifacts
- required before RL training

### Step 4 - Generate synthetic augmentation (optional but supported)
>>>>>>> Stashed changes
```bash
python src/data/generate_synthetic.py
```

<<<<<<< Updated upstream
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

=======
- fits the shared hierarchical reward model
- generates reality-anchored synthetic contexts from the train split
- outputs: `models/reward_model.json`, `data/processed/synthetic_clean.csv`, `data/processed/synthetic_state_vectors.npy`, `data/processed/synthetic_report.json`

### Step 5 - Train the RL agent
>>>>>>> Stashed changes
```bash
python train_agent.py --synthetic-weight 0.35
```

<<<<<<< Updated upstream
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

=======
- requires precomputed state vectors
- trains Double DQN on a one-step offline environment
- uses expected reward from the hierarchical reward model
- supports turning synthetic augmentation off with `--synthetic-weight 0`
- outputs: `models/agent.pt`, `models/training_log.csv`, `models/training_summary.json`

### Step 6 - Evaluate the agent
>>>>>>> Stashed changes
```bash
python eval_agent.py
```

<<<<<<< Updated upstream
Optional interactive evaluation mode:

=======
- reports held-out test metrics
- compares DQN against `state_prior`, `always7`, and uniform random expected reward
- writes `models/eval_report.json`

Optional:
>>>>>>> Stashed changes
```bash
python eval_agent.py --interactive
```

<<<<<<< Updated upstream
Presentation demo:

```bash
python demo.py
```
=======
### Step 7 - Run the demo
```bash
python demo.py
```

Runs fixed presentation scenarios showing:

- wrist context
- HMM belief state
- chosen bucket
- ranked track examples

### Step 8 - Run the simulation
```bash
python simulate_user.py
```

Runs deterministic multi-session user profiles using the same reward model used by training and evaluation.
>>>>>>> Stashed changes

Deterministic multi-session simulation:

```bash
python simulate_user.py
```

<<<<<<< Updated upstream
`eval_agent.py` writes:

- `models/eval_report.json`

## What Each Top-Level Script Does

### `train_agent.py`
=======
**Why HMM instead of direct classification?**  
Mood and context-response are hidden variables. Wrist signals are noisy and ambiguous. The HMM gives an uncertainty-aware belief state instead of a brittle hard label, which is a much better fit for this problem.

**Why 3 hidden states instead of 6?**  
The original six-state mood story sounded better than the data supported. In practice, SiTunes wrist data is much better at separating coarse activity-energy regimes than fine-grained internal emotions. A 3-state model is materially more defensible and more stable.

**Why wrist-only HMM with time explicit downstream?**  
When time was baked into the HMM emissions, the model leaned too heavily on time-of-day patterns. The rebuild keeps time as a downstream feature for the DQN, which preserves useful context without letting it dominate the latent state model.

**Why Double DQN?**  
Standard DQN overestimates Q-values because it uses the same network to both select and evaluate actions. Double DQN separates those roles with online and target networks, improving stability on a small offline dataset.

**Why use a hierarchical reward model for offline RL?**  
The old historical-rollout framing mostly rewarded matching what happened in the logged data. The rebuilt approach makes the chosen action matter by estimating context-action reward with hierarchical shrinkage over real train interactions.

**Why deterministic retrieval?**  
For demos, evaluation, and reproducibility, a recommendation system should not surface different tracks each run for the same context. The retrieval layer is now deterministic and context-aware.
>>>>>>> Stashed changes

- loads `interactions_clean.csv` and `state_vectors.npy`
- uses train users only for fitting the reward model
- optionally mixes in synthetic contexts with a lower sampling weight
- trains a Double DQN on one-step contexts
- uses validation expected reward for model selection
- reports held-out test metrics after training

### `eval_agent.py`

<<<<<<< Updated upstream
- loads the trained HMM, DQN, reward model, and music library
- evaluates the policy on held-out test users
- compares DQN against:
  - `state_prior`
  - `always7`
  - uniform random expected reward
- prints a scenario gallery using the same model artifacts
=======
Latest local rebuild results from the current generated artifacts:

| Metric | Value |
|--------|-------|
| HMM states used | 3 / 3 |
| Validation belief entropy mean | 0.166 |
| Test belief entropy mean | 0.146 |
| Unique rounded belief vectors | train 210 / val 105 / test 102 |
| Held-out DQN expected reward | +0.2692 |
| State-prior baseline | +0.2459 |
| Always-7 baseline | +0.1566 |
| Random uniform expected reward | +0.0691 |

These numbers come from the latest local rebuild and will change if the models are retrained.
>>>>>>> Stashed changes

### `demo.py`

- runs three fixed presentation scenarios
- shows the corrected belief vector
- shows the selected bucket
- prints top-ranked example tracks

<<<<<<< Updated upstream
### `simulate_user.py`
=======
| Member | Contribution |
|--------|-------------|
| Person 1 | Data pipeline and HMM implementation |
| Person 2 | RL environment, reward model, DQN training |
| Person 3 | Music library, evaluation, demo, simulation |
>>>>>>> Stashed changes

- runs fixed user profiles across multiple sessions
- samples rewards and mood deltas from the shared reward model
- updates session mood sequentially
- prints deterministic, ASCII-safe output for presentation/demo use

## Auxiliary Catalog Handling

<<<<<<< Updated upstream
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
=======
- Li, J. et al. (2024). *SiTunes: A Situational Music Recommendation Dataset with Physiological and Psychological Signals.* CHIIR 2024.
- Zhang, K. et al. (2018). *The PMEmo Dataset for Music Emotion Recognition.* ICMR 2018.
- van Hasselt, H. et al. (2016). *Deep Reinforcement Learning with Double Q-learning.* AAAI 2016.
- Russell, J.A. (1980). *A circumplex model of affect.* Journal of Personality and Social Psychology.
>>>>>>> Stashed changes
