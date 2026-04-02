# CS4100-Project-

# Ambient Music Recommendation Agent
### CS4100 Artificial Intelligence - Northeastern University

An AI system that recommends music based on the user's current biometrics, emotional state, and environment instead of only past listening history. A Hidden Markov Model (HMM) infers a coarse latent context-energy belief state from wrist sensor data, and a Deep Q-Network (DQN) chooses the music bucket most likely to help in the current situation. A deterministic retrieval layer then ranks concrete songs using the user's baseline preferences.

This project is intentionally built with from-scratch NumPy and PyTorch implementations for the core AI components. It does not use `hmmlearn`, `stable-baselines3`, or transformer-based models.

---

## The Problem

Spotify and Apple Music know what you liked last week. They do not know that you are tense at 2pm, low-energy in the evening, or trying to stay focused during a study block. Most recommenders ignore the most relevant signal: how you feel right now.

This project asks a harder question:

**Given your current biometric signals, current environment, and baseline taste profile, what kind of music is most likely to help right now?**

---

## Our Solution

A four-part ambient recommendation pipeline:

1. **Sense the current situation**  
   A 3-state HMM reads a 30-step wrist window and produces a belief distribution over coarse latent context-energy states.

2. **Remember the person**  
   Stage 1 of SiTunes is used to build a compact user preference profile from baseline ratings.

3. **Choose the right intervention**  
   A Double DQN receives a 14-dimensional state vector and chooses one of 8 music mood buckets.

4. **Retrieve real tracks**  
   A deterministic music library ranks concrete songs from SiTunes, Spotify, and PMEmo using bucket fit, scenario fit, and user preference fit.

The key idea is still POMDP-style: the user's internal state is partially hidden, so the system reasons over a belief state rather than pretending wrist data directly reveals mood.

---

## Architecture

```text
Biometrics + Context + Optional Check-In
 [HR, intensity, activity, weather, speed, pre-emotion]
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
    | DQN Agent (14D state, 8 actions) |
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

PMEmo is not treated as reliable hard bucket supervision for the policy. In the current system it is used more cautiously: valence/arousal guidance, EDA impact hints, and soft bucket hints built from PMEmo-internal tempo percentiles.

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

The DQN receives a 14-dimensional state vector:

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
]
```

### What the added features do

- `weather_norm`, `speed_norm`, and `hr_*`: explicit context and biometric features
- `pre_valence_norm`, `pre_arousal_norm`: the user's current reported emotion when available
- `pre_emotion_mask`: tells the model whether those emotion values are real self-report or passive / fallback inputs
- `user_valence_pref_norm`, `user_energy_pref_norm`: compact baseline taste profile from Stage 1

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
|   |-- TA_DEFENSE_GUIDE.md
|   `-- TA_QA_SHEET.md
|-- models/
|   |-- hmm.npz
|   |-- reward_model.json
|   |-- agent.pt
|   `-- eval_report.json
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

### Step 2 - Train the HMM

```bash
python src/hmm/hmm_train.py
```

This trains the 3-state, 60-observation HMM and saves:

- `models/hmm.npz`
- `models/hmm_metrics.json`
- `models/hmm_convergence.csv`

### Step 3 - Precompute beliefs and 14D state vectors

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

**Why a 14D state instead of the original 5D state?**  
The old state ignored weather, speed, HR summaries, pre-emotions, and Stage 1 preferences. The 14D state lets the policy separate cases that looked identical before, such as a stressed still person versus a balanced still person.

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
| Combined reward | +0.1630 | **+0.1645** | +0.0805 | +0.0778 |
| Emotion benefit | +0.0473 | +0.0438 | +0.0083 | - |
| Acceptance | +0.4329 | **+0.4462** | +0.2490 | - |
| Regret (vs oracle) | 0.0040 | **0.0024** | 0.0864 | - |

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
- The main qualitative win is that the policy now reacts to emotion and taste features that the old 5D pipeline ignored.

The strongest presentation results are qualitative:

- Same physical state + different mood -> different bucket
- Same context + different taste profile -> different track ranking, sometimes different bucket
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

---

## References

- Li, J. et al. (2024). *SiTunes: A Situational Music Recommendation Dataset with Physiological and Psychological Signals.*
- Zhang, K. et al. (2018). *The PMEmo Dataset for Music Emotion Recognition.*
- van Hasselt, H. et al. (2016). *Deep Reinforcement Learning with Double Q-learning.*
- Russell, J. A. (1980). *A circumplex model of affect.*

---

## Additional Documentation

For the code-accurate technical walkthrough and TA-defense prep, see:

- [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md)
- [docs/TA_DEFENSE_GUIDE.md](docs/TA_DEFENSE_GUIDE.md)
- [docs/TA_QA_SHEET.md](docs/TA_QA_SHEET.md)
