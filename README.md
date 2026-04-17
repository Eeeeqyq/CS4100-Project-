# CS4100-Project-

# Ambient Music Recommendation Agent
### CS4100 Artificial Intelligence | Northeastern University

This repo contains two music recommendation systems built around the same problem:

> given the user's current context and baseline taste, recommend music that is likely to help right now, not just music they liked before

The **current recommended system** is the rebuilt **`v2.2`** pipeline under `src/v2/`.

The original **HMM + DQN** pipeline is still included as the earlier course approach and baseline/original design.

The core AI components are implemented from scratch with NumPy and PyTorch. The project does not rely on `hmmlearn`, `stable-baselines3`, or transformer-based recommender models.

---

## What To Use

If you are new to the repo, use **`v2.2`**.

`v2.2` is:
- explicit-goal
- anchor-first
- support-aware about public-song transfer

High-level logic:
1. encode the current context from wrist, environment, and optional self-report
2. encode the user from Stage 1 taste history
3. retrieve strong historical **SiTunes** anchors
4. rerank anchors by predicted **benefit** and **acceptance**
5. transfer to **Spotify / PMEmo** only when that transfer is well-supported

Use the original HMM + DQN pipeline only if you specifically want the older course-design path.

---

## Current Verified `v2.2` Status

Latest saved rebuilt result:
- `ready = true`
- held-out test rows: `311`
- anchor query `recall@20 = 0.7460`
- anchor query weighted `recall@20 = 0.3011`
- anchor rerank `hit@10 = 0.7170`
- anchor rerank mean rank `= 5.9270`
- benefit MAE `= 0.1262`
- blended acceptance MAE `= 0.2947`
- public-transfer-supported share `= 0.6399`
- top-1 source max share `= 0.6399`

Current `v2.2` caveat:
- legacy exact-song rerank is still only a soft diagnostic and remains weak

Source-of-truth artifacts:
- `models/rebuild/offline_eval_v2.json`
- `models/rebuild/v2_readiness.json`

---

## Quick Start For `v2.2`

### Fastest path if rebuilt artifacts already exist

```bash
python eval_v2.py --no-rerun
python demo_v2.py
```

Use this path for:
- checking the latest verified `v2.2` results
- presentation prep
- demo recording

### Full fresh `v2.2` rebuild

```bash
python train_v2.py
python eval_v2.py
python demo_v2.py
```

What the wrappers do:
- `train_v2.py`: full rebuilt Block B stack
- `eval_v2.py`: clean offline summary + readiness output
- `demo_v2.py`: presentation-friendly goal-conditioned demo

Do **not** run `train_v2.py` live during a presentation.

---

## What The Demo Actually Shows

`demo_v2.py` fixes one held-out test context and changes only the explicit goal:
- `focus`
- `wind_down`
- `uplift`
- `movement`

That lets you show:
- goal-dependent recommendation changes
- fallback to SiTunes anchors when public transfer support is weak
- promotion of Spotify when transfer support is strong enough

Recommended presentation commands:

```bash
python eval_v2.py --no-rerun
python demo_v2.py
```

For the full presentation explanation:
- `docs/PRESENTATION_REPORT.md`
- `docs/PRESENTATION_SCRIPT.md`

---

## Systems In This Repo

### 1. Rebuilt `v2.2` system (recommended)

Main runtime surface:
- `train_v2.py`
- `eval_v2.py`
- `demo_v2.py`
- `scripts/build_v2_data.py`
- `src/v2/`

Scientific contract:
- explicit goal is the main interface
- SiTunes is the intervention-supervision source
- Spotify and PMEmo are transfer catalogs, not primary intervention labels
- exact-song recovery is a legacy diagnostic, not the main objective

### 2. Original HMM + DQN system (baseline/original design)

Main runtime surface:
- `train_agent.py`
- `eval_agent.py`
- `demo.py`
- `simulate_user.py`
- `src/data/`
- `src/hmm/`
- `src/rl_agent/`
- `src/music/music_library.py`

High-level logic:
1. preprocess SiTunes / PMEmo / Spotify
2. train a wrist-only HMM
3. convert contexts into a `16D` DQN state
4. train a Double DQN over `8` music buckets
5. retrieve concrete songs deterministically

This path is preserved for continuity and comparison, but it is not the main system we recommend now.

---

## Data

Raw-data roots:
- SiTunes: `data/raw/situnes/SiTunes/`
- PMEmo: `data/raw/pmemo/`
- Spotify Kaggle: `data/raw/spotify_kaggle/dataset.csv`

Requirements:
- **SiTunes is required**
- **PMEmo and Spotify are optional but recommended**

What each dataset is used for in `v2.2`:
- **SiTunes**
  - Stage 1: baseline taste history
  - Stage 2/3: intervention anchors and outcome supervision
- **PMEmo**
  - transfer-side affect cues
  - dynamic valence/arousal contours
  - optional EDA-derived signal
- **Spotify Kaggle**
  - large public transfer catalog

Important point:
- Spotify and PMEmo are used in `v2.2`, but **not** as primary intervention-label sources

---

## Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd CS4100-Project-
```

### 2. Create an environment

Python `3.10+` is recommended.

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

### 4. Place the datasets

**SiTunes**

Expected path:

```text
data/raw/situnes/SiTunes/
```

Matching clone command:

```bash
git clone https://github.com/JiayuLi-997/SiTunes_dataset data/raw/situnes/SiTunes
```

**PMEmo**

Expected root:

```text
data/raw/pmemo/
```

Expected current files include:
- `annotations/static_annotations.csv`
- `annotations/dynamic_annotations.csv`
- `features/static_features.csv`
- `metadata.csv`

**Spotify Kaggle**

Expected file:

```text
data/raw/spotify_kaggle/dataset.csv
```

---

## Running The Project

Always run commands from the project root.

### Recommended path: `v2.2`

Simplest wrapper commands:

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
```

Useful direct commands:

```bash
python eval_v2.py --no-rerun
python demo_v2.py
python src/v2/inference/recommend.py --split test --limit 3 --top-k 5 --candidate-k 50 --explicit-goal focus
```

### Original HMM + DQN path

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

---

## Main Outputs

### Rebuilt `v2.2`

- `data/processed/rebuild/decision_table.parquet`
- `data/processed/rebuild/anchor_table.parquet`
- `data/processed/rebuild/song_catalog.parquet`
- `models/rebuild/query_tower.pt`
- `models/rebuild/reranker.pt`
- `models/rebuild/offline_eval_v2.json`
- `models/rebuild/v2_readiness.json`

### Original HMM + DQN

- `models/hmm.npz`
- `models/reward_model.json`
- `models/agent.pt`
- `models/eval_report.json`

---

## Where To Read More

- `docs/PROJECT_STATE.md`
  - current operational truth, artifacts, and verified metrics
- `docs/PRESENTATION_REPORT.md`
  - full explanation of `v2.2`
- `docs/PRESENTATION_SCRIPT.md`
  - short 5-6 minute presentation talk track
- `docs/V2_EXECUTION_PLAN.md`
  - rebuilt `v2.2` execution contract

---

## Main Limitations

- `v2.2` is still built on a relatively small SiTunes intervention dataset
- `movement` remains a low-support goal
- public transfer is still engineered rather than fully learned
- exact-song recovery remains a legacy diagnostic, not the main success criterion
- the original HMM + DQN path is runnable, but it is not the strongest research story in the repo anymore
