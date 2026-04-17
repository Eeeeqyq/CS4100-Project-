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

## Methods At A Glance

The rebuilt `v2.2` system combines several AI methods, each chosen for a specific reason.

| Method | What it does in `v2.2` | Why we used it |
|--------|-------------------------|----------------|
| **Sequence modeling** | encodes the wrist time series with a learned context model | wrist data is temporal, so a static average would throw away important information |
| **Representation learning** | learns embeddings for context, users, and songs | the system needs a shared representation to compare current situations, historical anchors, and public songs |
| **Anchor retrieval** | retrieves relevant historical SiTunes interventions | SiTunes is the only dataset with intervention-outcome supervision, so it should drive intervention quality |
| **Multitask reranking** | predicts benefit, acceptance, and relevance for retrieved anchors | helpfulness and likability are different objectives, so we should not collapse them too early |
| **Calibrated public transfer** | allows Spotify / PMEmo songs to outrank anchors only when support is strong enough | public catalogs are useful, but they are not directly labeled with intervention outcomes |
| **Explicit-goal conditioning** | uses `focus`, `wind_down`, `uplift`, or `movement` as a direct input | the data is strong enough to support goal-conditioned recommendation, but not a strong claim about always inferring the user's goal automatically |

### How the `v2.2` pipeline fits together

1. **Context encoder**
   - input: wrist sequence + environment + optional self-report
   - output: context embedding
2. **User encoder**
   - input: Stage 1 taste history
   - output: user embedding
3. **Song encoder**
   - input: song metadata/features and PMEmo dynamics where available
   - output: song embedding
4. **Query tower**
   - input: context embedding + user embedding + explicit goal + target affect state
   - output: query embedding used to retrieve SiTunes anchors
5. **Reranker**
   - input: retrieved anchor candidates
   - output: predicted benefit, predicted acceptance, and anchor relevance
6. **Transfer gate**
   - input: anchor-conditioned support for public songs
   - output: whether Spotify / PMEmo candidates are allowed to outrank anchors

### Main logic behind the approach

The main reasoning is:
- learn intervention quality from **SiTunes**, because it is the only intervention-outcome dataset in the repo
- use **Spotify** and **PMEmo** to widen the recommendation space, but only through controlled transfer
- separate **benefit** from **acceptance**, because a song can be helpful but disliked, or liked but unhelpful
- use **retrieval + reranking** instead of exact-song classification, because multiple historical anchors can be valid in similar contexts

---

## Datasets

| Dataset | Role | Size |
|---------|------|------|
| [SiTunes](https://github.com/JiayuLi-997/SiTunes_dataset) | Core intervention dataset: biometrics, context, pre/post emotion labels, and baseline preference survey | ~2,000 raw interactions -> 1,406 cleaned Stage 2/3 interactions, 30 users |
| [PMEmo](https://github.com/HuiZhangDB/PMEmo) | Transfer-side music-affect support and optional EDA signal | 736 labeled tracks |
| [Spotify Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) | Large public music catalog for transfer candidates | ~89,500 tracks |

### How each dataset is actually used

- **SiTunes Stage 1**
  - baseline preference modeling
- **SiTunes Stage 2 and Stage 3**
  - intervention-outcome learning
  - anchor supervision for `v2.2`
- **PMEmo**
  - transfer-side affect cues
  - dynamic valence/arousal contours
  - optional EDA-derived signal
- **Spotify Kaggle**
  - large public transfer catalog

### Why this matters

For `v2.2`, the key rule is:
- learn intervention quality from **SiTunes**
- use **PMEmo** and **Spotify** for public transfer, not as primary intervention labels

That is why the rebuilt system is anchor-first instead of treating the mixed public catalog as if it were fully labeled.

---

## Original HMM + DQN Findings

Latest saved legacy result from `models/eval_report.json` on the held-out test split:

| Metric | DQN | State-Prior | Always-7 | Random |
|--------|-----|-------------|----------|--------|
| Combined reward | `+0.1639` | `+0.1641` | `+0.0821` | `+0.0781` |
| Emotion benefit | `+0.0465` | `+0.0439` | `+0.0104` | - |
| Acceptance | `+0.4377` | `+0.4446` | `+0.2494` | - |
| Regret | `0.0021` | `0.0019` | `0.0839` | - |

Additional saved HMM summary from the current generated artifacts:
- mean belief entropy: `0.1592`
- rounded unique belief vectors: `473`
- corrected state usage:
  - state 0: `656`
  - state 1: `481`
  - state 2: `269`

How to interpret this:
- the original DQN clearly beats trivial baselines such as `always7` and uniform random
- it is still slightly below the `state_prior` baseline on held-out combined reward
- the original pipeline is still runnable and scientifically interpretable, but it is no longer the strongest research story in the repo compared with `v2.2`

Source-of-truth artifact:
- `models/eval_report.json`

---

## `v2.2` Results And Interpretation

The rebuilt `v2.2` system is **not** mainly trying to recover the exact historical song. Its main job is:

1. retrieve strong historical SiTunes intervention anchors
2. rerank them by predicted benefit and acceptance
3. transfer to public songs only when that transfer is actually supported

Latest saved rebuilt result:

| Metric | Value | What it means in plain English | Why it matters |
|--------|-------|--------------------------------|----------------|
| `ready` | `true` | the rebuilt system passes its main readiness contract | shows that the primary `v2.2` gates are currently satisfied |
| held-out test rows | `311` | the main offline result is measured on 311 held-out decision rows from unseen test users | shows the result is a held-out evaluation, not a training metric |
| anchor query `recall@20` | `0.7460` | in about 75% of held-out test cases, retrieval finds at least one useful anchor in the top 20 | shows the first-stage retriever is usually finding relevant SiTunes interventions |
| anchor query weighted `recall@20` | `0.3011` | the retriever is not only finding any positive anchor, but also finding stronger tiered positives often enough | checks that the retriever is surfacing better anchors, not only weak neighbors |
| anchor rerank `hit@10` | `0.7170` | after reranking, a useful anchor appears in the top 10 in about 72% of held-out cases | shows the second stage usually improves the candidate list into a usable recommendation set |
| anchor rerank mean rank | `5.9270` | when a positive anchor is found, it tends to be near the top | lower is better; this checks that good anchors are not buried |
| benefit MAE | `0.1262` | predicted emotional benefit is fairly close to observed benefit | lower is better; this checks whether the model’s benefit estimates are numerically credible |
| blended acceptance MAE | `0.2947` | predicted user acceptance is reasonably calibrated | lower is better; this checks whether the system is balancing helpfulness with plausibility |
| public-transfer-supported share | `0.6399` | public songs win in about 64% of held-out cases, but only when support is strong enough | shows the system is actually using public transfer rather than staying anchor-only |
| top-1 source max share | `0.6399` | no single source completely dominates the top recommendation | checks that the system is not collapsing to one catalog |

Important caution:
- legacy exact-song metrics are still reported, but they are **diagnostic only**
- a weak exact-song metric does **not** mean `v2.2` failed, because exact historical song imitation is no longer the primary objective

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

How to explain the demo:
- if the top result stays in **SiTunes**, the model is saying the anchor evidence is stronger than the transfer evidence
- if the top result becomes **Spotify**, the model is saying public transfer support is strong enough to promote a public song
- the demo is not trying to prove that every goal must give a completely different song
- it is trying to prove that the system changes behavior in a goal-consistent, support-aware way

What to look for in the current verified demo:
- `focus` falls back to SiTunes anchors
- `wind_down` transfers to Spotify
- `uplift` and `movement` can still fall back to anchors when transfer support is weaker on that same row

That pattern is good for `v2.2`, because it shows the model is not blindly pushing public songs to the top.

Recommended presentation commands:

```bash
python eval_v2.py --no-rerun
python demo_v2.py
```

For the full presentation explanation:
- `docs/PRESENTATION_REPORT.md`

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

## Repo Structure

```text
CS4100-Project-/
|-- train_v2.py, eval_v2.py, demo_v2.py
|   `-- top-level entrypoints for the rebuilt v2.2 system
|-- scripts/
|   `-- build_v2_data.py
|       `-- rebuilt data-contract builder
|-- src/
|   |-- v2/
|   |   |-- data/
|   |   |-- train/
|   |   |-- inference/
|   |   `-- eval/
|   |       `-- rebuilt v2.2 pipeline
|   |-- data/
|   |-- hmm/
|   |-- rl_agent/
|   `-- music/
|       `-- original HMM + DQN pipeline modules
|-- train_agent.py, eval_agent.py, demo.py, simulate_user.py
|   `-- top-level entrypoints for the original pipeline
|-- docs/
|   |-- PROJECT_STATE.md
|   |-- PRESENTATION_REPORT.md
|   `-- V2_EXECUTION_PLAN.md
|-- models/
|   `-- rebuild/
|       `-- saved v2.2 evaluation/readiness artifacts
`-- data/
    `-- raw and processed datasets
```

If you only care about the current recommended system, start with:
- `train_v2.py`
- `eval_v2.py`
- `demo_v2.py`
- `src/v2/`

---

## Raw Data Paths

Raw-data roots:
- SiTunes: `data/raw/situnes/SiTunes/`
- PMEmo: `data/raw/pmemo/`
- Spotify Kaggle: `data/raw/spotify_kaggle/dataset.csv`

Requirements:
- **SiTunes is required**
- **PMEmo and Spotify are optional but recommended**

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
- `docs/V2_EXECUTION_PLAN.md`
  - rebuilt `v2.2` execution contract

---

## Main Limitations

- `v2.2` is still built on a relatively small SiTunes intervention dataset
- `movement` remains a low-support goal
- public transfer is still engineered rather than fully learned
- exact-song recovery remains a legacy diagnostic, not the main success criterion
- the original HMM + DQN path is runnable, but it is not the strongest research story in the repo anymore
