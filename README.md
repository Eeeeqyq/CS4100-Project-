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

## Original HMM + DQN Approach At A Glance

The original course pipeline uses a different AI design. It is simpler than `v2.2`, but still grounded in a clear logic.

| Method | What it does in the original pipeline | Why it was used |
|--------|----------------------------------------|-----------------|
| **Hidden Markov Model (HMM)** | converts wrist observations into a coarse latent belief over 3 hidden states | the user's internal state is only partially observable, so the system uses a belief state rather than treating raw wrist data as direct mood labels |
| **Feature-based state construction** | combines HMM belief with context, check-in, and preference features into a `16D` RL state | the policy needs more than wrist-only information to make a recommendation decision |
| **Hierarchical reward model** | estimates expected emotional benefit and acceptance for state-action pairs | offline recommendation needs a reward estimate before training a policy, and the data is too sparse for naive fine-grained estimates |
| **Double DQN** | chooses one of 8 music buckets from the offline state | the course setup is framed as a sequential decision problem, so the policy is learned over discrete actions |
| **Deterministic retrieval** | maps the chosen bucket to actual tracks from SiTunes, PMEmo, and Spotify | the policy chooses an intervention type first, then retrieval turns that into concrete songs |

### How the original pipeline fits together

1. **Preprocess the datasets**
   - clean SiTunes and build optional PMEmo / Spotify retrieval tables
2. **Train the HMM**
   - read wrist observations and estimate a coarse latent belief state
3. **Build the RL state**
   - combine the HMM belief with time, activity, weather, speed, self-report, and taste features
4. **Estimate reward**
   - model emotional benefit and acceptance from offline intervention data
5. **Train the DQN**
   - choose one of 8 music buckets
6. **Retrieve tracks**
   - rank actual songs consistent with the chosen bucket and the user's baseline taste

### Main logic behind the original approach

The original pipeline treats music recommendation as a POMDP-style problem:
- the user's true internal state is hidden
- wrist signals provide only partial evidence
- the HMM maintains a belief over coarse latent states
- the DQN then chooses an action bucket using that belief plus explicit context features

This approach is still useful to understand because it is the starting point that `v2.2` improves on.

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
- every episode in `MusicEnv` is one step (`done=True`), so the "DQN" is effectively an epsilon-greedy **contextual bandit**: `GAMMA`, the target network, and the Double-DQN target have no effect on the learned Q-values
- all policies, including the DQN, are scored by the same train-fit reward model the DQN was trained against, not by observed test outcomes; these numbers are model-estimated reward, not off-policy estimates

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
| benefit MAE | `0.1262` | error against the **candidate train anchors'** recorded benefit, not the test user's outcome | the anchor's own past outcome is also a reranker input, so this is closer to reconstruction than prediction |
| blended acceptance MAE | `0.2947` | same as above, for acceptance | same caveat as benefit MAE |
| public-transfer-supported share | `0.6399` | a public song is top-1 in about 64% of held-out cases | a **behaviour** metric, not a quality metric: public songs have no outcome labels; the `0.72` threshold was the top of a validation sweep (`0.62`-`0.72`) and the only value that kept one source under the 85% gate |
| top-1 source max share | `0.6399` | no single source completely dominates the top recommendation | checks that the system is not collapsing to one catalog |

Important caution:
- legacy exact-song metrics are still reported, but they are **diagnostic only**
- a weak exact-song metric does **not** mean `v2.2` failed, because exact historical song imitation is no longer the primary objective
- the goal is currently always assigned by the rule-based `goal_router_v1` in evaluation (`goal_source = router_fallback` for every row); the `--explicit-goal` override is only exercised in the demo

### Baselines

`python src/v2/eval/baselines.py` (also run by `python eval_v2.py`) compares the learned stages with simple alternatives on the same 311 test rows, using the same metric code as `offline_eval.py`. Brackets are 95% confidence intervals from resampling the **5 test users**.

| Retrieval (over all 815 train anchors) | recall@20 | weighted recall@20 |
|---|---|---|
| learned query tower | `0.746` [0.64, 0.81] | `0.301` [0.23, 0.35] |
| random over all anchors | `0.117` [0.11, 0.12] | `0.025` [0.02, 0.03] |
| random within the router goal | `0.369` [0.35, 0.41] | `0.118` [0.10, 0.13] |
| best historical anchors for the goal (no context) | `0.707` [0.67, 0.81] | `0.372` [0.32, 0.47] |
| nearest neighbours within goal (15-D context features) | `0.904` [0.78, 0.97] | `0.481` [0.38, 0.55] |

| Reranking (same learned top-50) | hit@10 |
|---|---|
| query order | `0.595` [0.52, 0.64] |
| learned reranker | `0.717` [0.64, 0.75] |
| sort by train anchor success | `0.871` [0.78, 0.91] |
| sort by context distance | `0.765` [0.67, 0.81] |

How to read this honestly:
- the learned retriever is far above chance, but **not** clearly above "best historical anchors for this goal", which ignores the user's context entirely
- the nearest-neighbour and success baselines win partly because tier-3 positives are *defined* as same-goal, high-success anchors that are close in that same 15-D feature space (`src/v2/data/anchors.py`); the labels encode the heuristic, so these baselines are a measure of label circularity, not proof that kNN is a better recommender
- with only 5 test users the intervals are wide; differences of a few points are not meaningful

Source-of-truth artifacts:
- `models/rebuild/offline_eval_v2.json`
- `models/rebuild/v2_readiness.json`
- `models/rebuild/baselines_v2.json`

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
python -m src.data.preprocess
python train_v2.py
python eval_v2.py
python demo_v2.py
```

`python -m src.data.preprocess` is required on a fresh clone: it writes `data/processed/split_manifest.json` (the user-level train/val/test split), which `src/v2/train/train_user_encoder.py` reads. Without it, `train_v2.py` fails at the user-encoder step.

What the wrappers do:
- `train_v2.py`: full rebuilt Block B stack
- `eval_v2.py`: clean offline summary + readiness output + presentation figures (`source_mix`, `goal_breakdown`)
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
- goal-conditioned: the goal comes from the rule-based router in training and evaluation; an explicit goal can override it at inference (demo)
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
- **PMEmo and Spotify are optional for running the code, but required to reproduce the reported `v2.2` numbers**

See [Setup](#setup) for the exact download commands and folder layout.

---

## Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd CS4100-Project-
```

### 2. Create an environment and install dependencies

Verified with Python `3.13`. (Python 3.14 is not verified; PyTorch wheels may not be available for it yet.)

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv -r requirements.txt
```

Or with the standard library:

```bash
python -m venv .venv
python -m pip install -r requirements.txt   # run after activating .venv
```

Activate it:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Place the datasets

The raw data is not in git (`data/` is gitignored). After this step, the layout must be exactly:

```text
data/raw/
|-- situnes/SiTunes/            Stage1/ Stage2/ Stage3/ music_metadata/
|-- pmemo/                      annotations/ features/ EDA/ metadata.csv
`-- spotify_kaggle/dataset.csv
```

**SiTunes (required)**

The SiTunes GitHub repo has its own `SiTunes/` subfolder, so clone it into `data/raw/situnes` (not `data/raw/situnes/SiTunes`):

```bash
git clone https://github.com/JiayuLi-997/SiTunes_dataset data/raw/situnes
```

Check: `data/raw/situnes/SiTunes/Stage2/wrist.npy` should exist.

**PMEmo (optional, but needed to reproduce the reported results)**

Download the PMEmo2019 release from the link in the [PMEmo README](https://github.com/HuiZhangDB/PMEmo) and unzip its contents into `data/raw/pmemo/`.

Files the code reads:
- `annotations/static_annotations.csv` (required if PMEmo is used)
- `annotations/static_annotations_std.csv` (required if PMEmo is used)
- `annotations/dynamic_annotations.csv`
- `annotations/dynamic_annotations_std.csv`
- `features/static_features.csv`
- `metadata.csv`
- `EDA/*_EDA.csv`: if this folder is missing, the EDA signal is **silently skipped** and your numbers will differ from the reported ones

The full download is about 1.2 GB, mostly `chorus/*.mp3`, which the code never reads. You can delete `chorus/` to save space.

**Spotify Kaggle (optional, but needed for v2.2 public transfer)**

Download the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (Kaggle account required) and save its CSV as:

```text
data/raw/spotify_kaggle/dataset.csv
```

Without Spotify, `v2.2` can only transfer to the 736 PMEmo songs, so the public-transfer metrics will not match the reported ones.

---

## Running The Project

Always run commands from the project root.

### Recommended path: `v2.2`

Simplest wrapper commands:

```bash
python -m src.data.preprocess
python train_v2.py
python eval_v2.py
python demo_v2.py
```

Equivalent underlying module sequence:

```bash
python -m src.data.preprocess
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
- `docs/V2_EXECUTION_PLAN.md`
  - rebuilt `v2.2` execution contract

---

## Main Limitations

- `v2.2` is still built on a relatively small SiTunes intervention dataset
- `movement` remains a low-support goal
- public transfer is still engineered rather than fully learned, and public-song recommendations cannot be validated because public songs have no outcome labels
- the learned retriever does not clearly beat, and the learned reranker loses to, simple goal + historical-success baselines (see [Baselines](#baselines)); the anchor labels are partly defined by those same heuristics
- the test set is 5 users, so confidence intervals are wide
- the original pipeline's "DQN" is a one-step contextual bandit and is evaluated with its own reward model
- exact-song recovery remains a legacy diagnostic, not the main success criterion
- the original HMM + DQN path is runnable, but it is not the strongest research story in the repo anymore
