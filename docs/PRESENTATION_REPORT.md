# V2.2 Presentation Report

Last updated: 2026-04-12

## Purpose

This report is the presentation-ready explanation of the rebuilt `v2.2` system.

It is written to answer four questions clearly:

1. What problem are we actually solving?
2. Why did we choose these AI methods?
3. What does the full system do from raw data to final recommendation?
4. What do the current results mean, and what should we claim honestly?

This report focuses on `v2.2`, not the legacy HMM + DQN grading pipeline. The old pipeline still exists in the repo for course compatibility, but the stronger research story is the rebuilt explicit-goal, anchor-first system under `src/v2/`.

---

## Executive Summary

The core problem is not "predict what song the user played before." That is too narrow and not aligned with the real purpose of the system.

The real problem is:

> Given the user's current biometric and environmental context, an explicit goal, and their baseline taste profile, recommend music that is likely to move their emotional state in the right direction while still being acceptable enough that they would realistically follow it.

The key design decision in `v2.2` is to separate the problem into two parts:

- learn intervention quality only from **SiTunes**, because that is the only dataset in the repo that contains context plus pre/post emotional outcome
- use **Spotify** and **PMEmo** only as public transfer catalogs, not as primary intervention labels

That leads to the final `v2.2` pipeline:

1. build a context representation from wrist, environment, and optional self-report
2. build a user representation from Stage 1 taste history
3. retrieve strong historical **SiTunes anchors**
4. rerank those anchors by predicted benefit and acceptance
5. transfer to public songs only if the anchor evidence is strong enough

That is the main scientific shift from the earlier bucket-first approach.

---

## 1. Motivation And Problem Setup

### The motivation

Standard music recommenders optimize for familiarity and engagement:

- what you liked before
- what similar users liked
- what fits your playlist history

They usually ignore the fact that the same person may want very different music depending on the situation:

- studying vs recovering
- anxious vs flat
- sedentary vs moving
- alone at night vs active during the day

This project is trying to build a **situational music recommender**, not just a preference recommender.

### The actual decision problem

We define the decision problem as:

> choose a music intervention that moves the user toward a context-appropriate target affect state while staying acceptable enough that they are likely to engage with it

This matters because "better mood" is too vague.

Examples:

- for `focus`, we usually want moderate valence and controlled, not spiky, arousal
- for `wind_down`, we usually want lower arousal
- for `uplift`, we want higher valence and some activation
- for `movement`, we want higher arousal and moderate-high valence

So the target is not "maximize valence." The target depends on the goal.

### The objective in plain language

For each context:

- compare the user's pre-music emotional state to the target state for the goal
- estimate how much closer the post-music state gets to that target
- separately estimate whether the user would plausibly accept the recommendation

Then recommend tracks with strong evidence on both.

### Why explicit goal is the right interface

The repo still contains a fallback goal router, but `v2.2` treats **explicit goal** as the main interface.

That choice is scientifically cleaner because:

- the data is not strong enough to justify a bold claim that the system can always infer the correct goal automatically
- explicit goal makes the objective well-defined
- it lets us evaluate the recommendation logic itself rather than entangling it with a weak goal-inference problem

For presentation, this is the correct framing.

---

## 2. How To Think About The Repo

There are **two systems** in the repo.

### System A: legacy course pipeline

This is the original:

- HMM over wrist observations
- 16D DQN state
- DQN action over 8 music buckets
- deterministic track retrieval

This path is still useful for course continuity and for showing the original AI framing, but it is not the main presentation story anymore.

### System B: rebuilt `v2.2` pipeline

This is the stronger research system and should be the main thing you present.

Its contract is:

> explicit goal + current context + user taste -> retrieve strong SiTunes anchors -> rerank anchors -> transfer to public songs only when supported

That is the system whose readiness is currently `true`.

### Presentation advice

For the class presentation:

- mention the legacy system briefly as the original baseline design
- present `v2.2` as the improved formulation
- explain that `v2.2` was built in response to the real methodological weaknesses of the bucket-first setup

---

## 3. Nature Of The Data

The repo currently uses three datasets.

### 3.1 SiTunes

SiTunes is the only dataset in the project that contains the full intervention story:

- biometric context
- environment
- pre-listening emotion
- post-listening emotion
- user feedback

That is why SiTunes is the only valid source for **intervention-quality supervision**.

In the rebuilt data:

- total decision rows: `1406`
- users: `30`
- user-level split:
  - train: `815`
  - val: `280`
  - test: `311`

SiTunes also has two different roles:

- **Stage 1** gives baseline taste history through ratings
- **Stage 2/3** gives intervention rows for learning benefit and acceptance

### 3.2 PMEmo

PMEmo is a music-emotion dataset. It is useful for:

- affect cues
- dynamic valence/arousal contours
- retrieval-side emotion guidance
- optional EDA response hints

But PMEmo is **not** used as the main source of intervention supervision because it does not tell us:

- this user
- in this context
- listened to this song
- and their state changed in this way

So PMEmo is a **transfer-side music representation source**, not the main decision-label source.

### 3.3 Spotify Kaggle

Spotify Kaggle is the large public catalog.

Its value is breadth:

- many songs
- dense metadata
- better chance of finding public transfer candidates

Its weakness is the same as PMEmo:

- it does not contain intervention-outcome supervision

So Spotify is also a transfer catalog, not the primary label source.

### Why this matters

This data structure is the main reason the rebuilt system is anchor-first.

If we had trained directly on a mixed catalog as though all non-SiTunes songs were true negatives, that would be mathematically wrong. Those public songs are mostly **unlabeled**, not "bad."

---

## 4. The Core AI Logic Behind V2.2

### 4.1 Goal-conditioned target state

The system maps each goal to a target affect state `(tau_valence, tau_arousal)` plus weights on valence vs arousal.

Current target table:

| Goal | Target Valence | Target Arousal | Valence Weight | Arousal Weight |
|------|----------------|----------------|----------------|----------------|
| `focus` | `0.35` | `0.05` | `0.35` | `0.65` |
| `wind_down` | `0.30` | `-0.45` | `0.40` | `0.60` |
| `uplift` | `0.70` | `0.20` | `0.60` | `0.40` |
| `movement` | `0.60` | `0.65` | `0.45` | `0.55` |

The target can be adjusted slightly based on the current context. For example:

- movement may demand a slightly higher arousal target when movement evidence is already high
- focus late in the day can slightly lower the arousal target

### 4.2 Benefit target

The system computes benefit as:

- distance from pre-state to goal target
- minus distance from post-state to goal target

Interpretation:

- positive benefit means the song moved the user closer to the desired state
- negative benefit means it moved them away from it

This is much better than vague reward shaping because it states exactly what "helping" means.

### 4.3 Acceptance target

Acceptance is modeled separately from benefit.

That is important because:

- a song can help emotionally but still be disliked
- a song can be liked but not be useful for the goal

`v2.2` stores two observed channels separately:

- `accept_pref_target`
- `accept_rating_target`

with masks:

- `accept_pref_mask`
- `accept_rating_mask`

This avoids collapsing different feedback types into one noisy label too early.

### 4.4 The final decision idea

The system tries to recommend songs with:

- high predicted benefit
- high predicted acceptance
- low uncertainty
- strong evidence from similar successful historical interventions

That is the real AI logic of the system.

---

## 5. End-To-End Pipeline From Start To Finish

This section is the most important one to understand the system mechanically.

### Step 1. Build the supervised decision table

Script:

```bash
python scripts/build_v2_data.py
```

Main outputs:

- `data/processed/rebuild/decision_table.parquet`
- `data/processed/rebuild/anchor_table.parquet`
- `data/processed/rebuild/anchor_positive_sets.npz`
- `data/processed/rebuild/anchor_negative_pools.npz`
- `data/processed/rebuild/situnes_anchor_index.npz`

Each SiTunes decision row stores:

- user id
- song id
- split
- explicit goal or fallback goal source
- pre/post valence and arousal
- benefit target
- separate acceptance targets and masks
- context fields
- target affect state

### Step 2. Build the context representation

Model:

- `src/v2/models/context_encoder.py`

Inputs:

- wrist sequence: `30 x 9`
- environment vector: `9`
- self-report vector: `3`

Wrist sequence features:

- normalized heart rate
- heart-rate delta
- normalized intensity
- normalized steps
- one-hot activity type

Environment features:

- time sin/cos
- weather one-hot
- temperature z-score
- humidity z-score
- speed
- weekend flag

Self-report features:

- pre-valence
- pre-arousal
- check-in mask

Architecture:

- 2-layer bidirectional GRU over the wrist sequence
- attention pooling over time
- environment projection
- self-report projection
- fusion into a `128D` context embedding

Why this method:

- wrist data is sequential, not static
- the same average HR can mean different things depending on the recent sequence
- attention makes the model focus on the most informative time steps

This is one of the clearest ways `v2.2` responds to the TA feedback about sequence modeling.

### Step 3. Build the song representation

Model:

- `src/v2/models/song_encoder.py`

Inputs:

- static song vector: `20D`
- dynamic affect sequence: `20 x 2`

Static song features include:

- valence and arousal estimates
- energy
- tempo
- danceability
- acousticness
- instrumentalness
- speechiness
- liveness
- loudness
- popularity
- explicit flag
- source flags
- compact genre embedding
- EDA impact hint

Dynamic sequence:

- 20-step valence/arousal contour

Architecture:

- MLP for static features
- Conv1D stack for dynamic trajectory
- learned fusion gate between static and dynamic branches
- output song embedding: `128D`

Why this method:

- many songs can look similar in static metadata but behave differently emotionally over time
- dynamic contours matter for goals like focus or wind-down, where volatility and peakiness change suitability

### Step 4. Build the user representation

Model:

- `src/v2/models/user_encoder.py`

Input:

- up to `20` Stage 1 rated songs per user

Architecture:

- project each rated song plus rating features
- attention over history items
- fuse into a `64D` user embedding
- also output taste-affect summary and confidence

Why this method:

- we want personalization without needing a huge per-user history
- attention lets the model focus on the most informative prior ratings
- this is a reasonable way to encode baseline taste from the data we actually have

### Step 5. Build SiTunes anchor supervision

This is the structural heart of `v2.2`.

Instead of saying "one exact historical song is the only correct answer," the system builds a set of acceptable anchor positives from SiTunes only.

Current anchor supervision:

- positive cap: `5`
- negative cap: `32`
- every row has at least one positive

Positive tiers:

- **Tier 1**: the factual intervention row itself
- **Tier 2**: same-song, same-goal, successful rows
- **Tier 3**: nearby successful context neighbors with the same goal

Negatives include:

- hard same-goal failures
- same-song mismatched-goal rows
- same-goal confounders

Observed anchor success is defined as:

- `0.7 * benefit_target + 0.3 * acceptance_obs`

Why this matters:

- it matches the real problem better than exact-song recovery
- it acknowledges that more than one song can be acceptable in a context
- it keeps supervision grounded in real intervention data

### Step 6. Retrieve anchors with the query tower

Model:

- `src/v2/models/query_tower.py`

Input:

- context embedding `128D`
- user embedding `64D`
- goal one-hot `4D`
- target state `2D`

Output:

- query embedding `128D`

Training logic:

- retrieve only from **train anchors**
- use multi-positive supervision
- weight positives by tier
- use hard negatives
- use a pairwise margin term against hard negatives

Why this method:

- retrieval is the right first stage when the decision space is large
- training against train anchors only preserves split safety
- tier weighting prevents all positives from being treated as equally informative

### Step 7. Rerank anchors

Model:

- `src/v2/models/reranker.py`

Current pair feature width:

- `346D`

Outputs:

- `benefit_hat`
- `accept_pref_hat`
- `accept_rating_hat`
- `anchor_relevance_logit`

Architecture:

- MLP backbone
- multitask heads
- separate calibration for benefit, preference, and rating

Why this method:

- retrieval gets plausible candidates
- reranking does the fine-grained decision work
- multitask prediction keeps the system interpretable:
  - one head explains usefulness
  - one head explains acceptance
  - one head explains relative anchor relevance

### Step 8. Transfer to public songs only when supported

Inference script:

- `src/v2/inference/recommend.py`

This is the final stage:

1. retrieve top SiTunes anchors
2. rerank anchors
3. inspect the best anchors
4. expand from those anchors into Spotify / PMEmo only if support is strong enough

Public transfer uses:

- similarity to the best anchors
- fit to the goal-conditioned target affect state
- fit to the user taste representation
- song quality
- PMEmo dynamic affect cues
- uncertainty penalty

This is the right logic because public songs are not primary labels. They are transfer candidates supported by anchor evidence.

---

## 6. Why These AI Methods Make Sense

### Why not train directly on exact song identity?

Because the data does not justify it.

The exact historical song is not the only plausible good answer. If the user was studying and a calm, familiar, medium-valence track worked, another similar track might also have worked. Treating all non-factual songs as wrong creates a bad supervision signal.

### Why not use PMEmo or Spotify as hard intervention labels?

Because they do not contain the right supervision. They tell us about songs, not about successful interventions in context.

### Why use sequence modeling over wrist data?

Because context is temporal. A single static average cannot distinguish:

- stable calm
- rising activation
- recovery after movement
- brief spikes

A GRU with attention is a reasonable sequential model for this scale of data.

### Why split benefit and acceptance?

Because the recommender must do two things at once:

- help the user reach the goal state
- remain plausible enough that the user would actually follow the recommendation

That is closer to the real decision problem than one merged reward with unclear semantics.

### Why use anchor retrieval plus reranking?

Because:

- the candidate space is large
- the labels are sparse
- the intervention evidence lives only in SiTunes

Anchor retrieval narrows the decision to relevant historical interventions, and reranking reasons more carefully within that set.

### Why allow public transfer at all?

Because SiTunes is too small to be the only music library. If we want practical recommendations, we need a larger catalog. The compromise is:

- learn intervention quality from SiTunes
- transfer only when anchor evidence is strong

That is a much better compromise than pretending public songs are directly supervised.

---

## 7. How V2.2 Responds To The TA Feedback

This section is useful if someone asks why the rebuild was necessary.

### TA point: use song-level supervision, not only coarse buckets

Response:

- yes, `v2.2` now does this
- it supervises on SiTunes intervention songs as anchors
- this is much closer to song-level intervention reasoning than the old bucket-only policy

### TA point: use neural sequence modeling on the wrist time series

Response:

- yes, `v2.2` does this with a 2-layer bidirectional GRU plus attention
- that replaces the idea that coarse HMM state alone should carry the whole context story

### TA point: define the objective clearly

Response:

- yes, `v2.2` now has a clear goal-conditioned objective
- benefit means movement toward a target affect state
- acceptance means plausible user uptake

### TA point: pretrain first, then personalize

Response:

- partially addressed
- the current repo learns shared song, user, and context representations and then personalizes through the user encoder and anchor retrieval
- but it does **not** yet integrate extra datasets such as DEAM or WESAD

That means the feedback was addressed in the right direction, but not fully exhausted.

### TA point: use a larger dataset

Response:

- partially addressed, but in a principled way
- we use public song catalogs for transfer
- we do not misuse them as fake intervention labels

This is the correct scientific compromise for the data currently wired into the repo.

---

## 8. Current Verified Demo Story

### Recommended demo commands

Use these for the presentation:

```bash
python eval_v2.py
python demo_v2.py
```

These were re-run and verified on `2026-04-12`.

### Example: same held-out test row, different explicit goals

For test row `128`, the system changes the top recommendation when we change the goal.

| Goal | Top-1 Song | Source | Transfer Supported | Reason |
|------|------------|--------|-------------------|--------|
| `focus` | `BUTTERFLY EFFECT` | Spotify | yes | stable low-volatility contour |
| `wind_down` | `Iragai Poley` | Spotify | yes | stable low-volatility contour |
| `uplift` | `BUTTERFLY EFFECT` | Spotify | yes | stable low-volatility contour |
| `movement` | `505` | Spotify | yes | supported by similar successful movement anchors |

### What this demo shows

It shows three important things:

1. the system is **goal-sensitive**, not just context-sensitive
2. public songs can win, but only with strong anchor support
3. the explanation text is tied to either:
   - public transfer evidence
   - or successful anchor support

### What to say during the demo

Say something like:

> The system is not trying to recover one memorized song. It first retrieves similar successful historical interventions from SiTunes, then checks whether a public-catalog song is well-supported by those anchors for the current goal.

That line captures the main design idea.

---

## 9. Current Results

These are the current verified `v2.2` test metrics from:

```bash
python src/v2/eval/offline_eval.py --split test --candidate-k 50
python src/v2/eval/check_readiness.py
```

### Primary results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| anchor query recall@20 | `0.7460` | retrieval usually surfaces at least one positive anchor early |
| anchor query recall@50 | `0.8810` | good anchor coverage by depth 50 |
| weighted anchor query recall@20 | `0.3011` | meaningful recovery of stronger-tier positives |
| anchor rerank hit@10 | `0.7170` | reranker usually places a positive anchor in the top 10 |
| anchor rerank conditional mean rank | `5.9270` | positives are near the top after reranking |
| weighted anchor rerank NDCG@10 | `0.2306` | reranking improves tier-aware ordering |
| benefit MAE | `0.1262` | benefit prediction is numerically credible |
| blended acceptance MAE | `0.2947` | acceptance calibration is good enough for the dual-objective story |
| top-1 predicted acceptance mean | `0.5101` | top recommendations are not systematically unacceptable |
| public-transfer-supported share | `0.6399` | public songs win often, but the source mix is still controlled |
| top-1 source max share | `0.6399` | no catalog collapse |

### Source distribution

Top-1 source distribution on the test set:

- SiTunes: `0.3601`
- Spotify: `0.6399`

This is a strong sign that the system is not simply collapsing to one source.

### Per-goal summary

| Goal | Rows | Weighted Query Recall@20 | Rerank Hit@10 | Rerank Conditional Rank |
|------|------|---------------------------|---------------|-------------------------|
| `focus` | `140` | `0.3058` | `0.7000` | `6.4841` |
| `wind_down` | `46` | `0.3955` | `0.9783` | `2.8913` |
| `uplift` | `107` | `0.2139` | `0.6355` | `5.9643` |
| `movement` | `18` | `0.5412` | `0.6667` | `9.6111` |

Interpretation:

- `wind_down` is currently the strongest goal
- `focus` is solid
- `uplift` is workable but less sharp
- `movement` is acceptable but still low-support and should not be oversold

### Readiness status

Current readiness:

- primary rebuilt contract: `ready = true`
- no hard blockers

Soft legacy diagnostic still weak:

- exact-song rerank hit@10: `0.0418`

This is no longer a hard failure because exact-song recovery is not the main contract of `v2.2`.

---

## 10. What The Results Mean

### The good news

- the rebuilt formulation works
- the system retrieves and reranks plausible anchors
- acceptance and benefit are both under control
- public transfer is active without collapsing to one source
- the explanation story is coherent

### The important nuance

The main success of `v2.2` is **not** exact song imitation.

The main success is:

- context-aware
- goal-aware
- taste-aware
- split-safe
- anchor-supported transfer

That is the right standard for the rebuilt system.

### The honest limitation

The legacy exact-song metric is still weak, and that tells us something real:

- the system is not a strong memorizer of historical song identity
- that is partly intentional, because the new objective is broader than exact history matching

That should be framed honestly, not hidden.

---

## 11. Main Limitations And What We Should Not Overclaim

### Limitation 1. SiTunes is still small

There are only `30` users and `1406` cleaned intervention rows.

So the system can be presented as a strong offline prototype, but not as a production-ready, universally generalizable recommender.

### Limitation 2. Movement is low support

The test split only has `18` movement rows.

So movement logic is promising, but still weaker evidence than focus or wind-down.

### Limitation 3. Public transfer is still engineered, not fully learned

The transfer stage uses a strong, thoughtful scoring function, but it is not yet a separately learned public-transfer model.

That is scientifically acceptable for a course project, but it is still a limitation.

### Limitation 4. Anchor positives are still neighbor-heavy

Current tier composition:

- tier 1: `0.116`
- tier 2: `0.1438`
- tier 3: `0.7402`

So most positives are still context-neighbor positives rather than factual same-song anchors.

That means the anchor task is much more honest than before, but it is still not maximally strict.

### Limitation 5. No extra public pretraining datasets yet

The repo does **not** currently use:

- DEAM
- WESAD
- Yambda
- CASE
- EmoWear

So the larger-dataset recommendation from the TA is only partially addressed.

### Limitation 6. Explicit goal is the main interface

This is a strength for clarity, but also a limitation:

- the system is not making a strong claim that it can always infer the user's true goal autonomously

That claim should not be made in the presentation.

---

## 12. What To Say In The Presentation

If you want a clean presentation flow, use this order.

### Slide 1. Motivation

Say:

> Standard recommenders optimize for preference history. We wanted a system that also reacts to the user's current physiological and emotional context.

### Slide 2. Problem formulation

Say:

> We reframed the task as goal-conditioned emotional intervention, not just next-song prediction.

### Slide 3. Why the original approach was not enough

Say:

> A bucket-first policy was a useful starting point, but it was too coarse. It did not use song-level supervision, and it did not clearly separate emotional benefit from user acceptance.

### Slide 4. Rebuilt v2.2 pipeline

Say:

> We now retrieve historical SiTunes intervention anchors, rerank them by predicted benefit and acceptance, and only transfer to public songs when that transfer is supported by the anchor evidence.

### Slide 5. Key AI methods

Highlight:

- bidirectional GRU + attention for wrist sequence modeling
- shared song representation with static + dynamic affect features
- attention-based user representation from Stage 1 ratings
- multi-positive anchor retrieval
- multitask reranker for benefit and acceptance
- support-aware public transfer

### Slide 6. Demo

Use the explicit-goal examples and show that changing the goal changes the recommendation.

### Slide 7. Results

Say:

> The rebuilt system passes its primary offline readiness contract, with strong anchor retrieval, strong reranking, acceptable calibration, and no source collapse.

### Slide 8. Honest conclusion

Say:

> The main success is not exact-song imitation. The main success is a more scientifically defensible decision pipeline for situational music intervention.

---

## 13. Questions You May Get, With Good Answers

### Q: Why not just use the exact song label?

A:

Because the exact historical song is an overly sharp target. More than one song may be suitable in a similar context. Anchor supervision is a better fit for the real recommendation problem.

### Q: Why not train on Spotify and PMEmo directly?

A:

Because they do not have the intervention-outcome labels we need. They are useful for transfer and song representation, but not as primary supervision for intervention quality.

### Q: Why use sequence modeling on the wrist data?

A:

Because the time pattern matters. A 30-step sequence can distinguish calm stability from rising activation or recovery, which a static average cannot do well.

### Q: Why separate benefit and acceptance?

A:

Because a helpful song can still be disliked, and a liked song can fail the emotional goal. A situational recommender should reason about both.

### Q: Why is exact-song hit@10 still low?

A:

Because the new system is not optimized mainly for memorizing one historical song identity. It is optimized for finding strong intervention anchors and transferring safely to public tracks.

### Q: Is this publishable as a real research system?

A:

As a course-scale offline AI system, yes, it is defensible. As a broader scientific claim about real-world emotional improvement, it would still need stronger datasets and a live user study.

---

## 14. Final Takeaway

The strongest way to describe the project is:

> We rebuilt the recommendation problem from a coarse bucket-selection task into a goal-conditioned intervention-retrieval task. The final system uses sequence modeling for context, separate modeling of benefit and acceptance, split-safe anchor retrieval from SiTunes, and cautious transfer to public catalogs.

That is the real intellectual contribution of the current repo.

If you keep the presentation centered on that point, the system is coherent, honest, and technically defensible.
