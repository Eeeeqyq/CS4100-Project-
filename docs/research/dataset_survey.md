# Dataset Survey and Project Redesign: Did the Music Help?

Status: research note, branch `research/dataset-survey`. Sources checked on **2026-11-12**.
Scope: public datasets that could support the project's core idea, which is to recommend music that improves the user's state right now (**benefit**) and not only music they like (**acceptance**). The note then proposes better-structured project designs.

How claims were checked:
- Every dataset fact below links to the page it came from, such as an official dataset page, README, Hugging Face API, PyPI or the dataset paper.
- Facts that came only from search snippets, secondary sources or unreachable pages are marked **UNVERIFIED**.
- Nothing was downloaded, no logins were used and no terms were accepted.

---

## 1. TL;DR

1. **SiTunes may be the only public dataset with music, a wearable signal and pre/post affect.**
   - It pairs a wristband signal with self-reported affect measured **before and after** a song, for the same listening event.
   - Nothing else we could verify has that triple.
   - Its 897 + 509 Stage 2/3 interactions match the 1,406 rows the current pipeline uses.
   - The dataset will not get bigger. The project has to get **more honest** about what 30 users can show.
2. **Large public music logs answer acceptance, not benefit.**
   - Yambda is the only large, open, *currently downloadable* option we found.
   - It has 4.79B events, Apache-2.0 and is not gated.
   - MSSD, MPD, LFM-1b and LFM-2b are all officially withdrawn.
3. **No public music dataset supports off-policy evaluation with logging propensities.**
   - The OPE method can be learned and validated on the **Open Bandit Dataset** (fashion, true propensities) or the **KuaiRand** random-exposure slice (short video).
   - On music data, OPE needs **estimated** propensities, and that caveat has to be stated.
4. **Recommendation: Design A** (§5), a pre-registered, user-level study on SiTunes.
   - Research question: *do wearable signals predict music benefit beyond pre-state, context and taste?*
   - It needs proper baselines (including a **regression-to-the-mean** baseline), confidence intervals across users and ablations.
   - Optional module: an OPE estimator validated on OBD before it is applied to SiTunes.

---

## 2. Comparison table

Legend for **Tests**:
- **(a)** affect prediction from biometrics
- **(b)** context-aware acceptance (like/skip/rating)
- **(c)** intervention benefit (did the state change after listening)
- **(d)** off-policy evaluation (propensities or randomized exposure)

A ● means the dataset supports that use; ◐ means partial or weak support.

### 2.1 Music + physiology / affect

| Dataset | Content | Scale | Labels | Pre/post affect? | Access / license | Size | Tests |
|---|---|---|---|---|---|---|---|
| **SiTunes** [1] | Wristband 30×4 matrix per interaction: HR (normalized per user), activity intensity, steps, activity type. Env: time, weather, rescaled GPS | 30 users; Stage 1/2/3 = 600/897/509 interactions (2,006); 307 songs; Stage 3 = 10 users | rating; Stage 3 adds 0–100 preference; pre/post valence and arousal in [-1, 1] (Stages 2–3) | **Yes** (Stages 2–3) | Open on GitHub. **No license** (GitHub API `license: null`), so it is unclear whether redistribution is allowed | ~2.2 MB repo | a ●, b ●, c ●, d ◐ (Stage 2 "traditional" vs Stage 3 "situational" recommenders; no propensities) |
| **PMEmo** [2] | Chorus clips (MP3); static and dynamic (0.5 s) annotations; listener EDA | 794 songs (repo); 457 subjects (paper) | Listener valence/arousal (V/A per paper) | No | Repo MIT; data on Google Drive (links not tested) | ~1.3 GB **UNVERIFIED** | a ● (music→EDA→V/A) |
| **HKU956** [3] | HR, EDA, BVP, IBI, skin temp while listening; CC-licensed music | 30 participants; "956" aligned sets | Per-song self-reported V/A [-10, 10]; TIPI | Not evident | HKU DataHub; **page 403, terms UNVERIFIED** | UNVERIFIED | a ●, c ◐ |
| **PhysioNet Brain Wearable Monitoring** [4] | 2× Empatica (EDA, HR, BVP, temp, ACC, IBI) + Muse EEG during n-back task | 15 + 13 participants | Task accuracy and RT. **No self-reported mood** | Conditions: no music / self-chosen relaxing / exciting / AI-generated relaxing | ODC-By 1.0, open | 558 MB zip, 2.7 GB raw | a ◐, c ◐ (physiological/performance "benefit", not mood) |
| **DEAP** [5] | 32-ch EEG + 12 peripheral (incl. GSR) @512 Hz; music-video stimuli | 32 participants × 40 one-minute clips | Self V/A/dominance, **liking**, familiarity | No (rated once after each clip) | Signed EULA; **permanent academic staff only** (search snippet; QMUL page returned 503/TLS error, **UNVERIFIED**) | UNVERIFIED | a ●, b ◐ |
| **AMIGOS** [6] | EEG, ECG, GSR + video; film stimuli (not music) | 40 participants; 16 short + 4 long videos | Self V/A, liking, emotions; PANAS baseline | No | EULA + request form (snippet, **UNVERIFIED**) | UNVERIFIED | a ● |
| **DREAMER** [7] | EEG (Emotiv) + ECG; film clips | 23 participants | Self V/A/dominance | No | Zenodo **restricted**: access request needed | Zenodo lists 3.7 TB (looks wrong; **UNVERIFIED**) | a ● |
| **WESAD** [8] | Chest (ECG, EDA, EMG, resp, temp, ACC @700 Hz) + wrist E4 (BVP 64 Hz, EDA 4 Hz…) | 15 subjects | Baseline / stress (TSST) / amusement (funny **videos**) + questionnaires | Meditation follows stress; no music | UCI mirror; non-commercial use | ~2.5 GB | a ● (wrist stress detection), out-of-domain check |
| **DEAM** [9] | Audio + per-second and per-song V/A; **no physiology** | 1,802 excerpts/songs | Crowd-annotated (MTurk) perceived V/A | No | Audio "Creative Commons" (variant unnamed) | Audio 1.3 GB; annotations 4.7 MB | Content prior only |
| **Daly et al. film-music EEG** (OpenNeuro ds002721) [10] | 19-ch EEG @1000 Hz; 12 s music clips | 31 subjects × 40 trials | 8 nine-point affect ratings | No | CC0 | 3.4 GB | a ● |
| **CASE** [11] | 8 physiological channels @1000 Hz; joystick V/A | 30 participants; 8 videos (1 music clip) | Continuous V/A | No | figshare/GitLab; paper CC BY 4.0, metadata CC0 | ~5 GB | a ● (not music) |

### 2.2 Music listening / recsys logs

| Dataset | Content | Scale | Labels | Access / license | Size | Tests |
|---|---|---|---|---|---|---|
| **Yambda** [12] | listens / likes / dislikes / unlikes / undislikes + audio embeddings; `is_organic` flag; `played_ratio_pct`; 5 s-binned timestamps | 4.79B events, 1M users, 9.39M tracks; subsets 50M / 500M / 5B | Listen fraction (skip proxy), explicit like/dislike, organic vs recommended | **Open, not gated** (HF API `gated: false`), Apache-2.0 | ~195 GB total; **50M subset: `listens.parquet` 369 MB, `multi_event.parquet` 384 MB** (HF tree API) | b ● (time-of-day derivable), d ◐ (organic vs rec, no propensities) |
| **Spotify MSSD** [13] | Sessions with skip_1/2/3, hour_of_day, context_type, shuffle, pause; audio features | 160M sessions, ~3.7M tracks, 8 weeks; "a subset … uniformly random recommendation setting" | Skips | **Withdrawn**: AIcrowd (8 Jul 2024) says "not available for download anymore" | — | b ●, d ● *if obtainable*; cite only |
| **Million Playlist Dataset** [14] | Playlists | 1M playlists, >2M tracks | None relevant | **Withdrawn** (AIcrowd) | — | — |
| **LFM-1b / LFM-2b** [15] | Last.fm listening events | LFM-2b ~2B events, 120k users | Plays | **Withdrawn**: "not available for download anymore due to license issues" | — | — |
| **Music4All-Onion** [16] | 26 audio/video/lyrics feature sets + Last.fm listening records | 109,269 tracks; 252,984,396 records; 119,140 users | Plays (no skips), lyric sentiment | Zenodo, **CC BY 4.0** | ~15 GB | b ◐ (content-affect proxy only) |
| **Music4All** (original) [17] | 30 s audio, lyrics, Spotify features, tags | 109,269 songs; 15,602 users | Plays | **Request by email** | UNVERIFIED | b ◐ |
| **#nowplaying-RS** [18] | Tweeted listening events + context + hashtag sentiment (4 lexicons) | 11.6M events, 139K users, 346K tracks | Sparse hashtag sentiment | Zenodo, CC BY 4.0 | ~1.4 GB | b ◐ (text-sentiment context) |
| **Culture-aware LFM-1b derivative** [19] | LFM-1b subset + Spotify acoustic features + country culture data | 55,190 users; 351M events | Plays | Zenodo, CC BY 4.0 | 3.1 GB | b ◐ |
| **MLHD / MLHD+** [20] | Last.fm logs | >27B logs, 583K users | Plays | **UNVERIFIED** (MetaBrainz page access-denied) | UNVERIFIED | — |
| **Yahoo! R3** [21] | Song ratings: MNAR train + random-exposure test | ~300k train ratings / 15.4k users / 1,000 songs; test 5.4k users × 10 random songs | 1–5 ratings | Yahoo Webscope (by request; site unreachable, **UNVERIFIED**) | — | d ◐ (MNAR debiasing, music!) |

### 2.3 OPE / logged-bandit methodology data (not music)

| Dataset | Logging | Propensities | Scale | License | Size | Tests |
|---|---|---|---|---|---|---|
| **Open Bandit Dataset** (ZOZOTOWN) [22] | Uniform Random and Bernoulli Thompson Sampling, run as an A/B test | **Yes, true** (`propensity_score` column) | ~26M impressions; 3-slot slate; 80 items in "All" campaign (ZOZO page says ids 0–80; exact count **UNVERIFIED**) | CC BY 4.0 (paper); code Apache-2.0 | Full size UNVERIFIED; **10k-row-per-policy sample ships in the GitHub repo** | d ● (gold standard: OPE estimate on BTS logs vs on-policy truth from Random) |
| **KuaiRand** [23] | 15 production policies + ~0.37% random replacement | `is_rand` flag, **no propensity column** | Pure: 27,285 users, 7,551 items, 1.44M interactions | CC BY-SA 4.0 | Pure ≈ 184–194 MB logs + 10 MB features | d ◐ (replay on random slice) |
| **Coat** [24] | Self-selected train + random test | Estimated propensities shipped | Small | CC BY-NC | Small | d ◐ |
| **Deezer carousel bandits** [25] | **Simulator**, not logs | n/a | 974,960 user embeddings, 862 playlists | Code/Zenodo license **UNVERIFIED** | user features 1.8 GB | Simulation only |
| **Yahoo! R6** [26] | Random traffic (basis of Li et al. replay) | Uniform | ">33 million events" | Webscope by request (**UNVERIFIED**, DNS failure) | — | d ● if approved |

### 2.4 Tooling notes (Python 3.13 / uv)

| Package | Latest | Python support | Verdict |
|---|---|---|---|
| `obp` [27] | 0.5.7 (2023-04-14) | Classifiers 3.8–3.10; repo pins `<3.10` | **Does not work on 3.13.** On 3.13, uv resolves `obp==0.4.1`, which crashes on pandas 2 (`DataFrame.drop()` TypeError, reproduced by the research agent). Either run it in a separate Python 3.10 uv environment *only to cross-check*, or reimplement IPS/SNIPS/DM/DR (~150 lines) and test them against obp's outputs. |
| `vowpalwabbit` [28] | 9.11.9 | 3.10–3.13 classifiers | Maintained; contextual bandits (`--cb_explore_adf`). |
| `d3rlpy` [29] | 2.8.1 (2025-03-02) | Classifiers 3.9–3.11 | Offline RL + FQE. Overkill for one-step decisions. |
| `scope-rl` [30] | 0.2.1 (2023-07-30) | 3.9–3.11 | Stale; skip. |

---

## 3. The honest gap: what "did the music help?" can and cannot answer

| Sub-question | Answerable with public data? | Best data | Why / caveat |
|---|---|---|---|
| Can wearables recognise affect or stress? | **Yes** (in the lab, well studied) | WESAD (open), PMEmo EDA, HKU956; DEAP/AMIGOS if staff sign EULAs | Lab stimuli are mostly video, sample sizes are 15–40 people, and per-subject normalization matters more than the model. |
| What emotion does a song *express*? | **Yes** | DEAM, PMEmo | This is *perceived* emotion, which is not the same as *induced change*. |
| Will this user *accept* this track in this context? | **Yes, at scale** | Yambda (listen fraction, dislikes, time); SiTunes ratings (small) | Yambda has no mood, activity or biometrics. Context is limited to time and the organic/recommended source. |
| Does affect *change* while listening, and can wearables predict how much? | **Partially** | **SiTunes only** (30 users, 1,406 events) | Pre/post are self-reports in [-1, 1]. HR is per-user normalized. **There is no no-music control arm**, so a change cannot be credited to the music rather than to time passing or **regression to the mean**. |
| Is song X *better* for this person than song Y or silence (a causal effect)? | **No** | — | This needs randomized assignment with a control. No public music dataset has it. PhysioNet BWM has music vs no-music conditions, but only physiological and task outcomes from 28 people. |
| Can OPE say whether a new policy beats the logged one? | **Methodology yes, music no** | OBD (true propensities), KuaiRand random slice | On SiTunes the logging propensities are unknown, so they must be estimated. That makes the estimate model-dependent. |
| Does benefit transfer to unlabeled catalog songs? | **No (not measurable)** | — | Catalog songs have no outcomes. The best available check is whether predicted benefit agrees with SiTunes songs that share features. That is a sanity check, not a validation. |

**Implication:** a credible project should
- make its main claim *predictive* and *within SiTunes* ("wearables add/do not add signal for post-listening affect change"),
- use public large data only for components with real labels (acceptance, affect recognition),
- state the causal gap explicitly rather than paper over it with a simulator.

**Why the current results look the way they do:** the kNN-beats-retriever result in the existing repo is consistent with this picture. With 30 users and labels built from the same feature space, the nearest-neighbour method is essentially reading the label definition back.

---

## 4. Three candidate designs

All three share these defaults:
- `uv` + Python 3.13, polars/pandas, scikit-learn, LightGBM, pytest.
- **User-level splits.** Leave-users-out grouped CV, or a time-based split for Yambda.
- **Confidence intervals by bootstrapping over users, not rows.** Every metric is computed per user first, then averaged.
- A frozen `eval/` protocol written *before* models are tuned.

### Design A: "Do wearables predict music benefit?" (SiTunes, pre-registered)

**Research questions (falsifiable):**
- **RQ1:** Does adding wrist features (HR, intensity, steps, activity) to a model with *pre-state + time/weather + user taste + song features* reduce error in predicting Δvalence and Δarousal on **unseen users**?
- Null hypothesis: no improvement. Threshold: the 95% CI of the per-user MAE difference excludes 0.
- **RQ2:** Is benefit separable from acceptance? Test this by checking whether a model ranks songs differently under the benefit and acceptance objectives (Kendall τ between the rankings), and whether rating predicts Δaffect.

**Data:**
- SiTunes Stages 2–3 (main).
- Stage 1 provides taste priors.
- PMEmo/DEAM give song-level perceived V/A features, joined only where tracks match. Otherwise use audio features from the SiTunes metadata.
- Optional: WESAD to pretrain a wrist-stress feature and test whether it transfers.

**Baselines (each must be beaten, with a CI):**
1. Global mean Δ.
2. **Regression-to-the-mean baseline:** Δ = β·(neutral − pre). This is the most important baseline, because pre-state alone will likely explain much of Δ.
3. Per-user mean (fit on that user's held-out-free history where available).
4. Same-goal / feature kNN (the existing strong baseline).
5. Ridge / logistic regression.
6. LightGBM on all non-wrist features.

**Main model:**
- LightGBM or a small MLP on pre-state + context + taste + song + wrist summary features.
- Optional: a tiny 1D-CNN/GRU over the 30×4 wrist matrix.
- With n ≈ 1.4k, gradient boosting on hand-summarized features will likely win. Report that honestly if so.

**Evaluation protocol:**
- Leave-users-out: 30 users, 5-fold GroupKFold repeated 10× with different seeds, or leave-one-user-out.
- Metrics: MAE/RMSE on Δ, Spearman ρ, plus *ranking* metrics per user, such as NDCG of true Δ among that user's songs.
- 95% CIs from 2,000 bootstrap resamples **of users**. Paired comparisons use per-user differences.
- **Ablations:** −wrist, −env, −pre-state, −taste, wrist-only. Add a **permutation test** that shuffles wrist features within users.
- Power note: with 30 users, only fairly large effects are detectable. Report the minimum detectable effect, not just p-values.
- Fix the circular-evaluation flaw: labels are raw post−pre self-reports, never built from model features.

**Optional OPE module** (makes the project complete):
- Validate IPS/SNIPS/DR implementations on the Open Bandit Dataset sample. Estimate the Random policy's value from BTS logs and compare it with the on-policy ground truth, following the OBP paper's protocol [22].
- Then apply DR with *estimated* propensities (behavior-cloned from Stage 2/3 logs) to compare "recommend max predicted benefit" vs "max predicted rating" policies on SiTunes.
- Report it as model-dependent.

**Effort and compute:**
- 3–4 weeks part-time; another +1–1.5 weeks for the OPE module.
- CPU only. All runs take minutes. Disk under 50 MB (plus the 1.3 GB PMEmo already on hand).

### Design B: "Context-aware acceptance at scale" (Yambda 50M)

**Research question:** Do session and time context features (hour, day-of-week, session position, previous-track listen ratios, organic vs recommended) improve **listen-through prediction** (`played_ratio_pct ≥ 50`) beyond user taste + item popularity, and is the gain uniform across users?

**Data:** Yambda `flat/50m`, `listens` + `likes`/`dislikes` + audio embeddings.
- **Download is about 0.75 GB+ for the 50M subset alone, so it needs your approval first.**
- Embeddings size is UNVERIFIED.

**Baselines:**
- Popularity
- User-mean listen ratio
- Item-kNN / implicit ALS
- Logistic regression on taste features

**Main model:**
- LightGBM ranker with context features.
- Stretch: SASRec-style sequence model (CPU-trainable on a user subsample).

**Evaluation:**
- Global temporal split (last N days as test) plus a user-cold slice.
- AUC / logloss for listen-through. NDCG@10 for next-track ranking.
- CIs by bootstrap over users.
- Separate metrics for organic vs recommended events (exposure-bias check).

**Relation to the core idea:** This answers "acceptance in context" convincingly, but **it says nothing about mood benefit.** It strengthens the acceptance half of the benefit/acceptance split.

**Effort and compute:**
- 4–6 weeks.
- 50M rows in polars needs about 8–16 GB RAM. Use lazy scans and downsample users if your laptop has less.
- LightGBM on CPU: tens of minutes per run.

### Design C: "Off-policy evaluation done right" (OBD → SiTunes)

**Research question:** How far off are IPS, SNIPS, DM and DR from on-policy ground truth when propensities are (i) true or (ii) estimated? Does that ordering hold on a semi-synthetic music bandit built from SiTunes?

**Data:**
- Open Bandit Dataset: start with the GitHub sample, and use the full zip only with approval. Full size is UNVERIFIED.
- KuaiRand-Pure random slice (~194 MB, borderline, so ask first).
- SiTunes.

**Baselines:** DM (what v1 effectively did), IPS, SNIPS, DR, Switch-DR.

**Evaluation:**
- Relative error vs ground truth over bootstrap resamples.
- Sensitivity to propensity clipping and to propensity-model misspecification.

**Effort and compute:**
- 3 weeks, CPU only.
- `obp` cannot be the runtime dependency on Python 3.13 (§2.4). Implement the estimators yourself, and keep a separate py3.10 uv environment only for parity tests.

**Weakness:** it is a strong *methods* project, but on its own it drifts from music and mood.

---

## 5. Recommendation: Design A (+ the optional OPE module)

Why:
- **It is the only design that tests the actual core idea.** "Benefit ≠ acceptance" and "wearables matter" are claims about *SiTunes-type data*, and SiTunes is the only dataset that has them.
- **It directly fixes the code-review findings.**
  - Circular labels are replaced by raw Δ self-reports.
  - The kNN-beats-model result becomes an explicit baseline rather than an embarrassment.
  - DM-only evaluation is replaced by DR with OBD-validated estimators.
- **It is honest about scale.** A well-run study of 30 users with user-level CIs, a regression-to-the-mean baseline and ablations is more convincing to a reviewer than a large pipeline with no proper baselines. A clear null result ("wrist features add no signal beyond pre-state, CI [−0.01, +0.02]") is still a publishable-quality portfolio finding.
- **It fits a student laptop.** Everything runs in minutes on CPU.

If time remains after Design A, add **Design B** as a second chapter (with download approval). It shows the same acceptance model scales to 1M users. Present it as the "acceptance at scale" counterpart, not as evidence about mood.

### Suggested repo structure (new project, src layout, uv)

```
music-benefit/
├── pyproject.toml            # uv-managed; ruff + pyright + pytest config
├── uv.lock
├── README.md                 # question, headline result with CI, how to reproduce
├── configs/                  # one YAML per experiment (seeds, folds, features)
├── data/
│   ├── raw/                  # gitignored; fetched by scripts/fetch_*.py
│   ├── processed/            # gitignored; parquet
│   └── DATA_CARD.md          # source, license, version/date, known issues
├── src/music_benefit/
│   ├── data/                 # loaders (situnes.py, pmemo.py, obd.py), schema checks (pandera)
│   ├── features/             # wrist summaries, context, taste, song features
│   ├── splits.py             # GroupKFold by user, manifest written to disk
│   ├── models/               # baselines.py (mean, RTM, kNN, ridge), gbm.py, seq.py
│   ├── ope/                  # ips.py, snips.py, dm.py, dr.py, propensity.py
│   ├── eval/                 # metrics.py, bootstrap_users.py, ablations.py
│   └── cli.py                # `uv run music-benefit run configs/a_main.yaml`
├── tests/
│   ├── test_splits.py        # no user leaks across folds (hypothesis property test)
│   ├── test_ope.py           # estimators unbiased on a synthetic bandit with known value
│   └── test_metrics.py
├── notebooks/                # EDA only; no logic that results depend on
├── reports/
│   ├── PREREGISTRATION.md    # RQs, metrics, thresholds — committed before tuning
│   ├── results/              # JSON per run (config hash, seed, metrics, CIs)
│   └── figures/
└── .github/workflows/ci.yml  # ruff, pyright, pytest on a tiny fixture dataset
```

Key structural rules:
1. Labels come only from raw outcome columns. Features never touch label construction.
2. Every reported number has a config file, a seed and a per-user CI.
3. `PREREGISTRATION.md` is committed before any model comparison. Changes after that are listed in a "deviations" section.

---

## 6. Open items for the user

- **Downloads over 200 MB that need your OK:** Yambda 50M (~0.75 GB+), KuaiRand-Pure (~194 MB, borderline), full Open Bandit Dataset (size unverified), WESAD (~2.5 GB), PhysioNet BWM (558 MB zip).
- **Gated or approval-based datasets** that we did *not* request:
  - DEAP and AMIGOS: EULA signed by permanent academic staff, so a professor would have to sign.
  - DREAMER: restricted Zenodo record.
  - Music4All (original): email request.
  - Yahoo R3/R6: Webscope request.
  - Spotify MSSD/MPD: withdrawn, with Spotify Research the only route.
  - **No Hugging Face login was needed for anything recommended**, since Yambda is ungated.
- **SiTunes has no license file.** Before publishing derived data or a public repo that includes it, ask the authors or only ship download scripts.
- **Not verified:** Melon Playlist Dataset, Echo Nest Taste Profile, and a full browse of Hugging Face Hub search results for music+physiology. The QMUL pages (DEAP/AMIGOS) were unreachable (TLS/503), and the HKU956 DataHub page returned 403.

---

## 7. Sources

1. SiTunes: https://github.com/JiayuLi-997/SiTunes_dataset (stats table); field docs https://raw.githubusercontent.com/JiayuLi-997/SiTunes_dataset/main/SiTunes/README.md ; license/size via https://api.github.com/repos/JiayuLi-997/SiTunes_dataset . Paper: Grigorev et al., CHIIR 2024, doi:10.1145/3627508.3638343 (venue from DBLP; ACM page 403).
2. PMEmo: https://github.com/HuiZhangDB/PMEmo ; paper https://arxiv.org/abs/1908.03692 (rater and EDA counts from the paper).
3. HKU956: https://datahub.hku.hk/articles/dataset/HKU956_Dataset_in_Detecting_Music-Induced_Emotion_Based_on_Acoustic_Analysis_and_Physiological_Sensing_A_Multimodal_Approach_/21080821 ; https://doaj.org/article/346745e2a56a4e0691cd342ab5c51d58 (**search-result level only**).
4. PhysioNet Brain Wearable Monitoring v1.0.0: https://www.physionet.org/content/brain-wearable-monitoring/1.0.0/
5. DEAP: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/ (returned 503 on 2026-11-12); https://research.utwente.nl/en/publications/deap-a-database-for-emotion-analysis-using-physiological-signals/
6. AMIGOS: https://arxiv.org/abs/1702.02510 ; access page https://www.eecs.qmul.ac.uk/mmv/datasets/amigos/download.html (unreachable).
7. DREAMER: https://zenodo.org/records/546113
8. WESAD: https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection
9. DEAM: https://cvml.unige.ch/databases/DEAM/
10. Daly et al. 2020, Scientific Data: https://pmc.ncbi.nlm.nih.gov/articles/PMC7295758 ; mirror https://huggingface.co/datasets/EEGDash/ds002721
11. CASE: https://pmc.ncbi.nlm.nih.gov/articles/PMC6785543
12. Yambda: https://huggingface.co/datasets/yandex/yambda ; API https://huggingface.co/api/datasets/yandex/yambda ; file sizes https://huggingface.co/api/datasets/yandex/yambda/tree/main/flat/50m ; paper https://arxiv.org/abs/2505.22238
13. Spotify MSSD: https://arxiv.org/abs/1901.09851 (random subset quote, 160M sessions, skip definitions); availability https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge
14. MPD: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
15. LFM-2b: https://www.cp.jku.at/datasets/LFM-2b/ ; LFM-1b: https://www.cp.jku.at/datasets/LFM-1b/
16. Music4All-Onion: https://zenodo.org/record/6609677
17. Music4All: https://sites.google.com/view/contact4music4all
18. #nowplaying-RS: https://zenodo.org/records/3248543
19. Culture-aware music rec dataset: https://zenodo.org/records/3477842
20. MLHD: https://zenodo.org/records/1417499
21. Yahoo R3: Marlin & Zemel 2009 (details from secondary sources; **UNVERIFIED**)
22. Open Bandit Dataset/Pipeline: https://research.zozo.com/data.html ; https://github.com/st-tech/zr-obp ; sample https://github.com/st-tech/zr-obp/blob/master/obd/README.md ; paper https://arxiv.org/abs/2008.07146
23. KuaiRand: https://kuairand.com/
24. Coat: https://www.cs.cornell.edu/~schnabts/mnar/
25. Deezer carousel bandits: https://github.com/deezer/carousel_bandits ; https://arxiv.org/abs/2009.06546 ; https://zenodo.org/records/4048678
26. Yahoo R6 / replay: Li et al. https://arxiv.org/abs/1003.0146 (Webscope unreachable)
27. obp: https://pypi.org/project/obp/ (JSON: version 0.5.7, uploaded 2023-04-14, classifiers 3.8–3.10)
28. vowpalwabbit: https://pypi.org/project/vowpalwabbit/
29. d3rlpy: https://pypi.org/project/d3rlpy/
30. scope-rl: https://pypi.org/project/scope-rl/
