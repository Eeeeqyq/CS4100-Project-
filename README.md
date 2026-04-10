# Ambient Music Mood Prediction

**CS4100 Artificial Intelligence — Northeastern University**

Given a user's physiological signals, environmental context, and emotional state, can we predict what kind of music they'll listen to? This project uses a two-stage pipeline — an unsupervised Hidden Markov Model for latent physiological state discovery, followed by a supervised feedforward neural network for mood bucket classification — to investigate whether wrist-sensor data and situational context carry predictive signal for music preference.

## Why Prediction Instead of Reinforcement Learning

The original project concept framed this as an RL problem: an agent recommends music and learns from user feedback over time. We pivoted away from RL for a structural reason. RL requires episodes with sequential decisions, temporal credit assignment, and terminal states. Music listening doesn't have this structure — a single recommendation is a one-shot context→action mapping, not a multi-step decision process where early actions affect later outcomes. An initial DQN implementation confirmed this concern: it performed slightly *worse* than a simple lookup table (combined reward 0.163 vs. 0.165), indicating that the RL machinery was solving a problem that didn't exist in the data.

Supervised classification is the honest framing: given your current situation, predict which category of music fits. The HMM provides the latent state modeling that makes the problem interesting, and the neural network handles the prediction. Both are core course topics (probabilistic graphical models, gradient descent, backpropagation) applied to a problem they actually fit.

## Architecture

```
Wrist Sensor Data (30 timesteps × 4 channels)
    │
    ▼
┌──────────────────────────────┐
│  Observation Discretization  │
│  HR (3 bins) × Intensity     │
│  (3 bins) × Activity (3 cat) │
│  = 27 possible observations  │
└──────────┬───────────────────┘
           │ 30-step discrete sequences
           ▼
┌──────────────────────────────┐
│  HMM (3 states, 27 obs)     │
│  Baum-Welch (unsupervised)   │
│  → 3D belief distribution    │
└──────────┬───────────────────┘
           │ P(state=0), P(state=1), P(state=2)
           ▼
┌──────────────────────────────────────────────┐
│  Feature Vector (~14–21 dimensions)          │
│  HMM beliefs + time + weather + temperature  │
│  + humidity + speed + HR stats + activity     │
│  + pre-emotion valence/arousal + mask         │
└──────────┬───────────────────────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Feedforward Neural Network  │
│  21 → 64 → 32 → 4           │
│  ReLU + Dropout(0.3)         │
│  CrossEntropyLoss + Adam     │
└──────────┬───────────────────┘
           │
           ▼
    Mood Bucket Prediction
    (4 classes from track valence × energy)
```

## Dataset

All data comes from **SiTunes** (Li et al., CHIIR 2024), a situational music recommendation dataset collected via a three-stage field study with smart wristbands.

| Component | Source | Shape | Content |
|-----------|--------|-------|---------|
| Interactions | Stage2/interactions.csv | 897 × 11 | user, track, rating, pre/post emotion V/A |
| Wrist physio | Stage2/wrist.npy | 897 × 30 × 4 | heart rate, activity intensity, steps, activity type |
| Environment | Stage2/env.json | 897 records | time period, weather, temperature, humidity, GPS, speed |
| Track metadata | music_metadata/ | 936 tracks | Spotify audio features (valence, energy, genre, etc.) |

Heart rate values are normalized per user (deviation from personal baseline). Activity types include still, lying, walking, running, and transitional states. The dataset is heavily sedentary: 86% of timesteps are still or lying, with only 15% walking and <1% running.

### Mood Bucket Target

Classification targets are derived from the Spotify audio features of the track each user listened to. Tracks are split into four quadrants using the median of valence (0.74) and energy (0.67):

| Bucket | Valence | Energy | Count | Example Genres |
|--------|---------|--------|-------|----------------|
| happy-energetic | high | high | 242 | hip-hop, rock, pop, country |
| calm-relaxed | high | low | 172 | jazz, rock, other |
| tense-dark | low | high | 168 | rock, alternative, electronic |
| sad-melancholic | low | low | 225 | pop, electronic, new age |

90 interactions (10%) had no matching track metadata and were excluded, leaving 807 interactions for training and evaluation.

## HMM Design Choices

The HMM is implemented from scratch in NumPy (no hmmlearn) using scaled forward-backward and Baum-Welch.

**3 hidden states** because the wrist data has three natural physiological regimes: rest/lying (low HR, minimal activity), general sedentary (normal HR, low-moderate activity), and walking/mobile (elevated HR, high activity). More states would overfit on 897 sequences; fewer would fail to distinguish rest from wakefulness.

**27 discrete observations** (3 × 3 × 3) from tercile-binning heart rate and activity intensity, plus collapsing activity type into sedentary/walking/running. This keeps the emission matrix small enough to estimate reliably (81 parameters for 3 states × 27 observations) while capturing the key physiological dimensions. The observation encoding is `hr_bin * 9 + intensity_bin * 3 + activity_bin`.

**Baum-Welch training** converged in 38 iterations with Laplace smoothing (ε=1e-6) on emission and transition counts to prevent zero probabilities. Initialization uses random Dirichlet draws to break symmetry.

The trained HMM discovered three states with clear physiological interpretations:

| State | Interactions | Mean HR | Activity | Interpretation |
|-------|-------------|---------|----------|----------------|
| 0 | 309 (34%) | -9.3 | 4.7 (very low) | Rest/lying — 86% lying timesteps |
| 1 | 474 (53%) | +5.0 | 31.4 (moderate) | Sedentary/awake — mixed still + transitional |
| 2 | 114 (13%) | +2.5 | 26.9 (moderate) | Walking/mobile — 88% walking timesteps |

Transition matrix self-loop probabilities are 0.989, 0.981, and 0.961 — states are sticky within 30-step windows, which makes physiological sense.

## Neural Network Design Choices

The classifier is a two-hidden-layer feedforward network implemented in PyTorch.

**Layer sizes: 21 → 64 → 32 → 4.** The input dimension (21) comes from the full feature set. The first hidden layer (64 units) is roughly 3× the input dimension — a common heuristic that gives the network enough capacity to learn nonlinear feature combinations without being so large that it memorizes the training set. The second hidden layer (32 units) compresses the representation toward the output dimension (4 classes), creating a bottleneck that forces the network to learn compact, generalizable features rather than preserving all input information. With only 543 training examples, a larger network (e.g., 128 → 64) would overfit faster; a smaller one (e.g., 32 → 16) might underfit on 21 input features.

**ReLU activations** because they're the standard default for feedforward networks — no vanishing gradient problem, computationally cheap, and there's no reason to use anything fancier for this problem.

**Dropout (0.3)** between hidden layers as regularization. With 543 training examples and ~3,000 learnable parameters, overfitting is the primary risk. 0.3 means each neuron has a 30% chance of being zeroed during training, which prevents co-adaptation and acts as implicit ensemble averaging.

**Adam optimizer (lr=0.001, weight_decay=1e-4)** because Adam adapts per-parameter learning rates and handles sparse gradients well. The learning rate 0.001 is Adam's recommended default. Weight decay adds L2 regularization to further constrain the model.

**Class-weighted CrossEntropyLoss** to handle mild class imbalance (242 happy-energetic vs. 168 tense-dark). Weights are inversely proportional to class frequency so the model doesn't just learn to predict the majority class.

**Early stopping (patience=30)** monitors validation loss and restores the best checkpoint. This is the most important regularization mechanism — it stops training before the model memorizes the training set.

**5-seed averaging** for reported results. With small datasets, a single random initialization can produce misleading numbers. Averaging across seeds gives mean ± std, which is more honest.

## Results

### Ablation Study

All models were evaluated on 264 held-out interactions from 10 users not seen during training.

| Model | Features | Accuracy | F1 Macro |
|-------|----------|----------|----------|
| Random baseline | — | 0.246 ± 0.025 | 0.244 |
| Majority baseline | — | 0.295 | 0.114 |
| **A: Full model** | HMM + context + physio + emotion | 0.270 ± 0.014 | 0.230 ± 0.039 |
| **B: No HMM** | context + physio + emotion | **0.292 ± 0.031** | **0.274 ± 0.028** |
| **C: No emotion** | HMM + context + physio | 0.283 ± 0.022 | 0.260 ± 0.040 |
| **D: Context only** | time + weather + activity | 0.275 ± 0.020 | 0.265 ± 0.026 |

### Interpretation

No model variant meaningfully exceeds the baselines. The full model (A) actually performs *worst*, suggesting that adding more features gives the network more noise to overfit to. The HMM belief states hurt rather than help (removing them improves accuracy by +0.023). All differences are within error bars.

The HMM successfully discovers physiologically meaningful latent states — rest, sedentary, and walking are cleanly separated. The problem is downstream: these physical activity states don't predict music mood preference. Whether someone is lying down or walking has near-zero correlation with whether they'll listen to happy-energetic vs. sad-melancholic music. The same is true for time of day, weather, temperature, and self-reported pre-emotion.

This is a negative result, but an informative one. It suggests that situational music preference in this dataset is driven primarily by personal taste and recommender system behavior rather than by immediate physiological or environmental context.

## Datasets That Could Provide Lift

We evaluated several supplementary datasets during project scoping:

**#nowplaying-RS** (11.6M listening events, CC BY 4.0 on Zenodo) has Spotify audio features, timestamps, geolocation, and hashtag-derived mood context. Its scale could support training a larger classifier, but it lacks physiological signals entirely.

**LifeSnaps** (71 participants, 4 months, Fitbit Sense data on Zenodo) has wristband heart rate, steps, and ecological momentary mood assessments from daily life. The signal format closely matches SiTunes' wrist data, making it suitable for HMM pretraining. However, mood labels are 96.8% sparse (only ~5,000 of 160,000 hourly records have mood annotations), and the hourly temporal resolution is much coarser than SiTunes' per-minute windowing.

**CASE** (30 subjects, continuous V/A annotations at 20Hz with 1000Hz physio) provides the densest physio→affect mapping available, but from chest-mounted lab sensors during passive video watching — the domain gap to consumer wristbands during daily life is too large for practical transfer.

**MuSe** (90,001 tracks with valence/arousal/dominance from Last.fm tags) validated that our four mood quadrants correspond to musically coherent categories (calm-relaxed maps to indie/ambient/folk; tense-dark maps to rock/alternative/electronic). We used this for methodology justification rather than as a pipeline dependency.

## Next Steps

**Change the prediction target.** The current target (track mood bucket) is a property of the music, largely determined by the recommender system rather than the user's situation. Predicting the user's post-emotion quadrant — "will this person feel happy-energetic or sad-melancholic after listening?" — has stronger theoretical grounding. The SiTunes paper found that subjective pre-emotion features significantly predict post-listening emotional outcomes.

**Add user-level features.** Per-user mood bucket distributions vary substantially (some users listen to 50%+ happy-energetic; others favor tense-dark). A user preference embedding or historical bucket proportions from training data would capture personal taste, which our exploration suggests is the dominant factor in music choice.

**Reduce classification granularity.** Binary classification (high-valence vs. low-valence, or high-energy vs. low-energy) may surface signal that 4-way classification is too noisy to detect.

**Pretrain the HMM on LifeSnaps.** Despite sparsity, LifeSnaps offers 5,000+ labeled physio windows from diverse daily-life activities. Since the HMM trains unsupervised, even the unlabeled physio sequences (60,000+ hours with heart rate and steps) would help it discover more robust physiological states than SiTunes' 897 sedentary-dominated sequences alone.

**Use Stage 3 data.** SiTunes Stage 3 (509 interactions from a situational recommender) may show stronger context→preference coupling since the recommender was explicitly designed to adapt to situations. Training on Stage 2 and evaluating on Stage 3 tests cross-recommender generalization.