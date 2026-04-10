"""
Generate reality-anchored synthetic contexts from the train split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.common import MODELS_DIR, PROCESSED_DIR, state_vector_from_components
from src.hmm.hmm_inference import corrected_belief
from src.hmm.hmm_model import HMM
from src.rl_agent.reward_model import HierarchicalRewardModel


SEED = 99
SYNTHETIC_ROWS = 900
REBALANCE_TEMPERATURE = 0.78
TIME_SMOOTH = 0.15


def tempered_probs(counts: pd.Series, temperature: float) -> np.ndarray:
    probs = counts.sort_index().to_numpy(dtype=np.float64)
    probs /= probs.sum()
    probs = np.power(probs, temperature)
    probs /= probs.sum()
    return probs


def build_reward_model(train_df: pd.DataFrame) -> HierarchicalRewardModel:
    reward_model = HierarchicalRewardModel(seed=SEED).fit(train_df)
    reward_model.save(MODELS_DIR / "reward_model.json")
    return reward_model


def build_time_transition(train_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    start_counts = np.ones(3, dtype=np.float64)
    trans = np.ones((3, 3), dtype=np.float64) * TIME_SMOOTH
    for _, user_df in train_df.sort_values(["user_id", "timestamp", "inter_id"]).groupby("user_id"):
        buckets = user_df["time_bucket"].astype(int).tolist()
        if not buckets:
            continue
        start_counts[buckets[0]] += 1.0
        for left, right in zip(buckets[:-1], buckets[1:]):
            trans[left, right] += 1.0
    start_probs = start_counts / start_counts.sum()
    trans /= trans.sum(axis=1, keepdims=True)
    return start_probs, trans


def sample_template_index(
    train_df: pd.DataFrame,
    rng: np.random.Generator,
    activity: int,
    time_bucket: int,
) -> int:
    candidates = train_df[(train_df["activity_majority"] == activity) & (train_df["time_bucket"] == time_bucket)]
    if candidates.empty:
        candidates = train_df[train_df["activity_majority"] == activity]
    if candidates.empty:
        candidates = train_df
    return int(rng.choice(candidates.index.to_numpy()))


def make_synthetic_rows(
    train_df: pd.DataFrame,
    train_wrist: np.ndarray,
    hmm: HMM,
    reward_model: HierarchicalRewardModel,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    rng = np.random.default_rng(SEED)

    activity_counts = train_df["activity_majority"].value_counts().sort_index()
    activity_probs = tempered_probs(activity_counts, temperature=REBALANCE_TEMPERATURE)
    time_activity = {}
    for time_bucket, group in train_df.groupby("time_bucket"):
        counts = group["activity_majority"].value_counts().reindex(range(5), fill_value=0) + 1
        probs = tempered_probs(counts, temperature=REBALANCE_TEMPERATURE)
        time_activity[int(time_bucket)] = probs

    session_counts = train_df.groupby("user_id").size().to_numpy(dtype=np.int32)
    start_probs, trans = build_time_transition(train_df)

    rows = []
    state_vectors = []
    synthetic_user_id = 5000
    total_rows = 0

    real_by_state = {
        state: group.copy()
        for state, group in train_df.groupby("hmm_state")
    }
    action_counts = train_df["action_bucket"].value_counts().sort_index()
    action_probs = action_counts.to_numpy(dtype=np.float64)
    action_probs /= action_probs.sum()

    while total_rows < SYNTHETIC_ROWS:
        n_sessions = int(rng.choice(session_counts))
        current_time = int(rng.choice(3, p=start_probs))

        for session_idx in range(n_sessions):
            if total_rows >= SYNTHETIC_ROWS:
                break

            activity = int(rng.choice(5, p=time_activity.get(current_time, activity_probs)))
            template_idx = sample_template_index(train_df, rng, activity=activity, time_bucket=current_time)
            template = train_df.loc[template_idx]
            obs_seq = train_wrist[template_idx]

            belief = corrected_belief(hmm, obs_seq, activity)
            hmm_state = int(belief.argmax())
            state_vector = state_vector_from_components(
                belief,
                current_time,
                activity,
                weather_bucket=int(template.get("weather_bucket", 1)),
                gps_speed=float(template.get("gps_speed", 0.0)),
                hr_mean_rel_user=float(template.get("hr_mean_rel_user", 0.0)),
                hr_std=float(template.get("hr_std", 0.0)),
                pre_valence=float(template.get("emo_pre_valence", 0.0)),
                pre_arousal=float(template.get("emo_pre_arousal", 0.0)),
                pre_emotion_mask=0.0,
                user_valence_pref=float(template.get("user_valence_pref", 0.0)),
                user_energy_pref=float(template.get("user_energy_pref", 0.0)),
            )

            action = int(rng.choice(8, p=action_probs))

            source_for_mood = real_by_state.get(hmm_state, train_df)
            mood_seed = source_for_mood.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
            reward = reward_model.sample_reward(
                hmm_state,
                current_time,
                activity,
                action,
                pre_valence=float(mood_seed["emo_pre_valence"]),
                pre_arousal=float(mood_seed["emo_pre_arousal"]),
            )
            components = reward_model.expected_components(
                hmm_state,
                current_time,
                activity,
                action,
                pre_valence=float(mood_seed["emo_pre_valence"]),
                pre_arousal=float(mood_seed["emo_pre_arousal"]),
                user_valence_pref=float(template.get("user_valence_pref", 0.0)),
                user_energy_pref=float(template.get("user_energy_pref", 0.0)),
            )
            dv, da = reward_model.sample_mood_delta(reward)
            pre_valence = float(mood_seed["emo_pre_valence"])
            pre_arousal = float(mood_seed["emo_pre_arousal"])
            post_valence = float(np.clip(pre_valence + dv, -0.99, 0.99))
            post_arousal = float(np.clip(pre_arousal + da, -0.99, 0.99))

            hour = {0: 9, 1: 14, 2: 21}[current_time]
            timestamp = 1_704_067_200 + (synthetic_user_id - 5000) * 86_400 + session_idx * 14_400 + hour * 3_600

            rows.append(
                {
                    "inter_id": total_rows + 1,
                    "user_id": synthetic_user_id,
                    "item_id": -1,
                    "timestamp": timestamp,
                    "timestamp_local": pd.to_datetime(timestamp, unit="s", utc=True).tz_convert("America/New_York").isoformat(),
                    "rating": 0,
                    "preference": 0,
                    "emo_pre_valence": round(pre_valence, 6),
                    "emo_pre_arousal": round(pre_arousal, 6),
                    "emo_post_valence": round(post_valence, 6),
                    "emo_post_arousal": round(post_arousal, 6),
                    "reward": reward,
                    "reward_score": float((post_valence - pre_valence) * 0.7 + (post_arousal - pre_arousal) * 0.3),
                    "emotion_benefit": float(components["emotion_benefit"]),
                    "acceptance_score": float(components["acceptance"]),
                    "combined_reward": float(components["combined_reward"]),
                    "action_bucket": action,
                    "time_bucket": current_time,
                    "time_label": {0: "morning", 1: "afternoon", 2: "evening"}[current_time],
                    "activity_majority": activity,
                    "activity_majority_raw": {0: 0, 1: 1, 2: 2, 3: 4, 4: 5}[activity],
                    "activity_last": activity,
                    "activity_last_raw": {0: 0, 1: 1, 2: 2, 3: 4, 4: 5}[activity],
                    "intensity_mean": float(template["intensity_mean"]),
                    "intensity_last": float(template["intensity_last"]),
                    "intensity_bucket_mean": int(template["intensity_bucket_mean"]),
                    "intensity_bucket_last": int(template["intensity_bucket_last"]),
                    "hr_mean": float(template.get("hr_mean", 0.0)),
                    "hr_std": float(template.get("hr_std", 0.0)),
                    "hr_min": float(template.get("hr_min", 0.0)),
                    "hr_max": float(template.get("hr_max", 0.0)),
                    "hr_last": float(template.get("hr_last", 0.0)),
                    "user_hr_baseline_mean": float(template.get("user_hr_baseline_mean", 0.0)),
                    "user_hr_baseline_std": float(template.get("user_hr_baseline_std", 0.0)),
                    "hr_mean_rel_user": float(template.get("hr_mean_rel_user", 0.0)),
                    "hr_std_rel_user": float(template.get("hr_std_rel_user", 0.0)),
                    "weather_bucket": int(template.get("weather_bucket", 1)),
                    "gps_speed": float(template.get("gps_speed", 0.0)),
                    "dataset_stage": "synthetic",
                    "split": "synthetic",
                    "session_order_user": session_idx,
                    "hmm_state": hmm_state,
                    "belief_0": float(belief[0]),
                    "belief_1": float(belief[1]),
                    "belief_2": float(belief[2]),
                    "pre_emotion_mask": 0.0,
                    "user_valence_pref": float(template.get("user_valence_pref", 0.0)),
                    "user_energy_pref": float(template.get("user_energy_pref", 0.0)),
                    "is_synthetic": True,
                    "source_template_idx": template_idx,
                }
            )
            state_vectors.append(state_vector)

            total_rows += 1
            current_time = int(rng.choice(3, p=trans[current_time]))

        synthetic_user_id += 1

    synthetic_df = pd.DataFrame(rows)
    state_vectors = np.asarray(state_vectors, dtype=np.float32)

    real_beliefs = train_df[["belief_0", "belief_1", "belief_2"]].to_numpy(dtype=np.float32)
    real_entropy = -np.sum(real_beliefs * np.log(real_beliefs + 1e-12), axis=1)
    synth_entropy = -np.sum(state_vectors[:, :3] * np.log(state_vectors[:, :3] + 1e-12), axis=1)

    report = {
        "synthetic_rows": int(len(synthetic_df)),
        "synthetic_users": int(synthetic_df["user_id"].nunique()),
        "activity_distribution_real": train_df["activity_majority"].value_counts(normalize=True).sort_index().to_dict(),
        "activity_distribution_synth": synthetic_df["activity_majority"].value_counts(normalize=True).sort_index().to_dict(),
        "time_distribution_real": train_df["time_bucket"].value_counts(normalize=True).sort_index().to_dict(),
        "time_distribution_synth": synthetic_df["time_bucket"].value_counts(normalize=True).sort_index().to_dict(),
        "reward_distribution_real": train_df["reward"].value_counts(normalize=True).sort_index().to_dict(),
        "reward_distribution_synth": synthetic_df["reward"].value_counts(normalize=True).sort_index().to_dict(),
        "belief_entropy_real_mean": float(real_entropy.mean()),
        "belief_entropy_synth_mean": float(synth_entropy.mean()),
        "session_length_real_mean": float(train_df.groupby("user_id").size().mean()),
        "session_length_synth_mean": float(synthetic_df.groupby("user_id").size().mean()),
    }
    return synthetic_df, state_vectors, report


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(PROCESSED_DIR / "interactions_clean.csv")
    states = np.load(PROCESSED_DIR / "state_vectors.npy")
    wrist_obs = np.load(PROCESSED_DIR / "wrist_obs_all.npy")
    df["hmm_state"] = states[:, :3].argmax(axis=1)
    df["belief_0"] = states[:, 0]
    df["belief_1"] = states[:, 1]
    df["belief_2"] = states[:, 2]
    df["is_synthetic"] = False

    train_mask = df["split"].to_numpy() == "train"
    train_df = df[train_mask].reset_index(drop=True)
    train_wrist = wrist_obs[train_mask]

    reward_model = build_reward_model(train_df)
    hmm = HMM.load(MODELS_DIR / "hmm.npz")

    synthetic_df, synthetic_states, report = make_synthetic_rows(train_df, train_wrist, hmm, reward_model)

    synthetic_df.to_csv(PROCESSED_DIR / "synthetic_clean.csv", index=False)
    np.save(PROCESSED_DIR / "synthetic_state_vectors.npy", synthetic_states)
    (PROCESSED_DIR / "synthetic_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Synthetic augmentation written:")
    print(f"  rows={len(synthetic_df)}  users={synthetic_df['user_id'].nunique()}")
    print(f"  entropy_real={report['belief_entropy_real_mean']:.4f}  entropy_synth={report['belief_entropy_synth_mean']:.4f}")
    print("  models/reward_model.json")
    print("  data/processed/synthetic_clean.csv")
    print("  data/processed/synthetic_state_vectors.npy")
    print("  data/processed/synthetic_report.json")


if __name__ == "__main__":
    main()
