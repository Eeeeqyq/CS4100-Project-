"""
Held-out evaluation and scenario gallery for the rebuilt ambient music agent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.common import ACTIVITY_LABELS, BUCKET_LABELS, PROCESSED_DIR, STATE_DIM, TIME_LABELS, state_vector_from_components
from src.hmm.hmm_inference import ACTIVITY_REMAP, corrected_belief, encode_obs_seq
from src.hmm.hmm_model import HMM
from src.music.music_library import MusicLibrary
from src.rl_agent.dqn_agent import DQNAgent
from src.rl_agent.reward_model import HierarchicalRewardModel


MODELS = Path("models")


def load_artifacts():
    df = pd.read_csv(PROCESSED_DIR / "interactions_clean.csv")
    states = np.load(PROCESSED_DIR / "state_vectors.npy")
    if "step_active" not in df.columns:
        if "step_nonzero_frac" in df.columns:
            df["step_active"] = (df["step_nonzero_frac"].astype(float) >= 0.05).astype(int)
        else:
            df["step_active"] = 0
    df["hmm_state"] = states[:, :3].argmax(axis=1)
    hmm = HMM.load(MODELS / "hmm.npz")
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=8, hidden=128)
    agent.load(MODELS / "agent.pt")
    agent.epsilon = 0.0
    reward_model = HierarchicalRewardModel.load(MODELS / "reward_model.json")
    library = MusicLibrary.build()
    return df, states, hmm, agent, reward_model, library


def evaluate_policy(actions: np.ndarray, df: pd.DataFrame, reward_model: HierarchicalRewardModel) -> dict:
    combined_rewards = []
    emotion_rewards = []
    acceptance_rewards = []
    regrets = []
    match_rates = []
    supports = []
    for action, (_, row) in zip(actions, df.iterrows()):
        per_action = [
            reward_model.expected_components(
                int(row["hmm_state"]),
                int(row["time_bucket"]),
                int(row["activity_majority"]),
                int(row.get("step_active", 0)),
                bucket,
                pre_valence=float(row.get("emo_pre_valence", 0.0)),
                pre_arousal=float(row.get("emo_pre_arousal", 0.0)),
                user_valence_pref=float(row.get("user_valence_pref", 0.0)),
                user_energy_pref=float(row.get("user_energy_pref", 0.0)),
            )
            for bucket in range(8)
        ]
        chosen = per_action[int(action)]
        combined_rewards.append(chosen["combined_reward"])
        emotion_rewards.append(chosen["emotion_benefit"])
        acceptance_rewards.append(chosen["acceptance"])
        regrets.append(max(item["combined_reward"] for item in per_action) - chosen["combined_reward"])
        match_rates.append(int(action) == int(row.get("action_bucket", -1)))
        supports.append(chosen["support"])

    action_counts = np.bincount(actions, minlength=8).astype(np.float64)
    action_probs = action_counts / max(action_counts.sum(), 1.0)
    return {
        "mean_expected_reward": float(np.mean(combined_rewards)),
        "mean_emotion_benefit": float(np.mean(emotion_rewards)),
        "mean_acceptance": float(np.mean(acceptance_rewards)),
        "mean_regret": float(np.mean(regrets)),
        "historical_match_rate": float(np.mean(match_rates)),
        "mean_support": float(np.mean(supports)),
        "action_entropy": float(-np.sum(action_probs * np.log(action_probs + 1e-12))),
        "action_counts": {str(i): int(c) for i, c in enumerate(action_counts.astype(int))},
    }


def state_prior_actions(train_df: pd.DataFrame, reward_model: HierarchicalRewardModel, df: pd.DataFrame) -> np.ndarray:
    best_by_state = {}
    for hmm_state in range(3):
        subset_state = train_df[train_df["hmm_state"] == hmm_state]
        if subset_state.empty:
            subset_state = train_df
        for step_active in [0, 1]:
            subset = subset_state[subset_state["step_active"] == step_active]
            if subset.empty:
                subset = subset_state
            time_mode = int(subset["time_bucket"].mode().iloc[0])
            activity_mode = int(subset["activity_majority"].mode().iloc[0])
            scores = [
                reward_model.expected_reward(hmm_state, time_mode, activity_mode, step_active, action)
                for action in range(8)
            ]
            best_by_state[(hmm_state, step_active)] = int(np.argmax(scores))
    return np.asarray(
        [best_by_state[(int(row["hmm_state"]), int(row.get("step_active", 0)))] for _, row in df.iterrows()],
        dtype=np.int32,
    )


def section_a_held_out(df: pd.DataFrame, states: np.ndarray, agent: DQNAgent, reward_model: HierarchicalRewardModel) -> dict:
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_states = states[df["split"].to_numpy() == "test"]

    dqn_actions = np.asarray([agent.greedy_action(state) for state in test_states], dtype=np.int32)
    always7 = np.full(len(test_df), 7, dtype=np.int32)
    state_prior = state_prior_actions(train_df, reward_model, test_df)

    uniform_expected = []
    for _, row in test_df.iterrows():
        uniform_expected.append(
            np.mean(
                [
                    reward_model.expected_reward(
                        int(row["hmm_state"]),
                        int(row["time_bucket"]),
                        int(row["activity_majority"]),
                        int(row.get("step_active", 0)),
                        action,
                        pre_valence=float(row.get("emo_pre_valence", 0.0)),
                        pre_arousal=float(row.get("emo_pre_arousal", 0.0)),
                        user_valence_pref=float(row.get("user_valence_pref", 0.0)),
                        user_energy_pref=float(row.get("user_energy_pref", 0.0)),
                    )
                    for action in range(8)
                ]
            )
        )

    metrics = {
        "dqn": evaluate_policy(dqn_actions, test_df, reward_model),
        "always7": evaluate_policy(always7, test_df, reward_model),
        "state_prior": evaluate_policy(state_prior, test_df, reward_model),
        "random_uniform_expected_reward": float(np.mean(uniform_expected)),
    }

    print("=" * 88)
    print("SECTION A - HELD-OUT TEST CONTEXTS")
    print("=" * 88)
    print(f"Rows: {len(test_df)}  Users: {test_df['user_id'].nunique()}")
    print(
        f"{'Policy':<16} {'Combined':>10} {'Emotion':>10} {'Accept':>10} "
        f"{'Regret':>10} {'Support':>10} {'Entropy':>9}"
    )
    print("-" * 88)
    for name in ["dqn", "state_prior", "always7"]:
        m = metrics[name]
        print(
            f"{name:<16} {m['mean_expected_reward']:>+10.4f} {m['mean_emotion_benefit']:>+10.4f} "
            f"{m['mean_acceptance']:>+10.4f} {m['mean_regret']:>10.4f} "
            f"{m['mean_support']:>10.2f} {m['action_entropy']:>9.3f}"
        )
    print(f"{'random_uniform':<16} {metrics['random_uniform_expected_reward']:>+10.4f}")
    print()
    print("DQN action counts:", metrics["dqn"]["action_counts"])
    return metrics


SCENARIOS = [
    {
        "name": "Still body, stressed mind",
        "intensity": 5,
        "activity": 0,
        "hr_mean": 18.0,
        "step_mean": 0.0,
        "step_nonzero_frac": 0.0,
        "step_active": 0,
        "time": 1,
        "weather": 2,
        "speed": 0.1,
        "pre_valence": -0.80,
        "pre_arousal": -0.60,
        "mask": 1.0,
        "mode": "wind_down",
        "user_profile": {"user_valence_pref": 0.20, "user_energy_pref": -0.10, "top_genres": ["ambient", "indie"]},
    },
    {
        "name": "Still body, balanced mood",
        "intensity": 5,
        "activity": 0,
        "hr_mean": 18.0,
        "step_mean": 0.0,
        "step_nonzero_frac": 0.0,
        "step_active": 0,
        "time": 1,
        "weather": 2,
        "speed": 0.1,
        "pre_valence": 0.00,
        "pre_arousal": 0.20,
        "mask": 1.0,
        "mode": "focus",
        "user_profile": {"user_valence_pref": 0.20, "user_energy_pref": -0.10, "top_genres": ["indie", "classical"]},
    },
    {
        "name": "Same signal, now with steps",
        "intensity": 3,
        "activity": 0,
        "hr_mean": 8.0,
        "step_mean": 8.0,
        "step_nonzero_frac": 0.30,
        "step_active": 1,
        "time": 1,
        "weather": 1,
        "speed": 0.0,
        "pre_valence": -0.80,
        "pre_arousal": 0.50,
        "mask": 1.0,
        "mode": "uplift",
        "user_profile": {"user_valence_pref": 0.10, "user_energy_pref": -0.10, "top_genres": ["indie", "pop", "ambient"]},
    },
]


def section_b_gallery(hmm: HMM, agent: DQNAgent, reward_model: HierarchicalRewardModel, library: MusicLibrary) -> None:
    print("\n" + "=" * 88)
    print("SECTION B - SCENARIO GALLERY")
    print("=" * 88)

    for scenario in SCENARIOS:
        activity_remapped = ACTIVITY_REMAP[int(scenario["activity"])]
        obs_seq = encode_obs_seq(scenario["intensity"], scenario["activity"], hr_mean=scenario["hr_mean"])
        belief = corrected_belief(hmm, obs_seq, activity_remapped)
        state = state_vector_from_components(
            belief,
            scenario["time"],
            activity_remapped,
            weather_bucket=scenario["weather"],
            gps_speed=scenario["speed"],
            hr_mean_rel_user=scenario["hr_mean"],
            hr_std=6.0,
            pre_valence=scenario["pre_valence"],
            pre_arousal=scenario["pre_arousal"],
            pre_emotion_mask=scenario["mask"],
            user_valence_pref=scenario["user_profile"]["user_valence_pref"],
            user_energy_pref=scenario["user_profile"]["user_energy_pref"],
            step_mean=scenario.get("step_mean", 0.0),
            step_nonzero_frac=scenario.get("step_nonzero_frac", 0.0),
        )
        action = int(agent.greedy_action(state))
        components = reward_model.expected_components(
            int(np.argmax(belief)),
            int(scenario["time"]),
            int(activity_remapped),
            int(scenario.get("step_active", 0)),
            action,
            pre_valence=float(scenario["pre_valence"]),
            pre_arousal=float(scenario["pre_arousal"]),
            user_valence_pref=float(scenario["user_profile"]["user_valence_pref"]),
            user_energy_pref=float(scenario["user_profile"]["user_energy_pref"]),
        )
        tracks = library.get_tracks(action, n=3, context={"mode": scenario["mode"], "user_profile": scenario["user_profile"]})

        print(f"\n{scenario['name']}")
        print(
            f"  Context: intensity={scenario['intensity']:<4} hr={scenario['hr_mean']:<5.1f} "
            f"activity={ACTIVITY_LABELS[activity_remapped]:<13} time={TIME_LABELS[scenario['time']]} "
            f"step_active={int(scenario.get('step_active', 0))}"
        )
        print(
            f"  Mood   : pre_valence={scenario['pre_valence']:+.2f} "
            f"pre_arousal={scenario['pre_arousal']:+.2f} mask={scenario['mask']:.0f}"
        )
        print(f"  Belief : [{belief[0]:.2f} {belief[1]:.2f} {belief[2]:.2f}]")
        print(
            f"  Action : bucket {action} ({BUCKET_LABELS[action]})  "
            f"combined={components['combined_reward']:+.3f}  "
            f"emotion={components['emotion_benefit']:+.3f}  accept={components['acceptance']:+.3f}"
        )
        for _, track in tracks.iterrows():
            explanation = f" | {track['dynamic_reason']}" if str(track.get("dynamic_reason", "")) else ""
            print(
                f"    - {track['track_name']} / {track['artist']} "
                f"[{track['source']}, score={track['score']:.2f}]{explanation}"
            )


def interactive_mode(hmm: HMM, agent: DQNAgent, reward_model: HierarchicalRewardModel, library: MusicLibrary) -> None:
    print("\n" + "=" * 88)
    print("INTERACTIVE MODE")
    print("=" * 88)
    print("Enter q to quit.")
    while True:
        raw = input("Intensity (default 5): ").strip()
        if raw.lower() == "q":
            break
        intensity = float(raw) if raw else 5.0

        raw = input("Activity raw code 0=still 1=trans 2=walk 4=lying 5=run (default 0): ").strip()
        if raw.lower() == "q":
            break
        activity = int(raw) if raw else 0

        raw = input("HR mean (default 0): ").strip()
        if raw.lower() == "q":
            break
        hr_mean = float(raw) if raw else 0.0

        raw = input("Step mean (default 0): ").strip()
        if raw.lower() == "q":
            break
        step_mean = float(raw) if raw else 0.0

        raw = input("Step nonzero fraction in [0,1] (default 0): ").strip()
        if raw.lower() == "q":
            break
        step_nonzero_frac = float(raw) if raw else 0.0

        raw = input("Time bucket 0=morning 1=afternoon 2=evening (default 1): ").strip()
        if raw.lower() == "q":
            break
        time_bucket = int(raw) if raw else 1

        raw = input("Pre-valence in [-1,1], blank for passive fallback: ").strip()
        if raw.lower() == "q":
            break
        pre_valence = float(raw) if raw else None

        raw = input("Pre-arousal in [-1,1], blank for passive fallback: ").strip()
        if raw.lower() == "q":
            break
        pre_arousal = float(raw) if raw else None

        mask = 1.0 if pre_valence is not None and pre_arousal is not None else 0.0
        step_active = int(step_nonzero_frac >= 0.05)
        activity_remapped = ACTIVITY_REMAP.get(activity, 0)
        belief = corrected_belief(hmm, encode_obs_seq(intensity, activity, hr_mean=hr_mean), activity_remapped)
        state = state_vector_from_components(
            belief,
            time_bucket,
            activity_remapped,
            weather_bucket=1,
            gps_speed=0.0,
            hr_mean_rel_user=hr_mean,
            hr_std=6.0,
            pre_valence=pre_valence,
            pre_arousal=pre_arousal,
            pre_emotion_mask=mask,
            user_valence_pref=0.0,
            user_energy_pref=0.0,
            step_mean=step_mean,
            step_nonzero_frac=step_nonzero_frac,
        )
        action = int(agent.greedy_action(state))
        components = reward_model.expected_components(
            int(np.argmax(belief)),
            time_bucket,
            activity_remapped,
            step_active,
            action,
            pre_valence=0.0 if pre_valence is None else float(pre_valence),
            pre_arousal=0.0 if pre_arousal is None else float(pre_arousal),
            user_valence_pref=0.0,
            user_energy_pref=0.0,
        )
        tracks = library.get_tracks(action, n=5)

        print(f"\nBelief: [{belief[0]:.2f} {belief[1]:.2f} {belief[2]:.2f}]")
        print(
            f"Recommended bucket: {action} ({BUCKET_LABELS[action]})  "
            f"combined={components['combined_reward']:+.3f}  "
            f"emotion={components['emotion_benefit']:+.3f}  accept={components['acceptance']:+.3f} "
            f"step_active={step_active}"
        )
        for _, track in tracks.iterrows():
            explanation = f" | {track['dynamic_reason']}" if str(track.get("dynamic_reason", "")) else ""
            print(f"  - {track['track_name']} / {track['artist']} [{track['source']}]{explanation}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained ambient music agent.")
    parser.add_argument("--interactive", action="store_true", help="Open an interactive context query loop after the report.")
    args = parser.parse_args()

    df, states, hmm, agent, reward_model, library = load_artifacts()
    metrics = section_a_held_out(df, states, agent, reward_model)
    section_b_gallery(hmm, agent, reward_model, library)

    report = {
        "held_out_metrics": metrics,
        "model_files": {
            "hmm": str(MODELS / "hmm.npz"),
            "agent": str(MODELS / "agent.pt"),
            "reward_model": str(MODELS / "reward_model.json"),
        },
    }
    (MODELS / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nSaved models/eval_report.json")

    if args.interactive:
        interactive_mode(hmm, agent, reward_model, library)


if __name__ == "__main__":
    main()
