"""
Held-out evaluation and scenario gallery for the rebuilt ambient music agent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.common import ACTIVITY_LABELS, BUCKET_LABELS, PROCESSED_DIR, TIME_LABELS
from src.hmm.hmm_inference import ACTIVITY_REMAP, corrected_belief, encode_obs_seq
from src.hmm.hmm_model import HMM
from src.music.music_library import MusicLibrary
from src.rl_agent.dqn_agent import DQNAgent
from src.rl_agent.reward_model import HierarchicalRewardModel


MODELS = Path("models")


def load_artifacts():
    df = pd.read_csv(PROCESSED_DIR / "interactions_clean.csv")
    states = np.load(PROCESSED_DIR / "state_vectors.npy")
    df["hmm_state"] = states[:, :3].argmax(axis=1)
    hmm = HMM.load(MODELS / "hmm.npz")
    agent = DQNAgent(state_dim=5, action_dim=8)
    agent.load(MODELS / "agent.pt")
    agent.epsilon = 0.0
    reward_model = HierarchicalRewardModel.load(MODELS / "reward_model.json")
    library = MusicLibrary.build()
    return df, states, hmm, agent, reward_model, library


def evaluate_policy(actions: np.ndarray, df: pd.DataFrame, reward_model: HierarchicalRewardModel) -> dict:
    expected_rewards = []
    positive_probs = []
    regrets = []
    match_rates = []
    for action, (_, row) in zip(actions, df.iterrows()):
        context_rewards = [
            reward_model.expected_reward(
                int(row["hmm_state"]),
                int(row["time_bucket"]),
                int(row["activity_majority"]),
                bucket,
            )
            for bucket in range(8)
        ]
        expected = context_rewards[int(action)]
        expected_rewards.append(expected)
        positive_probs.append(
            reward_model.positive_prob(
                int(row["hmm_state"]),
                int(row["time_bucket"]),
                int(row["activity_majority"]),
                int(action),
            )
        )
        regrets.append(max(context_rewards) - expected)
        match_rates.append(int(action) == int(row.get("action_bucket", -1)))

    action_counts = np.bincount(actions, minlength=8).astype(np.float64)
    action_probs = action_counts / action_counts.sum()
    return {
        "mean_expected_reward": float(np.mean(expected_rewards)),
        "mean_positive_prob": float(np.mean(positive_probs)),
        "mean_regret": float(np.mean(regrets)),
        "historical_match_rate": float(np.mean(match_rates)),
        "action_entropy": float(-np.sum(action_probs * np.log(action_probs + 1e-12))),
        "action_counts": {str(i): int(c) for i, c in enumerate(action_counts.astype(int))},
    }


def state_prior_actions(train_df: pd.DataFrame, reward_model: HierarchicalRewardModel, df: pd.DataFrame) -> np.ndarray:
    best_by_state = {}
    for hmm_state in range(3):
        subset = train_df[train_df["hmm_state"] == hmm_state]
        if subset.empty:
            subset = train_df
        time_mode = int(subset["time_bucket"].mode().iloc[0])
        activity_mode = int(subset["activity_majority"].mode().iloc[0])
        scores = [
            reward_model.expected_reward(hmm_state, time_mode, activity_mode, action)
            for action in range(8)
        ]
        best_by_state[hmm_state] = int(np.argmax(scores))
    return df["hmm_state"].map(best_by_state).to_numpy(dtype=np.int32)


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
                        action,
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

    print("=" * 72)
    print("SECTION A - HELD-OUT TEST CONTEXTS")
    print("=" * 72)
    print(f"Rows: {len(test_df)}  Users: {test_df['user_id'].nunique()}")
    print(
        f"{'Policy':<16} {'ExpReward':>10} {'PosProb':>10} {'Regret':>10} "
        f"{'Match':>9} {'Entropy':>9}"
    )
    print("-" * 72)
    for name in ["dqn", "state_prior", "always7"]:
        m = metrics[name]
        print(
            f"{name:<16} {m['mean_expected_reward']:>+10.4f} {m['mean_positive_prob']:>10.4f} "
            f"{m['mean_regret']:>10.4f} {m['historical_match_rate']:>8.1%} {m['action_entropy']:>9.3f}"
        )
    print(f"{'random_uniform':<16} {metrics['random_uniform_expected_reward']:>+10.4f}")
    print()
    print("DQN action counts:", metrics["dqn"]["action_counts"])
    return metrics


SCENARIOS = [
    {"name": "Late-night stressed study", "intensity": 4, "activity": 0, "time": 2, "mode": "wind_down"},
    {"name": "Afternoon deep work", "intensity": 7, "activity": 0, "time": 1, "mode": "focus"},
    {"name": "Morning commute walk", "intensity": 24, "activity": 2, "time": 0, "mode": "uplift"},
    {"name": "Evening recovery lie-down", "intensity": 3, "activity": 4, "time": 2, "mode": "wind_down"},
    {"name": "Morning run", "intensity": 118, "activity": 5, "time": 0, "mode": "exercise"},
    {"name": "Workout sprint set", "intensity": 145, "activity": 5, "time": 1, "mode": "exercise"},
]


def section_b_gallery(hmm: HMM, agent: DQNAgent, reward_model: HierarchicalRewardModel, library: MusicLibrary) -> None:
    print("\n" + "=" * 72)
    print("SECTION B - SCENARIO GALLERY")
    print("=" * 72)

    for scenario in SCENARIOS:
        activity_remapped = ACTIVITY_REMAP[int(scenario["activity"])]
        obs_seq = encode_obs_seq(scenario["intensity"], scenario["activity"])
        belief = corrected_belief(hmm, obs_seq, activity_remapped)
        state = np.asarray(
            [belief[0], belief[1], belief[2], scenario["time"] / 2.0, activity_remapped / 4.0],
            dtype=np.float32,
        )
        action = int(agent.greedy_action(state))
        expected_reward = reward_model.expected_reward(
            int(np.argmax(belief)),
            int(scenario["time"]),
            int(activity_remapped),
            action,
        )
        tracks = library.get_tracks(action, n=3, context={"mode": scenario["mode"]})

        print(f"\n{scenario['name']}")
        print(
            f"  Context: intensity={scenario['intensity']:<4} "
            f"activity={ACTIVITY_LABELS[activity_remapped]:<13} time={TIME_LABELS[scenario['time']]}"
        )
        print(f"  Belief : [{belief[0]:.2f} {belief[1]:.2f} {belief[2]:.2f}]")
        print(f"  Action : bucket {action} ({BUCKET_LABELS[action]})  expected_reward={expected_reward:+.3f}")
        for _, track in tracks.iterrows():
            print(
                f"    - {track['track_name']} / {track['artist']} "
                f"[{track['source']}, score={track['score']:.2f}]"
            )


def interactive_mode(hmm: HMM, agent: DQNAgent, reward_model: HierarchicalRewardModel, library: MusicLibrary) -> None:
    print("\n" + "=" * 72)
    print("INTERACTIVE MODE")
    print("=" * 72)
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

        raw = input("Time bucket 0=morning 1=afternoon 2=evening (default 1): ").strip()
        if raw.lower() == "q":
            break
        time_bucket = int(raw) if raw else 1

        activity_remapped = ACTIVITY_REMAP.get(activity, 0)
        belief = corrected_belief(hmm, encode_obs_seq(intensity, activity), activity_remapped)
        state = np.asarray([belief[0], belief[1], belief[2], time_bucket / 2.0, activity_remapped / 4.0], dtype=np.float32)
        action = int(agent.greedy_action(state))
        expected_reward = reward_model.expected_reward(int(np.argmax(belief)), time_bucket, activity_remapped, action)
        tracks = library.get_tracks(action, n=5)

        print(f"\nBelief: [{belief[0]:.2f} {belief[1]:.2f} {belief[2]:.2f}]")
        print(f"Recommended bucket: {action} ({BUCKET_LABELS[action]})  expected_reward={expected_reward:+.3f}")
        for _, track in tracks.iterrows():
            print(f"  - {track['track_name']} / {track['artist']} [{track['source']}]")
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
