"""
Train the Double DQN policy against the hierarchical context-action reward model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.common import MODELS_DIR, PROCESSED_DIR
from src.rl_agent.dqn_agent import DQNAgent
from src.rl_agent.environment import MusicEnv
from src.rl_agent.reward_model import HierarchicalRewardModel


N_EPISODES = 4500
BURN_IN = 400
LR = 1e-3
GAMMA = 0.90
EPS_START = 1.0
EPS_DECAY = 0.997
EPS_MIN = 0.05
BATCH_SIZE = 128
BUFFER_CAP = 20_000
TARGET_SYNC = 20
HIDDEN = 64
EVAL_EVERY = 150
SEED = 42


def load_real_data() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(PROCESSED_DIR / "interactions_clean.csv")
    states = np.load(PROCESSED_DIR / "state_vectors.npy")
    df["hmm_state"] = states[:, :3].argmax(axis=1)
    df["belief_0"] = states[:, 0]
    df["belief_1"] = states[:, 1]
    df["belief_2"] = states[:, 2]
    df["is_synthetic"] = False
    return df, states


def load_synthetic() -> tuple[pd.DataFrame | None, np.ndarray | None]:
    syn_csv = PROCESSED_DIR / "synthetic_clean.csv"
    syn_npy = PROCESSED_DIR / "synthetic_state_vectors.npy"
    if not syn_csv.exists() or not syn_npy.exists():
        return None, None
    df = pd.read_csv(syn_csv)
    states = np.load(syn_npy)
    if "hmm_state" not in df.columns:
        df["hmm_state"] = states[:, :3].argmax(axis=1)
    if "is_synthetic" not in df.columns:
        df["is_synthetic"] = True
    return df, states


def build_or_load_reward_model(train_df: pd.DataFrame) -> HierarchicalRewardModel:
    path = MODELS_DIR / "reward_model.json"
    if path.exists():
        return HierarchicalRewardModel.load(path)
    model = HierarchicalRewardModel(seed=SEED).fit(train_df)
    model.save(path)
    return model


def evaluate_policy(
    actions: np.ndarray,
    df: pd.DataFrame,
    reward_model: HierarchicalRewardModel,
) -> dict:
    expected_rewards = []
    positive_probs = []
    oracle_rewards = []
    historical_match = []
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
        expected_rewards.append(context_rewards[int(action)])
        positive_probs.append(
            reward_model.positive_prob(
                int(row["hmm_state"]),
                int(row["time_bucket"]),
                int(row["activity_majority"]),
                int(action),
            )
        )
        oracle_rewards.append(max(context_rewards))
        historical_match.append(int(action) == int(row.get("action_bucket", -1)))

    expected_rewards = np.asarray(expected_rewards, dtype=np.float64)
    positive_probs = np.asarray(positive_probs, dtype=np.float64)
    oracle_rewards = np.asarray(oracle_rewards, dtype=np.float64)
    historical_match = np.asarray(historical_match, dtype=np.float64)

    action_counts = np.bincount(actions, minlength=8).astype(np.float64)
    action_probs = action_counts / max(action_counts.sum(), 1.0)
    action_entropy = float(-np.sum(action_probs * np.log(action_probs + 1e-12)))

    return {
        "mean_expected_reward": float(expected_rewards.mean()),
        "mean_positive_prob": float(positive_probs.mean()),
        "mean_regret": float((oracle_rewards - expected_rewards).mean()),
        "historical_match_rate": float(historical_match.mean()),
        "action_entropy": action_entropy,
        "action_counts": {str(i): int(c) for i, c in enumerate(action_counts.astype(int))},
    }


def policy_actions(agent: DQNAgent, states: np.ndarray) -> np.ndarray:
    return np.asarray([agent.greedy_action(state) for state in states], dtype=np.int32)


def baseline_actions_state_prior(train_df: pd.DataFrame, reward_model: HierarchicalRewardModel, df: pd.DataFrame) -> np.ndarray:
    state_best = {}
    for hmm_state in range(3):
        scores = []
        subset = train_df[train_df["hmm_state"] == hmm_state]
        if subset.empty:
            subset = train_df
        time_mode = int(subset["time_bucket"].mode().iloc[0])
        activity_mode = int(subset["activity_majority"].mode().iloc[0])
        for action in range(8):
            scores.append(reward_model.expected_reward(hmm_state, time_mode, activity_mode, action))
        state_best[hmm_state] = int(np.argmax(scores))
    return df["hmm_state"].map(state_best).to_numpy(dtype=np.int32)


def train(args: argparse.Namespace) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    real_df, real_states = load_real_data()
    train_df = real_df[real_df["split"] == "train"].reset_index(drop=True)
    val_df = real_df[real_df["split"] == "val"].reset_index(drop=True)
    test_df = real_df[real_df["split"] == "test"].reset_index(drop=True)
    train_states = real_states[real_df["split"].to_numpy() == "train"]
    val_states = real_states[real_df["split"].to_numpy() == "val"]
    test_states = real_states[real_df["split"].to_numpy() == "test"]

    reward_model = build_or_load_reward_model(train_df)

    env_df = train_df.copy()
    env_states = train_states.copy()
    sample_weights = np.ones(len(env_df), dtype=np.float64)

    if args.synthetic_weight > 0:
        synthetic_df, synthetic_states = load_synthetic()
        if synthetic_df is not None and synthetic_states is not None:
            env_df = pd.concat([env_df, synthetic_df], ignore_index=True)
            env_states = np.vstack([env_states, synthetic_states])
            sample_weights = np.concatenate(
                [
                    np.ones(len(train_df), dtype=np.float64),
                    np.full(len(synthetic_df), float(args.synthetic_weight), dtype=np.float64),
                ]
            )

    env = MusicEnv(env_df, env_states, reward_model=reward_model, sample_weights=sample_weights, seed=SEED)
    print(env)
    print(f"  real_train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(f"  synthetic_weight={args.synthetic_weight:.2f}  total_env_contexts={len(env_df)}")

    agent = DQNAgent(
        state_dim=MusicEnv.STATE_DIM,
        action_dim=MusicEnv.ACTION_DIM,
        lr=LR,
        gamma=GAMMA,
        epsilon=EPS_START,
        epsilon_decay=EPS_DECAY,
        epsilon_min=EPS_MIN,
        buffer_cap=BUFFER_CAP,
        batch_size=BATCH_SIZE,
        target_sync=TARGET_SYNC,
        hidden=HIDDEN,
        seed=SEED,
    )

    for _ in range(BURN_IN):
        state = env.reset()
        action = env.sample_action()
        next_state, reward, done, _ = env.step(action)
        agent.replay.push(state, action, reward, next_state, done)

    best_val = -np.inf
    best_checkpoint = None
    log_rows = []

    print("\nTraining Double DQN on one-step contexts...")
    for episode in range(1, N_EPISODES + 1):
        state = env.reset()
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.replay.push(state, action, reward, next_state, done)

        loss = agent.update()
        if episode % 3 == 0:
            extra_loss = agent.update()
            if extra_loss:
                loss = float((loss + extra_loss) / 2.0) if loss else extra_loss

        agent.end_episode()

        if episode % EVAL_EVERY == 0 or episode == 1:
            val_actions = policy_actions(agent, val_states)
            val_metrics = evaluate_policy(val_actions, val_df, reward_model)
            train_actions = policy_actions(agent, train_states[: min(256, len(train_states))])
            preview_metrics = evaluate_policy(
                train_actions,
                train_df.iloc[: min(256, len(train_df))].reset_index(drop=True),
                reward_model,
            )

            row = {
                "episode": episode,
                "loss": float(loss or 0.0),
                "epsilon": float(agent.epsilon),
                "val_expected_reward": val_metrics["mean_expected_reward"],
                "val_positive_prob": val_metrics["mean_positive_prob"],
                "val_regret": val_metrics["mean_regret"],
                "val_action_entropy": val_metrics["action_entropy"],
                "preview_expected_reward": preview_metrics["mean_expected_reward"],
            }
            log_rows.append(row)
            print(
                f"  ep={episode:4d}  val_exp={val_metrics['mean_expected_reward']:+.4f}  "
                f"val_regret={val_metrics['mean_regret']:.4f}  "
                f"entropy={val_metrics['action_entropy']:.3f}  eps={agent.epsilon:.3f}"
            )

            if val_metrics["mean_expected_reward"] > best_val:
                best_val = val_metrics["mean_expected_reward"]
                best_checkpoint = {
                    "q_net": {k: v.detach().cpu().clone() for k, v in agent.q_net.state_dict().items()},
                    "target_net": {k: v.detach().cpu().clone() for k, v in agent.target_net.state_dict().items()},
                    "optimizer": agent.optimizer.state_dict(),
                    "episode": agent.episode,
                    "epsilon": agent.epsilon,
                }

    if best_checkpoint is None:
        raise RuntimeError("Training completed without producing a checkpoint.")

    agent.q_net.load_state_dict(best_checkpoint["q_net"])
    agent.target_net.load_state_dict(best_checkpoint["target_net"])
    agent.optimizer.load_state_dict(best_checkpoint["optimizer"])
    agent.episode = int(best_checkpoint["episode"])
    agent.epsilon = float(best_checkpoint["epsilon"])
    agent.save(MODELS_DIR / "agent.pt")

    test_actions = policy_actions(agent, test_states)
    test_metrics = evaluate_policy(test_actions, test_df, reward_model)
    always7_metrics = evaluate_policy(np.full(len(test_df), 7, dtype=np.int32), test_df, reward_model)
    random_expected = []
    for _, row in test_df.iterrows():
        per_action = [
            reward_model.expected_reward(
                int(row["hmm_state"]),
                int(row["time_bucket"]),
                int(row["activity_majority"]),
                action,
            )
            for action in range(8)
        ]
        random_expected.append(np.mean(per_action))
    random_metrics = {"mean_expected_reward": float(np.mean(random_expected))}
    state_prior_actions = baseline_actions_state_prior(train_df, reward_model, test_df)
    state_prior_metrics = evaluate_policy(state_prior_actions, test_df, reward_model)

    summary = {
        "config": {
            "episodes": N_EPISODES,
            "burn_in": BURN_IN,
            "lr": LR,
            "gamma": GAMMA,
            "epsilon_start": EPS_START,
            "epsilon_decay": EPS_DECAY,
            "epsilon_min": EPS_MIN,
            "batch_size": BATCH_SIZE,
            "buffer_cap": BUFFER_CAP,
            "target_sync": TARGET_SYNC,
            "hidden": HIDDEN,
            "synthetic_weight": args.synthetic_weight,
        },
        "best_val_expected_reward": best_val,
        "test_metrics": test_metrics,
        "baselines": {
            "always7": always7_metrics,
            "random_uniform_expected_reward": random_metrics,
            "state_prior": state_prior_metrics,
        },
    }

    pd.DataFrame(log_rows).to_csv(MODELS_DIR / "training_log.csv", index=False)
    (MODELS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nTest-set policy summary:")
    print(f"  DQN mean expected reward:      {test_metrics['mean_expected_reward']:+.4f}")
    print(f"  DQN mean regret:               {test_metrics['mean_regret']:.4f}")
    print(f"  DQN action entropy:            {test_metrics['action_entropy']:.3f}")
    print(f"  Always-7 expected reward:      {always7_metrics['mean_expected_reward']:+.4f}")
    print(f"  State-prior expected reward:   {state_prior_metrics['mean_expected_reward']:+.4f}")
    print(f"  Random uniform exp. reward:    {random_metrics['mean_expected_reward']:+.4f}")
    print("\nSaved:")
    print("  models/agent.pt")
    print("  models/training_log.csv")
    print("  models/training_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Double DQN policy.")
    parser.add_argument(
        "--synthetic-weight",
        type=float,
        default=0.35,
        help="Sampling weight applied to synthetic contexts relative to real contexts.",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
