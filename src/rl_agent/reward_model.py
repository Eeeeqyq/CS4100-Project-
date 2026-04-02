"""
Hierarchical context-action reward model for offline policy training and evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


OUTCOMES = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)


class HierarchicalRewardModel:
    def __init__(
        self,
        k_action: float = 32.0,
        k_state_action: float = 18.0,
        k_full: float = 10.0,
        seed: int = 42,
    ):
        self.k_action = float(k_action)
        self.k_state_action = float(k_state_action)
        self.k_full = float(k_full)
        self.rng = np.random.default_rng(seed)
        self.global_counts = np.ones(3, dtype=np.float64)
        self.action_counts: dict[int, np.ndarray] = {}
        self.state_action_counts: dict[tuple[int, int], np.ndarray] = {}
        self.full_counts: dict[tuple[int, int, int, int], np.ndarray] = {}
        self.delta_stats = {}
        self.metadata = {}

    @staticmethod
    def _reward_index(reward: int) -> int:
        return {-1: 0, 0: 1, 1: 2}[int(reward)]

    @staticmethod
    def _blend(child_counts: np.ndarray | None, parent_probs: np.ndarray, k: float) -> np.ndarray:
        if child_counts is None:
            return parent_probs
        child_total = float(np.sum(child_counts))
        if child_total <= 0:
            return parent_probs
        child_probs = child_counts / child_total
        weight = child_total / (child_total + k)
        return weight * child_probs + (1.0 - weight) * parent_probs

    def fit(self, df: pd.DataFrame) -> "HierarchicalRewardModel":
        work = df.copy()
        required = ["hmm_state", "time_bucket", "activity_majority", "action_bucket", "reward"]
        missing = [col for col in required if col not in work.columns]
        if missing:
            raise ValueError(f"Reward model fit missing columns: {missing}")

        self.global_counts = np.ones(3, dtype=np.float64)
        self.action_counts.clear()
        self.state_action_counts.clear()
        self.full_counts.clear()

        for _, row in work.iterrows():
            reward_idx = self._reward_index(int(row["reward"]))
            action = int(row["action_bucket"])
            hmm_state = int(row["hmm_state"])
            time_bucket = int(row["time_bucket"])
            activity = int(row["activity_majority"])

            self.global_counts[reward_idx] += 1.0
            self.action_counts.setdefault(action, np.ones(3, dtype=np.float64))[reward_idx] += 1.0
            self.state_action_counts.setdefault((hmm_state, action), np.ones(3, dtype=np.float64))[reward_idx] += 1.0
            self.full_counts.setdefault((hmm_state, time_bucket, activity, action), np.ones(3, dtype=np.float64))[reward_idx] += 1.0

        work["delta_valence"] = work["emo_post_valence"] - work["emo_pre_valence"]
        work["delta_arousal"] = work["emo_post_arousal"] - work["emo_pre_arousal"]
        self.delta_stats = {}
        for reward in [-1, 0, 1]:
            subset = work[work["reward"] == reward]
            if subset.empty:
                subset = work
            self.delta_stats[str(reward)] = {
                "mean_delta_valence": float(subset["delta_valence"].mean()),
                "std_delta_valence": float(subset["delta_valence"].std(ddof=0) + 1e-6),
                "mean_delta_arousal": float(subset["delta_arousal"].mean()),
                "std_delta_arousal": float(subset["delta_arousal"].std(ddof=0) + 1e-6),
                "count": int(len(subset)),
            }

        self.metadata = {
            "rows": int(len(work)),
            "users": int(work["user_id"].nunique()),
        }
        return self

    def probs(self, hmm_state: int, time_bucket: int, activity: int, action: int) -> np.ndarray:
        global_probs = self.global_counts / np.sum(self.global_counts)
        action_probs = self._blend(self.action_counts.get(int(action)), global_probs, self.k_action)
        state_action_probs = self._blend(
            self.state_action_counts.get((int(hmm_state), int(action))),
            action_probs,
            self.k_state_action,
        )
        probs = self._blend(
            self.full_counts.get((int(hmm_state), int(time_bucket), int(activity), int(action))),
            state_action_probs,
            self.k_full,
        )
        probs = np.asarray(probs, dtype=np.float32)
        probs /= probs.sum()
        return probs

    def expected_reward(self, hmm_state: int, time_bucket: int, activity: int, action: int) -> float:
        return float(np.dot(self.probs(hmm_state, time_bucket, activity, action), OUTCOMES))

    def positive_prob(self, hmm_state: int, time_bucket: int, activity: int, action: int) -> float:
        return float(self.probs(hmm_state, time_bucket, activity, action)[2])

    def sample_reward(self, hmm_state: int, time_bucket: int, activity: int, action: int) -> int:
        return int(self.rng.choice([-1, 0, 1], p=self.probs(hmm_state, time_bucket, activity, action)))

    def sample_mood_delta(self, reward: int) -> tuple[float, float]:
        stats = self.delta_stats[str(int(reward))]
        return (
            float(self.rng.normal(stats["mean_delta_valence"], max(stats["std_delta_valence"], 1e-4))),
            float(self.rng.normal(stats["mean_delta_arousal"], max(stats["std_delta_arousal"], 1e-4))),
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "k_action": self.k_action,
            "k_state_action": self.k_state_action,
            "k_full": self.k_full,
            "global_counts": self.global_counts.tolist(),
            "action_counts": {str(k): v.tolist() for k, v in self.action_counts.items()},
            "state_action_counts": {f"{k[0]}|{k[1]}": v.tolist() for k, v in self.state_action_counts.items()},
            "full_counts": {
                f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": v.tolist()
                for k, v in self.full_counts.items()
            },
            "delta_stats": self.delta_stats,
            "metadata": self.metadata,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HierarchicalRewardModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            k_action=payload["k_action"],
            k_state_action=payload["k_state_action"],
            k_full=payload["k_full"],
        )
        model.global_counts = np.asarray(payload["global_counts"], dtype=np.float64)
        model.action_counts = {
            int(k): np.asarray(v, dtype=np.float64) for k, v in payload["action_counts"].items()
        }
        model.state_action_counts = {
            tuple(int(piece) for piece in k.split("|")): np.asarray(v, dtype=np.float64)
            for k, v in payload["state_action_counts"].items()
        }
        model.full_counts = {
            tuple(int(piece) for piece in k.split("|")): np.asarray(v, dtype=np.float64)
            for k, v in payload["full_counts"].items()
        }
        model.delta_stats = payload["delta_stats"]
        model.metadata = payload.get("metadata", {})
        return model
