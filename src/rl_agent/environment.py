"""
One-step offline environment driven by a context-action reward model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MusicEnv:
    STATE_DIM = 5
    ACTION_DIM = 8

    def __init__(
        self,
        df: pd.DataFrame,
        states: np.ndarray,
        reward_model,
        sample_weights: np.ndarray | None = None,
        reward_mode: str = "expected",
        seed: int = 42,
    ):
        self.df = df.reset_index(drop=True)
        self.states = np.asarray(states, dtype=np.float32)
        self.reward_model = reward_model
        self.rng = np.random.default_rng(seed)
        self.reward_mode = reward_mode
        self.sample_weights = None
        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights, dtype=np.float64)
            self.sample_weights = sample_weights / sample_weights.sum()

        self._row_idx: int | None = None
        self._done = True

    def reset(self) -> np.ndarray:
        if self.sample_weights is None:
            self._row_idx = int(self.rng.integers(0, len(self.df)))
        else:
            self._row_idx = int(self.rng.choice(len(self.df), p=self.sample_weights))
        self._done = False
        return self._state(self._row_idx)

    def step(self, action: int):
        if self._done:
            raise RuntimeError("Call reset() before step().")
        assert 0 <= int(action) < self.ACTION_DIM, f"action must be in [0, {self.ACTION_DIM - 1}]"

        row = self.df.iloc[int(self._row_idx)]
        expected_reward = self.reward_model.expected_reward(
            int(row["hmm_state"]),
            int(row["time_bucket"]),
            int(row["activity_majority"]),
            int(action),
        )
        if self.reward_mode == "sample":
            reward = self.reward_model.sample_reward(
                int(row["hmm_state"]),
                int(row["time_bucket"]),
                int(row["activity_majority"]),
                int(action),
            )
        else:
            reward = expected_reward
        self._done = True

        info = {
            "historical_action": int(row.get("action_bucket", -1)),
            "expected_reward": expected_reward,
            "positive_prob": self.reward_model.positive_prob(
                int(row["hmm_state"]),
                int(row["time_bucket"]),
                int(row["activity_majority"]),
                int(action),
            ),
            "user_id": int(row.get("user_id", -1)),
            "is_synthetic": bool(row.get("is_synthetic", False)),
        }
        return np.zeros(self.STATE_DIM, dtype=np.float32), reward, True, info

    def sample_action(self) -> int:
        return int(self.rng.integers(0, self.ACTION_DIM))

    def _state(self, row_idx: int) -> np.ndarray:
        return self.states[row_idx].copy()

    def __repr__(self) -> str:
        return f"MusicEnv(contexts={len(self.df)}, state={self.STATE_DIM}, actions={self.ACTION_DIM})"
