"""
src/rl_agent/environment.py
Wraps cleaned SiTunes data as a gym-style MDP.

State  : 8-dim float32  [belief_0..5, time_norm, activity_norm]
Action : int 0-7        music mood bucket
Reward : -1 / 0 / +1   from pre/post psychological ratings
Episode: one user's consecutive session (sorted by timestamp)
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.hmm.hmm_model import HMM


class MusicEnv:

    STATE_DIM  = 8
    ACTION_DIM = 8

    def __init__(self, df: pd.DataFrame, wrist: np.ndarray,
                 hmm: HMM, seed: int = 42):
        """
        Parameters
        ----------
        df    : combined stage2+stage3 clean DataFrame
        wrist : combined wrist2+wrist3 encoded array, shape (N, 30)
        hmm   : trained HMM
        seed  : random seed
        """
        self.df    = df.reset_index(drop=True)
        self.wrist = wrist
        self.hmm   = hmm
        self.rng   = np.random.default_rng(seed)

        self._sessions = self._build_sessions()
        self._ptr      = 0
        self._rows     = []
        self._done     = True

    # ── Episode management ────────────────────────────────────────────────

    def _build_sessions(self):
        """Group row indices by user, sorted by timestamp."""
        sessions = {}
        for uid, grp in self.df.groupby("user_id"):
            idx = grp.sort_values("timestamp").index.tolist()
            if len(idx) >= 2:
                sessions[uid] = idx
        return sessions

    def reset(self):
        """Sample a random user session, return initial state."""
        uid         = self.rng.choice(list(self._sessions.keys()))
        self._rows  = self._sessions[uid]
        self._ptr   = 0
        self._done  = False
        return self._state(self._rows[0])

    def step(self, action: int):
        """
        Returns (next_state, reward, done, info).

        Reward has two components:
          - emotion_reward : +1/0/-1 from actual pre/post psych ratings
          - match_bonus    : +0.2 if recommended bucket matches what user listened to
        """
        if self._done:
            raise RuntimeError("Call reset() before step().")

        row_idx = self._rows[self._ptr]
        row     = self.df.iloc[row_idx]

        gt_action    = int(row["action_bucket"])
        base_reward  = float(row["reward"])
        match_bonus  = 0.2 if action == gt_action else 0.0
        reward       = base_reward + match_bonus

        self._ptr += 1
        done       = self._ptr >= len(self._rows)
        self._done = done

        next_state = (np.zeros(self.STATE_DIM, dtype=np.float32)
                      if done else self._state(self._rows[self._ptr]))

        info = {"gt_action": gt_action, "base_reward": base_reward,
                "user_id": int(row["user_id"])}
        return next_state, reward, done, info

    def sample_action(self):
        return int(self.rng.integers(0, self.ACTION_DIM))

    # ── State construction ────────────────────────────────────────────────

    def _state(self, row_idx: int) -> np.ndarray:
        """
        8-dim state: [hmm_belief(6), time_norm, activity_norm]

        time_norm     = time_bucket / 2.0    (0, 0.5, or 1.0)
        activity_norm = activity_type / 4.0  (0 to 1.0)
        """
        obs_seq  = self.wrist[row_idx]             # shape (30,)
        belief   = self.hmm.belief_state(obs_seq)  # shape (6,)

        # Recover time and activity from encoded observation (last timestep)
        # obs = wrist_obs*9 + time*3 + weather
        last_obs     = int(obs_seq[-1])
        time_bucket  = (last_obs % 9) // 3          # 0-2
        wrist_obs    = last_obs // 9                 # 0-19
        activity     = wrist_obs % 5                 # 0-4

        time_norm     = time_bucket / 2.0
        activity_norm = activity    / 4.0

        return np.concatenate([belief, [time_norm, activity_norm]]).astype(np.float32)

    # ── Info ──────────────────────────────────────────────────────────────

    @property
    def n_users(self):
        return len(self._sessions)

    def __repr__(self):
        return (f"MusicEnv(users={self.n_users}, "
                f"interactions={len(self.df)}, "
                f"state={self.STATE_DIM}, actions={self.ACTION_DIM})")