"""
Hierarchical context-action reward model for offline policy training and evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.common import (
    DEFAULT_ACCEPTANCE_WEIGHT,
    DEFAULT_EMOTION_WEIGHT,
    acceptance_score,
    bucket_targets,
    combine_outcomes,
    emotional_benefit,
)


OUTCOMES = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)


def _blank_summary() -> dict[str, float]:
    return {
        "count": 0.0,
        "emotion_sum": 0.0,
        "acceptance_sum": 0.0,
    }


def valence_bin(value: float) -> int:
    if value < -0.15:
        return 0
    if value > 0.15:
        return 2
    return 1


def arousal_bin(value: float) -> int:
    if value < -0.15:
        return 0
    if value > 0.15:
        return 2
    return 1


def preference_alignment(action: int, user_valence_pref: float | None, user_energy_pref: float | None) -> float:
    if user_valence_pref is None or user_energy_pref is None:
        return 0.0
    target = bucket_targets(int(action))
    pref_val_target = float(np.clip(0.5 + 0.25 * float(user_valence_pref), 0.0, 1.0))
    pref_energy_target = float(np.clip(0.5 + 0.25 * float(user_energy_pref), 0.0, 1.0))
    fit = 1.0 - (0.9 * abs(target["valence"] - pref_val_target) + 1.1 * abs(target["energy"] - pref_energy_target))
    return float(np.clip(fit, -1.0, 1.0))


class HierarchicalRewardModel:
    def __init__(
        self,
        k_action: float = 32.0,
        k_affect: float = 18.0,
        k_full: float = 12.0,
        alpha: float = DEFAULT_EMOTION_WEIGHT,
        beta: float = DEFAULT_ACCEPTANCE_WEIGHT,
        preference_blend: float = 0.35,
        seed: int = 42,
    ):
        self.k_action = float(k_action)
        self.k_affect = float(k_affect)
        self.k_full = float(k_full)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.preference_blend = float(preference_blend)
        self.rng = np.random.default_rng(seed)

        self.global_counts = np.ones(3, dtype=np.float64)
        self.action_counts: dict[int, np.ndarray] = {}
        self.affect_action_counts: dict[tuple[int, int, int, int], np.ndarray] = {}
        self.full_counts: dict[tuple[int, int, int, int, int, int, int], np.ndarray] = {}

        self.global_summary = _blank_summary()
        self.action_summary: dict[int, dict[str, float]] = {}
        self.affect_action_summary: dict[tuple[int, int, int, int], dict[str, float]] = {}
        self.full_summary: dict[tuple[int, int, int, int, int, int, int], dict[str, float]] = {}

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

    @staticmethod
    def _add_summary(summary: dict[str, float], emotion_value: float, acceptance_value: float) -> None:
        summary["count"] += 1.0
        summary["emotion_sum"] += float(emotion_value)
        summary["acceptance_sum"] += float(acceptance_value)

    @staticmethod
    def _mean(summary: dict[str, float] | None, key: str) -> float:
        if summary is None or summary["count"] <= 0:
            return 0.0
        return float(summary[key] / summary["count"])

    @staticmethod
    def _blend_mean(
        child_summary: dict[str, float] | None,
        parent_mean: float,
        key: str,
        k: float,
    ) -> float:
        if child_summary is None or child_summary["count"] <= 0:
            return parent_mean
        child_mean = float(child_summary[key] / child_summary["count"])
        weight = child_summary["count"] / (child_summary["count"] + k)
        return float(weight * child_mean + (1.0 - weight) * parent_mean)

    @staticmethod
    def _affect_key(hmm_state: int, pre_valence: float, pre_arousal: float, action: int) -> tuple[int, int, int, int]:
        return int(hmm_state), valence_bin(float(pre_valence)), arousal_bin(float(pre_arousal)), int(action)

    @staticmethod
    def step_bin(step_active: int | float) -> int:
        return int(bool(step_active))

    @classmethod
    def _full_key(
        cls,
        hmm_state: int,
        time_bucket: int,
        activity: int,
        step_active: int | float,
        pre_valence: float,
        pre_arousal: float,
        action: int,
    ) -> tuple[int, int, int, int, int, int, int]:
        return (
            int(hmm_state),
            int(time_bucket),
            int(activity),
            cls.step_bin(step_active),
            valence_bin(float(pre_valence)),
            arousal_bin(float(pre_arousal)),
            int(action),
        )

    def fit(self, df: pd.DataFrame) -> "HierarchicalRewardModel":
        work = df.copy()
        required = [
            "hmm_state",
            "time_bucket",
            "activity_majority",
            "action_bucket",
            "reward",
            "emo_pre_valence",
            "emo_pre_arousal",
        ]
        missing = [col for col in required if col not in work.columns]
        if missing:
            raise ValueError(f"Reward model fit missing columns: {missing}")
        if "step_active" not in work.columns:
            if "step_nonzero_frac" in work.columns:
                work["step_active"] = (work["step_nonzero_frac"].astype(float) >= 0.05).astype(int)
            else:
                work["step_active"] = 0

        if "emotion_benefit" not in work.columns:
            work["emotion_benefit"] = work["reward_score"].map(emotional_benefit)
        if "acceptance_score" not in work.columns:
            work["acceptance_score"] = [
                acceptance_score(row.get("preference"), row.get("rating"))
                for _, row in work.iterrows()
            ]
        if "combined_reward" not in work.columns:
            work["combined_reward"] = [
                combine_outcomes(row["emotion_benefit"], row["acceptance_score"], self.alpha, self.beta)
                for _, row in work.iterrows()
            ]

        self.global_counts = np.ones(3, dtype=np.float64)
        self.action_counts.clear()
        self.affect_action_counts.clear()
        self.full_counts.clear()
        self.global_summary = _blank_summary()
        self.action_summary.clear()
        self.affect_action_summary.clear()
        self.full_summary.clear()

        for _, row in work.iterrows():
            reward_idx = self._reward_index(int(row["reward"]))
            action = int(row["action_bucket"])
            hmm_state = int(row["hmm_state"])
            time_bucket = int(row["time_bucket"])
            activity = int(row["activity_majority"])
            step_active = self.step_bin(row.get("step_active", 0))
            emotion_value = float(row["emotion_benefit"])
            acceptance_value = float(row["acceptance_score"])
            pre_valence = float(row["emo_pre_valence"])
            pre_arousal = float(row["emo_pre_arousal"])

            affect_key = self._affect_key(hmm_state, pre_valence, pre_arousal, action)
            full_key = self._full_key(hmm_state, time_bucket, activity, step_active, pre_valence, pre_arousal, action)

            self.global_counts[reward_idx] += 1.0
            self.action_counts.setdefault(action, np.ones(3, dtype=np.float64))[reward_idx] += 1.0
            self.affect_action_counts.setdefault(affect_key, np.ones(3, dtype=np.float64))[reward_idx] += 1.0
            self.full_counts.setdefault(full_key, np.ones(3, dtype=np.float64))[reward_idx] += 1.0

            self._add_summary(self.global_summary, emotion_value, acceptance_value)
            self._add_summary(self.action_summary.setdefault(action, _blank_summary()), emotion_value, acceptance_value)
            self._add_summary(self.affect_action_summary.setdefault(affect_key, _blank_summary()), emotion_value, acceptance_value)
            self._add_summary(self.full_summary.setdefault(full_key, _blank_summary()), emotion_value, acceptance_value)

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
            "alpha": self.alpha,
            "beta": self.beta,
            "preference_blend": self.preference_blend,
            "uses_step_active": True,
            "emotion_benefit_mean": float(work["emotion_benefit"].mean()),
            "acceptance_score_mean": float(work["acceptance_score"].mean()),
            "combined_reward_mean": float(work["combined_reward"].mean()),
        }
        return self

    def probs(
        self,
        hmm_state: int,
        time_bucket: int,
        activity: int,
        step_active: int | float,
        action: int,
        pre_valence: float = 0.0,
        pre_arousal: float = 0.0,
    ) -> np.ndarray:
        global_probs = self.global_counts / np.sum(self.global_counts)
        action_probs = self._blend(self.action_counts.get(int(action)), global_probs, self.k_action)
        affect_probs = self._blend(
            self.affect_action_counts.get(self._affect_key(hmm_state, pre_valence, pre_arousal, action)),
            action_probs,
            self.k_affect,
        )
        probs = self._blend(
            self.full_counts.get(
                self._full_key(hmm_state, time_bucket, activity, step_active, pre_valence, pre_arousal, action)
            ),
            affect_probs,
            self.k_full,
        )
        probs = np.asarray(probs, dtype=np.float32)
        probs /= probs.sum()
        return probs

    def expected_components(
        self,
        hmm_state: int,
        time_bucket: int,
        activity: int,
        step_active: int | float,
        action: int,
        pre_valence: float = 0.0,
        pre_arousal: float = 0.0,
        user_valence_pref: float | None = None,
        user_energy_pref: float | None = None,
        alpha: float | None = None,
        beta: float | None = None,
    ) -> dict[str, float]:
        alpha = self.alpha if alpha is None else float(alpha)
        beta = self.beta if beta is None else float(beta)

        global_emotion = self._mean(self.global_summary, "emotion_sum")
        global_acceptance = self._mean(self.global_summary, "acceptance_sum")

        action_summary = self.action_summary.get(int(action))
        affect_summary = self.affect_action_summary.get(self._affect_key(hmm_state, pre_valence, pre_arousal, action))
        full_summary = self.full_summary.get(
            self._full_key(hmm_state, time_bucket, activity, step_active, pre_valence, pre_arousal, action)
        )

        emotion = self._blend_mean(action_summary, global_emotion, "emotion_sum", self.k_action)
        emotion = self._blend_mean(affect_summary, emotion, "emotion_sum", self.k_affect)
        emotion = self._blend_mean(full_summary, emotion, "emotion_sum", self.k_full)

        acceptance = self._blend_mean(action_summary, global_acceptance, "acceptance_sum", self.k_action)
        acceptance = self._blend_mean(affect_summary, acceptance, "acceptance_sum", self.k_affect)
        acceptance = self._blend_mean(full_summary, acceptance, "acceptance_sum", self.k_full)

        if user_valence_pref is not None and user_energy_pref is not None:
            pref_fit = preference_alignment(action, user_valence_pref, user_energy_pref)
            acceptance = float((1.0 - self.preference_blend) * acceptance + self.preference_blend * pref_fit)

        support = 0.0
        if full_summary is not None:
            support = float(full_summary["count"])
        elif affect_summary is not None:
            support = float(affect_summary["count"])
        elif action_summary is not None:
            support = float(action_summary["count"])

        return {
            "emotion_benefit": float(emotion),
            "acceptance": float(acceptance),
            "combined_reward": combine_outcomes(emotion, acceptance, alpha, beta),
            "support": float(support),
        }

    def expected_reward(
        self,
        hmm_state: int,
        time_bucket: int,
        activity: int,
        step_active: int | float,
        action: int,
        pre_valence: float = 0.0,
        pre_arousal: float = 0.0,
        user_valence_pref: float | None = None,
        user_energy_pref: float | None = None,
    ) -> float:
        return self.expected_components(
            hmm_state,
            time_bucket,
            activity,
            step_active,
            action,
            pre_valence=pre_valence,
            pre_arousal=pre_arousal,
            user_valence_pref=user_valence_pref,
            user_energy_pref=user_energy_pref,
        )["combined_reward"]

    def positive_prob(
        self,
        hmm_state: int,
        time_bucket: int,
        activity: int,
        step_active: int | float,
        action: int,
        pre_valence: float = 0.0,
        pre_arousal: float = 0.0,
    ) -> float:
        return float(self.probs(hmm_state, time_bucket, activity, step_active, action, pre_valence, pre_arousal)[2])

    def sample_reward(
        self,
        hmm_state: int,
        time_bucket: int,
        activity: int,
        step_active: int | float,
        action: int,
        pre_valence: float = 0.0,
        pre_arousal: float = 0.0,
    ) -> int:
        return int(
            self.rng.choice(
                [-1, 0, 1],
                p=self.probs(hmm_state, time_bucket, activity, step_active, action, pre_valence, pre_arousal),
            )
        )

    def sample_mood_delta(self, reward: int) -> tuple[float, float]:
        stats = self.delta_stats[str(int(reward))]
        return (
            float(self.rng.normal(stats["mean_delta_valence"], max(stats["std_delta_valence"], 1e-4))),
            float(self.rng.normal(stats["mean_delta_arousal"], max(stats["std_delta_arousal"], 1e-4))),
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "k_action": self.k_action,
            "k_affect": self.k_affect,
            "k_full": self.k_full,
            "alpha": self.alpha,
            "beta": self.beta,
            "preference_blend": self.preference_blend,
            "global_counts": self.global_counts.tolist(),
            "action_counts": {str(k): v.tolist() for k, v in self.action_counts.items()},
            "affect_action_counts": {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": v.tolist() for k, v in self.affect_action_counts.items()},
            "full_counts": {
                f"{k[0]}|{k[1]}|{k[2]}|{k[3]}|{k[4]}|{k[5]}|{k[6]}": v.tolist()
                for k, v in self.full_counts.items()
            },
            "global_summary": self.global_summary,
            "action_summary": {str(k): v for k, v in self.action_summary.items()},
            "affect_action_summary": {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": v for k, v in self.affect_action_summary.items()},
            "full_summary": {
                f"{k[0]}|{k[1]}|{k[2]}|{k[3]}|{k[4]}|{k[5]}|{k[6]}": v
                for k, v in self.full_summary.items()
            },
            "delta_stats": self.delta_stats,
            "metadata": self.metadata,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def _parse_full_key_text(cls, text: str) -> tuple[int, int, int, int, int, int, int]:
        pieces = tuple(int(piece) for piece in text.split("|"))
        if len(pieces) == 7:
            return pieces
        if len(pieces) == 6:
            return (pieces[0], pieces[1], pieces[2], 0, pieces[3], pieces[4], pieces[5])
        raise ValueError(f"Unexpected full key format: {text}")

    @classmethod
    def load(cls, path: str | Path) -> "HierarchicalRewardModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            k_action=payload["k_action"],
            k_affect=payload.get("k_affect", payload.get("k_state_action", 18.0)),
            k_full=payload["k_full"],
            alpha=payload.get("alpha", DEFAULT_EMOTION_WEIGHT),
            beta=payload.get("beta", DEFAULT_ACCEPTANCE_WEIGHT),
            preference_blend=payload.get("preference_blend", 0.35),
        )
        model.global_counts = np.asarray(payload["global_counts"], dtype=np.float64)
        model.action_counts = {
            int(k): np.asarray(v, dtype=np.float64) for k, v in payload["action_counts"].items()
        }
        model.affect_action_counts = {
            tuple(int(piece) for piece in k.split("|")): np.asarray(v, dtype=np.float64)
            for k, v in payload.get("affect_action_counts", {}).items()
        }
        model.full_counts = {
            cls._parse_full_key_text(k): np.asarray(v, dtype=np.float64)
            for k, v in payload["full_counts"].items()
        }
        model.global_summary = payload.get("global_summary", _blank_summary())
        model.action_summary = {int(k): v for k, v in payload.get("action_summary", {}).items()}
        model.affect_action_summary = {
            tuple(int(piece) for piece in k.split("|")): v
            for k, v in payload.get("affect_action_summary", {}).items()
        }
        model.full_summary = {
            cls._parse_full_key_text(k): v
            for k, v in payload.get("full_summary", {}).items()
        }
        model.delta_stats = payload["delta_stats"]
        model.metadata = payload.get("metadata", {})
        return model
