"""
Inference helpers for the wrist-only 3-state HMM.
"""

from __future__ import annotations

import numpy as np

from src.data.common import ACTIVITY_REMAP, encode_wrist_session, intensity_bucket


_ACTIVITY_PRIOR = np.asarray(
    [
        [0.72, 0.20, 0.08],  # still
        [0.22, 0.58, 0.20],  # transition
        [0.10, 0.60, 0.30],  # walking
        [0.70, 0.22, 0.08],  # lying
        [0.05, 0.20, 0.75],  # running
    ],
    dtype=np.float32,
)


def encode_obs_seq(
    intensity: float,
    activity_type: int,
    length: int = 30,
) -> np.ndarray:
    activity = ACTIVITY_REMAP.get(int(activity_type), 0)
    obs = int(intensity_bucket(float(intensity)) * 5 + activity)
    return np.full(length, obs, dtype=np.int32)


def session_obs_from_wrist(wrist_session: np.ndarray) -> np.ndarray:
    return encode_wrist_session(wrist_session)


def corrected_belief(
    hmm,
    obs_seq: np.ndarray,
    activity_remapped: int,
    temperature: float | None = None,
    prior_strength: float | None = None,
) -> np.ndarray:
    temperature = float(
        hmm.metadata.get("belief_temperature", 1.0) if temperature is None else temperature
    )
    prior_strength = float(
        hmm.metadata.get("belief_prior_strength", 1.0)
        if prior_strength is None
        else prior_strength
    )

    posterior = hmm.belief_state(obs_seq, temperature=temperature)
    prior = _ACTIVITY_PRIOR[int(activity_remapped)]

    log_posterior = np.log(np.clip(posterior, 1e-12, 1.0))
    log_prior = np.log(np.clip(prior, 1e-12, 1.0))
    fused = log_posterior + prior_strength * log_prior
    fused -= np.max(fused)
    fused = np.exp(fused)
    fused /= fused.sum()
    return fused.astype(np.float32)


def physical_target_state(
    activity_remapped: int,
    intensity_bucket_value: int,
) -> int:
    if activity_remapped == 4 or intensity_bucket_value >= 3:
        return 2
    if activity_remapped in {1, 2} or intensity_bucket_value >= 2:
        return 1
    return 0
