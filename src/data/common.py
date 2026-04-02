"""
Shared constants and helpers for the Ambient Music Agent data pipeline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

SITUNES_DIR = RAW_DIR / "situnes" / "SiTunes"
PMEMO_DIR = RAW_DIR / "pmemo"
SPOTIFY_DIR = RAW_DIR / "spotify_kaggle"

N_HMM_STATES = 3
N_WRIST_OBS = 20
STATE_DIM = 5
ACTION_DIM = 8

DEFAULT_SEED = 42
TRAIN_USERS = 20
VAL_USERS = 5
TEST_USERS = 5

REWARD_THRESHOLD = 0.10

ACTIVITY_REMAP = {
    0: 0,  # still
    1: 1,  # transition
    2: 2,  # walking
    3: 0,  # missing -> still
    4: 3,  # lying
    5: 4,  # running
}

ACTIVITY_LABELS = {
    0: "still",
    1: "transitioning",
    2: "walking",
    3: "lying",
    4: "running",
}

TIME_LABELS = {
    0: "morning",
    1: "afternoon",
    2: "evening",
}

STATE_NAMES = [
    "low-energy",
    "moderate",
    "high-energy",
]

BUCKET_LABELS = {
    0: "dark-slow",
    1: "dark-fast",
    2: "intense-slow",
    3: "aggressive",
    4: "chill-study",
    5: "indie",
    6: "soulful",
    7: "energetic",
}


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def intensity_bucket(intensity: float) -> int:
    if intensity < 10:
        return 0
    if intensity < 30:
        return 1
    if intensity < 80:
        return 2
    return 3


def encode_wrist_timestep(wrist_ts: Iterable[float]) -> int:
    wrist_ts = list(wrist_ts)
    activity_raw = int(wrist_ts[3])
    activity = ACTIVITY_REMAP.get(activity_raw, 0)
    ib = intensity_bucket(float(wrist_ts[1]))
    return int(ib * 5 + activity)


def encode_wrist_session(wrist_session: np.ndarray) -> np.ndarray:
    encoded = [encode_wrist_timestep(wrist_session[t]) for t in range(len(wrist_session))]
    return np.asarray(encoded, dtype=np.int32)


def majority_vote(values: Iterable[int], tie_break: int) -> int:
    values = list(int(v) for v in values)
    counts = pd.Series(values).value_counts()
    top_count = int(counts.max())
    winners = sorted(int(v) for v, c in counts.items() if int(c) == top_count)
    if tie_break in winners:
        return tie_break
    return winners[0]


def summarize_wrist_session(wrist_session: np.ndarray) -> dict:
    intensities = wrist_session[:, 1].astype(float)
    activities_raw = wrist_session[:, 3].astype(int)
    activities = np.vectorize(lambda x: ACTIVITY_REMAP.get(int(x), 0))(activities_raw)
    encoded = encode_wrist_session(wrist_session)

    activity_last_raw = int(activities_raw[-1])
    activity_last = ACTIVITY_REMAP.get(activity_last_raw, 0)
    activity_majority = majority_vote(activities, tie_break=activity_last)
    activity_majority_raw = majority_vote(activities_raw, tie_break=activity_last_raw)

    intensity_mean = float(np.mean(intensities))
    intensity_last = float(intensities[-1])

    return {
        "wrist_obs": encoded,
        "activity_last_raw": activity_last_raw,
        "activity_last": activity_last,
        "activity_majority_raw": activity_majority_raw,
        "activity_majority": int(activity_majority),
        "intensity_last": intensity_last,
        "intensity_mean": intensity_mean,
        "intensity_bucket_last": intensity_bucket(intensity_last),
        "intensity_bucket_mean": intensity_bucket(intensity_mean),
    }


def reward_from_emotions(
    pre_valence: float,
    pre_arousal: float,
    post_valence: float,
    post_arousal: float,
    threshold: float = REWARD_THRESHOLD,
) -> tuple[int, float]:
    score = (post_valence - pre_valence) * 0.7 + (post_arousal - pre_arousal) * 0.3
    if score > threshold:
        return 1, float(score)
    if score < -threshold:
        return -1, float(score)
    return 0, float(score)


def get_action_bucket(valence: float, energy: float, tempo: float) -> int:
    valence_level = 0 if valence < 0.33 else 1
    energy_level = 0 if energy < 0.4 else 1
    tempo_level = 0 if tempo < 100 else 1
    return int(valence_level * 4 + energy_level * 2 + tempo_level)


def parse_situnes_timestamp(timestamp_s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(timestamp_s.astype("int64"), unit="s", utc=True)
    return ts.dt.tz_convert("America/New_York")


def balanced_user_split(
    user_counts: pd.Series,
    stage3_users: set[int],
    seed: int = DEFAULT_SEED,
) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)

    def assign_group(users: list[int], quotas: dict[str, int], split_counts: dict[str, int]) -> dict[str, list[int]]:
        result = {k: [] for k in quotas}
        shuffled = users.copy()
        rng.shuffle(shuffled)
        ordered = sorted(shuffled, key=lambda u: (-int(user_counts.loc[u]), u))
        for user_id in ordered:
            candidates = [
                split
                for split, quota in quotas.items()
                if len(result[split]) < quota
            ]
            split = min(candidates, key=lambda name: (split_counts[name], len(result[name]), name))
            result[split].append(int(user_id))
            split_counts[split] += int(user_counts.loc[user_id])
        return result

    stage3 = sorted(int(u) for u in stage3_users)
    stage2_only = sorted(int(u) for u in user_counts.index if int(u) not in stage3_users)

    split_counts = {"train": 0, "val": 0, "test": 0}
    stage3_assign = assign_group(stage3, {"train": 6, "val": 2, "test": 2}, split_counts)
    stage2_assign = assign_group(stage2_only, {"train": 14, "val": 3, "test": 3}, split_counts)

    combined = {
        split: sorted(stage3_assign[split] + stage2_assign[split])
        for split in ["train", "val", "test"]
    }
    return combined


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1_048_576) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_vector_from_components(
    belief: np.ndarray,
    time_bucket: int,
    activity_remapped: int,
) -> np.ndarray:
    return np.asarray(
        [
            float(belief[0]),
            float(belief[1]),
            float(belief[2]),
            float(time_bucket) / 2.0,
            float(activity_remapped) / 4.0,
        ],
        dtype=np.float32,
    )


def bucket_targets(bucket: int) -> dict[str, float]:
    targets = {
        0: {"valence": 0.15, "energy": 0.15, "tempo": 75.0},
        1: {"valence": 0.15, "energy": 0.25, "tempo": 125.0},
        2: {"valence": 0.20, "energy": 0.75, "tempo": 85.0},
        3: {"valence": 0.20, "energy": 0.85, "tempo": 145.0},
        4: {"valence": 0.55, "energy": 0.20, "tempo": 75.0},
        5: {"valence": 0.55, "energy": 0.30, "tempo": 120.0},
        6: {"valence": 0.60, "energy": 0.65, "tempo": 90.0},
        7: {"valence": 0.60, "energy": 0.80, "tempo": 140.0},
    }
    return targets[int(bucket)]


def track_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column, default in [
        ("popularity", 50.0),
        ("acousticness", 0.5),
        ("instrumentalness", 0.0),
        ("speechiness", 0.08),
        ("danceability", 0.5),
        ("liveness", 0.15),
        ("energy", 0.5),
        ("valence", 0.5),
        ("tempo", 120.0),
    ]:
        if column not in out.columns:
            out[column] = default
        out[column] = out[column].fillna(default)
    if "explicit" not in out.columns:
        out["explicit"] = False
    out["explicit"] = out["explicit"].fillna(False).astype(bool)
    if "genre" not in out.columns:
        out["genre"] = ""
    out["genre"] = out["genre"].fillna("").astype(str)
    return out
