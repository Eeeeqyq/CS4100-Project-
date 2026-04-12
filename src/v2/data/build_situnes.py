"""
SiTunes -> V2 decision tables and context tensors.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.common import ACTIVITY_REMAP, SITUNES_DIR
from src.data.preprocess import clean_situnes

from .normalization import apply_zscore, fit_train_only_env_stats, save_stats
from .schema import (
    DECISION_COLUMNS,
    STAGE1_HISTORY_COLUMNS,
    STAGE1_MAX_LEN,
    ENV_DIM,
    SELF_DIM,
    WRIST_DIM,
    validate_decision_table,
    validate_stage1_history_table,
)
from .targets import (
    GoalContext,
    acceptance_observation,
    acceptance_target,
    adjusted_target,
    benefit_target,
    goal_router_v1,
    preference_target,
    rating_target,
)


def _load_stage_env(stage_name: str) -> dict:
    env_path = SITUNES_DIR / stage_name / "env.json"
    return json.loads(env_path.read_text(encoding="utf-8"))


def _load_stage_wrist(stage_name: str) -> np.ndarray:
    return np.load(SITUNES_DIR / stage_name / "wrist.npy", allow_pickle=True)


def _encode_weather_bucket(weather_raw: int) -> int:
    if weather_raw <= 0:
        return 0
    if weather_raw == 1:
        return 1
    return 2


def _make_env_vector(env_entry: dict, timestamp_local: pd.Timestamp) -> tuple[np.ndarray, dict[str, float]]:
    hour = float(timestamp_local.hour) + float(timestamp_local.minute) / 60.0
    angle = 2.0 * np.pi * hour / 24.0

    weather_list = list(env_entry.get("weather", [0, 0.0, 0.0, 0.0]))
    while len(weather_list) < 4:
        weather_list.append(0.0)
    weather_raw = int(weather_list[0])
    weather_bucket = _encode_weather_bucket(weather_raw)
    temp_raw = float(weather_list[1])
    humidity_raw = float(weather_list[2])
    gps = list(env_entry.get("GPS", [0.0, 0.0, 0.0]))
    speed = float(gps[-1]) if gps else 0.0

    env_vec = np.asarray(
        [
            np.sin(angle),
            np.cos(angle),
            float(weather_bucket == 0),
            float(weather_bucket == 1),
            float(weather_bucket == 2),
            temp_raw,
            humidity_raw,
            np.clip(speed / 10.0, 0.0, 1.0),
            float(timestamp_local.dayofweek >= 5),
        ],
        dtype=np.float32,
    )
    return env_vec, {
        "weather_bucket": float(weather_bucket),
        "temp_raw": temp_raw,
        "humidity_raw": humidity_raw,
        "speed_norm": float(env_vec[7]),
        "weekend_flag": float(env_vec[8]),
    }


def _make_wrist_tensor(wrist_session: np.ndarray) -> np.ndarray:
    hr = wrist_session[:, 0].astype(np.float32)
    intensity = wrist_session[:, 1].astype(np.float32)
    steps = wrist_session[:, 2].astype(np.float32)
    activity_raw = wrist_session[:, 3].astype(np.int32)

    hr_norm = (hr - hr.mean()) / max(float(hr.std(ddof=0)), 1e-6)
    hr_delta = np.concatenate([[0.0], np.diff(hr)]).astype(np.float32)
    hr_delta = np.clip(hr_delta / 20.0, -1.0, 1.0)
    intensity_norm = np.clip(intensity / 100.0, 0.0, 1.0)
    steps_norm = np.clip(steps / 25.0, 0.0, 1.0)

    activity_oh = np.zeros((len(activity_raw), 5), dtype=np.float32)
    for idx, raw in enumerate(activity_raw):
        activity_oh[idx, ACTIVITY_REMAP.get(int(raw), 0)] = 1.0

    wrist = np.column_stack(
        [
            hr_norm.astype(np.float32),
            hr_delta.astype(np.float32),
            intensity_norm.astype(np.float32),
            steps_norm.astype(np.float32),
            activity_oh,
        ]
    ).astype(np.float32)
    if wrist.shape != (30, WRIST_DIM):
        raise ValueError(f"Unexpected wrist tensor shape: {wrist.shape}")
    return wrist


def _make_self_vector(row: pd.Series) -> np.ndarray:
    vec = np.asarray(
        [
            float(np.clip(row["emo_pre_valence"], -1.0, 1.0)),
            float(np.clip(row["emo_pre_arousal"], -1.0, 1.0)),
            float(row.get("pre_emotion_mask", 1.0)),
        ],
        dtype=np.float32,
    )
    if vec.shape != (SELF_DIM,):
        raise ValueError(f"Unexpected self vector shape: {vec.shape}")
    return vec


def _goal_context_from_row(row: pd.Series) -> GoalContext:
    return GoalContext(
        pre_valence=float(row["emo_pre_valence"]),
        pre_arousal=float(row["emo_pre_arousal"]),
        time_bucket=int(row["time_bucket"]),
        weather_bucket=int(row["weather_bucket"]),
        speed_norm=float(row["speed_norm"]),
        weekend_flag=float(row["weekend_flag"]),
        step_nonzero_frac=float(row["step_nonzero_frac"]),
        step_mean_norm=float(np.clip(float(row["step_mean"]) / 25.0, 0.0, 1.0)),
        activity_majority=int(row["activity_majority"]),
        hr_mean_rel=float(row.get("hr_mean_rel_user", 0.0)),
        checkin_mask=float(row.get("pre_emotion_mask", 1.0)),
    )


def _build_stage1_history_table(stage1_enriched: pd.DataFrame) -> pd.DataFrame:
    history = stage1_enriched.copy()
    history["song_id"] = history["item_id"].map(lambda item_id: f"situnes_{int(item_id)}")
    history = history.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    history["hist_pos"] = history.groupby("user_id").cumcount()
    history = history[history["hist_pos"] < STAGE1_MAX_LEN].copy()
    out = history[
        ["user_id", "song_id", "rating", "timestamp", "emo_valence", "emo_arousal", "hist_pos"]
    ].copy()
    validate_stage1_history_table(out)
    return out


def _build_decision_rows(combined: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    env_lookup = {
        "stage2": _load_stage_env("Stage2"),
        "stage3": _load_stage_env("Stage3"),
    }
    wrist_lookup = {
        "stage2": _load_stage_wrist("Stage2"),
        "stage3": _load_stage_wrist("Stage3"),
    }

    decision_rows: list[dict] = []
    wrist_tensors = []
    env_tensors = []
    self_tensors = []

    for _, row in combined.iterrows():
        stage_key = str(row["dataset_stage"])
        env_entry = env_lookup[stage_key][str(int(row["inter_id"]))]
        wrist_session = wrist_lookup[stage_key][int(row["inter_id"]) - 1]

        timestamp_local = pd.to_datetime(row["timestamp_local"])
        env_vec, env_meta = _make_env_vector(env_entry, timestamp_local)
        self_vec = _make_self_vector(row)
        ctx_row = row.copy()
        ctx_row["weather_bucket"] = int(env_meta["weather_bucket"])
        ctx_row["speed_norm"] = float(env_meta["speed_norm"])
        ctx_row["weekend_flag"] = float(env_meta["weekend_flag"])
        ctx = _goal_context_from_row(ctx_row)
        goal = goal_router_v1(ctx)
        tau_v, tau_a = adjusted_target(goal, ctx)
        benefit = benefit_target(
            float(row["emo_pre_valence"]),
            float(row["emo_pre_arousal"]),
            float(row["emo_post_valence"]),
            float(row["emo_post_arousal"]),
            goal,
            ctx,
        )
        accept = acceptance_target(row.get("preference"), row.get("rating"))
        accept_pref = preference_target(row.get("preference"))
        accept_rating = rating_target(row.get("rating"))
        _, _, accept_pref_mask, accept_rating_mask = acceptance_observation(row.get("preference"), row.get("rating"))

        decision_rows.append(
            {
                "decision_id": f"{stage_key}_{int(row['user_id'])}_{int(row['inter_id'])}",
                "user_id": int(row["user_id"]),
                "song_id": f"situnes_{int(row['item_id'])}",
                "split": str(row["split"]),
                "goal_idx": int(goal),
                "goal_source": "router_fallback",
                "explicit_goal_idx": -1,
                "pre_valence": float(row["emo_pre_valence"]),
                "pre_arousal": float(row["emo_pre_arousal"]),
                "post_valence": float(row["emo_post_valence"]),
                "post_arousal": float(row["emo_post_arousal"]),
                "benefit_target": benefit,
                "acceptance_target": accept,
                "accept_pref_target": accept_pref,
                "accept_rating_target": accept_rating,
                "accept_pref_mask": float(accept_pref_mask),
                "accept_rating_mask": float(accept_rating_mask),
                "time_bucket": int(row["time_bucket"]),
                "weather_bucket": int(env_meta["weather_bucket"]),
                "temp_z": np.nan,
                "humidity_z": np.nan,
                "speed_norm": float(env_meta["speed_norm"]),
                "weekend_flag": float(env_meta["weekend_flag"]),
                "checkin_mask": float(self_vec[2]),
                "timestamp": int(row["timestamp"]),
                "dataset_stage": stage_key,
                "temp_raw": float(env_meta["temp_raw"]),
                "humidity_raw": float(env_meta["humidity_raw"]),
                "tau_valence": tau_v,
                "tau_arousal": tau_a,
            }
        )
        wrist_tensors.append(_make_wrist_tensor(wrist_session))
        env_tensors.append(env_vec)
        self_tensors.append(self_vec)

    decision_df = pd.DataFrame(decision_rows)
    env_stats = fit_train_only_env_stats(decision_df)
    decision_df["temp_z"] = apply_zscore(decision_df["temp_raw"].to_numpy(dtype=np.float64), env_stats["temp"])
    decision_df["humidity_z"] = apply_zscore(decision_df["humidity_raw"].to_numpy(dtype=np.float64), env_stats["humidity"])

    env_array = np.asarray(env_tensors, dtype=np.float32)
    env_array[:, 5] = decision_df["temp_z"].to_numpy(dtype=np.float32)
    env_array[:, 6] = decision_df["humidity_z"].to_numpy(dtype=np.float32)
    self_array = np.asarray(self_tensors, dtype=np.float32)
    wrist_array = np.asarray(wrist_tensors, dtype=np.float32)

    if env_array.shape[1] != ENV_DIM:
        raise ValueError(f"Unexpected env tensor shape: {env_array.shape}")
    if self_array.shape[1] != SELF_DIM:
        raise ValueError(f"Unexpected self tensor shape: {self_array.shape}")

    decision_df = decision_df.reset_index(drop=True)
    return decision_df, wrist_array, env_array, self_array


def materialize_stage1_histories(
    history_df: pd.DataFrame,
    song_id_to_index: dict[str, int],
    out_dir: Path,
) -> dict[str, int]:
    users = sorted(int(user_id) for user_id in history_df["user_id"].unique())
    hist_song_idx = np.full((len(users), STAGE1_MAX_LEN), -1, dtype=np.int32)
    hist_rating = np.zeros((len(users), STAGE1_MAX_LEN), dtype=np.float32)
    hist_mask = np.zeros((len(users), STAGE1_MAX_LEN), dtype=np.float32)
    user_to_row = {user_id: idx for idx, user_id in enumerate(users)}

    for _, row in history_df.iterrows():
        uidx = user_to_row[int(row["user_id"])]
        pos = int(row["hist_pos"])
        hist_song_idx[uidx, pos] = int(song_id_to_index.get(str(row["song_id"]), -1))
        hist_rating[uidx, pos] = float(np.clip((float(row["rating"]) - 3.0) / 2.0, -1.0, 1.0))
        hist_mask[uidx, pos] = 1.0

    np.savez(
        out_dir / "stage1_histories.npz",
        user_ids=np.asarray(users, dtype=np.int32),
        hist_song_idx=hist_song_idx,
        hist_rating=hist_rating,
        hist_mask=hist_mask,
    )
    return {"users": len(users), "non_padding_rows": int(hist_mask.sum())}


def build_situnes_v2(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stage1_enriched, _, _, combined, _, _, _ = clean_situnes()

    history_df = _build_stage1_history_table(stage1_enriched)
    decision_df, wrist_array, env_array, self_array = _build_decision_rows(combined)

    export_decisions = decision_df[DECISION_COLUMNS].copy()
    validate_decision_table(export_decisions)
    validate_stage1_history_table(history_df)

    history_df.to_parquet(out_dir / "stage1_history_table.parquet", index=False)
    export_decisions.to_parquet(out_dir / "decision_table.parquet", index=False)
    np.save(out_dir / "wrist_windows.npy", wrist_array)
    np.save(out_dir / "env_features.npy", env_array)
    np.save(out_dir / "self_report.npy", self_array)

    env_stats = fit_train_only_env_stats(decision_df)
    save_stats(env_stats, out_dir / "normalization_stats.json")

    split_counts = export_decisions["split"].value_counts().sort_index().to_dict()
    return {
        "decision_rows": int(len(export_decisions)),
        "users": int(export_decisions["user_id"].nunique()),
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "history_rows": int(len(history_df)),
    }
