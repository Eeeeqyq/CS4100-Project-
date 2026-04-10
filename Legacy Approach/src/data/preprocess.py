"""
Deterministic preprocessing pipeline for SiTunes, PMEmo, and Spotify.

This script is the canonical data build entrypoint. Notebooks are optional
for exploration, but all repo-tracked logic lives here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.common import (
    PMEMO_DIR,
    PROCESSED_DIR,
    SITUNES_DIR,
    SPOTIFY_DIR,
    acceptance_score,
    balanced_user_split,
    combine_outcomes,
    config_hash,
    emotion_score,
    emotional_benefit,
    ensure_dirs,
    file_sha256,
    get_action_bucket,
    parse_situnes_timestamp,
    reward_from_emotions,
    summarize_wrist_session,
)


PIPELINE_CONFIG = {
    "reward_threshold": 0.10,
    "bucket_thresholds": {
        "valence": 0.33,
        "energy": 0.4,
        "tempo": 100.0,
    },
    "split_seed": 42,
    "stage3_user_quota": {"train": 6, "val": 2, "test": 2},
    "stage2_only_user_quota": {"train": 14, "val": 3, "test": 3},
    "pmemo_std_cutoff": 0.25,
    "pmemo_energy_blend": 0.55,
    "pmemo_tempo_clip": [55.0, 190.0],
    "reward_alpha": 0.70,
    "reward_beta": 0.30,
    "hr_baseline_shrink": 3.0,
}


def _load_situnes_music() -> pd.DataFrame:
    base = pd.read_csv(SITUNES_DIR / "music_metadata" / "music_info.csv")
    named = pd.read_csv(SITUNES_DIR / "music_metadata" / "music_info_withname.csv").rename(
        columns={"i_id_c": "item_id"}
    )

    keep_names = named[["item_id", "music", "singer"]].copy()
    keep_names["item_id"] = keep_names["item_id"].astype(int)

    music = base.merge(keep_names, on="item_id", how="left")
    music = music.drop(columns=["Unnamed: 0"], errors="ignore")
    music["music"] = music["music"].fillna(music["item_id"].map(lambda x: f"Track {int(x)}"))
    music["singer"] = music["singer"].fillna("Unknown Artist")
    music["action_bucket"] = music.apply(
        lambda row: get_action_bucket(row["valence"], row["energy"], row["tempo"]),
        axis=1,
    )
    music["bucket_hint"] = music["action_bucket"].astype(int)
    music["bucket_is_soft"] = False
    return music


def _clean_stage(
    stage_name: str,
    interactions: pd.DataFrame,
    wrist: np.ndarray,
    env: dict,
    music: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    df = interactions.copy()
    df["timestamp"] = df["timestamp"].astype("int64")
    df["timestamp_local"] = parse_situnes_timestamp(df["timestamp"]).astype(str)

    music_keep = music[
        [
            "item_id",
            "music",
            "singer",
            "general_genre",
            "popularity",
            "loudness",
            "danceability",
            "energy",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "valence",
            "tempo",
            "action_bucket",
            "bucket_hint",
            "bucket_is_soft",
            "F0final_sma_amean",
            "F0final_sma_stddev",
            "audspec_lengthL1norm_sma_stddev",
            "pcm_RMSenergy_sma_stddev",
            "pcm_fftMag_psySharpness_sma_amean",
            "pcm_fftMag_psySharpness_sma_stddev",
            "pcm_zcr_sma_amean",
            "pcm_zcr_sma_stddev",
        ]
    ].copy()
    df = df.merge(music_keep, on="item_id", how="inner", validate="many_to_one")

    obs_list = []
    session_summaries = []
    time_buckets = []
    weather_buckets = []
    gps_speeds = []

    for _, row in df.iterrows():
        inter_id = int(row["inter_id"])
        env_entry = env[str(inter_id)]
        summary = summarize_wrist_session(wrist[inter_id - 1])
        obs_list.append(summary.pop("wrist_obs"))
        session_summaries.append(summary)

        time_bucket = max(0, min(2, int(env_entry.get("time", 2)) - 1))
        weather_bucket = max(0, min(2, int(env_entry.get("weather", [0])[0])))
        gps_speed = float(env_entry.get("GPS", [0.0, 0.0, 0.0])[-1])

        time_buckets.append(time_bucket)
        weather_buckets.append(weather_bucket)
        gps_speeds.append(gps_speed)

    summary_df = pd.DataFrame(session_summaries)
    obs = np.asarray(obs_list, dtype=np.int32)

    df = pd.concat([df.reset_index(drop=True), summary_df], axis=1)
    df["time_bucket"] = np.asarray(time_buckets, dtype=np.int32)
    df["weather_bucket"] = np.asarray(weather_buckets, dtype=np.int32)
    df["gps_speed"] = np.asarray(gps_speeds, dtype=np.float32)
    df["dataset_stage"] = stage_name
    df["pre_emotion_mask"] = 1.0

    rewards = [
        reward_from_emotions(
            row["emo_pre_valence"],
            row["emo_pre_arousal"],
            row["emo_post_valence"],
            row["emo_post_arousal"],
            threshold=PIPELINE_CONFIG["reward_threshold"],
        )
        for _, row in df.iterrows()
    ]
    df["reward"] = [reward for reward, _ in rewards]
    df["reward_score"] = [score for _, score in rewards]
    df["emotion_benefit"] = df["reward_score"].map(emotional_benefit)
    df["acceptance_score"] = [
        acceptance_score(
            preference=row.get("preference"),
            rating=row.get("rating"),
        )
        for _, row in df.iterrows()
    ]
    df["combined_reward"] = [
        combine_outcomes(
            emotion_value=row["emotion_benefit"],
            acceptance_value=row["acceptance_score"],
            alpha=PIPELINE_CONFIG["reward_alpha"],
            beta=PIPELINE_CONFIG["reward_beta"],
        )
        for _, row in df.iterrows()
    ]

    return df.reset_index(drop=True), obs


def build_user_preferences(stage1_clean: pd.DataFrame, situnes_music: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    music_keep = situnes_music[
        [
            "item_id",
            "music",
            "singer",
            "general_genre",
            "valence",
            "energy",
            "tempo",
            "acousticness",
            "instrumentalness",
            "popularity",
        ]
    ].copy()
    stage1_enriched = stage1_clean.merge(music_keep, on="item_id", how="left")

    global_stats = {
        "valence": float(stage1_enriched["valence"].mean()),
        "energy": float(stage1_enriched["energy"].mean()),
        "acousticness": float(stage1_enriched["acousticness"].mean()),
        "instrumentalness": float(stage1_enriched["instrumentalness"].mean()),
        "popularity": float(stage1_enriched["popularity"].mean() / 100.0),
    }

    rows = []
    profiles = {}

    for user_id, group in stage1_enriched.groupby("user_id"):
        liked = group[group["rating"] >= 4]
        disliked = group[group["rating"] <= 2]

        def pref_delta(feature: str) -> float:
            if not liked.empty and not disliked.empty:
                value = float(liked[feature].mean() - disliked[feature].mean())
            elif not liked.empty:
                value = float(liked[feature].mean() - global_stats[feature])
            elif not disliked.empty:
                value = float(global_stats[feature] - disliked[feature].mean())
            else:
                value = 0.0
            return float(np.clip(value, -1.0, 1.0))

        weights = np.clip(group["rating"].to_numpy(dtype=np.float64) - 2.5, -1.5, 2.5)
        positive_weights = np.clip(group["rating"].to_numpy(dtype=np.float64) - 2.0, 0.0, None) + 0.1

        genre_scores = (
            group.assign(score=group["rating"] - 3.0)
            .groupby("general_genre")["score"]
            .mean()
            .sort_values(ascending=False)
        )
        top_genres = [str(x) for x in genre_scores[genre_scores > 0].head(3).index.tolist()]

        artist_scores = (
            group.assign(score=group["rating"] - 3.0)
            .groupby("singer")["score"]
            .mean()
            .sort_values(ascending=False)
        )
        top_artists = [str(x) for x in artist_scores[artist_scores > 0].head(3).index.tolist()]

        acoustic_pref = float(
            np.average(group["acousticness"].to_numpy(dtype=np.float64), weights=positive_weights)
        )
        instrumental_pref = float(
            np.average(group["instrumentalness"].to_numpy(dtype=np.float64), weights=positive_weights)
        )
        popularity_tolerance = float(
            np.clip(
                np.average(group["popularity"].to_numpy(dtype=np.float64) / 100.0, weights=positive_weights),
                0.0,
                1.0,
            )
        )

        valence_pref = pref_delta("valence")
        energy_pref = pref_delta("energy")

        rows.append(
            {
                "user_id": int(user_id),
                "user_valence_pref": valence_pref,
                "user_energy_pref": energy_pref,
                "preferred_acousticness": acoustic_pref,
                "preferred_instrumentalness": instrumental_pref,
                "popularity_tolerance": popularity_tolerance,
            }
        )
        profiles[str(int(user_id))] = {
            "user_id": int(user_id),
            "user_valence_pref": valence_pref,
            "user_energy_pref": energy_pref,
            "preferred_acousticness": acoustic_pref,
            "preferred_instrumentalness": instrumental_pref,
            "popularity_tolerance": popularity_tolerance,
            "top_genres": top_genres,
            "top_artists": top_artists,
            "mean_stage1_valence": float(group["emo_valence"].mean()),
            "mean_stage1_arousal": float(group["emo_arousal"].mean()),
            "stage1_rows": int(len(group)),
        }

    pref_df = pd.DataFrame(rows).sort_values("user_id").reset_index(drop=True)
    stage1_enriched = stage1_enriched.merge(pref_df, on="user_id", how="left")
    return stage1_enriched, pref_df, profiles


def attach_hr_baselines(combined: pd.DataFrame) -> pd.DataFrame:
    shrink = float(PIPELINE_CONFIG["hr_baseline_shrink"])
    split_stats = (
        combined.groupby("split")[["hr_mean", "hr_std"]]
        .mean()
        .rename(columns={"hr_mean": "split_hr_mean", "hr_std": "split_hr_std"})
        .reset_index()
    )
    user_stats = (
        combined.groupby(["split", "user_id"])
        .agg(
            user_hr_mean_raw=("hr_mean", "mean"),
            user_hr_std_raw=("hr_std", "mean"),
            hr_session_count=("inter_id", "count"),
        )
        .reset_index()
        .merge(split_stats, on="split", how="left")
    )
    weight = user_stats["hr_session_count"] / (user_stats["hr_session_count"] + shrink)
    user_stats["user_hr_baseline_mean"] = (
        weight * user_stats["user_hr_mean_raw"] + (1.0 - weight) * user_stats["split_hr_mean"]
    )
    user_stats["user_hr_baseline_std"] = (
        weight * user_stats["user_hr_std_raw"] + (1.0 - weight) * user_stats["split_hr_std"]
    )

    merged = combined.merge(
        user_stats[
            [
                "split",
                "user_id",
                "hr_session_count",
                "user_hr_baseline_mean",
                "user_hr_baseline_std",
            ]
        ],
        on=["split", "user_id"],
        how="left",
    )
    merged["hr_mean_rel_user"] = merged["hr_mean"] - merged["user_hr_baseline_mean"]
    merged["hr_std_rel_user"] = merged["hr_std"] - merged["user_hr_baseline_std"]
    return merged


def clean_situnes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, dict, dict]:
    df1 = pd.read_csv(SITUNES_DIR / "Stage1" / "interactions.csv")
    df2 = pd.read_csv(SITUNES_DIR / "Stage2" / "interactions.csv")
    df3 = pd.read_csv(SITUNES_DIR / "Stage3" / "interactions.csv")

    wrist2 = np.load(SITUNES_DIR / "Stage2" / "wrist.npy", allow_pickle=True)
    wrist3 = np.load(SITUNES_DIR / "Stage3" / "wrist.npy", allow_pickle=True)

    with (SITUNES_DIR / "Stage2" / "env.json").open("r", encoding="utf-8") as handle:
        env2 = json.load(handle)
    with (SITUNES_DIR / "Stage3" / "env.json").open("r", encoding="utf-8") as handle:
        env3 = json.load(handle)

    situnes_music = _load_situnes_music()

    stage2_base, obs2 = _clean_stage("stage2", df2, wrist2, env2, situnes_music)
    stage3_base, obs3 = _clean_stage("stage3", df3, wrist3, env3, situnes_music)

    stage1_clean = df1.drop(columns=["duration"], errors="ignore").copy()
    stage1_clean["timestamp"] = stage1_clean["timestamp"].astype("int64")
    stage1_clean["timestamp_local"] = parse_situnes_timestamp(stage1_clean["timestamp"]).astype(str)
    stage1_enriched, pref_df, profiles = build_user_preferences(stage1_clean, situnes_music)

    obs_all_raw = np.vstack([obs2, obs3])
    stage2_base["obs_source_index"] = np.arange(len(stage2_base), dtype=np.int32)
    stage3_base["obs_source_index"] = np.arange(len(stage3_base), dtype=np.int32) + len(stage2_base)

    combined = pd.concat([stage2_base, stage3_base], ignore_index=True)
    combined = combined.sort_values(["user_id", "timestamp", "inter_id"]).reset_index(drop=True)

    user_counts = combined.groupby("user_id").size().sort_values(ascending=False)
    split_users = balanced_user_split(
        user_counts=user_counts,
        stage3_users=set(stage3_base["user_id"].unique()),
        seed=int(PIPELINE_CONFIG["split_seed"]),
    )
    user_to_split = {
        user_id: split_name
        for split_name, users in split_users.items()
        for user_id in users
    }
    combined["split"] = combined["user_id"].map(user_to_split)
    combined["session_order_user"] = combined.groupby("user_id").cumcount().astype(int)

    combined = combined.merge(pref_df, on="user_id", how="left")
    combined = attach_hr_baselines(combined)
    combined["pre_emotion_mask"] = 1.0

    wrist_obs_all = obs_all_raw[combined["obs_source_index"].to_numpy(dtype=np.int32)]
    combined["obs_idx"] = np.arange(len(combined), dtype=np.int32)
    combined = combined.drop(columns=["obs_source_index"], errors="ignore")

    stage2_clean = combined[combined["dataset_stage"] == "stage2"].copy().reset_index(drop=True)
    stage3_clean = combined[combined["dataset_stage"] == "stage3"].copy().reset_index(drop=True)

    audit = {
        "situnes": {
            "stage1_rows": int(len(stage1_enriched)),
            "stage2_rows": int(len(stage2_clean)),
            "stage3_rows": int(len(stage3_clean)),
            "combined_rows": int(len(combined)),
            "user_counts": {str(int(k)): int(v) for k, v in user_counts.items()},
            "split_users": {k: [int(u) for u in v] for k, v in split_users.items()},
            "split_rows": combined["split"].value_counts().sort_index().to_dict(),
            "reward_distribution": combined["reward"].value_counts().sort_index().to_dict(),
            "bucket_distribution": combined["action_bucket"].value_counts().sort_index().to_dict(),
            "time_distribution": combined["time_bucket"].value_counts().sort_index().to_dict(),
            "activity_distribution": combined["activity_majority"].value_counts().sort_index().to_dict(),
            "hr_mean_summary": {
                "mean": float(combined["hr_mean"].mean()),
                "std": float(combined["hr_mean"].std(ddof=0)),
                "min": float(combined["hr_mean"].min()),
                "max": float(combined["hr_mean"].max()),
            },
            "preference_profile_rows": int(len(pref_df)),
            "raw_hashes": {
                "stage1_interactions": file_sha256(SITUNES_DIR / "Stage1" / "interactions.csv"),
                "stage2_interactions": file_sha256(SITUNES_DIR / "Stage2" / "interactions.csv"),
                "stage3_interactions": file_sha256(SITUNES_DIR / "Stage3" / "interactions.csv"),
                "stage2_wrist": file_sha256(SITUNES_DIR / "Stage2" / "wrist.npy"),
                "stage3_wrist": file_sha256(SITUNES_DIR / "Stage3" / "wrist.npy"),
                "music_info": file_sha256(SITUNES_DIR / "music_metadata" / "music_info.csv"),
            },
        }
    }

    return stage1_enriched, stage2_clean, stage3_clean, combined, wrist_obs_all, audit, profiles


def _ridge_fit_predict(
    train_df: pd.DataFrame,
    target: str,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
    alpha: float = 1.0,
) -> tuple[np.ndarray, dict]:
    x_train = train_df[feature_cols].to_numpy(dtype=np.float64)
    y_train = train_df[target].to_numpy(dtype=np.float64)
    x_pred = predict_df[feature_cols].to_numpy(dtype=np.float64)

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-8] = 1.0

    x_train_z = (x_train - mean) / std
    x_pred_z = (x_pred - mean) / std

    x_aug = np.column_stack([np.ones(len(x_train_z)), x_train_z])
    p_aug = np.column_stack([np.ones(len(x_pred_z)), x_pred_z])

    reg = np.eye(x_aug.shape[1], dtype=np.float64) * alpha
    reg[0, 0] = 0.0
    weights = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y_train)

    fitted = x_aug @ weights
    predicted = p_aug @ weights

    ss_res = float(np.square(y_train - fitted).sum())
    ss_tot = float(np.square(y_train - y_train.mean()).sum() + 1e-12)
    r2 = 1.0 - ss_res / ss_tot

    metadata = {
        "target": target,
        "alpha": alpha,
        "r2_train": r2,
        "feature_cols": feature_cols,
    }
    return predicted, metadata


def compute_pmemo_eda_impact() -> dict[int, float]:
    eda_dir = PMEMO_DIR / "EDA"
    if not eda_dir.exists():
        return {}

    scores = {}
    for csv_path in sorted(eda_dir.glob("*_EDA.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        signal = df.drop(columns=[col for col in df.columns if str(col).lower().startswith("time")], errors="ignore")
        if signal.empty:
            continue
        listener_std = signal.std(axis=0, ddof=0).replace([np.inf, -np.inf], np.nan).dropna()
        if listener_std.empty:
            continue
        try:
            music_id = int(csv_path.stem.split("_")[0])
        except ValueError:
            continue
        scores[music_id] = float(listener_std.mean())

    if not scores:
        return {}
    values = np.asarray(list(scores.values()), dtype=np.float64)
    lo = float(values.min())
    hi = float(values.max())
    denom = max(hi - lo, 1e-8)
    return {music_id: float((value - lo) / denom) for music_id, value in scores.items()}


def clean_pmemo(situnes_music: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    static = pd.read_csv(
        PMEMO_DIR / "annotations" / "static_annotations.csv",
        usecols=["musicId", "Arousal(mean)", "Valence(mean)"],
    )
    static_std = pd.read_csv(
        PMEMO_DIR / "annotations" / "static_annotations_std.csv",
        usecols=["musicId", "Arousal(std)", "Valence(std)"],
    )
    metadata = pd.read_csv(PMEMO_DIR / "metadata.csv")

    common_feature_cols = [
        "musicId",
        "F0final_sma_amean",
        "F0final_sma_stddev",
        "audspec_lengthL1norm_sma_stddev",
        "pcm_RMSenergy_sma_stddev",
        "pcm_fftMag_psySharpness_sma_amean",
        "pcm_fftMag_psySharpness_sma_stddev",
        "pcm_zcr_sma_amean",
        "pcm_zcr_sma_stddev",
    ]
    pmemo_features = pd.read_csv(
        PMEMO_DIR / "features" / "static_features.csv",
        usecols=common_feature_cols,
    )

    merged = (
        static.merge(static_std, on="musicId", how="inner")
        .merge(metadata[["musicId", "title", "artist", "album", "duration"]], on="musicId", how="left")
        .merge(pmemo_features, on="musicId", how="inner")
    )
    merged = merged[merged["Valence(std)"] <= PIPELINE_CONFIG["pmemo_std_cutoff"]].copy()
    merged["valence_01"] = merged["Valence(mean)"].clip(0.01, 0.99)
    merged["arousal_01"] = merged["Arousal(mean)"].clip(0.01, 0.99)
    merged["valence_norm"] = (merged["valence_01"] * 2.0 - 1.0).clip(-0.99, 0.99)
    merged["arousal_norm"] = (merged["arousal_01"] * 2.0 - 1.0).clip(-0.99, 0.99)

    feature_cols = [col for col in common_feature_cols if col != "musicId"]
    situnes_train = situnes_music[feature_cols + ["energy", "tempo"]].dropna().copy()

    predicted_energy, energy_meta = _ridge_fit_predict(
        situnes_train,
        target="energy",
        predict_df=merged,
        feature_cols=feature_cols,
        alpha=2.0,
    )
    predicted_tempo, tempo_meta = _ridge_fit_predict(
        situnes_train,
        target="tempo",
        predict_df=merged,
        feature_cols=feature_cols,
        alpha=4.0,
    )

    blend = float(PIPELINE_CONFIG["pmemo_energy_blend"])
    merged["energy_pred"] = np.clip(predicted_energy, 0.0, 1.0)
    merged["energy"] = np.clip(
        blend * merged["energy_pred"] + (1.0 - blend) * merged["arousal_01"],
        0.0,
        1.0,
    )
    merged["tempo"] = np.clip(
        predicted_tempo,
        PIPELINE_CONFIG["pmemo_tempo_clip"][0],
        PIPELINE_CONFIG["pmemo_tempo_clip"][1],
    )
    # Use PMEmo-internal tempo ranking for soft bucket hints. The transferred
    # absolute tempo scale is weak, but relative tempo still helps diversify
    # retrieval without treating PMEmo as hard policy supervision.
    merged["tempo_pctl"] = merged["tempo"].rank(pct=True, method="average")
    valence_level = (merged["valence_01"] >= 0.33).astype(int)
    energy_level = (merged["energy"] >= 0.4).astype(int)
    tempo_level = (merged["tempo_pctl"] >= 0.5).astype(int)
    merged["bucket_hint"] = (valence_level * 4 + energy_level * 2 + tempo_level).astype(int)
    merged["action_bucket"] = merged["bucket_hint"].astype(int)
    merged["bucket_is_soft"] = True

    eda_impact = compute_pmemo_eda_impact()
    merged["eda_impact"] = merged["musicId"].map(eda_impact).fillna(0.0)

    out = merged.rename(columns={"musicId": "song_id"})[
        [
            "song_id",
            "title",
            "artist",
            "album",
            "duration",
            "valence_01",
            "valence_norm",
            "arousal_01",
            "arousal_norm",
            "energy",
            "tempo",
            "action_bucket",
            "bucket_hint",
            "bucket_is_soft",
            "eda_impact",
        ]
    ].copy()
    out["source"] = "pmemo"

    audit = {
        "pmemo": {
            "rows": int(len(out)),
            "bucket_distribution": out["bucket_hint"].value_counts().sort_index().to_dict(),
            "eda_coverage": int((out["eda_impact"] > 0).sum()),
            "transfer_models": {
                "energy": {
                    "r2_train": round(float(energy_meta["r2_train"]), 4),
                    "features": feature_cols,
                },
                "tempo": {
                    "r2_train": round(float(tempo_meta["r2_train"]), 4),
                    "features": feature_cols,
                },
            },
        }
    }
    return out, audit


def clean_spotify() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(SPOTIFY_DIR / "dataset.csv")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.drop_duplicates(subset="track_id", keep="first")
    df = df[df["tempo"] > 0].copy()
    df = df.dropna(
        subset=[
            "track_id",
            "track_name",
            "artists",
            "valence",
            "energy",
            "tempo",
        ]
    )
    for text_col in ["artists", "album_name", "track_name", "track_genre"]:
        df[text_col] = (
            df[text_col]
            .astype(str)
            .str.encode("utf-8", errors="ignore")
            .str.decode("utf-8")
            .str.strip()
        )

    df["action_bucket"] = df.apply(
        lambda row: get_action_bucket(row["valence"], row["energy"], row["tempo"]),
        axis=1,
    )
    df["bucket_hint"] = df["action_bucket"].astype(int)
    df["bucket_is_soft"] = False

    out = df[
        [
            "track_id",
            "track_name",
            "artists",
            "album_name",
            "track_genre",
            "popularity",
            "duration_ms",
            "explicit",
            "danceability",
            "energy",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
            "action_bucket",
            "bucket_hint",
            "bucket_is_soft",
        ]
    ].copy()
    out["source"] = "spotify"
    out = out.rename(columns={"track_genre": "genre"})

    audit = {
        "spotify": {
            "rows": int(len(out)),
            "bucket_distribution": out["action_bucket"].value_counts().sort_index().to_dict(),
        }
    }
    return out, audit


def write_split_artifacts(combined: pd.DataFrame) -> dict:
    split_manifest = {}
    for split_name, split_df in combined.groupby("split"):
        split_manifest[split_name] = {
            "users": sorted(int(u) for u in split_df["user_id"].unique()),
            "row_indices": split_df.index.astype(int).tolist(),
            "rows": int(len(split_df)),
        }
    return split_manifest


def build_audit(
    situnes_audit: dict,
    pmemo_audit: dict,
    spotify_audit: dict,
    split_manifest: dict,
) -> dict:
    audit = {
        "config": PIPELINE_CONFIG,
        "config_hash": config_hash(PIPELINE_CONFIG),
        "raw_roots": {
            "situnes": str(SITUNES_DIR),
            "pmemo": str(PMEMO_DIR),
            "spotify": str(SPOTIFY_DIR),
        },
        "splits": split_manifest,
    }
    audit.update(situnes_audit)
    audit.update(pmemo_audit)
    audit.update(spotify_audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic processed datasets.")
    parser.add_argument("--skip-pmemo", action="store_true", help="Skip PMEmo preprocessing.")
    parser.add_argument("--skip-spotify", action="store_true", help="Skip Spotify preprocessing.")
    args = parser.parse_args()

    ensure_dirs()

    stage1_clean, stage2_clean, stage3_clean, combined, wrist_obs_all, situnes_audit, user_profiles = clean_situnes()
    split_manifest = write_split_artifacts(combined)

    stage1_clean.to_csv(PROCESSED_DIR / "stage1_clean.csv", index=False)
    stage2_clean.to_csv(PROCESSED_DIR / "stage2_clean.csv", index=False)
    stage3_clean.to_csv(PROCESSED_DIR / "stage3_clean.csv", index=False)
    combined.to_csv(PROCESSED_DIR / "interactions_clean.csv", index=False)
    np.save(PROCESSED_DIR / "wrist_obs_all.npy", wrist_obs_all)
    np.save(PROCESSED_DIR / "wrist_stage2_obs.npy", wrist_obs_all[combined["dataset_stage"].to_numpy() == "stage2"])
    np.save(PROCESSED_DIR / "wrist_stage3_obs.npy", wrist_obs_all[combined["dataset_stage"].to_numpy() == "stage3"])

    situnes_music = _load_situnes_music()
    situnes_music.to_csv(PROCESSED_DIR / "music_situnes_clean.csv", index=False)

    pmemo_audit = {"pmemo": {"skipped": True}}
    if not args.skip_pmemo and (PMEMO_DIR / "annotations" / "static_annotations.csv").exists():
        pmemo_clean, pmemo_audit = clean_pmemo(situnes_music)
        pmemo_clean.to_csv(PROCESSED_DIR / "music_pmemo_clean.csv", index=False)

    spotify_audit = {"spotify": {"skipped": True}}
    if not args.skip_spotify and (SPOTIFY_DIR / "dataset.csv").exists():
        spotify_clean, spotify_audit = clean_spotify()
        spotify_clean.to_csv(PROCESSED_DIR / "music_spotify_clean.csv", index=False)

    audit = build_audit(situnes_audit, pmemo_audit, spotify_audit, split_manifest)
    with (PROCESSED_DIR / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=2)
    with (PROCESSED_DIR / "dataset_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    with (PROCESSED_DIR / "user_preferences.json").open("w", encoding="utf-8") as handle:
        json.dump(user_profiles, handle, indent=2)

    print("Processed datasets written to", PROCESSED_DIR)
    print("  stage1_clean.csv")
    print("  stage2_clean.csv")
    print("  stage3_clean.csv")
    print("  interactions_clean.csv")
    print("  wrist_obs_all.npy / wrist_stage2_obs.npy / wrist_stage3_obs.npy")
    print("  music_situnes_clean.csv")
    if not args.skip_pmemo:
        print("  music_pmemo_clean.csv")
    if not args.skip_spotify:
        print("  music_spotify_clean.csv")
    print("  user_preferences.json")
    print("  split_manifest.json")
    print("  dataset_audit.json")


if __name__ == "__main__":
    main()
