"""
Public and catalog music tensors for the V2 rebuild.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.common import PMEMO_DIR, PROCESSED_DIR
from src.data.preprocess import _load_situnes_music, clean_pmemo, clean_spotify

from .normalization import apply_zscore, fit_song_stats, save_stats
from .schema import SONG_CATALOG_COLUMNS, SONG_DYN_LEN, SONG_STATIC_FEATURES, validate_song_catalog


def _token_hash_embedding(text: str, dim: int = 4) -> np.ndarray:
    if not text:
        return np.zeros(dim, dtype=np.float32)
    vec = np.zeros(dim, dtype=np.float64)
    tokens = [token for token in str(text).lower().replace("/", " ").replace("-", " ").split() if token]
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        bucket = digest[0] % dim
        sign = 1.0 if digest[1] % 2 == 0 else -1.0
        vec[bucket] += sign
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec /= norm
    return vec.astype(np.float32)


def _resample_curve(values: np.ndarray, target_len: int = SONG_DYN_LEN) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(arr) == 1:
        return np.full(target_len, float(arr[0]), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(arr))
    x_new = np.linspace(0.0, 1.0, num=target_len)
    return np.interp(x_new, x_old, arr).astype(np.float32)


def _load_pmemo_dynamic_curves() -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    path = PMEMO_DIR / "annotations" / "dynamic_annotations.csv"
    if not path.exists():
        return {}, {}
    df = pd.read_csv(path, usecols=["musicId", "frameTime", "Arousal(mean)", "Valence(mean)"])
    df = df.sort_values(["musicId", "frameTime"])
    valence_curves = {}
    arousal_curves = {}
    for music_id, group in df.groupby("musicId", sort=True):
        valence_curves[int(music_id)] = _resample_curve(group["Valence(mean)"].to_numpy(dtype=np.float64))
        arousal_curves[int(music_id)] = _resample_curve(group["Arousal(mean)"].to_numpy(dtype=np.float64))
    return valence_curves, arousal_curves


def _build_situnes_catalog() -> pd.DataFrame:
    situnes = _load_situnes_music().copy()
    situnes = situnes.dropna(
        subset=[
            "item_id",
            "valence",
            "energy",
            "tempo",
            "danceability",
            "acousticness",
            "instrumentalness",
            "speechiness",
            "loudness",
            "popularity",
        ]
    )
    situnes = situnes[situnes["item_id"].astype(int) >= 0].copy()
    situnes["song_id"] = situnes["item_id"].map(lambda item_id: f"situnes_{int(item_id)}")
    return pd.DataFrame(
        {
            "song_id": situnes["song_id"],
            "source": "situnes",
            "title": situnes["music"],
            "artist": situnes["singer"],
            "genre": situnes.get("general_genre", ""),
            "valence_static": situnes["valence"].astype(float),
            "arousal_static": situnes["energy"].astype(float),
            "energy": situnes["energy"].astype(float),
            "tempo_raw": situnes["tempo"].astype(float),
            "danceability": situnes.get("danceability", 0.5).astype(float),
            "acousticness": situnes.get("acousticness", 0.5).astype(float),
            "instrumentalness": situnes.get("instrumentalness", 0.0).astype(float),
            "speechiness": situnes.get("speechiness", 0.08).astype(float),
            "liveness": 0.15,
            "loudness_raw": situnes.get("loudness", -8.0).astype(float),
            "popularity_raw": situnes.get("popularity", 45.0).astype(float),
            "explicit_flag": 0.0,
            "eda_impact_norm": 0.0,
            "dyn_valence_delta": 0.0,
            "dyn_arousal_delta": 0.0,
            "dyn_valence_vol": 0.0,
            "dyn_arousal_vol": 0.0,
            "dyn_arousal_peak": 0.0,
            "song_quality": 0.75,
            "has_dynamic": False,
            "trainable": True,
        }
    )


def _build_pmemo_catalog(situnes_music: pd.DataFrame) -> pd.DataFrame:
    if not (PMEMO_DIR / "annotations" / "static_annotations.csv").exists():
        fallback = PROCESSED_DIR / "music_pmemo_clean.csv"
        if not fallback.exists():
            return pd.DataFrame()
        pmemo = pd.read_csv(fallback)
    else:
        pmemo, _ = clean_pmemo(situnes_music)

    return pd.DataFrame(
        {
            "song_id": pmemo["song_id"].map(lambda song_id: f"pmemo_{int(song_id)}"),
            "source": "pmemo",
            "title": pmemo.get("title", "Unknown"),
            "artist": pmemo.get("artist", "Unknown"),
            "genre": "",
            "valence_static": pmemo["valence_01"].astype(float),
            "arousal_static": pmemo["arousal_01"].astype(float),
            "energy": pmemo["energy"].astype(float),
            "tempo_raw": pmemo["tempo"].astype(float),
            "danceability": 0.45,
            "acousticness": 0.55,
            "instrumentalness": 0.25,
            "speechiness": 0.05,
            "liveness": 0.15,
            "loudness_raw": -8.0,
            "popularity_raw": 35.0,
            "explicit_flag": 0.0,
            "eda_impact_norm": pmemo.get("eda_impact", 0.0).astype(float),
            "dyn_valence_delta": pmemo.get("dyn_valence_delta", 0.0).astype(float),
            "dyn_arousal_delta": pmemo.get("dyn_arousal_delta", 0.0).astype(float),
            "dyn_valence_vol": pmemo.get("dyn_valence_volatility", 0.0).astype(float),
            "dyn_arousal_vol": pmemo.get("dyn_arousal_volatility", 0.0).astype(float),
            "dyn_arousal_peak": pmemo.get("dyn_arousal_peak", 0.0).astype(float),
            "song_quality": pmemo.get("dyn_quality", 0.5).astype(float),
            "has_dynamic": True,
            "trainable": True,
        }
    )


def _build_spotify_catalog() -> pd.DataFrame:
    source_path = Path("data/raw/spotify_kaggle/dataset.csv")
    if not source_path.exists():
        fallback = PROCESSED_DIR / "music_spotify_clean.csv"
        if not fallback.exists():
            return pd.DataFrame()
        spotify = pd.read_csv(fallback)
        if "genre" not in spotify.columns and "track_genre" in spotify.columns:
            spotify = spotify.rename(columns={"track_genre": "genre"})
    else:
        spotify, _ = clean_spotify()

    out = pd.DataFrame(
        {
            "song_id": spotify["track_id"].map(lambda track_id: f"spotify_{track_id}"),
            "source": "spotify",
            "title": spotify["track_name"],
            "artist": spotify["artists"],
            "genre": spotify.get("genre", ""),
            "valence_static": spotify["valence"].astype(float),
            "arousal_static": spotify["energy"].astype(float),
            "energy": spotify["energy"].astype(float),
            "tempo_raw": spotify["tempo"].astype(float),
            "danceability": spotify.get("danceability", 0.5).astype(float),
            "acousticness": spotify.get("acousticness", 0.5).astype(float),
            "instrumentalness": spotify.get("instrumentalness", 0.0).astype(float),
            "speechiness": spotify.get("speechiness", 0.08).astype(float),
            "liveness": spotify.get("liveness", 0.15).astype(float),
            "loudness_raw": spotify["loudness"].astype(float) if "loudness" in spotify.columns else pd.Series(-8.0, index=spotify.index),
            "popularity_raw": spotify.get("popularity", 50.0).astype(float),
            "explicit_flag": spotify.get("explicit", False).astype(float),
            "eda_impact_norm": 0.0,
            "dyn_valence_delta": 0.0,
            "dyn_arousal_delta": 0.0,
            "dyn_valence_vol": 0.0,
            "dyn_arousal_vol": 0.0,
            "dyn_arousal_peak": 0.0,
            "song_quality": 0.65,
            "has_dynamic": False,
            "trainable": True,
        }
    )
    return out


def build_public_music_v2(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    situnes_music = _load_situnes_music()
    frames = [_build_situnes_catalog()]

    pmemo_df = _build_pmemo_catalog(situnes_music)
    if not pmemo_df.empty:
        frames.append(pmemo_df)

    spotify_df = _build_spotify_catalog()
    if not spotify_df.empty:
        frames.append(spotify_df)

    catalog = pd.concat(frames, ignore_index=True)
    song_stats = fit_song_stats(catalog)
    catalog["tempo_norm"] = apply_zscore(catalog["tempo_raw"].to_numpy(dtype=np.float64), song_stats["tempo"])
    catalog["loudness_norm"] = apply_zscore(catalog["loudness_raw"].to_numpy(dtype=np.float64), song_stats["loudness"])
    catalog["popularity_norm"] = apply_zscore(catalog["popularity_raw"].to_numpy(dtype=np.float64), song_stats["popularity"])

    genre_emb = np.stack([_token_hash_embedding(text) for text in catalog["genre"].fillna("")], axis=0)
    for idx in range(4):
        catalog[f"genre_emb_{idx + 1}"] = genre_emb[:, idx].astype(np.float32)
    catalog["source_situnes"] = (catalog["source"] == "situnes").astype(float)
    catalog["source_spotify"] = (catalog["source"] == "spotify").astype(float)
    catalog["source_pmemo"] = (catalog["source"] == "pmemo").astype(float)

    val_curves, aro_curves = _load_pmemo_dynamic_curves()
    song_dynamic = np.zeros((len(catalog), SONG_DYN_LEN, 2), dtype=np.float32)
    song_dynamic_mask = np.zeros((len(catalog), SONG_DYN_LEN, 1), dtype=np.float32)

    for idx, row in catalog.iterrows():
        if row["source"] != "pmemo":
            continue
        raw_song_id = int(str(row["song_id"]).split("_", 1)[1])
        if raw_song_id not in val_curves or raw_song_id not in aro_curves:
            continue
        song_dynamic[idx, :, 0] = val_curves[raw_song_id]
        song_dynamic[idx, :, 1] = aro_curves[raw_song_id]
        song_dynamic_mask[idx, :, 0] = 1.0

    song_static = catalog[SONG_STATIC_FEATURES].to_numpy(dtype=np.float32)
    export_catalog = catalog[SONG_CATALOG_COLUMNS].copy()
    validate_song_catalog(export_catalog)

    export_catalog.to_parquet(out_dir / "song_catalog.parquet", index=False)
    np.save(out_dir / "song_static.npy", song_static)
    np.save(out_dir / "song_dynamic.npy", song_dynamic)
    np.save(out_dir / "song_dynamic_mask.npy", song_dynamic_mask)
    (out_dir / "song_id_map.json").write_text(
        json.dumps({song_id: idx for idx, song_id in enumerate(export_catalog["song_id"].tolist())}, indent=2),
        encoding="utf-8",
    )
    save_stats(song_stats, out_dir / "song_normalization_stats.json")

    return {
        "songs": int(len(export_catalog)),
        "sources": export_catalog["source"].value_counts().sort_index().to_dict(),
        "dynamic_rows": int(export_catalog["has_dynamic"].sum()),
    }
