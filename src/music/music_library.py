"""
Deterministic, preference-aware music retrieval on top of the cleaned catalogs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.common import BUCKET_LABELS, PROCESSED_DIR, bucket_targets, track_quality_features


MODE_GENRE_BOOSTS = {
    "focus": ["ambient", "classical", "study", "piano", "acoustic", "chill", "indie", "lofi"],
    "wind_down": ["ambient", "acoustic", "singer-songwriter", "folk", "classical", "lofi", "jazz"],
    "exercise": ["dance", "edm", "electro", "house", "hip-hop", "rock", "pop", "workout"],
    "exercise-lite": ["dance", "indie", "rock", "pop", "electro"],
    "uplift": ["soul", "funk", "disco", "indie", "pop"],
}


class MusicLibrary:
    def __init__(self, df: pd.DataFrame, user_profiles: dict[str, dict] | None = None):
        self.df = track_quality_features(df.reset_index(drop=True))
        self.df["bucket_label"] = self.df["action_bucket"].map(BUCKET_LABELS).fillna("soft-match")
        self.user_profiles = user_profiles or {}

    @classmethod
    def build(cls, processed_dir: str | Path = PROCESSED_DIR):
        processed = Path(processed_dir)
        frames = []

        situnes_path = processed / "music_situnes_clean.csv"
        if not situnes_path.exists():
            raise FileNotFoundError("music_situnes_clean.csv not found. Run preprocessing first.")
        situnes = pd.read_csv(situnes_path)
        frames.append(
            pd.DataFrame(
                {
                    "track_name": situnes["music"],
                    "artist": situnes["singer"],
                    "genre": situnes.get("general_genre", ""),
                    "valence": situnes["valence"],
                    "energy": situnes["energy"],
                    "tempo": situnes["tempo"],
                    "danceability": situnes.get("danceability", 0.5),
                    "speechiness": situnes.get("speechiness", 0.08),
                    "acousticness": situnes.get("acousticness", 0.5),
                    "instrumentalness": situnes.get("instrumentalness", 0.0),
                    "popularity": situnes.get("popularity", 45.0),
                    "action_bucket": situnes["action_bucket"].astype(int),
                    "bucket_hint": situnes.get("bucket_hint", situnes["action_bucket"]).astype(int),
                    "bucket_is_soft": situnes.get("bucket_is_soft", False),
                    "eda_impact": situnes.get("eda_impact", 0.0),
                    "dyn_valence_delta": 0.0,
                    "dyn_arousal_delta": 0.0,
                    "dyn_arousal_volatility": 0.0,
                    "dyn_arousal_peak": 0.0,
                    "dyn_quality": 0.0,
                    "source": "situnes",
                    "explicit": False,
                }
            )
        )

        pmemo_path = processed / "music_pmemo_clean.csv"
        if pmemo_path.exists():
            pmemo = pd.read_csv(pmemo_path)
            frames.append(
                pd.DataFrame(
                    {
                        "track_name": pmemo.get("title", pd.Series(["Unknown"] * len(pmemo))),
                        "artist": pmemo.get("artist", pd.Series(["Unknown"] * len(pmemo))),
                        "genre": "",
                        "valence": pmemo["valence_01"],
                        "energy": pmemo["energy"],
                        "tempo": pmemo["tempo"],
                        "danceability": 0.45,
                        "speechiness": 0.05,
                        "acousticness": 0.55,
                        "instrumentalness": 0.25,
                        "popularity": 35.0,
                        "action_bucket": pmemo["action_bucket"].astype(int),
                        "bucket_hint": pmemo.get("bucket_hint", pmemo["action_bucket"]).astype(int),
                        "bucket_is_soft": pmemo.get("bucket_is_soft", True),
                        "eda_impact": pmemo.get("eda_impact", 0.0),
                        "dyn_valence_delta": pmemo.get("dyn_valence_delta", 0.0),
                        "dyn_arousal_delta": pmemo.get("dyn_arousal_delta", 0.0),
                        "dyn_arousal_volatility": pmemo.get("dyn_arousal_volatility", 0.0),
                        "dyn_arousal_peak": pmemo.get("dyn_arousal_peak", 0.0),
                        "dyn_quality": pmemo.get("dyn_quality", 0.5),
                        "source": "pmemo",
                        "explicit": False,
                    }
                )
            )

        spotify_path = processed / "music_spotify_clean.csv"
        if spotify_path.exists():
            spotify = pd.read_csv(spotify_path)
            frames.append(
                pd.DataFrame(
                    {
                        "track_name": spotify["track_name"],
                        "artist": spotify["artists"],
                        "genre": spotify.get("genre", spotify.get("track_genre", "")),
                        "valence": spotify["valence"],
                        "energy": spotify["energy"],
                        "tempo": spotify["tempo"],
                        "danceability": spotify.get("danceability", 0.5),
                        "speechiness": spotify.get("speechiness", 0.08),
                        "acousticness": spotify.get("acousticness", 0.5),
                        "instrumentalness": spotify.get("instrumentalness", 0.0),
                        "popularity": spotify.get("popularity", 50.0),
                        "action_bucket": spotify["action_bucket"].astype(int),
                        "bucket_hint": spotify.get("bucket_hint", spotify["action_bucket"]).astype(int),
                        "bucket_is_soft": spotify.get("bucket_is_soft", False),
                        "eda_impact": spotify.get("eda_impact", 0.0),
                        "dyn_valence_delta": 0.0,
                        "dyn_arousal_delta": 0.0,
                        "dyn_arousal_volatility": 0.0,
                        "dyn_arousal_peak": 0.0,
                        "dyn_quality": 0.0,
                        "source": "spotify",
                        "explicit": spotify.get("explicit", False),
                    }
                )
            )

        combined = pd.concat(frames, ignore_index=True)
        user_profiles = {}
        pref_path = processed / "user_preferences.json"
        if pref_path.exists():
            user_profiles = json.loads(pref_path.read_text(encoding="utf-8"))
        return cls(combined, user_profiles=user_profiles)

    @staticmethod
    def _mode_for_bucket(bucket: int, context: dict | str | None = None) -> str:
        if isinstance(context, str) and context:
            return context
        if isinstance(context, dict) and context.get("mode"):
            return str(context["mode"])
        if bucket in {0, 1}:
            return "wind_down"
        if bucket in {4, 5, 6}:
            return "focus"
        if bucket in {3, 7}:
            return "exercise"
        return "uplift"

    @staticmethod
    def _genre_bonus(genre: str, mode: str) -> float:
        text = str(genre).lower()
        return sum(0.10 for keyword in MODE_GENRE_BOOSTS.get(mode, []) if keyword in text)

    def _resolve_user_profile(self, context: dict | None) -> dict:
        if context is None:
            return {}
        if context.get("user_profile"):
            return dict(context["user_profile"])
        user_id = context.get("user_id")
        if user_id is None:
            return {}
        return dict(self.user_profiles.get(str(int(user_id)), {}))

    @staticmethod
    def _genre_preference_bonus(genre: str, profile: dict) -> float:
        text = str(genre).lower()
        return sum(0.12 for keyword in profile.get("top_genres", []) if keyword.lower() in text)

    @staticmethod
    def _pmemo_dynamic_bonus(pool: pd.DataFrame, mode: str) -> pd.Series:
        bonus = pd.Series(np.zeros(len(pool), dtype=np.float64), index=pool.index)
        mask = pool["source"].eq("pmemo")
        if not mask.any():
            return bonus

        sub = pool.loc[mask]
        q = np.clip(sub["dyn_quality"].to_numpy(dtype=np.float64), 0.0, 1.0)
        valence_rise = np.clip(sub["dyn_valence_delta"].to_numpy(dtype=np.float64) / 0.10, 0.0, 1.0)
        arousal_rise = np.clip(sub["dyn_arousal_delta"].to_numpy(dtype=np.float64) / 0.10, 0.0, 1.0)
        arousal_fall = np.clip(-sub["dyn_arousal_delta"].to_numpy(dtype=np.float64) / 0.10, 0.0, 1.0)
        arousal_peak_norm = np.clip((sub["dyn_arousal_peak"].to_numpy(dtype=np.float64) - 0.20) / 0.80, 0.0, 1.0)
        arousal_stability = 1.0 - np.clip(sub["dyn_arousal_volatility"].to_numpy(dtype=np.float64) / 0.05, 0.0, 1.0)

        if mode == "focus":
            values = q * (0.10 * arousal_stability + 0.06 * (1.0 - arousal_peak_norm) + 0.04 * arousal_fall)
        elif mode == "wind_down":
            values = q * (0.12 * arousal_fall + 0.06 * (1.0 - arousal_peak_norm) + 0.06 * arousal_stability)
        elif mode == "exercise":
            values = q * (0.10 * arousal_rise + 0.08 * arousal_peak_norm + 0.04 * valence_rise)
        elif mode == "exercise-lite":
            values = q * (0.08 * arousal_rise + 0.05 * arousal_peak_norm + 0.03 * valence_rise)
        else:
            values = q * (0.10 * valence_rise + 0.05 * arousal_rise + 0.03 * arousal_stability)

        bonus.loc[mask] = np.clip(values, 0.0, 0.18)
        return bonus

    @staticmethod
    def _pmemo_dynamic_reason(row: pd.Series, mode: str) -> str:
        if str(row.get("source", "")) != "pmemo":
            return ""
        if float(row.get("dynamic_bonus", 0.0)) <= 0.01:
            return ""
        if mode == "focus":
            return "stable low-volatility contour"
        if mode == "wind_down":
            return "gentle calming trajectory"
        if mode in {"exercise", "exercise-lite"}:
            return "rising arousal trajectory"
        return "positive rising trajectory"

    def _score_tracks(self, bucket: int, mode: str, context: dict | None = None) -> tuple[pd.Series, pd.Series]:
        target = bucket_targets(bucket)
        user_profile = self._resolve_user_profile(context or {})
        pool = self.df.copy()

        valence_fit = 1.0 - np.abs(pool["valence"] - target["valence"]) / 0.7
        energy_fit = 1.0 - np.abs(pool["energy"] - target["energy"]) / 0.7
        tempo_fit = 1.0 - np.abs(pool["tempo"] - target["tempo"]) / 90.0
        score = 1.8 * valence_fit + 2.0 * energy_fit + 1.4 * tempo_fit

        hard_match = (pool["action_bucket"] == bucket) & (~pool["bucket_is_soft"])
        soft_match = (pool["bucket_hint"] == bucket) & (pool["bucket_is_soft"])
        score += hard_match.astype(float) * 0.90
        score += soft_match.astype(float) * 0.20

        if mode == "focus":
            score += 0.9 * pool["instrumentalness"] + 0.6 * pool["acousticness"]
            score -= 1.0 * pool["speechiness"] + 0.2 * pool["explicit"].astype(float)
        elif mode == "wind_down":
            score += 0.8 * pool["acousticness"] + 0.7 * pool["instrumentalness"]
            score -= 0.8 * pool["speechiness"] + 0.4 * pool["energy"]
        elif mode in {"exercise", "exercise-lite"}:
            score += 0.9 * pool["danceability"] + 0.3 * pool["popularity"] / 100.0
            score -= 0.3 * pool["acousticness"]
        else:
            score += 0.3 * pool["popularity"] / 100.0

        if user_profile:
            val_pref = float(user_profile.get("user_valence_pref", 0.0))
            energy_pref = float(user_profile.get("user_energy_pref", 0.0))
            pref_val_target = np.clip(0.5 + 0.25 * val_pref, 0.0, 1.0)
            pref_energy_target = np.clip(0.5 + 0.25 * energy_pref, 0.0, 1.0)
            score += 0.45 * (1.0 - np.abs(pool["valence"] - pref_val_target))
            score += 0.55 * (1.0 - np.abs(pool["energy"] - pref_energy_target))

            acoustic_pref = float(user_profile.get("preferred_acousticness", 0.5))
            instrumental_pref = float(user_profile.get("preferred_instrumentalness", 0.1))
            popularity_tolerance = float(user_profile.get("popularity_tolerance", 0.5))
            score += 0.25 * (1.0 - np.abs(pool["acousticness"] - acoustic_pref))
            score += 0.20 * (1.0 - np.abs(pool["instrumentalness"] - instrumental_pref))
            score += 0.18 * (1.0 - np.abs(pool["popularity"] / 100.0 - popularity_tolerance))
            score += pool["genre"].map(lambda text: self._genre_preference_bonus(text, user_profile))

        score += pool["genre"].map(lambda text: self._genre_bonus(text, mode))
        score += 0.18 * pool["eda_impact"]
        dynamic_bonus = self._pmemo_dynamic_bonus(pool, mode)
        score += dynamic_bonus
        score += pool["source"].map({"situnes": 0.10, "pmemo": 0.04, "spotify": 0.0}).fillna(0.0)
        return score, dynamic_bonus

    def get_tracks(
        self,
        bucket: int,
        n: int = 5,
        context: dict | str | None = None,
        exclude_ids: list[int] | None = None,
    ) -> pd.DataFrame:
        mode = self._mode_for_bucket(bucket, context=context)
        pool = self.df.copy()
        if exclude_ids:
            pool = pool[~pool.index.isin(exclude_ids)].copy()
        if pool.empty:
            return pd.DataFrame()

        score, dynamic_bonus = self._score_tracks(
            bucket,
            mode,
            context=context if isinstance(context, dict) else {"mode": mode},
        )
        pool = pool.assign(
            score=score.loc[pool.index].to_numpy(),
            dynamic_bonus=dynamic_bonus.loc[pool.index].to_numpy(),
        )
        pool = pool.sort_values(
            ["score", "popularity", "track_name", "artist"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        pool = pool.drop_duplicates(subset=["track_name", "artist"], keep="first")
        result = pool.head(n).copy()
        result["dynamic_reason"] = result.apply(lambda row: self._pmemo_dynamic_reason(row, mode), axis=1)
        return result[
            [
                "track_name",
                "artist",
                "genre",
                "valence",
                "energy",
                "tempo",
                "action_bucket",
                "bucket_hint",
                "bucket_is_soft",
                "source",
                "popularity",
                "eda_impact",
                "dynamic_bonus",
                "dynamic_reason",
                "score",
            ]
        ].reset_index(drop=True)

    def bucket_size(self, bucket: int) -> int:
        return int((self.df["action_bucket"] == bucket).sum())

    def describe(self) -> None:
        print(f"MusicLibrary - {len(self.df)} tracks")
        for bucket, label in BUCKET_LABELS.items():
            print(f"  {bucket} {label:<15}: {self.bucket_size(bucket)}")
