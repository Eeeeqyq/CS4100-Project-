"""
Deterministic, context-aware music retrieval on top of the cleaned catalogs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.common import BUCKET_LABELS, bucket_targets, track_quality_features


MODE_GENRE_BOOSTS = {
    "focus": ["ambient", "classical", "study", "piano", "acoustic", "chill", "indie"],
    "wind_down": ["ambient", "acoustic", "singer-songwriter", "folk", "classical", "lofi"],
    "exercise": ["dance", "edm", "electro", "house", "hip-hop", "rock", "pop", "workout"],
    "uplift": ["soul", "funk", "disco", "indie", "pop"],
}


class MusicLibrary:
    def __init__(self, df: pd.DataFrame):
        self.df = track_quality_features(df.reset_index(drop=True))
        self.df["bucket_label"] = self.df["action_bucket"].map(BUCKET_LABELS)

    @classmethod
    def build(cls, processed_dir="data/processed"):
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
                        "source": "spotify",
                        "explicit": spotify.get("explicit", False),
                    }
                )
            )

        combined = pd.concat(frames, ignore_index=True)
        return cls(combined)

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

    def _score_tracks(self, bucket: int, mode: str) -> pd.Series:
        target = bucket_targets(bucket)
        bucket_df = self.df[self.df["action_bucket"] == bucket].copy()

        valence_fit = 1.0 - np.abs(bucket_df["valence"] - target["valence"]) / 0.7
        energy_fit = 1.0 - np.abs(bucket_df["energy"] - target["energy"]) / 0.7
        tempo_fit = 1.0 - np.abs(bucket_df["tempo"] - target["tempo"]) / 90.0
        score = 2.0 * valence_fit + 2.2 * energy_fit + 1.6 * tempo_fit

        if mode == "focus":
            score += 0.8 * bucket_df["instrumentalness"] + 0.5 * bucket_df["acousticness"]
            score -= 0.9 * bucket_df["speechiness"] + 0.2 * bucket_df["explicit"].astype(float)
        elif mode == "wind_down":
            score += 0.7 * bucket_df["acousticness"] + 0.6 * bucket_df["instrumentalness"]
            score -= 0.8 * bucket_df["speechiness"] + 0.4 * bucket_df["energy"]
        elif mode == "exercise":
            score += 0.9 * bucket_df["danceability"] + 0.3 * bucket_df["popularity"] / 100.0
            score -= 0.3 * bucket_df["acousticness"]
        else:
            score += 0.3 * bucket_df["popularity"] / 100.0

        score += bucket_df["genre"].map(lambda text: self._genre_bonus(text, mode))
        score += bucket_df["source"].map({"situnes": 0.12, "pmemo": 0.08, "spotify": 0.0}).fillna(0.0)
        return score

    def get_tracks(
        self,
        bucket: int,
        n: int = 5,
        context: dict | str | None = None,
        exclude_ids: list[int] | None = None,
    ) -> pd.DataFrame:
        mode = self._mode_for_bucket(bucket, context=context)
        pool = self.df[self.df["action_bucket"] == int(bucket)].copy()
        if exclude_ids:
            pool = pool[~pool.index.isin(exclude_ids)].copy()
        if pool.empty:
            return pd.DataFrame()

        pool["score"] = self._score_tracks(bucket, mode)
        pool = pool.sort_values(
            ["score", "popularity", "track_name", "artist"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        return pool.head(n)[
            [
                "track_name",
                "artist",
                "genre",
                "valence",
                "energy",
                "tempo",
                "action_bucket",
                "source",
                "popularity",
                "score",
            ]
        ].reset_index(drop=True)

    def bucket_size(self, bucket: int) -> int:
        return int((self.df["action_bucket"] == bucket).sum())

    def describe(self) -> None:
        print(f"MusicLibrary - {len(self.df)} tracks")
        for bucket, label in BUCKET_LABELS.items():
            print(f"  {bucket} {label:<15}: {self.bucket_size(bucket)}")
