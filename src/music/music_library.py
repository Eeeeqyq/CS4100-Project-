"""
src/music/music_library.py
Loads all cleaned music sources and maps action_bucket → tracks.

Auto-detects which sources are available in data/processed/.
Works with just SiTunes alone; PMEmo and Spotify add more variety.
"""

import pandas as pd
import numpy as np
from pathlib import Path


BUCKET_LABELS = {
    0: "dark-slow",      # low valence, low energy, slow
    1: "dark-fast",      # low valence, low energy, fast
    2: "intense-slow",   # low valence, high energy, slow
    3: "aggressive",     # low valence, high energy, fast
    4: "chill-study",    # med valence, low energy, slow
    5: "indie",          # med valence, low energy, fast
    6: "soulful",        # med valence, high energy, slow
    7: "energetic",      # med valence, high energy, fast
}


class MusicLibrary:

    def __init__(self, df: pd.DataFrame):
        self.df      = df.reset_index(drop=True)
        self._index  = {b: df[df["action_bucket"] == b].index.tolist()
                        for b in range(8)}

    @classmethod
    def build(cls, processed_dir="data/processed"):
        """
        Load all available music CSVs from data/processed/.
        Each source is optional except SiTunes.
        """
        p      = Path(processed_dir)
        frames = []

        # SiTunes — required
        path = p / "music_situnes_clean.csv"
        if path.exists():
            df = pd.read_csv(path)
            frames.append(pd.DataFrame({
                "track_name":    df["music"],
                "artist":        df["singer"],
                "valence":       df["valence"],
                "energy":        df["energy"],
                "tempo":         df["tempo"],
                "action_bucket": df["action_bucket"].astype(int),
                "source":        "situnes",
                "popularity":    df.get("popularity",
                                  pd.Series([50.0]*len(df))).fillna(50),
            }))
            print(f"  SiTunes:  {len(frames[-1]):>6} tracks")
        else:
            raise FileNotFoundError(
                "music_situnes_clean.csv not found. Run cleaning first.")

        # PMEmo — optional
        path = p / "music_pmemo_clean.csv"
        if path.exists():
            df = pd.read_csv(path)
            frames.append(pd.DataFrame({
                "track_name":    df.get("title",
                                  pd.Series(["Unknown"]*len(df))),
                "artist":        df.get("artist",
                                  pd.Series(["Unknown"]*len(df))),
                "valence":       df["valence_01"],
                "energy":        df.get("energy",
                                  pd.Series([0.5]*len(df))),
                "tempo":         df.get("tempo",
                                  pd.Series([120.0]*len(df))),
                "action_bucket": df["action_bucket"].astype(int),
                "source":        "pmemo",
                "popularity":    pd.Series([50.0]*len(df)),
            }))
            print(f"  PMEmo:    {len(frames[-1]):>6} tracks")

        # Spotify — optional
        path = p / "music_spotify_clean.csv"
        if path.exists():
            df = pd.read_csv(path)
            frames.append(pd.DataFrame({
                "track_name":    df["track_name"],
                "artist":        df["artists"],
                "valence":       df["valence"],
                "energy":        df["energy"],
                "tempo":         df["tempo"],
                "action_bucket": df["action_bucket"].astype(int),
                "source":        "spotify",
                "popularity":    df["popularity"].fillna(50),
            }))
            print(f"  Spotify:  {len(frames[-1]):>6} tracks")

        combined = pd.concat(frames, ignore_index=True)
        print(f"  Total:    {len(combined):>6} tracks")
        return cls(combined)

    # ── Retrieval ─────────────────────────────────────────────────────────

    def get_tracks(self, bucket: int, n=5,
                   exclude_ids=None, prefer_popular=True):
        """
        Return up to n tracks from the given bucket.

        Parameters
        ----------
        bucket        : int 0-7
        n             : how many tracks to return
        exclude_ids   : list of row indices to skip (recently played)
        prefer_popular: sort by popularity before sampling
        """
        idx  = self._index.get(bucket, [])
        pool = self.df.loc[idx].copy()

        if exclude_ids:
            pool = pool[~pool.index.isin(exclude_ids)]
        if pool.empty:
            return pd.DataFrame()

        if prefer_popular:
            pool   = pool.sort_values("popularity", ascending=False)
            pool   = pool.head(min(50, len(pool)))

        result = pool.sample(min(n, len(pool)))
        return result[["track_name", "artist", "valence", "energy",
                        "tempo", "action_bucket", "source", "popularity"]]

    def bucket_size(self, bucket):
        return len(self._index.get(bucket, []))

    def describe(self):
        print(f"MusicLibrary — {len(self.df)} tracks")
        for b, label in BUCKET_LABELS.items():
            print(f"  {b} {label:<15}: {self.bucket_size(b)}")