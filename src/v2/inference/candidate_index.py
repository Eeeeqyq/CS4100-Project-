"""
Simple cosine candidate index for the V2 rebuild.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class CandidateIndex:
    def __init__(self) -> None:
        self.song_ids: list[str] = []
        self.norm_song_emb: np.ndarray | None = None

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(x, axis=1, keepdims=True)
        denom = np.clip(denom, 1e-8, None)
        return x / denom

    def build(self, song_emb: np.ndarray, song_ids: list[str]) -> None:
        self.song_ids = list(song_ids)
        self.norm_song_emb = self._normalize(np.asarray(song_emb, dtype=np.float32))

    def topk(self, query_emb: np.ndarray, k: int = 100) -> tuple[np.ndarray, np.ndarray]:
        if self.norm_song_emb is None:
            raise RuntimeError("CandidateIndex has not been built.")
        query = self._normalize(np.asarray(query_emb, dtype=np.float32))
        sims = query @ self.norm_song_emb.T
        k = min(int(k), sims.shape[1])
        top_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        top_scores = np.take_along_axis(sims, top_idx, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)
        return top_idx.astype(np.int32), top_scores.astype(np.float32)

    def save(self, path: str | Path) -> None:
        if self.norm_song_emb is None:
            raise RuntimeError("CandidateIndex has not been built.")
        path = Path(path)
        np.savez(path, song_ids=np.asarray(self.song_ids), norm_song_emb=self.norm_song_emb)

    @classmethod
    def load(cls, path: str | Path) -> "CandidateIndex":
        obj = np.load(path, allow_pickle=True)
        out = cls()
        out.song_ids = obj["song_ids"].tolist()
        out.norm_song_emb = obj["norm_song_emb"].astype(np.float32)
        return out
