"""
Build the V2 rebuild data contracts and tensors.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v2.data.build_public_music import build_public_music_v2
from src.v2.data.build_situnes import build_situnes_v2, materialize_stage1_histories
from src.v2.data.anchors import build_anchor_supervision
from src.v2.data.schema import (
    validate_anchor_table,
    validate_decision_table,
    validate_song_catalog,
    validate_stage1_history_table,
    validate_tensor_shapes,
)


OUT_DIR = PROJECT_ROOT / "data" / "processed" / "rebuild"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    situnes_outputs = build_situnes_v2(OUT_DIR)
    music_outputs = build_public_music_v2(OUT_DIR)

    decision_df = pd.read_parquet(OUT_DIR / "decision_table.parquet")
    history_df = pd.read_parquet(OUT_DIR / "stage1_history_table.parquet")
    song_catalog = pd.read_parquet(OUT_DIR / "song_catalog.parquet")

    song_id_to_index = {song_id: idx for idx, song_id in enumerate(song_catalog["song_id"].tolist())}
    history_stats = materialize_stage1_histories(history_df, song_id_to_index, OUT_DIR)

    decision_df["factual_song_idx"] = decision_df["song_id"].map(song_id_to_index).astype(int)
    decision_df.to_parquet(OUT_DIR / "decision_table.parquet", index=False)
    np.save(OUT_DIR / "factual_song_idx.npy", decision_df["factual_song_idx"].to_numpy(dtype=np.int32))

    anchor_stats = build_anchor_supervision(
        decision_df=decision_df,
        song_static=np.load(OUT_DIR / "song_static.npy").astype(np.float32),
        out_dir=OUT_DIR,
    )

    validate_decision_table(decision_df)
    validate_stage1_history_table(history_df)
    validate_song_catalog(song_catalog)
    validate_anchor_table(pd.read_parquet(OUT_DIR / "anchor_table.parquet"))
    validate_tensor_shapes(
        np.load(OUT_DIR / "wrist_windows.npy"),
        np.load(OUT_DIR / "env_features.npy"),
        np.load(OUT_DIR / "self_report.npy"),
        np.load(OUT_DIR / "song_static.npy"),
        np.load(OUT_DIR / "song_dynamic.npy"),
        np.load(OUT_DIR / "song_dynamic_mask.npy"),
    )

    summary = {
        "decision_rows": situnes_outputs["decision_rows"],
        "users": situnes_outputs["users"],
        "songs": music_outputs["songs"],
        "split_counts": situnes_outputs["split_counts"],
        "stage1_histories": history_stats,
        "anchor_supervision": anchor_stats,
    }
    (OUT_DIR / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
