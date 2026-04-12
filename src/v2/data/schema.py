"""
Canonical tensor and table schemas for the V2 rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final

import numpy as np
import pandas as pd
import torch


class Goal(IntEnum):
    FOCUS = 0
    WIND_DOWN = 1
    UPLIFT = 2
    MOVEMENT = 3


GOAL_NAMES: Final[dict[int, str]] = {
    Goal.FOCUS: "focus",
    Goal.WIND_DOWN: "wind_down",
    Goal.UPLIFT: "uplift",
    Goal.MOVEMENT: "movement",
}
GOAL_SOURCE_VALUES: Final[tuple[str, ...]] = ("explicit", "router_fallback")

WRIST_SEQ_LEN: Final[int] = 30
WRIST_DIM: Final[int] = 9
ENV_DIM: Final[int] = 9
SELF_DIM: Final[int] = 3
GOAL_DIM: Final[int] = 4
TAU_DIM: Final[int] = 2
SONG_STATIC_DIM: Final[int] = 20
SONG_DYN_LEN: Final[int] = 20
SONG_DYN_DIM: Final[int] = 2
SONG_EMB_DIM: Final[int] = 128
CTX_EMB_DIM: Final[int] = 128
USER_EMB_DIM: Final[int] = 64
STAGE1_MAX_LEN: Final[int] = 20

WRIST_FEATURES: Final[list[str]] = [
    "hr_norm",
    "hr_delta",
    "intensity_norm",
    "steps_norm",
    "activity_still",
    "activity_transition",
    "activity_walk",
    "activity_lying",
    "activity_run",
]

ENV_FEATURES: Final[list[str]] = [
    "time_sin",
    "time_cos",
    "weather_clear",
    "weather_cloudy",
    "weather_rainy_other",
    "temp_z",
    "humidity_z",
    "speed_norm",
    "weekend_flag",
]

SELF_FEATURES: Final[list[str]] = [
    "pre_valence",
    "pre_arousal",
    "checkin_mask",
]

SONG_STATIC_FEATURES: Final[list[str]] = [
    "valence_static",
    "arousal_static",
    "energy",
    "tempo_norm",
    "danceability",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "loudness_norm",
    "popularity_norm",
    "explicit_flag",
    "source_situnes",
    "source_spotify",
    "source_pmemo",
    "genre_emb_1",
    "genre_emb_2",
    "genre_emb_3",
    "genre_emb_4",
    "eda_impact_norm",
]

SONG_DYN_SUMMARY_FEATURES: Final[list[str]] = [
    "dyn_valence_delta",
    "dyn_arousal_delta",
    "dyn_valence_vol",
    "dyn_arousal_vol",
    "dyn_arousal_peak",
    "eda_impact_norm",
]

DECISION_COLUMNS: Final[list[str]] = [
    "decision_id",
    "user_id",
    "song_id",
    "split",
    "goal_idx",
    "goal_source",
    "explicit_goal_idx",
    "pre_valence",
    "pre_arousal",
    "post_valence",
    "post_arousal",
    "benefit_target",
    "acceptance_target",
    "accept_pref_target",
    "accept_rating_target",
    "accept_pref_mask",
    "accept_rating_mask",
    "time_bucket",
    "weather_bucket",
    "temp_z",
    "humidity_z",
    "speed_norm",
    "weekend_flag",
    "checkin_mask",
    "timestamp",
    "dataset_stage",
    "tau_valence",
    "tau_arousal",
]

ANCHOR_COLUMNS: Final[list[str]] = [
    "anchor_idx",
    "decision_id",
    "user_id",
    "song_id",
    "factual_song_idx",
    "split",
    "available_for_index",
    "goal_idx",
    "goal_source",
    "explicit_goal_idx",
    "pre_valence",
    "pre_arousal",
    "post_valence",
    "post_arousal",
    "benefit_target",
    "accept_pref_target",
    "accept_rating_target",
    "accept_pref_mask",
    "accept_rating_mask",
    "acceptance_obs",
    "acceptance_source",
    "anchor_success_obs",
    "time_bucket",
    "weather_bucket",
    "temp_z",
    "humidity_z",
    "speed_norm",
    "weekend_flag",
    "checkin_mask",
    "tau_valence",
    "tau_arousal",
    "timestamp",
    "dataset_stage",
    "local_support_count",
    "support_count",
    "support_norm",
    "positive_tier1_count",
    "positive_tier2_count",
    "positive_tier3_count",
    "factual_positive_available",
]

STAGE1_HISTORY_COLUMNS: Final[list[str]] = [
    "user_id",
    "song_id",
    "rating",
    "timestamp",
    "emo_valence",
    "emo_arousal",
    "hist_pos",
]

SONG_CATALOG_COLUMNS: Final[list[str]] = [
    "song_id",
    "source",
    "title",
    "artist",
    "genre",
    "valence_static",
    "arousal_static",
    "energy",
    "tempo_norm",
    "danceability",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "loudness_norm",
    "popularity_norm",
    "explicit_flag",
    "eda_impact_norm",
    "dyn_valence_delta",
    "dyn_arousal_delta",
    "dyn_valence_vol",
    "dyn_arousal_vol",
    "dyn_arousal_peak",
    "song_quality",
    "has_dynamic",
    "trainable",
]


@dataclass(frozen=True)
class DecisionBatch:
    x_wrist: torch.Tensor
    x_env: torch.Tensor
    x_self: torch.Tensor
    goal_idx: torch.Tensor
    goal_onehot: torch.Tensor
    tau: torch.Tensor
    user_hist_song_idx: torch.Tensor
    user_hist_rating: torch.Tensor
    user_hist_mask: torch.Tensor
    factual_song_idx: torch.Tensor
    benefit_target: torch.Tensor
    acceptance_target: torch.Tensor
    accept_pref_target: torch.Tensor | None = None
    accept_rating_target: torch.Tensor | None = None
    accept_pref_mask: torch.Tensor | None = None
    accept_rating_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class SongCatalogTensors:
    x_song_static: torch.Tensor
    x_song_dyn: torch.Tensor
    x_song_dyn_mask: torch.Tensor
    song_ids: list[str]


def validate_feature_order() -> None:
    assert len(WRIST_FEATURES) == WRIST_DIM
    assert len(ENV_FEATURES) == ENV_DIM
    assert len(SELF_FEATURES) == SELF_DIM
    assert len(SONG_STATIC_FEATURES) == SONG_STATIC_DIM
    assert len(SONG_DYN_SUMMARY_FEATURES) == 6


def validate_decision_table(df: pd.DataFrame) -> None:
    missing = [column for column in DECISION_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Decision table missing columns: {missing}")
    if df["decision_id"].duplicated().any():
        raise ValueError("decision_id must be unique")
    if not df["goal_idx"].isin([0, 1, 2, 3]).all():
        raise ValueError("goal_idx must be in {0,1,2,3}")
    if not df["goal_source"].isin(GOAL_SOURCE_VALUES).all():
        raise ValueError(f"goal_source must be one of {GOAL_SOURCE_VALUES}")
    if not df["split"].isin(["train", "val", "test"]).all():
        raise ValueError("split must be one of train/val/test")
    numeric_cols = [
        "pre_valence",
        "pre_arousal",
        "post_valence",
        "post_arousal",
        "benefit_target",
        "acceptance_target",
        "accept_pref_target",
        "accept_rating_target",
        "accept_pref_mask",
        "accept_rating_mask",
        "temp_z",
        "humidity_z",
        "speed_norm",
        "weekend_flag",
        "checkin_mask",
        "tau_valence",
        "tau_arousal",
    ]
    if df[numeric_cols].isna().any().any():
        raise ValueError("Decision table contains NaNs in model-consumed numeric fields")


def validate_anchor_table(df: pd.DataFrame) -> None:
    missing = [column for column in ANCHOR_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Anchor table missing columns: {missing}")
    if df["anchor_idx"].duplicated().any():
        raise ValueError("anchor_idx must be unique")
    if df["decision_id"].duplicated().any():
        raise ValueError("Anchor table should have exactly one row per decision")
    if not df["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Anchor split must be one of train/val/test")
    if not df["goal_idx"].isin([0, 1, 2, 3]).all():
        raise ValueError("Anchor goal_idx must be in {0,1,2,3}")
    if not df["goal_source"].isin(GOAL_SOURCE_VALUES).all():
        raise ValueError(f"Anchor goal_source must be one of {GOAL_SOURCE_VALUES}")
    if not df["available_for_index"].isin([0, 1]).all():
        raise ValueError("available_for_index must be binary")
    numeric_cols = [
        "pre_valence",
        "pre_arousal",
        "post_valence",
        "post_arousal",
        "benefit_target",
        "accept_pref_target",
        "accept_rating_target",
        "accept_pref_mask",
        "accept_rating_mask",
        "acceptance_obs",
        "anchor_success_obs",
        "temp_z",
        "humidity_z",
        "speed_norm",
        "weekend_flag",
        "checkin_mask",
        "tau_valence",
        "tau_arousal",
        "local_support_count",
        "support_count",
        "support_norm",
        "positive_tier1_count",
        "positive_tier2_count",
        "positive_tier3_count",
        "factual_positive_available",
    ]
    if df[numeric_cols].isna().any().any():
        raise ValueError("Anchor table contains NaNs in model-consumed numeric fields")


def validate_stage1_history_table(df: pd.DataFrame) -> None:
    missing = [column for column in STAGE1_HISTORY_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Stage1 history table missing columns: {missing}")
    if not df["hist_pos"].between(0, STAGE1_MAX_LEN - 1).all():
        raise ValueError("hist_pos must be in [0, 19]")


def validate_song_catalog(df: pd.DataFrame) -> None:
    missing = [column for column in SONG_CATALOG_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Song catalog missing columns: {missing}")
    if df["song_id"].duplicated().any():
        raise ValueError("song_id must be unique in song catalog")
    numeric_cols = [col for col in SONG_CATALOG_COLUMNS if col not in {"song_id", "source", "title", "artist", "genre", "has_dynamic", "trainable"}]
    if df[numeric_cols].isna().any().any():
        raise ValueError("Song catalog contains NaNs in model-consumed numeric fields")


def validate_tensor_shapes(
    wrist: np.ndarray,
    env: np.ndarray,
    self_arr: np.ndarray,
    song_static: np.ndarray,
    song_dyn: np.ndarray,
    song_dyn_mask: np.ndarray,
) -> None:
    if wrist.ndim != 3 or wrist.shape[1:] != (WRIST_SEQ_LEN, WRIST_DIM):
        raise ValueError(f"Expected wrist tensor shape [N,{WRIST_SEQ_LEN},{WRIST_DIM}], got {wrist.shape}")
    if env.ndim != 2 or env.shape[1] != ENV_DIM:
        raise ValueError(f"Expected env tensor width {ENV_DIM}, got {env.shape}")
    if self_arr.ndim != 2 or self_arr.shape[1] != SELF_DIM:
        raise ValueError(f"Expected self tensor width {SELF_DIM}, got {self_arr.shape}")
    if song_static.ndim != 2 or song_static.shape[1] != SONG_STATIC_DIM:
        raise ValueError(f"Expected song static width {SONG_STATIC_DIM}, got {song_static.shape}")
    if song_dyn.ndim != 3 or song_dyn.shape[1:] != (SONG_DYN_LEN, SONG_DYN_DIM):
        raise ValueError(f"Expected song dynamic shape [N,{SONG_DYN_LEN},{SONG_DYN_DIM}], got {song_dyn.shape}")
    if song_dyn_mask.ndim != 3 or song_dyn_mask.shape[1:] != (SONG_DYN_LEN, 1):
        raise ValueError(f"Expected song dynamic mask shape [N,{SONG_DYN_LEN},1], got {song_dyn_mask.shape}")
