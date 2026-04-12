"""
Goal routing and target-state objective helpers for the V2 rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from .schema import Goal


@dataclass(frozen=True)
class GoalSpec:
    goal: Goal
    name: str
    tau_valence: float
    tau_arousal: float
    w_valence: float
    w_arousal: float


GOAL_SPECS: Final[dict[Goal, GoalSpec]] = {
    Goal.FOCUS: GoalSpec(Goal.FOCUS, "focus", 0.35, 0.05, 0.35, 0.65),
    Goal.WIND_DOWN: GoalSpec(Goal.WIND_DOWN, "wind_down", 0.30, -0.45, 0.40, 0.60),
    Goal.UPLIFT: GoalSpec(Goal.UPLIFT, "uplift", 0.70, 0.20, 0.60, 0.40),
    Goal.MOVEMENT: GoalSpec(Goal.MOVEMENT, "movement", 0.60, 0.65, 0.45, 0.55),
}


@dataclass(frozen=True)
class GoalContext:
    pre_valence: float
    pre_arousal: float
    time_bucket: int
    weather_bucket: int
    speed_norm: float
    weekend_flag: float
    step_nonzero_frac: float
    step_mean_norm: float
    activity_majority: int
    hr_mean_rel: float
    checkin_mask: float


def movement_evidence(ctx: GoalContext) -> float:
    score = (
        0.45 * float(ctx.step_nonzero_frac)
        + 0.25 * float(ctx.step_mean_norm)
        + 0.20 * float(ctx.speed_norm)
        + 0.10 * float(ctx.activity_majority in {2, 4})
    )
    return float(np.clip(score, 0.0, 1.0))


def recovery_evidence(ctx: GoalContext) -> float:
    score = (
        0.40 * float(ctx.time_bucket == 2)
        + 0.25 * float(ctx.activity_majority in {0, 3})
        + 0.20 * float(ctx.pre_arousal > 0.15)
        + 0.15 * float(movement_evidence(ctx) < 0.20)
    )
    return float(np.clip(score, 0.0, 1.0))


def goal_router_v1(ctx: GoalContext, explicit_goal: int | None = None) -> Goal:
    if explicit_goal is not None and explicit_goal >= 0:
        return Goal(int(explicit_goal))

    move = movement_evidence(ctx)
    if move >= 0.55:
        return Goal.MOVEMENT
    if ctx.time_bucket == 2 and ctx.pre_arousal > 0.15 and move < 0.20:
        return Goal.WIND_DOWN
    if ctx.pre_valence < -0.10:
        return Goal.UPLIFT
    if ctx.time_bucket in {0, 1} and move < 0.20:
        return Goal.FOCUS
    return Goal.UPLIFT


def adjusted_target(goal: Goal, ctx: GoalContext) -> tuple[float, float]:
    spec = GOAL_SPECS[goal]
    tau_v = spec.tau_valence
    tau_a = spec.tau_arousal

    if goal == Goal.MOVEMENT and movement_evidence(ctx) >= 0.35:
        tau_a += 0.10
    if goal == Goal.FOCUS and ctx.time_bucket == 2:
        tau_a -= 0.05
    if goal == Goal.WIND_DOWN and ctx.pre_arousal < -0.40:
        tau_a = max(tau_a, -0.35)
    if goal == Goal.UPLIFT and movement_evidence(ctx) < 0.15 and ctx.hr_mean_rel > 0.0:
        tau_a -= 0.10

    return float(np.clip(tau_v, -1.0, 1.0)), float(np.clip(tau_a, -1.0, 1.0))


def goal_weights(goal: Goal) -> tuple[float, float]:
    spec = GOAL_SPECS[goal]
    return float(spec.w_valence), float(spec.w_arousal)


def goal_distance(
    valence: float,
    arousal: float,
    tau_v: float,
    tau_a: float,
    w_v: float,
    w_a: float,
) -> float:
    return float(w_v * (valence - tau_v) ** 2 + w_a * (arousal - tau_a) ** 2)


def benefit_target(
    pre_v: float,
    pre_a: float,
    post_v: float,
    post_a: float,
    goal: Goal,
    ctx: GoalContext,
) -> float:
    tau_v, tau_a = adjusted_target(goal, ctx)
    w_v, w_a = goal_weights(goal)
    pre_d = goal_distance(pre_v, pre_a, tau_v, tau_a, w_v, w_a)
    post_d = goal_distance(post_v, post_a, tau_v, tau_a, w_v, w_a)
    return float(np.clip(pre_d - post_d, -2.0, 2.0))


def preference_target(preference: float | None) -> float:
    if preference is None or np.isnan(preference):
        return 0.0
    return float(np.clip((float(preference) - 50.0) / 50.0, -1.0, 1.0))


def rating_target(rating: float | None) -> float:
    if rating is None or np.isnan(rating):
        return 0.0
    return float(np.clip((float(rating) - 3.0) / 2.0, -1.0, 1.0))


def acceptance_observation(
    preference: float | None,
    rating: float | None,
) -> tuple[float, str, float, float]:
    if preference is not None and not np.isnan(preference):
        pref = preference_target(preference)
        return pref, "preference", 1.0, 0.0
    if rating is not None and not np.isnan(rating):
        rate = rating_target(rating)
        return rate, "rating", 0.0, 1.0
    return 0.0, "missing", 0.0, 0.0


def acceptance_target(preference: float | None, rating: float | None) -> float:
    value, _, _, _ = acceptance_observation(preference, rating)
    return float(value)
