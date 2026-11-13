"""Tests for the v2.2 anchor-retrieval baselines."""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from src.v2.eval.baselines import (
    knn_scores,
    order_metrics,
    random_order,
    rank_with_goal_priority,
    user_bootstrap_ci,
)


def test_goal_priority_puts_same_goal_first_then_sorts_by_score() -> None:
    scores = np.array([0.9, 0.1, 0.5, 0.8], dtype=np.float32)
    goals = np.array([1, 0, 0, 1])
    order = rank_with_goal_priority(scores, goals, query_goal=0)
    # same-goal anchors (1, 2) by score desc, then the rest (0, 3) by score desc
    assert order.tolist() == [2, 1, 0, 3]


def test_goal_priority_without_goal_is_plain_descending_sort() -> None:
    scores = np.array([0.2, 0.9, 0.5], dtype=np.float32)
    order = rank_with_goal_priority(scores, np.zeros(3, dtype=int), query_goal=None)
    assert order.tolist() == [1, 2, 0]


def test_goal_priority_breaks_ties_by_index_for_determinism() -> None:
    order = rank_with_goal_priority(
        np.ones(4, dtype=np.float32), np.zeros(4, dtype=int), query_goal=0
    )
    assert order.tolist() == [0, 1, 2, 3]


def test_knn_scores_prefer_closer_anchors() -> None:
    anchors = np.array([[0.0, 0.0], [3.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    scores = knn_scores(np.array([0.0, 0.0], dtype=np.float32), anchors)
    assert np.argsort(-scores).tolist() == [0, 2, 1]


def test_random_order_restricted_to_goal_ranks_goal_block_first() -> None:
    rng = np.random.default_rng(0)
    goals = np.array([0, 1, 0, 1, 0])
    order = random_order(rng, goals, query_goal=1)
    assert set(order[:2].tolist()) == {1, 3}
    assert sorted(order.tolist()) == [0, 1, 2, 3, 4]


def test_order_metrics_on_known_ranking() -> None:
    order = np.arange(100)
    positives = {4, 30}
    gains = {4: 1.0, 30: 0.35}
    m = order_metrics(order, positives, gains)
    assert m["hit_at_10"] == 1.0
    assert m["recall_at_20"] == 1.0
    assert m["weighted_recall_at_20"] == pytest.approx(1.0 / 1.35)
    assert m["best_rank"] == 5


def test_random_hit_rate_matches_hypergeometric() -> None:
    """Monte Carlo random ranking must agree with the closed form 1 - C(n-m,k)/C(n,k)."""
    n, k = 815, 20
    positives = {3, 100, 400, 500, 800}
    gains = {idx: 1.0 for idx in positives}
    rng = np.random.default_rng(42)
    goals = np.zeros(n, dtype=int)
    hits = [
        order_metrics(random_order(rng, goals, None), positives, gains)["recall_at_20"]
        for _ in range(4000)
    ]
    expected = 1.0 - comb(n - len(positives), k) / comb(n, k)
    assert np.mean(hits) == pytest.approx(expected, abs=0.015)


def test_user_bootstrap_ci_brackets_the_mean_and_is_degenerate_for_constant_data() -> (
    None
):
    rng = np.random.default_rng(0)
    values = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    users = np.array([1, 1, 2, 2, 3, 3])
    lo, hi = user_bootstrap_ci(values, users, rng, n_boot=500)
    assert lo <= values.mean() <= hi
    lo_c, hi_c = user_bootstrap_ci(np.ones(6), users, rng, n_boot=200)
    assert lo_c == hi_c == 1.0
