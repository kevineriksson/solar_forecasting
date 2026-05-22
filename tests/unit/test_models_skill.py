"""Unit tests for src/models/skill.py."""

from __future__ import annotations

import pytest

from src.models.skill import skill_score


def test_skill_score_zero_when_equal():
    assert skill_score(10.0, 10.0) == 0.0


def test_skill_score_positive_when_candidate_better():
    # candidate has half the RMSE of baseline -> skill = 0.5
    assert skill_score(5.0, 10.0) == pytest.approx(0.5)


def test_skill_score_negative_when_candidate_worse():
    assert skill_score(20.0, 10.0) == pytest.approx(-1.0)


def test_skill_score_zero_baseline_returns_neg_inf():
    # Degenerate guard: a baseline with zero RMSE is undefined for skill score.
    assert skill_score(1.0, 0.0) == float("-inf")
    assert skill_score(1.0, -0.5) == float("-inf")


def test_skill_score_perfect_candidate():
    assert skill_score(0.0, 10.0) == 1.0
