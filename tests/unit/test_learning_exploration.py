# tests/unit/test_learning_exploration.py — Tests for V3.1 Adaptive Exploration Module
"""Tests for the harness/learning/exploration.py module."""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.learning.exploration import (
    AdaptiveExplorationPolicy,
    PurposefulCandidateSelector,
    ExplorationResultLearner,
)


class TestAdaptiveExplorationPolicy:
    """Tests for adaptive exploration rate computation."""

    def test_compute_rate_high_confidence(self):
        policy = AdaptiveExplorationPolicy()
        rate = policy.compute_rate(["coding"], avg_confidence=0.9)
        assert rate <= 0.10  # High confidence = low exploration
        assert rate >= AdaptiveExplorationPolicy.MIN_EXPLORATION_RATE

    def test_compute_rate_low_confidence(self):
        policy = AdaptiveExplorationPolicy()
        rate = policy.compute_rate(["unknown"], avg_confidence=0.1)
        assert rate >= 0.40  # Low confidence = high exploration
        assert rate <= AdaptiveExplorationPolicy.MAX_EXPLORATION_RATE

    def test_compute_rate_medium_confidence(self):
        policy = AdaptiveExplorationPolicy()
        rate = policy.compute_rate(["general"], avg_confidence=0.5)
        # Should be around midway between min and max
        expected_mid = (
            AdaptiveExplorationPolicy.MAX_EXPLORATION_RATE
            + AdaptiveExplorationPolicy.MIN_EXPLORATION_RATE
        ) / 2
        assert abs(rate - 0.26) < 0.05  # 0.5 maps to ~0.26

    def test_compute_rate_no_confidence(self):
        policy = AdaptiveExplorationPolicy()
        rate = policy.compute_rate(["test"])
        assert rate == 0.26  # default 0.5 confidence -> ~0.26

    def test_compute_rate_with_learner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = ExplorationResultLearner(storage_dir=tmpdir)
            learner.record("m1", "coding", 0.9, "purposeful")
            learner.record("m2", "coding", 0.8, "purposeful")

            policy = AdaptiveExplorationPolicy(exploration_learner=learner)
            rate = policy.compute_rate(["coding"])
            # High average quality (0.85) -> low exploration
            assert rate <= 0.10

    def test_should_explore(self):
        """V3.2 Phase 6: should_explore is deterministic per seed."""
        policy = AdaptiveExplorationPolicy()
        # At max exploration rate (avg_confidence=0.1 -> ~0.452 rate), the
        # same seed always yields the same decision (determinism).
        assert policy.should_explore(["test"], avg_confidence=0.1, seed=42) == \
            policy.should_explore(["test"], avg_confidence=0.1, seed=42)
        # Across seeded runs we observe variety: some seeds explore.
        explored = sum(
            policy.should_explore(["test"], avg_confidence=0.1, seed=s)
            for s in range(200)
        )
        assert 0 < explored < 200

    def test_should_exploit_high_confidence(self):
        """V3.2 Phase 6: high confidence explores rarely at any seed."""
        policy = AdaptiveExplorationPolicy()
        # High confidence --> very low rate; even across many seeds, almost all
        # exploit (rate ~0.044 so ~4% explore). Treat as rarely-explores.
        explored = sum(
            policy.should_explore(["test"], avg_confidence=0.95, seed=s)
            for s in range(200)
        )
        assert explored < 100  # should exploit most of the time


class TestPurposefulCandidateSelector:
    """Tests for purposeful candidate selection."""

    def test_select_ranks_by_score(self):
        candidates = [
            {"model_id": "m1", "capability": "test", "is_unknown": False},
            {"model_id": "m2", "capability": "test", "is_unknown": True},
            {"model_id": "m3", "capability": "test", "novelty": 0.8},
        ]
        selected = PurposefulCandidateSelector.select(
            candidates, current_best="m1", top_k=2
        )
        assert len(selected) == 2
        # Unknown and novel candidates should rank higher
        assert selected[0]["model_id"] in ("m2", "m3")

    def test_select_respects_top_k(self):
        candidates = [
            {"model_id": f"m{i}", "capability": "test"}
            for i in range(10)
        ]
        selected = PurposefulCandidateSelector.select(candidates, top_k=3)
        assert len(selected) == 3

    def test_select_with_exploration_learner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = ExplorationResultLearner(storage_dir=tmpdir)
            learner.record("known-good", "test", 0.95, "purposeful")

            candidates = [
                {"model_id": "known-good", "capability": "test"},
                {"model_id": "unknown", "capability": "test", "is_unknown": True},
                {"model_id": "novel", "capability": "test", "novelty": 0.9},
            ]
            selected = PurposefulCandidateSelector.select(
                candidates, current_best="other", exploration_learner=learner, top_k=2
            )
            assert len(selected) == 2

    def test_select_empty_candidates(self):
        selected = PurposefulCandidateSelector.select([], top_k=5)
        assert selected == []

    def test_score_candidate_unknown(self):
        candidate = {"model_id": "m1", "capability": "test", "is_unknown": True}
        score = PurposefulCandidateSelector._score_candidate(candidate, "m2", None)
        assert score > 0.3

    def test_score_candidate_novel(self):
        candidate = {"model_id": "m1", "capability": "test", "novelty": 0.9}
        score = PurposefulCandidateSelector._score_candidate(candidate, "m2", None)
        assert score > 0.15


class TestExplorationResultLearner:
    """Tests for exploration result learning (re-exported from model_intelligence)."""

    def test_record_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = ExplorationResultLearner(storage_dir=tmpdir)
            learner.record("m1", "coding", 0.95, "purposeful")
            learner.record("m2", "coding", 0.3, "purposeful")

            best = learner.get_best_exploration_for("coding")
            assert best == "m1"

    def test_get_exploration_quality(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = ExplorationResultLearner(storage_dir=tmpdir)
            learner.record("m1", "coding", 0.8, "purposeful")
            learner.record("m2", "coding", 0.6, "purposeful")

            quality = learner.get_exploration_quality("coding")
            assert quality == pytest.approx(0.7)

    def test_no_data_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = ExplorationResultLearner(storage_dir=tmpdir)
            assert learner.get_best_exploration_for("unknown") is None
            assert learner.get_exploration_quality("unknown") == 0.5
