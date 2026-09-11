# tests/unit/test_model_intelligence_v31.py — Tests for V3.1 Adaptive Exploration
"""Tests for V3.1 Adaptive Exploration enhancements."""

import sys
import os
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.planner.model_intelligence import (
    AdaptiveModelPolicyV3, ExplorationResultLearner,
    FreeModelInvariantEnforcer,
)


def test_exploration_result_learner_record_and_retrieve():
    with tempfile.TemporaryDirectory() as tmpdir:
        learner = ExplorationResultLearner(storage_dir=tmpdir)
        learner.record("model_a", "coding", 0.9, "purposeful")
        learner.record("model_b", "coding", 0.3, "purposeful")

        best = learner.get_best_exploration_for("coding")
        assert best == "model_a"

        quality = learner.get_exploration_quality("coding")
        assert quality == pytest.approx(0.6, abs=0.01)  # (0.9 + 0.3) / 2


def test_exploration_no_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        learner = ExplorationResultLearner(storage_dir=tmpdir)
        best = learner.get_best_exploration_for("unknown")
        assert best is None

        quality = learner.get_exploration_quality("unknown")
        assert quality == 0.5  # neutral default


def test_exploration_result_learner_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        learner = ExplorationResultLearner(storage_dir=tmpdir)
        learner.record("m1", "test", 0.8, "random")

        # New instance with same dir should load data
        learner2 = ExplorationResultLearner(storage_dir=tmpdir)
        best = learner2.get_best_exploration_for("test")
        assert best == "m1"


class TestAdaptiveExplorationRate:
    def test_high_confidence_low_rate(self):
        # Mock perf registry with high confidence data
        class MockPerfRegistry:
            def rank_models(self, cap):
                class MockStats:
                    @staticmethod
                    def confidence_score():
                        return 0.9
                    min_samples_met = True
                return [MockStats()]

        policy = AdaptiveModelPolicyV3(None, MockPerfRegistry(), None)
        rate = policy._compute_exploration_rate(["coding"])
        assert rate < 0.10  # should be below default

    def test_low_confidence_high_rate(self):
        class MockPerfRegistry:
            def rank_models(self, cap):
                return []  # no data

        policy = AdaptiveModelPolicyV3(None, MockPerfRegistry(), None)
        rate = policy._compute_exploration_rate(["unknown"])
        assert rate >= 0.15  # should be closer to 20%

    def test_no_caps_default_rate(self):
        policy = AdaptiveModelPolicyV3(None, None, None)
        rate = policy._compute_exploration_rate([])
        assert rate == 0.10


class TestPurposefulExploration:
    def test_explore_selects_among_free_models(self):
        class MockCatalog:
            def list_free(self):
                class Model:
                    def __init__(self, mid):
                        self.model_id = mid
                return [Model("m1"), Model("m2"), Model("m3")]

        class MockPerfRegistry:
            def get_sample_count(self, mid, cap):
                return 0
            def rank_models(self, cap):
                return []

        policy = AdaptiveModelPolicyV3(MockCatalog(), MockPerfRegistry(), None)
        selection = policy._explore(["coding"], "coder", "task1")
        assert selection is not None
        assert hasattr(selection, 'model_id')
        assert selection.is_exploration is True

    def test_explore_empty_catalog_falls_back(self):
        class MockCatalog:
            def list_free(self):
                return []

        class MockRouter:
            def select(self, caps, **kw):
                class Sel:
                    model_id = "fallback"
                    is_exploration = False
                return Sel()

        policy = AdaptiveModelPolicyV3(MockCatalog(), None, MockRouter())
        selection = policy._explore(["coding"], "coder", "t1")
        assert selection.model_id == "fallback"
