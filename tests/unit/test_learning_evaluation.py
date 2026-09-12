# tests/unit/test_learning_evaluation.py — Tests for V3.1 Learning Evaluation
"""Tests for Learning Evaluation (Phase F)."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.learning.evaluation import (
    LearningGainCalculator, GeneralizationGain, FailureAvoidanceMetric,
    DecisionInfluenceMetric, LearningEfficiency, LearningEvaluator,
    NOT_AVAILABLE, NOT_APPLICABLE,
)


class TestLearningGainCalculator:
    def test_positive_gain(self):
        baseline = {"success_rate": 0.74, "avg_score": 0.7184}
        learning = {"success_rate": 0.80, "avg_score": 0.75}
        result = LearningGainCalculator.calculate(baseline, learning)
        assert result.success_rate_improvement > 0
        assert result.score_improvement > 0
        assert result.gain_ratio > 1.0

    def test_negative_gain(self):
        baseline = {"success_rate": 0.80, "avg_score": 0.75}
        learning = {"success_rate": 0.74, "avg_score": 0.70}
        result = LearningGainCalculator.calculate(baseline, learning)
        assert result.success_rate_improvement < 0
        assert result.score_improvement < 0

    def test_no_change(self):
        baseline = {"success_rate": 0.75, "avg_score": 0.70}
        learning = {"success_rate": 0.75, "avg_score": 0.70}
        result = LearningGainCalculator.calculate(baseline, learning)
        assert result.success_rate_improvement == 0
        assert result.score_improvement == 0

    def test_significant_threshold(self):
        baseline = {"success_rate": 0.70, "avg_score": 0.70}
        learning = {"success_rate": 0.76, "avg_score": 0.75}
        result = LearningGainCalculator.calculate(baseline, learning)
        assert result.significant, "6pp improvement should be significant"

    def test_not_significant(self):
        baseline = {"success_rate": 0.74, "avg_score": 0.70}
        learning = {"success_rate": 0.76, "avg_score": 0.72}
        result = LearningGainCalculator.calculate(baseline, learning)
        assert not result.significant, "2pp should not be significant"


class TestGeneralizationGain:
    def test_positive_generalization(self):
        results = [
            {"task_id": "train-1", "score": 0.8},
            {"task_id": "train-2", "score": 0.7},
            {"task_id": "eval-1", "score": 0.75},
            {"task_id": "eval-2", "score": 0.72},
        ]
        # V3.2 Phase 5: the hard-coded 0.5 random baseline is removed. The
        # baseline is now the caller-supplied EMPIRICAL COLD value; the
        # generalization gain is the unseen/train ratio above that baseline.
        gen = GeneralizationGain.calculate(
            results,
            train_task_ids=["train-1", "train-2"],
            eval_task_ids=["eval-1", "eval-2"],
            cold_baseline_score=0.3,  # empirical COLD-derived baseline
        )
        assert gen.generalization_gain > 0

    def test_no_cold_baseline_is_not_applicable(self):
        # Without a supplied empirical COLD baseline the generalization gain is
        # NOT_APPLICABLE, never an invented 0.5.
        results = [
            {"task_id": "train-1", "score": 0.8},
            {"task_id": "train-2", "score": 0.7},
            {"task_id": "eval-1", "score": 0.75},
            {"task_id": "eval-2", "score": 0.72},
        ]
        gen = GeneralizationGain.calculate(
            results,
            train_task_ids=["train-1", "train-2"],
            eval_task_ids=["eval-1", "eval-2"],
        )
        assert gen.baseline_generalization_score == NOT_APPLICABLE
        assert gen.generalization_gain == NOT_APPLICABLE

    def test_overfitting(self):
        results = [
            {"task_id": "train-1", "score": 0.9},
            {"task_id": "train-2", "score": 0.95},
            {"task_id": "eval-1", "score": 0.4},
            {"task_id": "eval-2", "score": 0.3},
        ]
        gen = GeneralizationGain.calculate(
            results,
            train_task_ids=["train-1", "train-2"],
            eval_task_ids=["eval-1", "eval-2"],
            cold_raw_score=0.5,
        )
        assert gen.gap > 0.4  # large gap indicates overfitting

    def test_no_eval(self):
        results = [{"task_id": "train-1", "score": 0.8}]
        gen = GeneralizationGain.calculate(
            results,
            train_task_ids=["train-1"],
            eval_task_ids=["eval-1"],
        )
        # No eval observations -> no valid unseen statistic, not a forced 0.0.
        assert gen.unseen_performance == NOT_AVAILABLE


class TestFailureAvoidanceMetric:
    def test_fewer_failures(self):
        baseline = [{"capability": "deploy", "error_message": "timeout"}]
        learning = []  # no failures, learning worked
        result = FailureAvoidanceMetric.calculate(baseline, learning)
        assert result.failures_avoided == 1
        assert result.failure_avoidance_rate == 1.0

    def test_repeated_failure_prevention(self):
        baseline = [
            {"capability": "deploy", "error_message": "timeout"},
            {"capability": "deploy", "error_message": "timeout"},
        ]
        learning = [
            {"capability": "test", "error_message": "assertion_error"},
        ]
        result = FailureAvoidanceMetric.calculate(baseline, learning)
        assert result.repeated_failures_prevented > 0

    def test_new_failure_patterns(self):
        baseline = [{"capability": "deploy", "error_message": "timeout"}]
        learning = [
            {"capability": "deploy", "error_message": "timeout"},
            {"capability": "new", "error_message": "new_error"},
        ]
        result = FailureAvoidanceMetric.calculate(baseline, learning)
        assert result.new_failure_patterns == 1

    def test_no_baseline_failures(self):
        result = FailureAvoidanceMetric.calculate([], [{"capability": "test", "error_message": "err"}])
        assert result.failures_avoided == 0
        assert result.failure_avoidance_rate == 0.0


class TestDecisionInfluenceMetric:
    def test_positive_influence(self):
        log = [
            {"task_id": "t1", "decision_type": "model", "influenced_by_learning": True, "outcome_improved": True},
            {"task_id": "t2", "decision_type": "agent", "influenced_by_learning": True, "outcome_improved": True},
            {"task_id": "t3", "decision_type": "model", "influenced_by_learning": False},
        ]
        result = DecisionInfluenceMetric.calculate(log)
        assert result.decisions_changed == 2
        assert result.net_improvement_rate > 0

    def test_negative_influence(self):
        log = [
            {"task_id": "t1", "decision_type": "model", "influenced_by_learning": True, "outcome_improved": False},
        ]
        result = DecisionInfluenceMetric.calculate(log)
        assert result.net_improvement_rate < 0

    def test_no_decisions(self):
        result = DecisionInfluenceMetric.calculate([])
        assert result.total_decisions == 0
        assert result.change_rate == 0.0


class TestLearningEfficiency:
    def test_efficiency_calculation(self):
        result = LearningEfficiency.calculate(
            total_experiences=100,
            extracted_experiences=50,
            high_quality_count=20,
            success_improvement_pct=5.0,
        )
        assert 0 < result.extraction_efficiency <= 1.0
        assert result.high_quality_fraction == 0.4
        assert result.learning_gain_per_experience == 0.1
        assert result.cost_per_gain > 0

    def test_zero_cases(self):
        result = LearningEfficiency.calculate(
            total_experiences=0, extracted_experiences=0,
            high_quality_count=0, success_improvement_pct=0.0,
        )
        assert result.extraction_efficiency == 0.0
        assert result.high_quality_fraction == 0.0


class TestLearningEvaluator:
    def test_evaluate_all(self):
        baseline = {
            "success_rate": 0.74, "avg_score": 0.7184,
            "failures": [{"capability": "deploy", "error_message": "timeout"}],
            "total_executions": 50,
        }
        learning = {
            "success_rate": 0.80, "avg_score": 0.75,
            "failures": [{"capability": "test", "error_message": "assertion"}],
            "extracted_experiences": 30,
            "high_quality_experiences": 15,
            "total_executions": 50,
        }
        decision_log = [
            {"task_id": "t1", "decision_type": "model", "influenced_by_learning": True, "outcome_improved": True},
            {"task_id": "t2", "decision_type": "agent", "influenced_by_learning": False},
        ]

        results = LearningEvaluator.evaluate_all(baseline, learning, decision_log)
        assert "learning_gain" in results
        assert "failure_avoidance" in results
        assert "decision_influence" in results
        assert "learning_efficiency" in results

        # Verify learning gain
        assert results["learning_gain"]["success_rate_improvement_pct"] == 6.0
        assert results["learning_gain"]["significant"] is True

        # Verify failure avoidance
        assert results["failure_avoidance"]["failures_avoided"] == 0  # baseline has 1, learning has 1

    def test_evaluate_without_decisions(self):
        baseline = {"success_rate": 0.7, "avg_score": 0.7, "failures": []}
        learning = {"success_rate": 0.75, "avg_score": 0.72, "failures": [], "extracted_experiences": 10, "high_quality_experiences": 5}
        results = LearningEvaluator.evaluate_all(baseline, learning)
        assert "learning_gain" in results
        assert "failure_avoidance" in results
        assert "decision_influence" not in results  # no decision log
        assert "learning_efficiency" in results
