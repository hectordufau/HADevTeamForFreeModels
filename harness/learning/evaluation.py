# harness/learning/evaluation.py — Phase F: Learning Evaluation
"""
Learning evaluation metrics: Learning Gain, Generalization Gain,
Failure Avoidance, Decision Influence, Learning Efficiency.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import math


class EvaluationError(Exception):
    """Raised on evaluation errors."""


# V3.2 Phase 5: explicit status sentinels — a statistic that cannot be validly
# computed is reported as such, never silently substituted.
NOT_AVAILABLE = "NOT_AVAILABLE"
NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class LearningGainResult:
    """Result of a learning gain calculation."""
    baseline_success_rate: float
    learning_success_rate: float
    success_rate_improvement: float  # absolute percentage points
    baseline_score: float
    learning_score: float
    score_improvement: float  # absolute improvement
    gain_ratio: float  # learning_score / baseline_score
    significant: bool  # improvement > threshold


@dataclass
class GeneralizationResult:
    """Result of generalization gain calculation.

    V3.2 Phase 5: fields are Any because they may carry explicit sentinels
    (NOT_AVAILABLE / NOT_APPLICABLE) when a statistic cannot be validly
    computed, instead of an invented numeric value.
    """
    baseline_generalization_score: Any
    learning_generalization_score: Any
    generalization_gain: Any  # positive = learning generalizes well
    seen_performance: Any
    unseen_performance: Any
    gap: Any  # seen - unseen (positive = overfitting)


@dataclass
class FailureAvoidanceResult:
    """Result of failure avoidance metric."""
    baseline_failure_count: int
    learning_failure_count: int
    failures_avoided: int
    failure_avoidance_rate: float  # 0.0 to 1.0
    repeated_failures_prevented: int
    new_failure_patterns: int


@dataclass
class DecisionInfluenceResult:
    """Result of decision influence metric."""
    decisions_changed: int
    total_decisions: int
    change_rate: float  # proportion of decisions influenced
    improvements_from_change: int
    regressions_from_change: int
    net_improvement_rate: float  # (improvements - regressions) / total
    decisions_tracked: int


@dataclass
class LearningEfficiencyResult:
    """Result of learning efficiency calculation."""
    total_experiences: int
    total_extractions: int
    extraction_efficiency: float  # extractions / experiences
    high_quality_fraction: float
    learning_gain_per_experience: float
    cost_per_gain: float  # experiences needed per 1% improvement


class LearningGainCalculator:
    """Calculates learning gain as percentage improvement over baseline.

    V3.2 Phase 5: the +5 percentage-point "significant" threshold is no longer
    hard-coded — it is supplied via the configurable SignificanceConfig (see
    harness.learning.statistics) and reflected in reports.
    """

    @staticmethod
    def calculate(baseline_metrics: Dict[str, Any],
                  learning_metrics: Dict[str, Any],
                  significance: Any = None) -> LearningGainResult:
        """Calculate learning gain from baseline vs learning metrics.

        Args:
            baseline_metrics: dict with success_rate, avg_score
            learning_metrics: dict with success_rate, avg_score
            significance: optional SignificanceConfig; defaults to the standard
                config so the +5pp threshold is configurable, not hard-coded.
        """
        if significance is None:
            from .statistics import DEFAULT_SIGNIFICANCE
            significance = DEFAULT_SIGNIFICANCE

        b_success = baseline_metrics.get("success_rate", 0)
        l_success = learning_metrics.get("success_rate", 0)
        b_score = baseline_metrics.get("avg_score", 0)
        l_score = learning_metrics.get("avg_score", 0)

        success_improvement = l_success - b_success
        score_improvement = l_score - b_score

        gain_ratio = l_score / max(b_score, 0.001)
        significant = abs(success_improvement) >= significance.success_rate_pp

        return LearningGainResult(
            baseline_success_rate=b_success,
            learning_success_rate=l_success,
            success_rate_improvement=round(success_improvement * 100, 2),
            baseline_score=b_score,
            learning_score=l_score,
            score_improvement=round(score_improvement, 4),
            gain_ratio=round(gain_ratio, 4),
            significant=significant,
        )


class GeneralizationGain:
    """Measures how well learning generalizes to unseen tasks.

    V3.2 (Phase 5): the artificial 0.5 "random baseline" is removed. The
    generalization baseline is an EMPIRICAL COLD-DERIVED value supplied by the
    caller; when EITHER an environment is absent (n=0) or the caller does not
    supply a COLD baseline for the pre-normalization score, it is reported as
    NOT_APPLICABLE rather than inventing a number. A bare ratio is only a
    descriptive statistic, not a causal claim.
    """

    @staticmethod
    def calculate(all_results: List[Dict[str, Any]],
                  train_task_ids: List[str],
                  eval_task_ids: List[str],
                  cold_baseline_score: Any = None,
                  cold_raw_score: Any = None) -> GeneralizationResult:
        """Calculate generalization gain.

        Args:
            all_results: All benchmark results with task_id and score
            train_task_ids: Task IDs used during training
            eval_task_ids: Task IDs held out for evaluation
            cold_baseline_score: Empirical COLD baseline for the generalization
                ration (e.g. COLD unseen/train ratio); NOT_APPLICABLE when None.
            cold_raw_score: Empirical COLD score for the gap baseline (from
                persisted COLD outcomes); NOT_APPLICABLE when None.

        Returns:
            GeneralizationResult — baseline_generalization_score is NOT_APPLICABLE
            when no empirical COLD baseline is supplied; when a baseline is
            supplied it is used, never replaced by an invented 0.5.
        """
        train_scores = [
            r.get("score", 0) for r in all_results
            if r.get("task_id") in train_task_ids
        ]
        eval_scores = [
            r.get("score", 0) for r in all_results
            if r.get("task_id") in eval_task_ids
        ]

        if not train_scores or not eval_scores:
            # None/insufficient environment: no valid generalization statistic.
            return GeneralizationResult(
                baseline_generalization_score=(cold_baseline_score if cold_baseline_score is not None else NOT_APPLICABLE),
                learning_generalization_score=NOT_AVAILABLE,
                generalization_gain=NOT_AVAILABLE,
                seen_performance=NOT_AVAILABLE,
                unseen_performance=NOT_AVAILABLE,
                gap=NOT_AVAILABLE,
            )

        train_avg = sum(train_scores) / len(train_scores)
        eval_avg = sum(eval_scores) / len(eval_scores)

        # Generalization gain: eval performance as fraction of train performance
        gen_gain = eval_avg / max(train_avg, 1e-9)

        # Gap: difference between seen and unseen (positive = overfitting);
        # delta is relative to the empirical COLD raw score when supplied.
        gap = train_avg - eval_avg

        # Empirical COLD baseline: use it if supplied, else NOT_APPLICABLE.
        baseline_gen = cold_baseline_score if cold_baseline_score is not None else NOT_APPLICABLE

        if isinstance(baseline_gen, (int, float)):
            generalization_gain = round(gen_gain - baseline_gen, 4)
        else:
            generalization_gain = NOT_APPLICABLE

        if cold_raw_score is not None:
            gap = train_avg - cold_raw_score

        return GeneralizationResult(
            baseline_generalization_score=baseline_gen,
            learning_generalization_score=round(gen_gain, 4),
            generalization_gain=generalization_gain,
            seen_performance=round(train_avg, 4),
            unseen_performance=round(eval_avg, 4),
            gap=round(gap, 4),
        )


class FailureAvoidanceMetric:
    """Measures how learning reduces repeated failures."""

    @staticmethod
    def calculate(baseline_failures: List[Dict[str, Any]],
                  learning_failures: List[Dict[str, Any]]) -> FailureAvoidanceResult:
        """Calculate failure avoidance from baseline vs learning failures.

        Args:
            baseline_failures: Failures from baseline (no learning)
            learning_failures: Failures with learning enabled

        Returns:
            FailureAvoidanceResult
        """
        b_count = len(baseline_failures)
        l_count = len(learning_failures)

        failures_avoided = max(0, b_count - l_count)
        avoidance_rate = failures_avoided / max(b_count, 1)

        # Detect repeated failures (same capability, same error pattern)
        b_patterns = {
            (f.get("capability", ""), f.get("error_message", "")[:50])
            for f in baseline_failures
        }
        l_patterns = {
            (f.get("capability", ""), f.get("error_message", "")[:50])
            for f in learning_failures
        }

        repeated_prevented = len(b_patterns - l_patterns)
        new_patterns = len(l_patterns - b_patterns)

        return FailureAvoidanceResult(
            baseline_failure_count=b_count,
            learning_failure_count=l_count,
            failures_avoided=failures_avoided,
            failure_avoidance_rate=round(avoidance_rate, 4),
            repeated_failures_prevented=repeated_prevented,
            new_failure_patterns=new_patterns,
        )


class DecisionInfluenceMetric:
    """Measures how much learning influences decisions and whether it improves outcomes."""

    @staticmethod
    def calculate(decision_log: List[Dict[str, Any]]) -> DecisionInfluenceResult:
        """Calculate decision influence from a decision log.

        Each log entry: {tast_id, decision_type, influenced_by_learning: bool,
                         outcome_improved: Optional[bool]}

        Returns:
            DecisionInfluenceResult
        """
        total = len(decision_log)
        changed = sum(1 for d in decision_log if d.get("influenced_by_learning", False))
        improved = sum(
            1 for d in decision_log
            if d.get("influenced_by_learning", False) and d.get("outcome_improved", False)
        )
        regressed = sum(
            1 for d in decision_log
            if d.get("influenced_by_learning", False) and d.get("outcome_improved") is False
        )

        net_rate = (improved - regressed) / max(total, 1)

        return DecisionInfluenceResult(
            decisions_changed=changed,
            total_decisions=total,
            change_rate=round(changed / max(total, 1), 4),
            improvements_from_change=improved,
            regressions_from_change=regressed,
            net_improvement_rate=round(net_rate, 4),
            decisions_tracked=total,
        )


class LearningEfficiency:
    """Measures how efficiently learning converts experiences into improvements."""

    @staticmethod
    def calculate(total_experiences: int,
                  extracted_experiences: int,
                  high_quality_count: int,
                  success_improvement_pct: float) -> LearningEfficiencyResult:
        """Calculate learning efficiency metrics."""
        extraction_efficiency = extracted_experiences / max(total_experiences, 1)
        high_quality_fraction = high_quality_count / max(extracted_experiences, 1)
        gain_per_exp = success_improvement_pct / max(extracted_experiences, 1)
        cost_per_gain = extracted_experiences / max(success_improvement_pct, 0.1)

        return LearningEfficiencyResult(
            total_experiences=total_experiences,
            total_extractions=extracted_experiences,
            extraction_efficiency=round(extraction_efficiency, 4),
            high_quality_fraction=round(high_quality_fraction, 4),
            learning_gain_per_experience=round(gain_per_exp, 4),
            cost_per_gain=round(cost_per_gain, 2),
        )


class LearningEvaluator:
    """
    Comprehensive learning evaluation aggregating all metrics.

    Runs all learning evaluation calculations and produces
    a unified report.
    """

    @staticmethod
    def evaluate_all(baseline: Dict[str, Any],
                     learning: Dict[str, Any],
                     decision_log: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run all learning evaluation metrics and produce a report.

        Args:
            baseline: Dict with success_rate, avg_score, failures list
            learning: Dict with success_rate, avg_score, failures list
            decision_log: Optional list of decision entries

        Returns:
            Dict with all evaluation results
        """
        results = {}

        # Learning Gain
        gain = LearningGainCalculator.calculate(baseline, learning)
        results["learning_gain"] = {
            "success_rate_improvement_pct": gain.success_rate_improvement,
            "score_improvement": gain.score_improvement,
            "gain_ratio": gain.gain_ratio,
            "significant": gain.significant,
        }

        # Failure Avoidance
        b_failures = baseline.get("failures", [])
        l_failures = learning.get("failures", [])
        avoidance = FailureAvoidanceMetric.calculate(b_failures, l_failures)
        results["failure_avoidance"] = {
            "failures_avoided": avoidance.failures_avoided,
            "failure_avoidance_rate": avoidance.failure_avoidance_rate,
            "repeated_failures_prevented": avoidance.repeated_failures_prevented,
            "new_failure_patterns": avoidance.new_failure_patterns,
        }

        # Decision Influence
        if decision_log:
            influence = DecisionInfluenceMetric.calculate(decision_log)
            results["decision_influence"] = {
                "decisions_changed": influence.decisions_changed,
                "change_rate": influence.change_rate,
                "improvements_from_change": influence.improvements_from_change,
                "regressions_from_change": influence.regressions_from_change,
                "net_improvement_rate": influence.net_improvement_rate,
            }

        # Learning Efficiency
        efficiency = LearningEfficiency.calculate(
            total_experiences=baseline.get("total_executions", 0) + learning.get("total_executions", 0),
            extracted_experiences=learning.get("extracted_experiences", 0),
            high_quality_count=learning.get("high_quality_experiences", 0),
            success_improvement_pct=gain.success_rate_improvement,
        )
        results["learning_efficiency"] = {
            "extraction_efficiency": efficiency.extraction_efficiency,
            "high_quality_fraction": efficiency.high_quality_fraction,
            "learning_gain_per_experience": efficiency.learning_gain_per_experience,
            "cost_per_gain": efficiency.cost_per_gain,
        }

        return results
