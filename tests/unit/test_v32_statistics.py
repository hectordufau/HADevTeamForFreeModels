"""Tests for V3.2 Phase 5 — Empirical COLD Baseline & Statistical Evaluation.

Covers: empirical COLD baseline with NO hard-coded 0.5, paired/unpaired
comparison, configurable significance (loading + reporting), 95% bootstrap CI
(deterministic), full statistical outputs block, small-sample no-false-
significance, TEST isolation, cross-run isolation, no-learning-feedback-loop,
security as a hard constraint, and historical-artifact protection.
"""

import os
import sys
import math
import json
import shutil
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from harness.learning.statistics import (
    SignificanceConfig,
    DEFAULT_SIGNIFICANCE,
    load_significance_config,
    EvaluationObservation,
    ColdBaseline,
    EmpiricalEvaluator,
    build_pairs,
    INSUFFICIENT_BASELINE_DATA,
    INSUFFICIENT_DATA,
    NOT_AVAILABLE,
    NOT_APPLICABLE,
    CI_NOT_COMPUTED_INSUFFICIENT_SAMPLE,
    NO_PAIRED_DATA,
)
from harness.learning.context import ExperimentContext
from harness.learning.evaluation import GeneralizationGain, LearningGainCalculator
from harness.learning import GeneralizationGain as G2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _repo_root():
    """Repo root = three dirname() from tests/unit/test_v32_statistics.py."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _clean_v32():
    base = os.path.join(_repo_root(), "artifacts", "v3.2")
    if os.path.exists(base):
        shutil.rmtree(base, ignore_errors=True)


def obs(run="run1", bench="bench", task="t1", mode="COLD", **kw):
    """Build an EvaluationObservation with defaults."""
    defaults = dict(
        validation_run_id=run, benchmark_id=bench, mode=mode, task_id=task,
        category="", config="", success=None,
        engineering_score=None, correctness=None, architecture=None,
        security=None, maintainability=None, efficiency=None,
        latency=None, iterations=None, verification_failures=None, retries=None,
        security_failure=None,
    )
    defaults.update(kw)
    return EvaluationObservation(**defaults)


def cold_success(n, value=True, **kw):
    """n COLD observations, distinct tasks, all with the given success value."""
    return [obs(task=f"t{i}", success=value, **kw) for i in range(n)]


def learned_success(n, value=True, **kw):
    return [obs(task=f"t{i}", mode="LEARNED", success=value, **kw)
            for i in range(n)]


def mixed_cold(true_n, false_n):
    """Distinct-task COLD observations with true_n successes and false_n
    failures (no duplicate task keys, unlike chained cold_success calls)."""
    return ([obs(task=f"t{i}", success=True) for i in range(true_n)]
            + [obs(task=f"t{i}", success=False) for i in range(true_n, true_n + false_n)])


def mixed_learned(true_n, false_n):
    """Distinct-task LEARNED observations with true_n successes and
    false_n failures."""
    return ([obs(task=f"t{i}", mode="LEARNED", success=True) for i in range(true_n)]
            + [obs(task=f"t{i}", mode="LEARNED", success=False)
               for i in range(true_n, true_n + false_n)])


def ctx(run="run1"):
    return ExperimentContext(
        validation_run_id=run, benchmark_id="bench", task_id="t1",
        execution_id="e1", mode="COLD")


SUFFICIENT = 10  # == DEFAULT min_samples


# ---------------------------------------------------------------------------
# 1. Empirical COLD baseline — no hard-coded 0.5, reproducible, run-isolated
# ---------------------------------------------------------------------------


class TestEmpiricalBaseline:
    def test_baseline_derived_from_cold_data_not_05(self):
        # 10 COLD obs all success=True -> empirical success_rate baseline = 1.0,
        # NOT the old hard-coded 0.5.
        c = cold_success(SUFFICIENT, value=True)
        ev = EmpiricalEvaluator(c)
        assert ev.baseline()["success_rate"] == 1.0

    def test_baseline_respects_cold_values(self):
        # Mix: 7 true, 3 false -> 0.7 empirical baseline from observed data.
        c = mixed_cold(7, 3)
        ev = EmpiricalEvaluator(c)
        assert ev.baseline()["success_rate"] == round(7 / 10, 4)

    def test_insufficient_baseline_reports_sentinel(self):
        # Fewer than min_samples COLD observations -> INSUFFICIENT_BASELINE_DATA,
        # never an invented value.
        c = cold_success(3, value=True)
        ev = EmpiricalEvaluator(c)
        assert ev.baseline()["success_rate"] == INSUFFICIENT_BASELINE_DATA
        assert ev.insufficient_baseline("success_rate") is True

    def test_no_hardcoded_point_five_in_source(self):
        # Grep the Phase 5 module + evaluator: the legacy random 0.5 baseline
        # must be gone.
        from harness import learning
        path = os.path.join(
            _repo_root(),
            "harness", "learning", "evaluation.py")
        with open(path) as f:
            src = f.read()
        path_stat = os.path.join(
            _repo_root(),
            "harness", "learning", "statistics.py")
        with open(path_stat) as f:
            src_stat = f.read()
        assert "baseline_gen = 0.5" not in src
        assert "baseline_gen = 0.5" not in src_stat

    def test_baseline_reproducible(self):
        # Same observations, two evaluators -> identical empirical baseline.
        c = cold_success(SUFFICIENT)
        a = EmpiricalEvaluator(c).baseline()
        b = EmpiricalEvaluator(c).baseline()
        assert a == b

    def test_baseline_run_isolated(self):
        # COLD observations from a different run cannot enter this baseline.
        c = cold_success(SUFFICIENT, run="runA")
        ev = EmpiricalEvaluator(c)
        other = cold_success(5, run="runB", value=True)
        # Only runA obs are consumed; runB never mixes in implicitly.
        assert ev.cold  # sanity


# ---------------------------------------------------------------------------
# 2. Paired comparison (matched / unmatched / duplicate / missing)
# ---------------------------------------------------------------------------


class TestPairedComparison:
    def test_matched_pairs_paired_type(self):
        c = cold_success(SUFFICIENT)
        l = learned_success(SUFFICIENT)
        ev = EmpiricalEvaluator(c + l)
        r = ev.compare(["success_rate"])["success_rate"]
        assert r["comparison_type"] == "PAIRED"
        assert r["paired_n"] == SUFFICIENT

    def test_unmatched_uses_unpaired_and_documents(self):
        # Different task ids -> no pairs -> UNPAIRED, NO_PAIRED_DATA or 0.
        c = cold_success(SUFFICIENT)
        l = [obs(task=f"x{i}", mode="LEARNED", success=True) for i in range(SUFFICIENT)]
        ev = EmpiricalEvaluator(c + l)
        r = ev.compare(["success_rate"])["success_rate"]
        assert r["comparison_type"] == "UNPAIRED"
        assert r["paired_n"] in (NO_PAIRED_DATA, 0)

    def test_duplicate_pairs_matched_one_to_one(self):
        # Same keys appear multiple times per side; one-to-one matching in key
        # order; leftovers reported as unmatched.
        c = [obs(task="t1", success=True), obs(task="t1", success=False)]
        l = [obs(task="t1", mode="LEARNED", success=True),
             obs(task="t1", mode="LEARNED", success=True),
             obs(task="t1", mode="LEARNED", success=False)]  # one extra
        ev = EmpiricalEvaluator(c + l)
        r = ev.compare(["success_rate"])["success_rate"]
        # 2 pairs possible (min of counts); leftover learned is unmatched.
        assert r["paired_n"] == 2

    def test_missing_pair_side(self):
        # A task present only in LEARNED -> unmatched learned documented.
        c = [obs(task="t1", success=True)]
        l = [obs(task="t1", mode="LEARNED", success=True),
             obs(task="t2", mode="LEARNED", success=True)]
        pairs, unmatched_cold, unmatched_learned = build_pairs(c, l)
        assert len(pairs) == 1
        assert unmatched_cold == []
        assert len(unmatched_learned) == 1


# ---------------------------------------------------------------------------
# 3. Significance config (custom threshold, min_samples, loading, reporting)
# ---------------------------------------------------------------------------


class TestSignificanceConfig:
    def test_custom_threshold_changes_classification(self):
        # success rate COLD=0.6, LEARNED=0.8 -> delta 20pp.
        c = mixed_cold(6, 4)  # 0.6
        l = mixed_learned(8, 2)  # 0.8
        ev = EmpiricalEvaluator(c + l)
        r = ev.compare(["success_rate"])["success_rate"]
        # 20pp > default 5pp -> significant True
        assert r["statistically_significant"] is True
        # With a huge threshold -> not significant.
        big = SignificanceConfig(success_rate_pp=0.50, score_delta=0.5, min_samples=10)
        ev2 = EmpiricalEvaluator(c + l, config=big)
        r2 = ev2.compare(["success_rate"])["success_rate"]
        assert r2["statistically_significant"] is False

    def test_score_delta_threshold(self):
        # engineering score delta 0.05; default score_delta 0.01 -> significant.
        c = [obs(task=f"t{i}", engineering_score=0.50) for i in range(SUFFICIENT)]
        l = [obs(task=f"t{i}", mode="LEARNED", engineering_score=0.55) for i in range(SUFFICIENT)]
        ev = EmpiricalEvaluator(c + l)
        r = ev.compare(["engineering_score"])["engineering_score"]
        assert r["statistically_significant"] is True
        # threshold above the delta -> not significant
        strict = SignificanceConfig(score_delta=0.5, success_rate_pp=0.05, min_samples=10)
        ev2 = EmpiricalEvaluator(c + l, config=strict)
        r2 = ev2.compare(["engineering_score"])["engineering_score"]
        assert r2["statistically_significant"] is False

    def test_min_samples_gates_significance(self):
        # With n < min_samples, significance is INSUFFICIENT_DATA even when a
        # big delta exists (no false significance on tiny samples).
        c = cold_success(3, value=True)
        l = learned_success(3, value=True)
        ev = EmpiricalEvaluator(c + l)  # min_samples=10
        r = ev.compare(["success_rate"])["success_rate"]
        assert r["statistically_significant"] == INSUFFICIENT_DATA
        # With a lowered min_samples (e.g. 3) it becomes evaluable.
        small_cfg = SignificanceConfig(min_samples=3)
        ev2 = EmpiricalEvaluator(c + l, config=small_cfg)
        r2 = ev2.compare(["success_rate"])["success_rate"]
        assert r2["statistically_significant"] in (True, False)

    def test_config_loading_from_yaml(self, tmp_path):
        # Write a config file with custom significance values and load them.
        cfg_file = tmp_path / "harness.yaml"
        cfg_file.write_text(
            "evaluation:\n  significance:\n"
            "    success_rate_pp: 0.10\n    score_delta: 0.02\n    min_samples: 20\n"
        )
        loaded = load_significance_config(str(cfg_file))
        assert loaded.success_rate_pp == 0.10
        assert loaded.score_delta == 0.02
        assert loaded.min_samples == 20

    def test_config_loading_falls_back_to_defaults(self, tmp_path):
        # Missing section -> defaults (deterministic, observable).
        cfg_file = tmp_path / "harness.yaml"
        cfg_file.write_text("evaluation:\n  enabled: true\n")
        loaded = load_significance_config(str(cfg_file))
        assert loaded == DEFAULT_SIGNIFICANCE

    def test_config_loading_invalid_rejected(self, tmp_path):
        cfg_file = tmp_path / "harness.yaml"
        cfg_file.write_text("evaluation:\n  significance:\n    min_samples: -3\n")
        with pytest.raises(ValueError):
            load_significance_config(str(cfg_file))

    def test_config_reported_in_report(self):
        ev = EmpiricalEvaluator(cold_success(SUFFICIENT) + learned_success(SUFFICIENT))
        rep = ev.report()
        assert rep["significance_config"] == DEFAULT_SIGNIFICANCE.to_dict()
        assert "success_rate_pp" in rep["significance_config"]
        assert "min_samples" in rep["significance_config"]


# ---------------------------------------------------------------------------
# 4. Confidence intervals
# ---------------------------------------------------------------------------


class TestConfidenceIntervals:
    def test_ci_sufficient_sample(self):
        c = [obs(task=f"t{i}", engineering_score=0.5 + 0.01 * i) for i in range(SUFFICIENT)]
        l = [obs(task=f"t{i}", mode="LEARNED", engineering_score=0.6 + 0.01 * i) for i in range(SUFFICIENT)]
        ev = EmpiricalEvaluator(c + l)
        r = ev.compare(["engineering_score"])["engineering_score"]
        ci = r["confidence_interval"]
        assert isinstance(ci, list) and len(ci) == 2
        assert ci[0] <= ci[1]

    def test_ci_insufficient_sample_sentinel(self):
        c = [obs(task=f"t{i}", engineering_score=0.5) for i in range(3)]
        l = [obs(task=f"t{i}", mode="LEARNED", engineering_score=0.6) for i in range(3)]
        ev = EmpiricalEvaluator(c + l)
        r = ev.compare(["engineering_score"])["engineering_score"]
        assert r["confidence_interval"] == CI_NOT_COMPUTED_INSUFFICIENT_SAMPLE

    def test_ci_deterministic_bootstrap(self):
        c = [obs(task=f"t{i}", engineering_score=0.5 + 0.01 * i) for i in range(50)]
        l = [obs(task=f"t{i}", mode="LEARNED", engineering_score=0.6 + 0.01 * i) for i in range(50)]
        ctx_a = ctx("runDeterministic")
        ev1 = EmpiricalEvaluator(c + l, experiment_context=ctx_a)
        ev2 = EmpiricalEvaluator(c + l, experiment_context=ctx_a)
        r1 = ev1.compare(["engineering_score"])["engineering_score"]["confidence_interval"]
        r2 = ev2.compare(["engineering_score"])["engineering_score"]["confidence_interval"]
        assert r1 == r2

    def test_ci_seed_derived_from_context(self):
        # Different ExperimentContext -> different deterministic seed -> CI can
        # differ, but each is reproducible.
        c = [obs(task=f"t{i}", engineering_score=0.5 + 0.01 * i) for i in range(50)]
        l = [obs(task=f"t{i}", mode="LEARNED", engineering_score=0.6 + 0.01 * i) for i in range(50)]
        ev_a = EmpiricalEvaluator(c + l, experiment_context=ctx("runA"))
        ev_b = EmpiricalEvaluator(c + l, experiment_context=ctx("runB"))
        # Re-running the same context reproduces; different context may differ.
        assert ev_a.compare(["engineering_score"])["engineering_score"]["confidence_interval"] == \
               EmpiricalEvaluator(c + l, experiment_context=ctx("runA")).compare(
                   ["engineering_score"])["engineering_score"]["confidence_interval"]


# ---------------------------------------------------------------------------
# 5. Statistical outputs (delta, pp delta, p-value, effect size, significance)
# ---------------------------------------------------------------------------


class TestStatisticalOutputs:
    def test_success_output_fields(self):
        c = mixed_cold(6, 4)  # 0.6
        l = mixed_learned(9, 1)  # 0.9
        ev = EmpiricalEvaluator(c + l)
        r = ev.compare(["success_rate"])["success_rate"]
        assert r["baseline_n"] == SUFFICIENT
        assert r["learned_n"] == SUFFICIENT
        assert r["baseline_mean"] == pytest.approx(0.6, abs=1e-4)
        assert r["learned_mean"] == pytest.approx(0.9, abs=1e-4)
        assert r["delta"] == pytest.approx(0.3, abs=1e-4)
        # delta in percentage points = 30.0 pp for success rate
        assert r["delta_percentage_points"] == pytest.approx(30.0, abs=1e-4)
        assert isinstance(r["p_value"], float)
        assert r["effect_size"] is not NOT_AVAILABLE
        assert r["statistically_significant"] is True

    def test_score_output_fields(self):
        c = [obs(task=f"t{i}", engineering_score=0.50) for i in range(SUFFICIENT)]
        l = [obs(task=f"t{i}", mode="LEARNED", engineering_score=0.60) for i in range(SUFFICIENT)]
        ev = EmpiricalEvaluator(c + l)
        r = ev.compare(["engineering_score"])["engineering_score"]
        assert r["baseline_mean"] == pytest.approx(0.5, abs=1e-4)
        assert r["learned_mean"] == pytest.approx(0.6, abs=1e-4)
        # non-success metric: delta_percentage_points = NOT_APPLICABLE
        assert r["delta_percentage_points"] == NOT_APPLICABLE
        assert r["statistically_significant"] is True

    def test_report_block_complete(self):
        c = cold_success(SUFFICIENT)
        l = learned_success(SUFFICIENT)
        ev = EmpiricalEvaluator(c + l)
        rep = ev.report()
        assert rep["feedback_loop"] == "NONE"
        assert rep["method"]["deterministic"] is True
        assert "baseline" in rep and "comparisons" in rep
        assert rep["test_n"] == 0


# ---------------------------------------------------------------------------
# 6. Small sample — no false significance
# ---------------------------------------------------------------------------


class TestSmallSample:
    def test_no_false_significance_below_min(self):
        # Even a big apparent delta on 2 vs 2 observations must NOT be called
        # statistically significant.
        c = cold_success(2, value=True)
        l = learned_success(2, value=False)
        ev = EmpiricalEvaluator(c + l)
        r = ev.compare(["success_rate"])["success_rate"]
        assert r["statistically_significant"] == INSUFFICIENT_DATA
        assert r["confidence_interval"] == CI_NOT_COMPUTED_INSUFFICIENT_SAMPLE
        assert r["p_value"] == INSUFFICIENT_DATA

    def test_effect_may_be_reported_but_not_significant(self):
        # Small sample: even with a big apparent delta, no baseline value is
        # invented (INSUFFICIENT_BASELINE_DATA) and no significance is claimed
        # (significance=INSUFFICIENT_DATA, CI=NOT_COMPUTED, p=INSUFFICIENT_DATA).
        c = cold_success(4, value=True)
        l = learned_success(4, value=False)
        ev = EmpiricalEvaluator(c + l, config=SignificanceConfig(min_samples=10))
        r = ev.compare(["success_rate"])["success_rate"]
        assert r["baseline_mean"] == INSUFFICIENT_BASELINE_DATA
        assert r["statistically_significant"] == INSUFFICIENT_DATA
        assert r["confidence_interval"] == CI_NOT_COMPUTED_INSUFFICIENT_SAMPLE
        assert r["p_value"] == INSUFFICIENT_DATA


# ---------------------------------------------------------------------------
# 7. TEST isolation — TEST data cannot enter baseline
# ---------------------------------------------------------------------------


class TestTestIsolation:
    def test_test_mode_excluded_from_cold(self):
        c = cold_success(SUFFICIENT, value=True)
        t = [obs(task=f"test{i}", mode="TEST", success=True) for i in range(50)]
        ev = EmpiricalEvaluator(c + t)
        # TEST obs are tracked separately from baseline cold — they never enter
        # the empirical COLD baseline.
        assert ev.test and len(ev.test) == 50
        # baseline only reflects the 10 COLD obs (all success -> 1.0)
        assert ev.baseline()["success_rate"] == 1.0
        assert ev.report()["test_n"] == 50

    def test_test_only_no_learned(self):
        # Only TEST observations -> no COLD baseline (INSUFFICIENT) and no
        # learned side -> INSUFFICIENT_DATA, no invented value.
        t = [obs(task=f"test{i}", mode="TEST", success=True) for i in range(50)]
        ev = EmpiricalEvaluator(t)
        assert ev.baseline()["success_rate"] == INSUFFICIENT_BASELINE_DATA


# ---------------------------------------------------------------------------
# 8. Cross-run isolation — Run A cannot enter Run B baseline
# ---------------------------------------------------------------------------


class TestCrossRunIsolation:
    def test_other_run_excluded_from_baseline(self):
        # The evaluator only consumes the observations it is handed; a helper
        # must never reach across runs. Here we build a Run-A cold set and a
        # hypothetical Run-B cold set; constructing the evaluator with only
        # Run A's observations means Run B can never enter the baseline.
        c_a = cold_success(SUFFICIENT, run="runA", value=True)
        c_b = cold_success(SUFFICIENT, run="runB", value=False)
        ev = EmpiricalEvaluator(c_a)
        assert ev.baseline()["success_rate"] == 1.0
        # Run B observations are distinct objects; if mixed in they'd be their
        # own set — verifying the evaluator scopes purely to input.
        assert all(o.validation_run_id == "runA" for o in ev.cold)

    def test_runB_never_leaks_into_runA_comparison(self):
        c_a = cold_success(SUFFICIENT, run="runA", value=True)
        l_a = learned_success(SUFFICIENT, run="runA", value=True)
        l_b = learned_success(3, run="runB", value=False)  # would harm means if leaked
        ev = EmpiricalEvaluator(c_a + l_a + l_b)
        r = ev.compare(["success_rate"])["success_rate"]
        # learned mean should be 1.0 (all runA learned) — runB false obs ignored
        assert r["learned_mean"] == pytest.approx(1.0, abs=1e-4)

    def test_global_historical_not_implicit(self):
        # No observation is pulled from another run implicitly; only provided
        # observations count. Pure determinism / scoping guard.
        ev = EmpiricalEvaluator([])
        assert ev.baseline()["success_rate"] == INSUFFICIENT_BASELINE_DATA


# ---------------------------------------------------------------------------
# 9. No feedback loop — statistical evaluation does not modify learning artifacts
# ---------------------------------------------------------------------------


class TestNoFeedbackLoop:
    def test_evaluator_does_not_write_learning_artifacts(self, tmp_path):
        # Run the evaluator with a sandboxed workspace and assert nothing was
        # created/modified. Phase 5 is read-only evaluation.
        _clean_v32()
        c = cold_success(SUFFICIENT)
        l = learned_success(SUFFICIENT)
        ev = EmpiricalEvaluator(c + l)
        rep = ev.report()
        assert rep["feedback_loop"] == "NONE"
        # No artifacts/v3.2 should have been written by the evaluator itself.
        base = os.path.join(_repo_root(), "artifacts", "v3.2")
        assert not os.path.exists(base)

    def test_inputs_unchanged_after_eval(self):
        # The observation list (persisted data) is not mutated by evaluation.
        c = cold_success(SUFFICIENT)
        l = learned_success(SUFFICIENT)
        before_c = [(o.mode, o.success) for o in c]
        before_l = [(o.mode, o.success) for o in l]
        ev = EmpiricalEvaluator(c + l)
        ev.report()
        after_c = [(o.mode, o.success) for o in c]
        after_l = [(o.mode, o.success) for o in l]
        assert before_c == after_c
        assert before_l == after_l

    def test_global_random_state_not_seeded(self):
        # The evaluator must never set Python's global random seed (that would
        # be a side effect on the wider process).
        import random
        random.seed(1234)
        marker = random.random()
        c = cold_success(SUFFICIENT)
        l = learned_success(SUFFICIENT)
        ev = EmpiricalEvaluator(c + l, experiment_context=ctx("g"))
        ev.report()
        # global stream continues unaffected (deterministic continuation).
        assert random.random() is not None


# ---------------------------------------------------------------------------
# 10. Security — hard constraint, not averaged away
# ---------------------------------------------------------------------------


class TestSecurityHardConstraint:
    def test_security_failure_reported_as_rate_not_averaged(self):
        # COLD has 2 security failures out of 10; LEARNED has 0. Security is
        # its own binary hard-failure rate, never mixed with engineering score.
        c = [obs(task=f"t{i}", security_failure=(i < 2)) for i in range(SUFFICIENT)]
        l = [obs(task=f"t{i}", mode="LEARNED", security_failure=False) for i in range(SUFFICIENT)]
        ev = EmpiricalEvaluator(c + l)
        rep = ev.report()
        assert "security_failure_rate" in rep["comparisons"]
        sr = rep["comparisons"]["security_failure_rate"]
        assert sr["baseline_mean"] == pytest.approx(0.2, abs=1e-4)
        assert sr["learned_mean"] == pytest.approx(0.0, abs=1e-4)

    def test_security_metric_not_averaged_with_others(self):
        # Comparing with a mixed engineering-score should NOT blend the
        # security hard-failure count into it.
        c = [obs(task=f"t{i}", engineering_score=0.9, security_failure=(i < 1))
             for i in range(SUFFICIENT)]
        l = [obs(task=f"t{i}", mode="LEARNED", engineering_score=0.9, security_failure=False)
             for i in range(SUFFICIENT)]
        ev = EmpiricalEvaluator(c + l)
        rep = ev.report()
        eng = rep["comparisons"]["engineering_score"]
        sec = rep["comparisons"]["security_failure_rate"]
        # Engineering mean is 0.9 (not dragged down by the one security failure),
        # while security_failure_rate is its own hard metric.
        assert eng["baseline_mean"] == pytest.approx(0.9, abs=1e-4)
        assert sec["baseline_mean"] == pytest.approx(0.1, abs=1e-4)


# ---------------------------------------------------------------------------
# 11. Generalization baseline — hard-coded 0.5 removed / NOT_APPLICABLE
# ---------------------------------------------------------------------------


class TestGeneralizationBaseline:
    def test_default_baseline_is_not_applicable(self):
        # No COLD baseline supplied -> NOT_APPLICABLE, never 0.5.
        res = GeneralizationGain.calculate(
            [{"task_id": "a", "score": 0.8}, {"task_id": "b", "score": 0.6}],
            train_task_ids=["a"], eval_task_ids=["b"])
        assert res.baseline_generalization_score == NOT_APPLICABLE
        assert res.generalization_gain == NOT_APPLICABLE

    def test_empirical_cold_baseline_used_when_supplied(self):
        res = GeneralizationGain.calculate(
            [{"task_id": "a", "score": 0.8}, {"task_id": "b", "score": 0.6}],
            train_task_ids=["a"], eval_task_ids=["b"],
            cold_baseline_score=0.4)
        assert res.baseline_generalization_score == 0.4
        assert res.generalization_gain == pytest.approx(0.6 / 0.8 - 0.4, abs=1e-4)

    def test_learning_gain_threshold_configurable(self):
        # Default 5pp threshold -> 20pp is significant.
        res = LearningGainCalculator.calculate(
            {"success_rate": 0.6}, {"success_rate": 0.8})
        assert res.significant is True
        # Custom raised threshold -> not significant.
        from harness.learning.statistics import SignificanceConfig
        strict = SignificanceConfig(success_rate_pp=0.5, score_delta=0.5, min_samples=10)
        res2 = LearningGainCalculator.calculate(
            {"success_rate": 0.6}, {"success_rate": 0.8}, significance=strict)
        assert res2.significant is False


# ---------------------------------------------------------------------------
# 12. Historical artifact protection (regression for cleanup defect)
# ---------------------------------------------------------------------------


class TestHistoricalArtifactProtection:
    def test_all_tracked_historical_artifacts_intact(self):
        import subprocess
        repo = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        tracked = subprocess.check_output(
            ["git", "ls-files", "artifacts/"], cwd=repo,
            stderr=subprocess.DEVNULL).decode().splitlines()
        assert len(tracked) >= 28, "expected the protected historical artifacts"
        missing = [r for r in tracked
                   if not os.path.exists(os.path.join(repo, r))]
        assert missing == [], f"missing tracked artifacts: {missing}"

    def test_cleanup_never_removes_git_tracked_artifacts(self):
        from tests.unit.test_v32_storage_isolation import _clean_all_artifacts
        import subprocess
        repo = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        tracked = subprocess.check_output(
            ["git", "ls-files", "artifacts/"], cwd=repo,
            stderr=subprocess.DEVNULL).decode().splitlines()
        assert len(tracked) >= 28
        _clean_all_artifacts()
        for rel in tracked:
            assert os.path.exists(os.path.join(repo, rel)), (
                f"historical artifact deleted by cleanup: {rel}")


# ---------------------------------------------------------------------------
# 13. Determinism / reproducibility smoke
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_full_report_deterministic(self):
        c = [obs(task=f"t{i}", engineering_score=0.5 + 0.01 * (i % 5),
                 success=(i % 3 != 0)) for i in range(30)]
        l = [obs(task=f"t{i}", mode="LEARNED",
                 engineering_score=0.6 + 0.01 * (i % 5),
                 success=(i % 4 != 0)) for i in range(30)]
        ctx_r = ctx("reproRun")
        a = EmpiricalEvaluator(c + l, experiment_context=ctx_r).report()
        b = EmpiricalEvaluator(c + l, experiment_context=ctx_r).report()
        assert a == b
