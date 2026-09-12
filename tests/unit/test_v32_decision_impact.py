"""Tests for V3.2 Decision Impact Persistence and Real Outcomes (Phase 3)."""

import sys
import os
import json
import tempfile
import shutil
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from harness.learning.context import ExperimentContext
from harness.learning.isolation import namespace_path, get_v32_base
from harness.planner.v3_integration import (
    DecisionImpact,
    DecisionImpactTracker,
    UnifiedDecisionPipeline,
    V3IntegrationError,
    OUTCOMES,
    ATTRIBUTION_LEVELS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_v32():
    base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "artifacts", "v3.2")
    if os.path.exists(base):
        shutil.rmtree(base, ignore_errors=True)


COLD_CTX = ExperimentContext(
    validation_run_id="run_cold", benchmark_id="bench", task_id="t1",
    execution_id="e1", mode="COLD")
LEARNED_CTX = ExperimentContext(
    validation_run_id="run_learned", benchmark_id="bench", task_id="t2",
    execution_id="e2", mode="LEARNED")
TEST_CTX = ExperimentContext(
    validation_run_id="run_test", benchmark_id="bench", task_id="t3",
    execution_id="e3", mode="TEST")
COLD_CTX_RUN2 = ExperimentContext(
    validation_run_id="run_sister", benchmark_id="bench", task_id="t4",
    execution_id="e4", mode="COLD")


def make_impact(did="d1", mode="COLD", run_id="run_cold", **kw: Any):
    base = dict(
        decision_id=did, context_id="c1", decision_type="model",
        baseline_choice="baseline", actual_choice="learned",
        influenced_by_learning=True,
        validation_run_id=run_id, mode=mode,
    )
    base.update(kw)
    return DecisionImpact(**base)


def _read_disk_file(ctx, did):
    store_dir = namespace_path("decisions", experiment_context=ctx)
    path = os.path.join(store_dir, f"{did}.json")
    assert os.path.exists(path), f"expected {path} to exist"
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Persistence: impact survives process restart / disk round-trip
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_impact_persists_to_namespaced_disk(self):
        _clean_v32()
        tracker = DecisionImpactTracker(experiment_context=COLD_CTX)
        tracker.record(make_impact("p1", mode="COLD", run_id="run_cold"),
                       experiment_context=COLD_CTX)
        data = _read_disk_file(COLD_CTX, "p1")
        assert data["decision_id"] == "p1"
        assert data["validation_run_id"] == "run_cold"
        assert data["mode"] == "COLD"
        assert data["benchmark_id"] == "bench"
        assert data["execution_id"] == "e1"
        # outcome semantics serialized
        assert data["outcome"] in OUTCOMES

    def test_impact_reloads_on_new_instance(self):
        """Tracker A records+persists; tracker B loads and impact exists."""
        _clean_v32()
        tracker_a = DecisionImpactTracker(experiment_context=COLD_CTX)
        tracker_a.record(make_impact("reload1", mode="COLD", run_id="run_cold"),
                         experiment_context=COLD_CTX)
        # Simulate process restart with a brand-new instance
        tracker_b = DecisionImpactTracker(experiment_context=COLD_CTX)
        impact = tracker_b.find("reload1")
        assert impact is not None
        assert impact.decision_id == "reload1"
        assert impact.mode == "COLD"
        assert impact.validation_run_id == "run_cold"
        assert len(tracker_b._impacts) == 1

    def test_legacy_flat_persistence_reload(self):
        """Legacy (no context) impacts persist to and reload from flat dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            a = DecisionImpactTracker(storage_dir=tmpdir)
            a.record(DecisionImpact(
                decision_id="legacy1", context_id="c1", decision_type="model",
                baseline_choice="a", actual_choice="b",
                influenced_by_learning=True))
            b = DecisionImpactTracker(storage_dir=tmpdir)
            assert b.find("legacy1") is not None

    def test_duplicate_record_is_idempotent(self):
        _clean_v32()
        tracker = DecisionImpactTracker(experiment_context=COLD_CTX)
        tracker.record(make_impact("dup", mode="COLD", run_id="run_cold"),
                       experiment_context=COLD_CTX)
        tracker.record(make_impact("dup", mode="COLD", run_id="run_cold"),
                       experiment_context=COLD_CTX)
        assert len(tracker._impacts) == 1


# ---------------------------------------------------------------------------
# 2. Multiple impacts, runs, and modes
# ---------------------------------------------------------------------------


class TestMultipleImpacts:
    def test_multiple_impacts_same_mode(self):
        _clean_v32()
        tr = DecisionImpactTracker(experiment_context=COLD_CTX)
        for i in range(5):
            tr.record(make_impact(f"m{i}", mode="COLD", run_id="run_cold"),
                      experiment_context=COLD_CTX)
        assert len(tr._impacts) == 5
        tr2 = DecisionImpactTracker(experiment_context=COLD_CTX)
        assert len(tr2._impacts) == 5

    def test_impacts_isolated_across_runs_and_modes(self):
        _clean_v32()
        cold = DecisionImpactTracker(experiment_context=COLD_CTX)
        learned = DecisionImpactTracker(experiment_context=LEARNED_CTX)
        cold.record(make_impact("r1m", mode="COLD", run_id="run_cold"),
                    experiment_context=COLD_CTX)
        cold.record(make_impact("r2m", mode="COLD", run_id="run_sister"),
                    experiment_context=COLD_CTX_RUN2)
        learned.record(make_impact("lm", mode="LEARNED", run_id="run_learned"),
                       experiment_context=LEARNED_CTX)

        # Each context only sees its own namespace
        assert cold.find("r1m") is not None
        assert cold.find("r2m") is not None
        assert learned.find("lm") is not None
        # COLD context must NOT see LEARNED impacts
        assert DecisionImpactTracker(experiment_context=COLD_CTX).find("lm") is None
        # LEARNED context must NOT see COLD impacts
        assert DecisionImpactTracker(experiment_context=LEARNED_CTX).find("r1m") is None


# ---------------------------------------------------------------------------
# 3. record_outcome / record_outcomes persistence + reload
# ---------------------------------------------------------------------------


class TestOutcomePersistence:
    def test_record_outcome_updates_and_persists(self):
        _clean_v32()
        tr = DecisionImpactTracker(experiment_context=LEARNED_CTX)
        tr.record(make_impact("o1", mode="LEARNED", run_id="run_learned"),
                  experiment_context=LEARNED_CTX)
        tr.record_outcome(
            "o1", outcome="IMPROVED", attribution_level="PROBABLE",
            score_delta=0.12, execution_id="e2",
            evaluation={"score": 0.9}, verification={"passed": True},
            observed_outcome={"score": 0.9, "status": "COMPLETED"},
            comparative_outcome="SCORE_ABOVE_COLD")
        # Persisted
        data = _read_disk_file(LEARNED_CTX, "o1")
        assert data["outcome"] == "IMPROVED"
        assert data["attribution_level"] == "PROBABLE"
        assert data["score_delta"] == pytest.approx(0.12)
        assert data["outcome_improved"] is True
        # Evidence references persisted
        sources = {e["source"] for e in data["evidence"]}
        assert "execution" in sources
        assert "evaluation" in sources
        assert "verification" in sources

    def test_outcome_survives_reload(self):
        _clean_v32()
        a = DecisionImpactTracker(experiment_context=LEARNED_CTX)
        a.record(make_impact("o2", mode="LEARNED", run_id="run_learned"),
                 experiment_context=LEARNED_CTX)
        a.record_outcome("o2", outcome="WORSENED", attribution_level="POSSIBLE",
                         score_delta=-0.05)
        b = DecisionImpactTracker(experiment_context=LEARNED_CTX)
        imp = b.find("o2")
        assert imp.outcome == "WORSENED"
        assert imp.attribution_level == "POSSIBLE"
        assert imp.outcome_improved is False

    def test_record_outcomes_pipeline_persists(self):
        _clean_v32()
        tracker = DecisionImpactTracker(experiment_context=LEARNED_CTX)
        pipeline = UnifiedDecisionPipeline(
            cap_recommender=None, model_policy=None, plan_selector=None,
            replan_optimizer=None, governance=None, decision_tracker=tracker)
        tracker.record(make_impact("p1", mode="LEARNED", run_id="run_learned"),
                       experiment_context=LEARNED_CTX)
        pipeline.record_outcomes([
            {"decision_id": "p1", "outcome": "IMPROVED",
             "attribution_level": "CONFIRMED", "score_delta": 0.2}
        ])
        data = _read_disk_file(LEARNED_CTX, "p1")
        assert data["outcome"] == "IMPROVED"
        # A new instance sees the persisted outcome
        assert DecisionImpactTracker(experiment_context=LEARNED_CTX).find("p1").outcome == "IMPROVED"

    def test_record_outcomes_legacy_boolean_encoding(self):
        """Legacy outcome_improved True/False/None encoded losslessly."""
        _clean_v32()
        tr = DecisionImpactTracker(experiment_context=COLD_CTX)
        for did, val in (("l_true", True), ("l_false", False), ("l_none", None)):
            tr.record(make_impact(did, mode="COLD", run_id="run_cold"),
                      experiment_context=COLD_CTX)
        pipeline = UnifiedDecisionPipeline(
            cap_recommender=None, model_policy=None, plan_selector=None,
            replan_optimizer=None, governance=None, decision_tracker=tr)
        pipeline.record_outcomes([
            {"decision_id": "l_true", "outcome_improved": True},
            {"decision_id": "l_false", "outcome_improved": False},
            {"decision_id": "l_none", "outcome_improved": None},
        ])
        assert tr.find("l_true").outcome == "IMPROVED"
        assert tr.find("l_false").outcome == "WORSENED"
        assert tr.find("l_none").outcome == "UNKNOWN"
        assert tr.find("l_none").outcome_improved is None


# ---------------------------------------------------------------------------
# 4. COLD baseline recording
# ---------------------------------------------------------------------------


class TestColdBaseline:
    def test_cold_baseline_recorded_and_distinguishable(self):
        _clean_v32()
        tr = DecisionImpactTracker(experiment_context=COLD_CTX)
        baseline = make_impact("cold_base", mode="COLD", run_id="run_cold",
                               influenced_by_learning=False)
        tr.record(baseline, experiment_context=COLD_CTX)
        assert tr.find("cold_base") is not None
        assert tr.find("cold_base").mode == "COLD"
        assert tr.find("cold_base").influenced_by_learning is False

    def test_cold_vs_learned_kept_separate(self):
        _clean_v32()
        cold = DecisionImpactTracker(experiment_context=COLD_CTX)
        learned = DecisionImpactTracker(experiment_context=LEARNED_CTX)
        cold.record(make_impact("cb", mode="COLD", run_id="run_cold",
                                influenced_by_learning=False),
                    experiment_context=COLD_CTX)
        learned.record(make_impact("lb", mode="LEARNED", run_id="run_learned"),
                       experiment_context=LEARNED_CTX)
        # COLD baseline must not leak into LEARNED and vice versa
        assert DecisionImpactTracker(experiment_context=LEARNED_CTX).find("cb") is None
        assert DecisionImpactTracker(experiment_context=COLD_CTX).find("lb") is None
        # raw COLD baseline data exists with no artificial 0.5 injected
        data = _read_disk_file(COLD_CTX, "cb")
        assert data["mode"] == "COLD"
        assert "baseline_choice" in data
        assert data["outcome"] == "UNKNOWN"  # no artificial baseline outcome


# ---------------------------------------------------------------------------
# 5. Outcome semantics
# ---------------------------------------------------------------------------


class TestOutcomeSemantics:
    def test_explicit_outcomes_valid(self):
        for o in OUTCOMES:
            _clean_v32()
            tr = DecisionImpactTracker(experiment_context=COLD_CTX)
            tr.record(make_impact(f"o_{o}", mode="COLD", run_id="run_cold"),
                      experiment_context=COLD_CTX)
            tr.record_outcome(f"o_{o}", outcome=o)
            assert tr.find(f"o_{o}").outcome == o

    def test_invalid_outcome_rejected(self):
        with pytest.raises(ValueError, match="Invalid outcome"):
            DecisionImpact(
                decision_id="x", context_id="c", decision_type="model",
                baseline_choice="a", actual_choice="b",
                influenced_by_learning=False, outcome="BAD")

    def test_attribution_levels_valid(self):
        for lvl in ATTRIBUTION_LEVELS:
            _clean_v32()
            tr = DecisionImpactTracker(experiment_context=COLD_CTX)
            tr.record(make_impact(f"a_{lvl}", mode="COLD", run_id="run_cold"),
                      experiment_context=COLD_CTX)
            tr.record_outcome(f"a_{lvl}", outcome="IMPROVED", attribution_level=lvl)
            assert tr.find(f"a_{lvl}").attribution_level == lvl

    def test_attribution_separate_from_outcome(self):
        """attribution (confidence in cause) is distinct from outcome (result)."""
        _clean_v32()
        tr = DecisionImpactTracker(experiment_context=COLD_CTX)
        tr.record(make_impact("sep", mode="COLD", run_id="run_cold"),
                  experiment_context=COLD_CTX)
        tr.record_outcome("sep", outcome="IMPROVED", attribution_level="NONE")
        imp = tr.find("sep")
        assert imp.outcome == "IMPROVED"
        assert imp.attribution_level == "NONE"  # no causal claim

    def test_legacy_outcome_improved_preserved(self):
        imp = DecisionImpact(
            decision_id="x", context_id="c", decision_type="model",
            baseline_choice="a", actual_choice="b", influenced_by_learning=True,
            outcome_improved=True)
        assert imp.outcome_improved is True
        assert imp.outcome == "IMPROVED"

    def test_summary_aggregates_semantics(self):
        _clean_v32()
        tr = DecisionImpactTracker(experiment_context=COLD_CTX)
        outcomes = ["IMPROVED", "IMPROVED", "UNCHANGED", "WORSENED", "UNKNOWN"]
        for i, o in enumerate(outcomes):
            tr.record(make_impact(f"s{i}", mode="COLD", run_id="run_cold"),
                      experiment_context=COLD_CTX)
            tr.record_outcome(f"s{i}", outcome=o)
        summary = tr.get_summary()
        assert summary["improvements"] == 2
        assert summary["unchanged"] == 1
        assert summary["regressions"] == 1
        assert summary["unknown"] == 1


# ---------------------------------------------------------------------------
# 6. UNKNOWN semantics
# ---------------------------------------------------------------------------


class TestUnknownSemantics:
    def test_unknown_explicit_not_false(self):
        _clean_v32()
        tr = DecisionImpactTracker(experiment_context=COLD_CTX)
        tr.record(make_impact("u1", mode="COLD", run_id="run_cold",
                              outcome_improved=None),
                  experiment_context=COLD_CTX)
        imp = tr.find("u1")
        assert imp.outcome == "UNKNOWN"
        assert imp.outcome_improved is None  # never coerced to False

    def test_unknown_not_counted_as_worsened(self):
        _clean_v32()
        tr = DecisionImpactTracker(experiment_context=COLD_CTX)
        for did in ("uk1", "uk2"):
            tr.record(make_impact(did, mode="COLD", run_id="run_cold"),
                      experiment_context=COLD_CTX)
            tr.record_outcome(did, outcome="UNKNOWN")
        tr.record(make_impact("w1", mode="COLD", run_id="run_cold"),
                  experiment_context=COLD_CTX)
        tr.record_outcome("w1", outcome="WORSENED")
        summary = tr.get_summary()
        assert summary["regressions"] == 1
        assert summary["unknown"] == 2

    def test_unknown_does_not_trigger_learning_harm(self):
        """UNKNOWN must not be treated as regressed/harm in impact metrics."""
        _clean_v32()
        tr = DecisionImpactTracker(experiment_context=COLD_CTX)
        for did in ("h1", "h2"):
            tr.record(make_impact(did, mode="COLD", run_id="run_cold"),
                      experiment_context=COLD_CTX)
            tr.record_outcome(did, outcome="UNKNOWN")
        summary = tr.get_summary()
        assert summary["net_improvement"] == 0
        assert summary["regressions"] == 0


# ---------------------------------------------------------------------------
# 7. Evidence provenance traceability
# ---------------------------------------------------------------------------


class TestEvidenceProvenance:
    def test_outcome_references_execution_evaluation_verification(self):
        _clean_v32()
        tr = DecisionImpactTracker(experiment_context=LEARNED_CTX)
        tr.record(make_impact("ev1", mode="LEARNED", run_id="run_learned"),
                  experiment_context=LEARNED_CTX)
        tr.record_outcome(
            "ev1", outcome="IMPROVED", execution_id="exec-99",
            evaluation={"score": 0.92, "verdict": "pass"},
            verification={"correct": True, "retries": 0})
        imp = tr.find("ev1")
        sources = {e["source"] for e in imp.evidence}
        assert "execution" in sources
        assert "evaluation" in sources
        assert "verification" in sources
        # execution_id traceable on the evidence
        assert all(e.get("execution_id") == "exec-99" for e in imp.evidence)

    def test_outcome_provenance_survives_reload(self):
        _clean_v32()
        a = DecisionImpactTracker(experiment_context=LEARNED_CTX)
        a.record(make_impact("ev2", mode="LEARNED", run_id="run_learned"),
                 experiment_context=LEARNED_CTX)
        a.record_outcome("ev2", outcome="IMPROVED", execution_id="exec-77",
                         evaluation={"score": 0.8}, verification={"passed": True})
        b = DecisionImpactTracker(experiment_context=LEARNED_CTX)
        imp = b.find("ev2")
        sources = {e["source"] for e in imp.evidence}
        assert {"execution", "evaluation", "verification"} <= sources


# ---------------------------------------------------------------------------
# 8. COLD/LEARNED comparison fields kept separate
# ---------------------------------------------------------------------------


class TestComparisonSeparation:
    def test_decision_fields_kept_separate(self):
        _clean_v32()
        tr = DecisionImpactTracker(experiment_context=LEARNED_CTX)
        tr.record(DecisionImpact(
            decision_id="cmp", context_id="c", decision_type="model",
            baseline_choice="default_model", actual_choice="learned_model",
            influenced_by_learning=True,
            validation_run_id="run_learned", mode="LEARNED"),
            experiment_context=LEARNED_CTX)
        tr.record_outcome(
            "cmp", outcome="IMPROVED",
            observed_outcome={"learned_score": 0.8, "cold_score": 0.6},
            comparative_outcome="SCORE_ABOVE_COLD", attribution_level="POSSIBLE")
        imp = tr.find("cmp")
        # Separate fields — comparative comparison vs causal attribution
        assert imp.selected_decision == "learned_model"
        assert imp.baseline_decision == "default_model"
        assert imp.observed_outcome["learned_score"] == 0.8
        assert imp.comparative_outcome == "SCORE_ABOVE_COLD"
        # comparison does not auto-elevate attribution to causal
        assert imp.attribution_level == "POSSIBLE"


# ---------------------------------------------------------------------------
# 9. TEST isolation
# ---------------------------------------------------------------------------


class TestTestIsolation:
    def test_test_decisions_persist_as_evaluation_evidence_only(self):
        _clean_v32()
        tr = DecisionImpactTracker(experiment_context=TEST_CTX)
        tr.record(make_impact("td1", mode="TEST", run_id="run_test"),
                  experiment_context=TEST_CTX)
        tr.record_outcome("td1", outcome="IMPROVED")
        # Persisted into TEST namespace (evaluation evidence)
        data = _read_disk_file(TEST_CTX, "td1")
        assert data["mode"] == "TEST"
        # TEST must NOT appear in training (COLD/LEARNED) namespaces
        assert DecisionImpactTracker(experiment_context=COLD_CTX).find("td1") is None
        assert DecisionImpactTracker(experiment_context=LEARNED_CTX).find("td1") is None

    def test_test_outcomes_do_not_feed_training_summary(self):
        _clean_v32()
        test = DecisionImpactTracker(experiment_context=TEST_CTX)
        test.record(make_impact("tt1", mode="TEST", run_id="run_test"),
                    experiment_context=TEST_CTX)
        test.record_outcome("tt1", outcome="IMPROVED")
        learned = DecisionImpactTracker(experiment_context=LEARNED_CTX)
        assert learned.find("tt1") is None
        assert learned.get_summary()["total_decisions"] == 0


# ---------------------------------------------------------------------------
# 10. Fail-closed & backward compatibility (covered additionally)
# ---------------------------------------------------------------------------


class TestFailClosedAndCompat:
    def test_v32_impact_without_context_fails_closed(self):
        """The missing-context flat fallback never swallows a V3.2 scoped impact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tr = DecisionImpactTracker(storage_dir=tmpdir)
            v32 = DecisionImpact(
                decision_id="v", context_id="c", decision_type="model",
                baseline_choice="a", actual_choice="b",
                influenced_by_learning=True,
                validation_run_id="run_x", mode="COLD")
            with pytest.raises(V3IntegrationError, match="requires an ExperimentContext"):
                tr.record(v32)

    def test_tracker_context_prevents_fallback_even_if_impact_drops_it(self):
        """Once a tracker is experiment-scoped, recording NEVER falls back to
        shared flat V3.1 storage even if the caller forgets to pass context."""
        _clean_v32()
        # Tracker constructed with a context
        tr = DecisionImpactTracker(experiment_context=COLD_CTX)
        tr.record(make_impact("scoped_no_arg", mode="COLD", run_id="run_cold"))
        # Persisted to namespaced path, not flat decisions_v31
        data = _read_disk_file(COLD_CTX, "scoped_no_arg")
        assert data["mode"] == "COLD"
        flat_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "artifacts", "decisions_v31")
        assert not os.path.exists(os.path.join(flat_dir, "scoped_no_arg.json"))

    def test_legacy_flat_still_works(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tr = DecisionImpactTracker(storage_dir=tmpdir)
            tr.record(DecisionImpact(
                decision_id="lg", context_id="c", decision_type="model",
                baseline_choice="a", actual_choice="b",
                influenced_by_learning=False))
            assert tr.find("lg") is not None

    def test_from_dict_reconstructs_full_impact(self):
        imp = make_impact("fd", mode="LEARNED", run_id="run_learned",
                          outcome="IMPROVED", attribution_level="CONFIRMED")
        restored = DecisionImpact.from_dict(imp.to_dict())
        assert restored == imp
