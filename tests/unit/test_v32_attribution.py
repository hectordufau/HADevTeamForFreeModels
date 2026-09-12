"""Tests for V3.2 Phase 4 — Decision Attribution and Confidence.

Covers the deterministic attribution engine, confidence explainability, metrics
denominator semantics, isolation, persistence, TEST protection, comparative-vs-
causal discipline, and historical-artifact cleanup protection.
"""

import sys
import os
import json
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from harness.learning.context import ExperimentContext
from harness.learning.isolation import namespace_path
from harness.planner.v3_integration import (
    DecisionImpact,
    DecisionImpactTracker,
    ATTRIBUTION_LEVELS,
    OUTCOMES,
)
from harness.learning.attribution import (
    classify_attribution,
    AttributionResult,
    AttributionMetrics,
    AttributionInputError,
    document_thresholds,
    compute_attribution_confidence,
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


def ev(sources):
    """Build an evidence list with the given source kinds."""
    return [{"source": s} for s in sources]


COMPLETE = ev(["execution", "verification", "evaluation"])
PARTIAL = ev(["execution"])


def call(**kw):
    """Call classify_attribution with common defaults."""
    defaults = dict(
        decision_id="d", learning_influenced=True, outcome="UNKNOWN",
        baseline_decision="", selected_decision="", comparative_outcome="",
        evidence=[], retrieval_confidence=None, strategy_confidence=None,
        experience_ids=None, strategy_ids=None,
    )
    defaults.update(kw)
    return classify_attribution(**defaults)


RUN1_COLD = ExperimentContext(
    validation_run_id="run1", benchmark_id="bench", task_id="t1",
    execution_id="e1", mode="COLD")
RUN1_LEARNED = ExperimentContext(
    validation_run_id="run1", benchmark_id="bench", task_id="t2",
    execution_id="e2", mode="LEARNED")
RUN2_LEARNED = ExperimentContext(
    validation_run_id="run2", benchmark_id="bench", task_id="t3",
    execution_id="e3", mode="LEARNED")
TEST_CTX = ExperimentContext(
    validation_run_id="run_test", benchmark_id="bench", task_id="t4",
    execution_id="e4", mode="TEST")


def make_impact(did, ctx, *, influenced=True, outcome="IMPROVED",
                baseline="base", selected="learned",
                retrieval_conf=0.8, strategy_conf=0.8,
                evidence=COMPLETE, experience_ids=("exp1",),
                strategy_ids=("strat1",), **kw):
    base = dict(
        decision_id=did, context_id="c1", decision_type="model",
        baseline_choice=baseline, actual_choice=selected,
        influenced_by_learning=influenced,
        validation_run_id=ctx.validation_run_id,
        benchmark_id=ctx.benchmark_id, mode=ctx.mode,
        execution_id=ctx.execution_id,
        outcome=outcome, selected_decision=selected, baseline_decision=baseline,
        retrieval_confidence=retrieval_conf, strategy_confidence=strategy_conf,
        experience_ids=list(experience_ids or []),
        strategy_ids=list(strategy_ids or []),
        evidence=evidence,
    )
    base.update(kw)
    return DecisionImpact(**base)


# ---------------------------------------------------------------------------
# 1. Attribution levels: CONFIRMED / PROBABLE / POSSIBLE / NONE
# ---------------------------------------------------------------------------


class TestAttributionLevels:
    def test_none_when_not_influenced(self):
        r = call(learning_influenced=False, outcome="IMPROVED")
        assert r.attribution_level == "NONE"

    def test_none_when_decision_unchanged(self):
        r = call(learning_influenced=True, outcome="IMPROVED",
                 baseline_decision="x", selected_decision="x",
                 retrieval_confidence=0.9, evidence=COMPLETE)
        assert r.attribution_level == "NONE"

    def test_confirmed_full_chain_high_confidence(self):
        r = call(outcome="IMPROVED", baseline_decision="x",
                 selected_decision="y", experience_ids=["e"],
                 evidence=COMPLETE, retrieval_confidence=0.8,
                 strategy_confidence=0.85)
        assert r.attribution_level == "CONFIRMED"

    def test_probable_partial_chain_moderate_confidence(self):
        r = call(outcome="IMPROVED", baseline_decision="x",
                 selected_decision="y", experience_ids=["e"],
                 evidence=PARTIAL, retrieval_confidence=0.6)
        assert r.attribution_level == "PROBABLE"

    def test_possible_when_no_evidence(self):
        r = call(outcome="IMPROVED", baseline_decision="x",
                 selected_decision="y", experience_ids=["e"],
                 retrieval_confidence=0.99, evidence=[])
        assert r.attribution_level == "POSSIBLE"
        assert r.causal_evidence_insufficient is True

    def test_possible_when_outcome_unknown(self):
        r = call(outcome="UNKNOWN", baseline_decision="x",
                 selected_decision="y", experience_ids=["e"],
                 evidence=COMPLETE, retrieval_confidence=0.9)
        assert r.attribution_level == "POSSIBLE"

    def test_possible_when_weak_confidence(self):
        r = call(outcome="IMPROVED", baseline_decision="x",
                 selected_decision="y", experience_ids=["e"],
                 evidence=COMPLETE, retrieval_confidence=0.2,
                 strategy_confidence=0.1)
        assert r.attribution_level == "POSSIBLE"

    def test_possible_when_influence_unverified(self):
        # claims influence but no ids and no confidence identify what
        r = call(outcome="IMPROVED", baseline_decision="x",
                 selected_decision="y", experience_ids=None,
                 strategy_ids=None, retrieval_confidence=None,
                 strategy_confidence=None, evidence=COMPLETE)
        assert r.attribution_level == "POSSIBLE"


class TestLevelValidity:
    def test_attribution_levels_canonical(self):
        assert set(ATTRIBUTION_LEVELS) == {
            "CONFIRMED", "PROBABLE", "POSSIBLE", "NONE"}

    def test_invalid_level_rejected(self):
        with pytest.raises(ValueError):
            AttributionResult("d", "DEFINITELY", 0.5, [], input_digest={})


# ---------------------------------------------------------------------------
# 2. Outcome combos IMPROVED / UNCHANGED / WORSENED / UNKNOWN
# ---------------------------------------------------------------------------


class TestOutcomeCombos:
    @pytest.mark.parametrize("outcome", list(OUTCOMES))
    def test_all_outcomes_acceptable(self, outcome):
        r = call(outcome=outcome, baseline_decision="x",
                 selected_decision="y", experience_ids=["e"],
                 evidence=COMPLETE, retrieval_confidence=0.8)
        assert r.attribution_level in ATTRIBUTION_LEVELS

    def test_unknown_is_not_harm(self):
        m = AttributionMetrics(
            [call(decision_id="u", outcome="UNKNOWN",
                  baseline_decision="x", selected_decision="y",
                  experience_ids=["e"], evidence=PARTIAL,
                  retrieval_confidence=0.6)],
            {"u": "UNKNOWN"}).to_dict()
        assert m["unknown_decisions"] == 1
        assert m["worsened_decisions"] == 0
        assert m["learning_harm_rate"] == 0.0

    def test_unknown_counts_explicitly_only_in_unknown_counter(self):
        res = [
            call(decision_id="a", outcome="IMPROVED", baseline_decision="x",
                 selected_decision="y", experience_ids=["e"], evidence=COMPLETE,
                 retrieval_confidence=0.8),
            call(decision_id="b", outcome="WORSENED", baseline_decision="x",
                 selected_decision="y", experience_ids=["e"], evidence=COMPLETE,
                 retrieval_confidence=0.8),
            call(decision_id="c", outcome="UNKNOWN", baseline_decision="x",
                 selected_decision="y", experience_ids=["e"],
                 evidence=COMPLETE, retrieval_confidence=0.8),
        ]
        outcomes = {"a": "IMPROVED", "b": "WORSENED", "c": "UNKNOWN"}
        m = AttributionMetrics(res, outcomes).to_dict()
        assert m["improved_decisions"] == 1
        assert m["worsened_decisions"] == 1
        assert m["unknown_decisions"] == 1


# ---------------------------------------------------------------------------
# 3. Learning influence: influenced / not influenced / missing evidence
# ---------------------------------------------------------------------------


class TestLearningInfluence:
    def test_influenced_leads_to_attribution(self):
        r = call(outcome="IMPROVED", baseline_decision="x",
                 selected_decision="y", experience_ids=["e"],
                 evidence=COMPLETE, retrieval_confidence=0.8)
        assert r.attribution_level != "NONE"

    def test_not_influenced_is_none(self):
        assert call(learning_influenced=False).attribution_level == "NONE"

    def test_missing_influence_evidence_capped_possible(self):
        # influenced=true but nothing identifies the source -> POSSIBLE
        r = call(outcome="IMPROVED", baseline_decision="x",
                 selected_decision="y", experience_ids=None,
                 strategy_ids=None, retrieval_confidence=None,
                 strategy_confidence=None, evidence=COMPLETE)
        assert r.attribution_level == "POSSIBLE"
        assert r.causal_evidence_insufficient is True


# ---------------------------------------------------------------------------
# 4. Evidence quality: complete chain / partial / insufficient
# ---------------------------------------------------------------------------


class TestEvidenceQuality:
    def test_complete_chain_confirmed(self):
        r = call(outcome="IMPROVED", baseline_decision="x",
                 selected_decision="y", experience_ids=["e"],
                 evidence=COMPLETE, retrieval_confidence=0.8)
        assert r.attribution_level == "CONFIRMED"

    def test_partial_chain_probable(self):
        r = call(outcome="IMPROVED", baseline_decision="x",
                 selected_decision="y", experience_ids=["e"],
                 evidence=ev(["evaluation"]), retrieval_confidence=0.6)
        assert r.attribution_level == "PROBABLE"

    def test_insufficient_chain_possible(self):
        r = call(outcome="IMPROVED", baseline_decision="x",
                 selected_decision="y", experience_ids=["e"],
                 evidence=[], retrieval_confidence=0.9)
        assert r.attribution_level == "POSSIBLE"
        assert r.causal_evidence_insufficient is True


# ---------------------------------------------------------------------------
# 5. Confidence deterministic + rationale + no synthetic evidence
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_confidence_deterministic(self):
        a = compute_attribution_confidence(0.7, 0.8, decision_changed=True)
        b = compute_attribution_confidence(0.7, 0.8, decision_changed=True)
        assert a == b

    def test_confidence_in_range(self):
        c = compute_attribution_confidence(0.9, 0.9, decision_changed=True)
        assert 0.0 <= c <= 1.0

    def test_confidence_monotonic_in_confidence(self):
        low = compute_attribution_confidence(0.3, None, decision_changed=True)
        high = compute_attribution_confidence(0.9, None, decision_changed=True)
        assert high > low

    def test_rationale_present_and_explanatory(self):
        r = call(outcome="IMPROVED", baseline_decision="x",
                 selected_decision="y", experience_ids=["e"],
                 evidence=COMPLETE, retrieval_confidence=0.8)
        assert isinstance(r.rationale, list) and r.rationale
        assert any("CONFIRMED" not in x for x in r.rationale)  # reasons, not just labels
        assert len(r.rationale[0]) > 10

    def test_no_synthetic_evidence_in_digest(self):
        r = call(outcome="IMPROVED", baseline_decision="x",
                 selected_decision="y", experience_ids=["e"],
                 evidence=COMPLETE, retrieval_confidence=0.8)
        # the input digest reflects exactly what was passed; no invented scores
        assert r.input_digest["retrieval_confidence"] == 0.8
        assert r.input_digest["experience_ids"] == ["e"]
        # a decision with no comparative outcome was not claimed comparative
        assert r.input_digest["comparative_outcome"] == ""

    def test_thresholds_documented(self):
        t = document_thresholds()
        assert t["confident_causal_confidence"] == 0.70
        assert t["moderate_causal_confidence"] == 0.40
        assert "CONFIRMED" in t["naming"]

    def test_invalid_confidence_rejected(self):
        with pytest.raises(AttributionInputError):
            call(learning_influenced=True, outcome="IMPROVED",
                 retrieval_confidence=1.5)


# ---------------------------------------------------------------------------
# 6. Comparative vs causal: LEARNED>COLD does NOT auto become CONFIRMED
# ---------------------------------------------------------------------------


class TestComparativeVsCausal:
    def test_learned_beats_cold_alone_is_possible_not_confirmed(self):
        # Strong comparative signal, but NO evidence chain -> POSSIBLE
        r = call(outcome="IMPROVED", baseline_decision="x",
                 selected_decision="y", experience_ids=["e"],
                 comparative_outcome="LEARNED>COLD",
                 retrieval_confidence=0.99, strategy_confidence=0.99,
                 evidence=[])
        assert r.attribution_level == "POSSIBLE"
        assert r.attribution_level != "CONFIRMED"
        assert r.causal_evidence_insufficient is True
        # comparative signal itself is positive -> /an improvement in score/
        # but not counted as causal.

    def test_learned_beats_cold_still_needs_evidence_for_confirm(self):
        r = call(outcome="IMPROVED", baseline_decision="x",
                 selected_decision="y", experience_ids=["e"],
                 comparative_outcome="LEARNED>COLD",
                 retrieval_confidence=0.8,
                 evidence=COMPLETE)
        assert r.attribution_level == "CONFIRMED"
        assert r.comparative_used is True

    def test_learned_above_cold_not_harm(self):
        # A positive comparative result is never labeled harm; UNKNOWN and
        # IMPROVED are distinct and IMPROVED is never counted as a regression.
        r = call(outcome="IMPROVED", baseline_decision="x",
                 selected_decision="y", experience_ids=["e"],
                 comparative_outcome="LEARNED>COLD", evidence=PARTIAL,
                 retrieval_confidence=0.6)
        assert r.attribution_level in ("CONFIRMED", "PROBABLE", "POSSIBLE")
        assert r.attribution_level != "NONE"


# ---------------------------------------------------------------------------
# 7. Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def _populate(self):
        res = [
            call(decision_id="a", outcome="IMPROVED", baseline_decision="x",
                 selected_decision="y", experience_ids=["e"],
                 evidence=COMPLETE, retrieval_confidence=0.8),
            call(decision_id="b", outcome="IMPROVED", baseline_decision="x",
                 selected_decision="y", experience_ids=["e"],
                 evidence=COMPLETE, retrieval_confidence=0.8),
            call(decision_id="c", outcome="WORSENED", baseline_decision="x",
                 selected_decision="y", experience_ids=["e"],
                 evidence=COMPLETE, retrieval_confidence=0.8),
            call(decision_id="d", outcome="UNKNOWN", baseline_decision="x",
                 selected_decision="y", experience_ids=["e"],
                 evidence=COMPLETE, retrieval_confidence=0.8),
            call(decision_id="e", learning_influenced=False, outcome="IMPROVED"),
        ]
        outcomes = {"a": "IMPROVED", "b": "IMPROVED", "c": "WORSENED",
                    "d": "UNKNOWN", "e": "IMPROVED"}
        return res, outcomes

    def test_improvement_rate_denominator_is_influenced_set(self):
        res, outcomes = self._populate()
        m = AttributionMetrics(res, outcomes).to_dict()
        # learning influenced = 4 (a,b,c,d); e is NONE -> excluded
        assert m["learning_influenced_decisions"] == 4
        assert m["improved_decisions"] == 2
        assert m["worsened_decisions"] == 1
        assert m["unknown_decisions"] == 1
        assert m["decision_improvement_rate"] == round(2 / 4, 4)
        assert m["learning_harm_rate"] == round(1 / 4, 4)

    def test_unknown_not_forced_into_improvement_or_harm(self):
        res, outcomes = self._populate()
        m = AttributionMetrics(res, outcomes).to_dict()
        # unknown contributes to neither improved nor worsened numerator
        assert m["improved_decisions"] + m["worsened_decisions"] == 3
        assert m["unknown_decisions"] == 1

    def test_attribution_counts(self):
        res, outcomes = self._populate()
        m = AttributionMetrics(res, outcomes).to_dict()
        assert m["attribution_confirmed"] == 3  # a,b,c (d UNKNOWN -> possible)
        assert m["attribution_none"] == 1       # e
        assert m["attribution_probable"] == 0
        assert m["attribution_possible"] == 1   # d

    def test_zero_denominator_is_zero_not_error(self):
        m = AttributionMetrics([]).to_dict()
        assert m["decision_improvement_rate"] == 0.0
        assert m["learning_harm_rate"] == 0.0


# ---------------------------------------------------------------------------
# 8. Isolation: run A != run B, COLD != LEARNED, TEST != training
# ---------------------------------------------------------------------------


class TestIsolation:
    def test_cross_run_isolation(self):
        _clean_v32()
        ta = DecisionImpactTracker(experiment_context=RUN1_LEARNED)
        ta.record(make_impact("a1", RUN1_LEARNED))
        tb = DecisionImpactTracker(experiment_context=RUN2_LEARNED)
        tb.record(make_impact("b1", RUN2_LEARNED))
        # run A's tracker never sees run B's impact
        a_sums = ta.attribution_summary()
        assert a_sums["learning_influenced_decisions"] == 1
        assert ta.find("b1") is None
        assert tb.find("a1") is None

    def test_cold_vs_learned_isolated(self):
        _clean_v32()
        tc = DecisionImpactTracker(experiment_context=RUN1_COLD)
        # COLD baseline impact is recorded as not-influenced or with baseline
        tc.record(make_impact("cold1", RUN1_COLD, influenced=False,
                              selected="baseline", baseline="baseline"))
        tl = DecisionImpactTracker(experiment_context=RUN1_LEARNED)
        tl.record(make_impact("learn1", RUN1_LEARNED))
        assert tc.find("learn1") is None
        assert tl.find("cold1") is None

    def test_test_never_becomes_training(self):
        _clean_v32()
        tt = DecisionImpactTracker(experiment_context=TEST_CTX)
        tt.record(make_impact("t1", TEST_CTX))
        # TEST attribution may be computed but is excluded from training-side
        # learning_influenced metrics
        res = tt.run_attribution(tt.find("t1"))
        assert res.attribution_level in ATTRIBUTION_LEVELS
        summary = tt.attribution_summary()
        # attribution_summary excludes TEST mode by design
        assert summary["learning_influenced_decisions"] == 0


# ---------------------------------------------------------------------------
# 9. Persistence: attribution survives tracker/process reload
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_attribution_survives_reload(self):
        _clean_v32()
        ctx = RUN1_LEARNED
        a = DecisionImpactTracker(experiment_context=ctx)
        impact = make_impact("persist1", ctx)
        a.record(impact)
        a.run_attribution(impact)
        assert impact.attribution_level == "CONFIRMED"
        # new tracker reloads from disk with attribution retained
        b = DecisionImpactTracker(experiment_context=ctx)
        loaded = b.find("persist1")
        assert loaded.attribution_level == "CONFIRMED"
        assert loaded.attribution_confidence == impact.attribution_confidence
        assert loaded.attribution_rationale
        assert loaded.retrieval_confidence == 0.8
        assert loaded.experience_ids == ["exp1"]

    def test_attribution_persisted_to_disk(self):
        _clean_v32()
        ctx = RUN1_LEARNED
        a = DecisionImpactTracker(experiment_context=ctx)
        impact = make_impact("p2", ctx)
        a.record(impact)
        a.run_attribution(impact)
        store_dir = namespace_path("decisions", experiment_context=ctx)
        path = os.path.join(store_dir, "p2.json")
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["attribution_level"] == "CONFIRMED"
        assert data["attribution_confidence"] > 0


# ---------------------------------------------------------------------------
# 10. Runner-level integration: attribution_summary from tracker
# ---------------------------------------------------------------------------


class TestTrackerIntegration:
    def test_attribution_summary_via_tracker(self):
        _clean_v32()
        ctx = RUN1_LEARNED
        t = DecisionImpactTracker(experiment_context=ctx)
        t.record(make_impact("s1", ctx, outcome="IMPROVED"))
        t.record(make_impact("s2", ctx, outcome="UNKNOWN"))
        m = t.attribution_summary()
        assert m["learning_influenced_decisions"] == 2
        assert m["improved_decisions"] == 1
        assert m["unknown_decisions"] == 1
        assert m["attribution_confirmed"] == 1  # s1 (complete)
        assert m["attribution_possible"] == 1   # s2 (UNKNOWN outcome)

    def test_run_attributions_persists_all(self):
        _clean_v32()
        ctx = RUN1_LEARNED
        t = DecisionImpactTracker(experiment_context=ctx)
        t.record(make_impact("r1", ctx, outcome="IMPROVED"))
        t.record(make_impact("r2", ctx, outcome="WORSENED"))
        res = t.run_attributions()
        assert set(res.keys()) == {"r1", "r2"}
        assert res["r1"].attribution_level == "CONFIRMED"


# ---------------------------------------------------------------------------
# 11. Historical artifact protection (regression for cleanup defect)
# ---------------------------------------------------------------------------


class TestHistoricalArtifactProtection:
    def test_cleanup_never_removes_git_tracked_artifacts(self):
        # Reuses the guarded cleaner from the storage-isolation module (same
        # defect that previously removed the 28 restored V3.0/V3.1 artifacts).
        from tests.unit.test_v32_storage_isolation import _clean_all_artifacts
        import subprocess
        repo = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        tracked = subprocess.check_output(
            ["git", "ls-files", "artifacts/"], cwd=repo,
            stderr=subprocess.DEVNULL).decode().splitlines()
        assert len(tracked) >= 28, "expected the protected historical artifacts"
        # Run the guarded cleanup.
        _clean_all_artifacts()
        for rel in tracked:
            assert os.path.exists(os.path.join(repo, rel)), (
                f"historical artifact deleted by cleanup: {rel}")

    def test_all_tracked_historical_artifacts_intact(self):
        # Direct invariant check: no git-tracked artifact is missing from disk.
        import subprocess
        repo = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        tracked = subprocess.check_output(
            ["git", "ls-files", "artifacts/"], cwd=repo,
            stderr=subprocess.DEVNULL).decode().splitlines()
        missing = [r for r in tracked
                   if not os.path.exists(os.path.join(repo, r))]
        assert missing == [], f"missing tracked artifacts: {missing}"
