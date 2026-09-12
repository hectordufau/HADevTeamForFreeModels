"""Tests for V3.2 Storage Isolation (Phase 2)."""

import sys
import os
import json
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from harness.learning.context import ExperimentContext
from harness.learning.isolation import (
    namespace_path,
    verify_isolation,
    retrieve_filtered,
    check_test_protection,
    get_v32_base,
    find_namespaced_dirs,
)
from harness.learning.experience import (
    StructuredExperience,
    ExperiencePipeline,
    ExecutionTrace,
    Evidence,
)
from harness.learning.strategy import Strategy, StrategyStore
from harness.planner.v3_integration import (
    DecisionImpact, DecisionImpactTracker, V3IntegrationError,
)
from harness.evidence.unified import (
    CanonicalExecutionRecord,
    EvidencePackage,
    EvidenceStore,
    EvidenceQuery,
)
from harness.planner.model_intelligence import ModelExperiment, ExperimentTracker, ModelProfileStore
from harness.routing.performance_v3 import PerformanceSample, PerformanceRegistryV3
from harness.memory.v3 import ExperienceRecord, ExperienceStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_v32():
    """Clean up V3.2 artifacts to avoid cross-test contamination."""
    base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "artifacts", "v3.2"
    )
    if os.path.exists(base):
        shutil.rmtree(base, ignore_errors=True)


def _git_tracked_artifact_paths(artifacts_dir):
    """Return the set of paths under artifacts/ that are tracked by git.

    These are the historical V3.0/V3.1 benchmark artifacts (PROTECTED) and the
    V3 performance registry. The test cleanup must NEVER delete/rewrite them.
    The set is derived from `git ls-files artifacts/` (run from the repo root)
    so it stays correct even as tracked files change; if git is unavailable or
    the repo is not resolvable it returns None and the caller must be
    conservative (refuse to delete).
    """
    import subprocess
    # Resolve the repo root as the parent of artifacts/ (artifacts/ is always
    # a first-level dir of the repo for this project).
    repo_root = os.path.dirname(artifacts_dir)
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "artifacts/"], cwd=repo_root,
            stderr=subprocess.DEVNULL).decode().splitlines()
    except Exception:
        return None  # unknown -> caller must be conservative
    tracked = set()
    for rel in out:
        abs_ = os.path.join(repo_root, *rel.split("/"))
        parts = rel.split("/")
        for i in range(1, len(parts)):
            tracked.add(os.path.join(repo_root, *parts[:i]))
        tracked.add(abs_)
    return tracked


def _clean_all_artifacts():
    """Clean up V3.2 artifact directories for cross-test isolation.

    SAFETY: this must never delete Git-tracked historical benchmark artifacts
    (the 28 restored V3.0/V3.1 artifacts under artifacts/experiences_v3/,
    artifacts/gen_experiences/, artifacts/exploration_results/,
    artifacts/experiences/, and artifacts/performance_v3/registry.json). Those
    are PROTECTED. We only remove directories that contain NO git-tracked
    files (e.g. artifacts/v3.2/ test scratch space).
    """
    artifacts = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "artifacts"
    )
    if not os.path.exists(artifacts):
        return
    tracked = _git_tracked_artifact_paths(artifacts)
    # If tracking is unknown, refuse to delete anything (conservative).
    if tracked is None:
        return
    for name in os.listdir(artifacts):
        path = os.path.join(artifacts, name)
        if not os.path.isdir(path):
            continue
        if path in tracked:
            continue  # PROTECTED: contains git-tracked historical artifacts
        shutil.rmtree(path, ignore_errors=True)


# Clean before ALL tests in this module to ensure clean state
_clean_all_artifacts()

COLD_CTX = ExperimentContext(
    validation_run_id="run_cold",
    benchmark_id="bench",
    task_id="t1",
    execution_id="e1",
    mode="COLD",
)

LEARNED_CTX = ExperimentContext(
    validation_run_id="run_learned",
    benchmark_id="bench",
    task_id="t2",
    execution_id="e2",
    mode="LEARNED",
)

TEST_CTX = ExperimentContext(
    validation_run_id="run_test",
    benchmark_id="bench",
    task_id="t3",
    execution_id="e3",
    mode="TEST",
)

COLD_CTX_RUN2 = ExperimentContext(
    validation_run_id="run_sister",
    benchmark_id="bench",
    task_id="t4",
    execution_id="e4",
    mode="COLD",
)


def ns_path(ctx, store_type):
    """Shorthand for namespace_path."""
    return namespace_path(store_type, experiment_context=ctx)


def make_exp_record(task_id="t1", mode="COLD", run_id="run_cold"):
    return ExperienceRecord(
        task_id=task_id,
        task_objective=f"Task {task_id}",
        capabilities_used=["code_gen"],
        result_status="COMPLETED",
        score=0.85,
        duration_seconds=10.0,
        lessons=["Lesson 1"],
    )


def make_structured_exp(task_id="t1", mode="COLD", run_id="run_cold"):
    return StructuredExperience(
        task_id=task_id,
        task_context="Test task",
        capabilities_used=["code_gen"],
        strategy={},
        outcome={"status": "COMPLETED", "score": 0.85, "duration": 10.0},
    )


def make_strategy(sid="s1", task_class="code_gen", mode="COLD", run_id="run_cold"):
    return Strategy(
        strategy_id=sid,
        task_class=task_class,
        capabilities=["code_gen"],
        conditions={},
        workflow={"name": "default", "steps": [], "ordering": "sequential"},
        validation_run_id=run_id,
        mode=mode,
    )


def make_decision_impact(did="d1", mode="COLD", run_id="run_cold"):
    return DecisionImpact(
        decision_id=did,
        context_id="c1",
        decision_type="model",
        baseline_choice="a",
        actual_choice="b",
        influenced_by_learning=True,
        validation_run_id=run_id,
        mode=mode,
    )


def make_evidence_pkg(nid="n1", mode="COLD", run_id="run_cold"):
    return EvidencePackage(
        node_id=nid,
        capability="code_gen",
        agent="agent_a",
        task_id="t1",
        execution_id="e1",
        validation_run_id=run_id,
        mode=mode,
    )


def make_canonical_record(eid="e1", mode="COLD", run_id="run_cold"):
    return CanonicalExecutionRecord(
        task_id="t1",
        execution_id=eid,
        workflow_name="wf1",
        nodes=[],
        capability_requirements={},
        assignments={},
        results={},
        evidence_hashes=[],
        validation_run_id=run_id,
        mode=mode,
    )


def make_model_experiment(eid="me1", mode="COLD", run_id="run_cold"):
    return ModelExperiment(
        experiment_id=eid,
        model_id="m1",
        capability="code_gen",
        task_id="t1",
        result_score=0.85,
        is_exploration=False,
        validation_run_id=run_id,
        mode=mode,
    )


def make_perf_sample(mode="COLD", run_id="run_cold"):
    return PerformanceSample(
        task_id="t1",
        model_id="m1",
        capability="code_gen",
        success=True,
        score=0.95,
        latency_ms=100.0,
        iterations=1,
        validation_run_id=run_id,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Namespace Path Construction
# ---------------------------------------------------------------------------

class TestNamespacePaths:
    """Test that namespace_path builds correct paths."""

    def test_namespace_path_cold(self):
        path = ns_path(COLD_CTX, "experiences")
        assert path.endswith("v3.2/run_cold/COLD/experiences")

    def test_namespace_path_learned(self):
        path = ns_path(LEARNED_CTX, "experiences")
        assert path.endswith("v3.2/run_learned/LEARNED/experiences")

    def test_namespace_path_test(self):
        path = ns_path(TEST_CTX, "experiences")
        assert path.endswith("v3.2/run_test/TEST/experiences")

    def test_namespace_path_all_store_types(self):
        for store in ("experiences", "strategies", "decisions", "evidence",
                      "experiments", "performance", "plan_learning"):
            path = ns_path(COLD_CTX, store)
            assert path.endswith(f"v3.2/run_cold/COLD/{store}")

    def test_namespace_path_different_run_ids(self):
        cold_path = ns_path(COLD_CTX, "experiences")
        sister_path = ns_path(COLD_CTX_RUN2, "experiences")
        assert cold_path != sister_path
        assert "run_cold" in cold_path
        assert "run_sister" in sister_path

    def test_namespace_path_missing_context_raises(self):
        with pytest.raises(ValueError, match="ExperimentContext required"):
            namespace_path("experiences")

    def test_namespace_path_invalid_mode_raises(self):
        """Creating an ExperimentContext with invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            ExperimentContext(
                validation_run_id="r1",
                benchmark_id="b",
                task_id="t",
                execution_id="e",
                mode="INVALID",
            )

    def test_namespace_path_create_dirs(self):
        path = ns_path(TEST_CTX, "experiences")
        assert os.path.exists(path)
        # Subdirs created
        assert os.path.exists(os.path.dirname(os.path.dirname(path)))

    def test_find_namespaced_dirs_exact(self):
        """find_namespaced_dirs returns exact match when both params given."""
        path = ns_path(LEARNED_CTX, "strategies")
        os.makedirs(path, exist_ok=True)
        found = find_namespaced_dirs("strategies", validation_run_id="run_learned", mode="LEARNED")
        assert len(found) == 1
        assert found[0] == path

    def test_find_namespaced_dirs_empty(self):
        found = find_namespaced_dirs("nonexistent")
        assert found == []


# ---------------------------------------------------------------------------
# Write Isolation — COLD writes don't appear in LEARNED
# ---------------------------------------------------------------------------

class TestWriteIsolation:
    """Test that writes in one mode don't leak into another."""

    def test_experience_pipeline_write_isolation(self):
        pipe = ExperiencePipeline()
        exp = make_structured_exp()
        pipe._store(exp, experiment_context=COLD_CTX)
        # Should not be in LEARNED path
        cold_dir = ns_path(COLD_CTX, "experiences")
        learned_dir = ns_path(LEARNED_CTX, "experiences")
        assert os.path.exists(os.path.join(cold_dir, "t1.json"))
        assert not os.path.exists(os.path.join(learned_dir, "t1.json"))

    def test_strategy_store_write_isolation(self):
        store = StrategyStore()
        s = make_strategy("s1", "code_gen", "COLD", "run_cold")
        store.save(s, experiment_context=COLD_CTX)
        cold_dir = ns_path(COLD_CTX, "strategies")
        learned_dir = ns_path(LEARNED_CTX, "strategies")
        assert os.path.exists(os.path.join(cold_dir, "s1.json"))
        assert not os.path.exists(os.path.join(learned_dir, "s1.json"))

    def test_decision_tracker_write_isolation(self):
        tracker = DecisionImpactTracker()
        impact = make_decision_impact("d1")
        tracker.record(impact, experiment_context=COLD_CTX)
        cold_dir = ns_path(COLD_CTX, "decisions")
        learned_dir = ns_path(LEARNED_CTX, "decisions")
        assert os.path.exists(os.path.join(cold_dir, "d1.json"))
        assert not os.path.exists(os.path.join(learned_dir, "d1.json"))

    def test_evidence_store_write_isolation(self):
        store = EvidenceStore()
        pkg = make_evidence_pkg("n1")
        store.store_evidence(pkg, experiment_context=COLD_CTX)
        cold_dir = ns_path(COLD_CTX, "evidence")
        learned_dir = ns_path(LEARNED_CTX, "evidence")
        assert os.path.exists(os.path.join(cold_dir, "packages", "t1", "e1", "n1.json"))
        assert not os.path.exists(os.path.join(learned_dir, "packages", "t1", "e1", "n1.json"))

    def test_experiment_tracker_write_isolation(self):
        tracker = ExperimentTracker()
        me = make_model_experiment("me1")
        tracker.record(me, experiment_context=COLD_CTX)
        cold_dir = ns_path(COLD_CTX, "experiments")
        learned_dir = ns_path(LEARNED_CTX, "experiments")
        assert os.path.exists(os.path.join(cold_dir, "me1.json"))
        assert not os.path.exists(os.path.join(learned_dir, "me1.json"))

    def test_v3_experience_store_write_isolation(self):
        store = ExperienceStore()
        exp = make_exp_record()
        store.store(exp, experiment_context=COLD_CTX)
        cold_dir = ns_path(COLD_CTX, "experiences")
        learned_dir = ns_path(LEARNED_CTX, "experiences")
        assert os.path.exists(os.path.join(cold_dir, "t1.json"))
        assert not os.path.exists(os.path.join(learned_dir, "t1.json"))


# ---------------------------------------------------------------------------
# Retrieval Isolation — COLD can't retrieve LEARNED
# ---------------------------------------------------------------------------

class TestRetrievalIsolation:
    """Test that retrieval from one mode doesn't return data from another."""

    def test_experience_pipeline_retrieval_isolation(self):
        pipe = ExperiencePipeline()
        pipe._store(make_structured_exp("cold_task"), experiment_context=COLD_CTX)
        pipe._store(make_structured_exp("learned_task"), experiment_context=LEARNED_CTX)

        # COLD can't retrieve LEARNED
        assert pipe.retrieve("cold_task", experiment_context=COLD_CTX) is not None
        assert pipe.retrieve("learned_task", experiment_context=COLD_CTX) is None
        # LEARNED can't retrieve COLD
        assert pipe.retrieve("learned_task", experiment_context=LEARNED_CTX) is not None
        assert pipe.retrieve("cold_task", experiment_context=LEARNED_CTX) is None

    def test_strategy_store_retrieval_isolation(self):
        store = StrategyStore()
        store.save(make_strategy("cold_s", mode="COLD", run_id="run_cold"), experiment_context=COLD_CTX)
        store.save(make_strategy("learned_s", mode="LEARNED", run_id="run_learned"), experiment_context=LEARNED_CTX)

        assert store.load("cold_s", experiment_context=COLD_CTX) is not None
        assert store.load("learned_s", experiment_context=COLD_CTX) is None
        assert store.load("learned_s", experiment_context=LEARNED_CTX) is not None

    def test_strategy_store_list_isolation(self):
        store = StrategyStore()
        store.save(make_strategy("s2", mode="COLD", run_id="run_cold"), experiment_context=COLD_CTX)
        store.save(make_strategy("s3", mode="LEARNED", run_id="run_learned"), experiment_context=LEARNED_CTX)
        assert len(store.list_strategies(experiment_context=COLD_CTX)) == 1
        assert len(store.list_strategies(experiment_context=LEARNED_CTX)) == 1

    def test_evidence_store_retrieval_isolation(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EvidenceStore(storage_dir=tmpdir)
            store.store_evidence(make_evidence_pkg("n1", mode="COLD", run_id="run_cold"), experiment_context=COLD_CTX)
            store.store_evidence(make_evidence_pkg("n2", mode="LEARNED", run_id="run_learned"), experiment_context=LEARNED_CTX)

            # Same task/execution but different node → 2 packages in flat view
            assert len(store.get_evidence("t1", "e1", experiment_context=COLD_CTX)) == 1
            assert len(store.get_evidence("t1", "e1", experiment_context=LEARNED_CTX)) == 1

            # COLD context: only one package returned
            pkgs_cold = store.get_evidence("t1", "e1", experiment_context=COLD_CTX)
            assert all(p.mode == "COLD" for p in pkgs_cold)
            assert pkgs_cold[0].node_id == "n1"

            pkgs_learned = store.get_evidence("t1", "e1", experiment_context=LEARNED_CTX)
            assert all(p.mode == "LEARNED" for p in pkgs_learned)
            assert pkgs_learned[0].node_id == "n2"

    def test_experiment_tracker_retrieval_isolation(self):
        tracker = ExperimentTracker()
        tracker.record(make_model_experiment("cold_exp", mode="COLD", run_id="run_cold"), experiment_context=COLD_CTX)
        tracker.record(make_model_experiment("learned_exp", mode="LEARNED", run_id="run_learned"), experiment_context=LEARNED_CTX)

        assert len(tracker.get_results("m1", "code_gen", experiment_context=COLD_CTX)) == 1
        assert len(tracker.get_results("m1", "code_gen", experiment_context=LEARNED_CTX)) == 1

    def test_v3_experience_store_retrieval_isolation(self):
        store = ExperienceStore()
        store.store(make_exp_record("cold_rec"), experiment_context=COLD_CTX)
        store.store(make_exp_record("learned_rec"), experiment_context=LEARNED_CTX)

        assert store.retrieve("cold_rec", experiment_context=COLD_CTX) is not None
        assert store.retrieve("learned_rec", experiment_context=COLD_CTX) is None

    def test_v3_experience_store_search_isolation(self):
        store = ExperienceStore()
        store.store(make_exp_record("cold_rec"), experiment_context=COLD_CTX)
        store.store(make_exp_record("learned_rec"), experiment_context=LEARNED_CTX)

        cold_results = store.search("Task", experiment_context=COLD_CTX)
        learned_results = store.search("Task", experiment_context=LEARNED_CTX)
        assert len(cold_results) == 1
        assert len(learned_results) == 1


# ---------------------------------------------------------------------------
# Cross-Run Isolation
# ---------------------------------------------------------------------------

class TestCrossRunIsolation:
    """Test that artifacts from Run A are invisible to Run B."""

    def test_cross_run_experience_pipeline(self):
        pipe = ExperiencePipeline()
        pipe._store(make_structured_exp("task_run1"), experiment_context=COLD_CTX)
        pipe._store(make_structured_exp("task_run2"), experiment_context=COLD_CTX_RUN2)

        assert pipe.retrieve("task_run1", experiment_context=COLD_CTX) is not None
        assert pipe.retrieve("task_run2", experiment_context=COLD_CTX) is None
        assert pipe.retrieve("task_run2", experiment_context=COLD_CTX_RUN2) is not None

    def test_cross_run_stategy_store(self):
        store = StrategyStore()
        store.save(make_strategy("s_a", mode="COLD", run_id="run_cold"), experiment_context=COLD_CTX)
        store.save(make_strategy("s_b", mode="COLD", run_id="run_sister"), experiment_context=COLD_CTX_RUN2)

        assert len(store.list_strategies(experiment_context=COLD_CTX)) == 1
        assert len(store.list_strategies(experiment_context=COLD_CTX_RUN2)) == 1

    def test_cross_run_evidence_store(self):
        store = EvidenceStore()
        store.store_evidence(
            make_evidence_pkg("n1", mode="COLD", run_id="run_cold"),
            experiment_context=COLD_CTX,
        )
        store.store_evidence(
            make_evidence_pkg("n1", mode="COLD", run_id="run_sister"),
            experiment_context=COLD_CTX_RUN2,
        )

        pkgs_cold = store.get_evidence("t1", "e1", experiment_context=COLD_CTX)
        pkgs_sister = store.get_evidence("t1", "e1", experiment_context=COLD_CTX_RUN2)
        assert len(pkgs_cold) == 1
        assert len(pkgs_sister) == 1
        assert pkgs_cold[0].validation_run_id == "run_cold"
        assert pkgs_sister[0].validation_run_id == "run_sister"


# ---------------------------------------------------------------------------
# TEST Protection
# ---------------------------------------------------------------------------

class TestTestProtection:
    """Test that TEST-mode artifacts never enter training/learning stores."""

    def test_check_test_protection_learn_raises(self):
        """check_test_protection raises for learn operation in TEST mode."""
        with pytest.raises(ValueError, match="TEST-mode artifacts cannot enter"):
            check_test_protection(TEST_CTX, operation="learn")

    def test_check_test_protection_train_raises(self):
        with pytest.raises(ValueError, match="TEST-mode artifacts cannot enter"):
            check_test_protection(TEST_CTX, operation="train")

    def test_check_test_protection_store_ok(self):
        """store is always allowed even in TEST mode."""
        check_test_protection(TEST_CTX, operation="store")  # should not raise

    def test_check_test_protection_none_ok(self):
        """None context never raises."""
        check_test_protection(None, operation="train")  # should not raise

    def test_check_test_protection_cold_ok(self):
        """COLD mode allows learn."""
        check_test_protection(COLD_CTX, operation="learn")  # should not raise

    def test_v3_experience_store_test_protection(self):
        """ExperienceStore.store with TEST context doesn't raise since store is allowed."""
        store = ExperienceStore()
        # Should not raise — store is a valid operation
        store.store(make_exp_record("test_task"), experiment_context=TEST_CTX)
        # Verify it was stored in TEST namespace
        test_dir = ns_path(TEST_CTX, "experiences")
        assert os.path.exists(os.path.join(test_dir, "test_task.json"))

    def test_strategy_store_test_protection(self):
        """StrategyStore.save with TEST context should work."""
        store = StrategyStore()
        store.save(make_strategy("test_s"), experiment_context=TEST_CTX)

    def test_pipeline_test_protection(self):
        """Pipeline._store with TEST context should work."""
        pipe = ExperiencePipeline()
        pipe._store(make_structured_exp("test_exp"), experiment_context=TEST_CTX)
        test_dir = ns_path(TEST_CTX, "experiences")
        assert os.path.exists(os.path.join(test_dir, "test_exp.json"))

    def test_evidence_store_test_protection(self):
        """EvidenceStore with TEST context should work."""
        store = EvidenceStore()
        store.store_evidence(make_evidence_pkg("test_pkg"), experiment_context=TEST_CTX)
        test_dir = ns_path(TEST_CTX, "evidence")
        assert os.path.exists(os.path.join(test_dir, "packages", "t1", "e1", "test_pkg.json"))

    def test_experiment_tracker_test_protection(self):
        """ExperimentTracker with TEST context should work."""
        tracker = ExperimentTracker()
        tracker.record(make_model_experiment("test_me"), experiment_context=TEST_CTX)


# ---------------------------------------------------------------------------
# Fail-Closed Behavior
# ---------------------------------------------------------------------------

class TestFailClosedBehavior:
    """Test that missing context causes error or defaults safely."""

    def test_namespace_path_no_context_raises(self):
        """No context + no manual params raises."""
        with pytest.raises(ValueError, match="ExperimentContext required"):
            namespace_path("experiences")

    def test_namespace_path_no_mode_raises(self):
        """Only run_id without mode raises."""
        with pytest.raises(ValueError, match="ExperimentContext required"):
            namespace_path("experiences", validation_run_id="r1")

    def test_pipeline_without_context_uses_v31_dir(self):
        """No context → falls back to V3.1 flat storage."""
        pipe = ExperiencePipeline()
        exp = make_structured_exp("legacy_task")
        pipe._store(exp)  # should not raise
        # V3.1 dir
        v31_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "artifacts", "experiences_v31"
        )
        assert os.path.exists(os.path.join(v31_dir, "legacy_task.json"))

    def test_strategy_store_without_context_uses_flat_dir(self):
        store = StrategyStore()
        s = make_strategy("legacy_s")
        store.save(s)  # should not raise
        flat_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "artifacts", "strategies"
        )
        assert os.path.exists(os.path.join(flat_dir, "legacy_s.json"))

    def test_evidence_store_without_context_uses_flat_dir(self):
        store = EvidenceStore()
        store.store_evidence(make_evidence_pkg("legacy_pkg"))
        flat_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "artifacts", "evidence_v3"
        )
        assert os.path.exists(os.path.join(flat_dir, "packages", "t1", "e1", "legacy_pkg.json"))


# ---------------------------------------------------------------------------
# Isolation Verifier
# ---------------------------------------------------------------------------

class TestIsolationVerifier:
    """Test that verify_isolation detects contamination."""

    def test_verify_isolation_clean(self):
        """Fresh environment has no contamination."""
        result = verify_isolation()
        assert result["passed"] is True
        assert result["contamination_count"] == 0

    def test_verify_isolation_detects_cold_learned_contamination(self):
        """Same artifact in COLD and LEARNED = contamination."""
        os.makedirs(ns_path(COLD_CTX, "experiences"), exist_ok=True)
        os.makedirs(ns_path(LEARNED_CTX, "experiences"), exist_ok=True)
        # Write same filename to both
        with open(os.path.join(ns_path(COLD_CTX, "experiences"), "same.json"), "w") as f:
            json.dump({"data": "cold"}, f)
        with open(os.path.join(ns_path(LEARNED_CTX, "experiences"), "same.json"), "w") as f:
            json.dump({"data": "learned"}, f)

        result = verify_isolation()
        assert result["passed"] is False
        assert result["contamination_count"] > 0
        assert result["checks"]["cold_learned"] > 0 or result["checks"]["learned_cold"] > 0

    def test_verify_isolation_detects_test_training(self):
        """Same filename in TEST and COLD = test_training contamination."""
        os.makedirs(ns_path(COLD_CTX, "experiences"), exist_ok=True)
        os.makedirs(ns_path(TEST_CTX, "experiences"), exist_ok=True)
        with open(os.path.join(ns_path(COLD_CTX, "experiences"), "shared.json"), "w") as f:
            json.dump({"data": "cold"}, f)
        with open(os.path.join(ns_path(TEST_CTX, "experiences"), "shared.json"), "w") as f:
            json.dump({"data": "test"}, f)

        result = verify_isolation()
        assert result["passed"] is False
        assert result["checks"]["test_training"] > 0

    def test_verify_isolation_detects_cross_run(self):
        """Same artifact in different runs = cross_run contamination."""
        os.makedirs(ns_path(COLD_CTX, "experiences"), exist_ok=True)
        os.makedirs(ns_path(COLD_CTX_RUN2, "experiences"), exist_ok=True)
        with open(os.path.join(ns_path(COLD_CTX, "experiences"), "cross.json"), "w") as f:
            json.dump({"data": "run1"}, f)
        with open(os.path.join(ns_path(COLD_CTX_RUN2, "experiences"), "cross.json"), "w") as f:
            json.dump({"data": "run2"}, f)

        result = verify_isolation()
        assert result["passed"] is False
        assert result["checks"]["cross_run"] > 0

    def test_verify_isolation_partial_clean(self):
        """Separate artifact names should be clean."""
        os.makedirs(ns_path(COLD_CTX, "experiences"), exist_ok=True)
        os.makedirs(ns_path(LEARNED_CTX, "experiences"), exist_ok=True)
        with open(os.path.join(ns_path(COLD_CTX, "experiences"), "a.json"), "w") as f:
            json.dump({"data": "cold"}, f)
        with open(os.path.join(ns_path(LEARNED_CTX, "experiences"), "b.json"), "w") as f:
            json.dump({"data": "learned"}, f)

        result = verify_isolation()
        # Different filenames → no cross-contamination detected
        assert result["passed"] is True
        assert result["contamination_count"] == 0


# ---------------------------------------------------------------------------
# Test plan_learning store isolation
# ---------------------------------------------------------------------------

class TestPlanLearningStoreIsolation:
    """Test PlanLearningStore namespace isolation."""

    def test_plan_learning_record_outcome_namespaced(self):
        from harness.planner.plan_intelligence import PlanScore, PlanLearningStore
        _clean_v32()
        store = PlanLearningStore()
        score = PlanScore(overall=0.8, capability_coverage=0.7, agent_fit=0.6,
                          model_quality=0.9, efficiency=0.5, risk=0.2)
        store.record_outcome("plan1", score, "COMPLETED", experiment_context=LEARNED_CTX)
        cold_dir = ns_path(COLD_CTX, "plan_learning")
        learned_dir = ns_path(LEARNED_CTX, "plan_learning")
        assert os.path.exists(os.path.join(learned_dir, "plan1.json"))
        assert not os.path.exists(os.path.join(cold_dir, "plan1.json"))

    def test_plan_learning_best_scoring_pattern_isolation(self):
        from harness.planner.plan_intelligence import PlanScore, PlanLearningStore
        store = PlanLearningStore()
        score = PlanScore(overall=0.9, capability_coverage=0.7, agent_fit=0.6,
                          model_quality=0.9, efficiency=0.5, risk=0.1)
        store.record_outcome("plan_best", score, "COMPLETED", experiment_context=COLD_CTX)
        # Different context gets None
        assert store.get_best_scoring_pattern(experiment_context=LEARNED_CTX) is None
        # Same context finds it
        assert store.get_best_scoring_pattern(experiment_context=COLD_CTX) is not None


# ---------------------------------------------------------------------------
# Test performance registry isolation
# ---------------------------------------------------------------------------

class TestPerformanceRegistryIsolation:
    """Test PerformanceRegistryV3 namespace isolation."""

    def test_performance_registry_namespaced_save(self):
        _clean_v32()
        reg = PerformanceRegistryV3()
        reg.record(make_perf_sample(mode="COLD", run_id="run_cold"), experiment_context=COLD_CTX)
        perf_dir = ns_path(COLD_CTX, "performance")
        assert os.path.exists(os.path.join(perf_dir, "registry.json"))

    def test_performance_registry_rank_models_isolation(self):
        _clean_v32()
        reg = PerformanceRegistryV3()
        for i in range(6):
            sample = PerformanceSample(
                task_id="t1",
                model_id="m1",
                capability="code_gen",
                success=True,
                score=0.5 + i * 0.1,
                latency_ms=100.0,
                iterations=1,
                validation_run_id="run_cold",
                mode="COLD",
            )
            reg.record(sample, experiment_context=COLD_CTX)

        # Same context returns results
        assert len(reg.rank_models("code_gen", experiment_context=COLD_CTX)) > 0
        # Different context returns empty (no data)
        assert len(reg.rank_models("code_gen", experiment_context=LEARNED_CTX)) == 0


# ---------------------------------------------------------------------------
# Test Retrieval Filtering utility
# ---------------------------------------------------------------------------

class TestRetrievalFiltering:
    """Test the retrieve_filtered utility function."""

    def test_retrieve_filtered_all_without_context(self):
        items = [
            make_evidence_pkg("n1", mode="COLD", run_id="run_cold"),
            make_evidence_pkg("n2", mode="LEARNED", run_id="run_learned"),
        ]
        filtered = retrieve_filtered(items)  # no context → all items
        assert len(filtered) == 2

    def test_retrieve_filtered_cold(self):
        items = [
            make_evidence_pkg("n1", mode="COLD", run_id="run_cold"),
            make_evidence_pkg("n2", mode="LEARNED", run_id="run_learned"),
        ]
        filtered = retrieve_filtered(items, COLD_CTX)
        assert len(filtered) == 1
        assert filtered[0].mode == "COLD"

    def test_retrieve_filtered_learned(self):
        items = [
            make_evidence_pkg("n1", mode="COLD", run_id="run_cold"),
            make_evidence_pkg("n2", mode="LEARNED", run_id="run_learned"),
        ]
        filtered = retrieve_filtered(items, LEARNED_CTX)
        assert len(filtered) == 1
        assert filtered[0].mode == "LEARNED"

    def test_retrieve_filtered_empty(self):
        assert retrieve_filtered([], COLD_CTX) == []

    def test_retrieve_filtered_items_without_mode(self):
        """Items without mode field pass through with context filtering."""
        items = [{"name": "no_mode"}]
        filtered = retrieve_filtered(items, COLD_CTX)
        assert len(filtered) == 1  # passes through since no mode to check

    def test_retrieve_filtered_none_context(self):
        items = [make_evidence_pkg("n1", mode="COLD", run_id="run_cold")]
        assert retrieve_filtered(items, None) == items  # no filtering


# ---------------------------------------------------------------------------
# Test v3.1 Backward Compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Test that V3.1 behavior is preserved when no experiment context."""

    def test_v3_experience_store_no_context(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ExperienceStore(storage_dir=tmpdir)
            exp = make_exp_record("legacy")
            store.store(exp)
            assert store.retrieve("legacy") is not None
            assert len(store.search("Task")) == 1

    def test_strategy_store_no_context(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StrategyStore(storage_dir=tmpdir)
            s = make_strategy("legacy_s")
            store.save(s)
            assert store.load("legacy_s") is not None

    def test_decision_tracker_no_context(self):
        """V3.1 backward compat: a legacy impact (no V3.2 provenance) records
        into flat storage when no ExperimentContext is given. A V3.2 impact
        (one carrying provenance) without a context must NOT silently fall back
        to shared V3.1 storage — it fails closed instead.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = DecisionImpactTracker(storage_dir=tmpdir)
            impact = DecisionImpact(
                decision_id="legacy_d", context_id="c1", decision_type="model",
                baseline_choice="a", actual_choice="b",
                influenced_by_learning=True,
                # NOTE: no V3.2 provenance fields → treated as legacy V3.1
            )
            tracker.record(impact)
            assert len(tracker._impacts) == 1
            # V3.2 experiment-scoped impact WITHOUT a context must fail closed
            v32_impact = DecisionImpact(
                decision_id="v32_d", context_id="c2", decision_type="model",
                baseline_choice="a", actual_choice="b",
                influenced_by_learning=True,
                validation_run_id="run_cold", mode="COLD",
            )
            with pytest.raises(V3IntegrationError, match="requires an ExperimentContext"):
                tracker.record(v32_impact)

    def test_evidence_store_no_context(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EvidenceStore(storage_dir=tmpdir)
            store.store_evidence(make_evidence_pkg("legacy_pkg"))
            assert len(store.get_evidence("t1", "e1")) == 1

    def test_experiment_tracker_no_context(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(storage_dir=tmpdir)
            tracker.record(make_model_experiment("legacy_me"))
            assert len(tracker.get_results("m1", "code_gen")) == 1

    def test_performance_registry_no_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = PerformanceRegistryV3(storage_path=os.path.join(tmpdir, "reg.json"))
            for i in range(6):
                reg.record(PerformanceSample(
                    task_id="t1", model_id="m1", capability="code_gen",
                    success=True, score=0.8, latency_ms=100, iterations=1,
                ))
            assert len(reg.rank_models("code_gen")) > 0
