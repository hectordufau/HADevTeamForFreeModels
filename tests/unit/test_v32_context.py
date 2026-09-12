"""Tests for V3.2 ExperimentContext (Phase 1)."""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from harness.learning.context import (
    ExperimentContext,
    ExperimentContextError,
    VALID_MODES,
)
from harness.learning.experience import StructuredExperience, ExperiencePipeline
from harness.learning.strategy import Strategy, StrategyStore
from harness.planner.v3_integration import DecisionImpact, DecisionImpactTracker
from harness.evidence.unified import (
    CanonicalExecutionRecord,
    EvidencePackage,
    EvidenceStore,
)
from harness.planner.model_intelligence import ModelExperiment, ExperimentTracker
from harness.routing.performance_v3 import PerformanceSample, PerformanceRegistryV3
from harness.planner.failure_intelligence import StructuredFailure


class TestExperimentContextCreation:
    """Test ExperimentContext creation and validation."""

    def test_experiment_context_creation(self):
        """Create an ExperimentContext with all fields."""
        ctx = ExperimentContext(
            validation_run_id="run_001",
            benchmark_id="bench_code_gen",
            task_id="task_42",
            execution_id="exec_abc123",
            mode="COLD",
            reproducibility_seed=42,
        )
        assert ctx.validation_run_id == "run_001"
        assert ctx.benchmark_id == "bench_code_gen"
        assert ctx.task_id == "task_42"
        assert ctx.execution_id == "exec_abc123"
        assert ctx.mode == "COLD"
        assert ctx.reproducibility_seed == 42
        assert ctx.context_id.startswith("expctx_")
        assert len(ctx.context_id) == 19  # "expctx_" + 12 hex chars
        assert ctx.timestamp  # non-empty timestamp

    def test_experiment_context_all_modes(self):
        """All three valid modes should work."""
        for mode in ["COLD", "LEARNED", "TEST"]:
            ctx = ExperimentContext(
                validation_run_id="run_001",
                benchmark_id="bench",
                task_id="task",
                execution_id="exec",
                mode=mode,
            )
            assert ctx.mode == mode

    def test_experiment_context_validation_invalid_mode(self):
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            ExperimentContext(
                validation_run_id="run_001",
                benchmark_id="bench",
                task_id="task",
                execution_id="exec",
                mode="INVALID",
            )

    def test_experiment_context_validation_empty_run_id(self):
        """Empty validation_run_id raises ValueError."""
        with pytest.raises(ValueError, match="validation_run_id must be a non-empty string"):
            ExperimentContext(
                validation_run_id="",
                benchmark_id="bench",
                task_id="task",
                execution_id="exec",
                mode="COLD",
            )

    def test_experiment_context_validation_empty_benchmark_id(self):
        """Empty benchmark_id raises ValueError."""
        with pytest.raises(ValueError, match="benchmark_id must be a non-empty string"):
            ExperimentContext(
                validation_run_id="run_001",
                benchmark_id="",
                task_id="task",
                execution_id="exec",
                mode="COLD",
            )

    def test_experiment_context_validation_empty_task_id(self):
        """Empty task_id raises ValueError."""
        with pytest.raises(ValueError, match="task_id must be a non-empty string"):
            ExperimentContext(
                validation_run_id="run_001",
                benchmark_id="bench",
                task_id="",
                execution_id="exec",
                mode="COLD",
            )

    def test_experiment_context_validation_empty_execution_id(self):
        """Empty execution_id raises ValueError."""
        with pytest.raises(ValueError, match="execution_id must be a non-empty string"):
            ExperimentContext(
                validation_run_id="run_001",
                benchmark_id="bench",
                task_id="task",
                execution_id="",
                mode="COLD",
            )


class TestExperimentContextSerialization:
    """Test serialization round-trips."""

    def test_experiment_context_serialization(self):
        """to_dict and from_dict round-trip."""
        ctx = ExperimentContext(
            validation_run_id="run_001",
            benchmark_id="bench_code_gen",
            task_id="task_42",
            execution_id="exec_abc123",
            mode="LEARNED",
            reproducibility_seed=99,
        )
        d = ctx.to_dict()
        # Dict has expected keys
        assert d["validation_run_id"] == "run_001"
        assert d["benchmark_id"] == "bench_code_gen"
        assert d["task_id"] == "task_42"
        assert d["execution_id"] == "exec_abc123"
        assert d["mode"] == "LEARNED"
        assert d["reproducibility_seed"] == 99
        assert d["context_id"] == ctx.context_id
        assert d["timestamp"] == ctx.timestamp

        # JSON serialization
        json_str = ctx.to_json()
        parsed = json.loads(json_str)
        assert parsed["validation_run_id"] == "run_001"

        # from_dict round-trip — context_id and timestamp are regenerated
        ctx2 = ExperimentContext.from_dict(d)
        assert ctx2.validation_run_id == ctx.validation_run_id
        assert ctx2.benchmark_id == ctx.benchmark_id
        assert ctx2.task_id == ctx.task_id
        assert ctx2.execution_id == ctx.execution_id
        assert ctx2.mode == ctx.mode
        assert ctx2.reproducibility_seed == ctx.reproducibility_seed
        # context_id and timestamp are auto-generated, so they differ
        assert ctx2.context_id != ctx.context_id

    def test_experiment_context_deterministic_dict(self):
        """Same fields produce same dict."""
        ctx_a = ExperimentContext(
            validation_run_id="run_001",
            benchmark_id="bench",
            task_id="task",
            execution_id="exec",
            mode="COLD",
        )
        ctx_b = ExperimentContext(
            validation_run_id="run_001",
            benchmark_id="bench",
            task_id="task",
            execution_id="exec",
            mode="COLD",
        )
        # context_id and timestamp are auto-generated, so exclude them from comparison
        dict_a = {k: v for k, v in ctx_a.to_dict().items() if k not in ("context_id", "timestamp")}
        dict_b = {k: v for k, v in ctx_b.to_dict().items() if k not in ("context_id", "timestamp")}
        assert dict_a == dict_b

    def test_experiment_context_defaults(self):
        """Timestamp and seed have sensible defaults."""
        ctx = ExperimentContext(
            validation_run_id="run_001",
            benchmark_id="bench",
            task_id="task",
            execution_id="exec",
            mode="TEST",
        )
        assert ctx.reproducibility_seed == 0
        assert ctx.timestamp  # should have a timestamp string

    def test_experiment_context_equality(self):
        """Two contexts with same fields are equal."""
        ctx1 = ExperimentContext(
            validation_run_id="run_001",
            benchmark_id="bench",
            task_id="task",
            execution_id="exec",
            mode="COLD",
            reproducibility_seed=0,
        )
        ctx2 = ExperimentContext(
            validation_run_id="run_001",
            benchmark_id="bench",
            task_id="task",
            execution_id="exec",
            mode="COLD",
            reproducibility_seed=0,
        )
        # Manually set same context_id and timestamp for comparison
        ctx2.context_id = ctx1.context_id
        ctx2.timestamp = ctx1.timestamp
        assert ctx1 == ctx2

    def test_experiment_context_hashable(self):
        """ExperimentContext can be used as a dict key."""
        ctx = ExperimentContext(
            validation_run_id="run_001",
            benchmark_id="bench",
            task_id="task",
            execution_id="exec",
            mode="COLD",
        )
        d = {ctx: "value"}
        assert d[ctx] == "value"

    def test_experiment_context_non_string_ids(self):
        """Non-string ids should raise ValueError."""
        with pytest.raises(ValueError, match="validation_run_id must be a non-empty string"):
            ExperimentContext(
                validation_run_id=123,  # type: ignore
                benchmark_id="bench",
                task_id="task",
                execution_id="exec",
                mode="COLD",
            )


class TestExperimentContextBackwardsCompatibility:
    """Test that all stores accept None ExperimentContext gracefully."""

    def test_experience_pipeline_backwards_compatible(self):
        """ExperiencePipeline accepts optional ExperimentContext."""
        pipe = ExperiencePipeline()
        assert pipe.experiment_context is None
        pipe2 = ExperiencePipeline(experiment_context=None)
        assert pipe2.experiment_context is None
        ctx = ExperimentContext(
            validation_run_id="run_001",
            benchmark_id="bench",
            task_id="task",
            execution_id="exec",
            mode="COLD",
        )
        pipe3 = ExperiencePipeline(experiment_context=ctx)
        assert pipe3.experiment_context is ctx

    def test_experience_pipeline_retrieve_backwards_compatible(self):
        """ExperiencePipeline.retrieve() accepts optional ExperimentContext."""
        pipe = ExperiencePipeline()
        # Should not raise error; returns None since no data
        result = pipe.retrieve("nonexistent")
        assert result is None
        result2 = pipe.retrieve("nonexistent", experiment_context=None)
        assert result2 is None

    def test_strategy_store_backwards_compatible(self):
        """StrategyStore accepts optional ExperimentContext."""
        store = StrategyStore()
        store2 = StrategyStore(experiment_context=None)
        ctx = ExperimentContext(
            validation_run_id="run_001",
            benchmark_id="bench",
            task_id="task",
            execution_id="exec",
            mode="COLD",
        )
        store3 = StrategyStore(experiment_context=ctx)

    def test_strategy_store_find_by_task_class_backwards_compatible(self):
        """StrategyStore.find_by_task_class() accepts optional ExperimentContext."""
        store = StrategyStore()
        result = store.find_by_task_class("test")
        assert result == []
        result2 = store.find_by_task_class("test", experiment_context=None)
        assert result2 == []

    def test_decision_impact_tracker_backwards_compatible(self):
        """DecisionImpactTracker accepts optional ExperimentContext."""
        tracker = DecisionImpactTracker()
        tracker2 = DecisionImpactTracker(experiment_context=None)
        ctx = ExperimentContext(
            validation_run_id="run_001",
            benchmark_id="bench",
            task_id="task",
            execution_id="exec",
            mode="COLD",
        )
        tracker3 = DecisionImpactTracker(experiment_context=ctx)

    def test_decision_impact_tracker_record_stores_context(self):
        """DecisionImpactTracker.record() stores context fields."""
        tracker = DecisionImpactTracker()
        impact = DecisionImpact(
            decision_id="d1",
            context_id="c1",
            decision_type="model",
            baseline_choice="model_a",
            actual_choice="model_b",
            influenced_by_learning=True,
        )
        ctx = ExperimentContext(
            validation_run_id="run_001",
            benchmark_id="bench",
            task_id="task",
            execution_id="exec",
            mode="LEARNED",
        )
        tracker.record(impact, experiment_context=ctx)
        assert impact.validation_run_id == "run_001"
        assert impact.benchmark_id == "bench"
        assert impact.mode == "LEARNED"
        assert impact.execution_id == "exec"

    def test_decision_impact_tracker_record_no_context(self):
        """DecisionImpactTracker.record() works without context."""
        tracker = DecisionImpactTracker()
        impact = DecisionImpact(
            decision_id="d2",
            context_id="c2",
            decision_type="model",
            baseline_choice="model_a",
            actual_choice="model_b",
            influenced_by_learning=False,
        )
        tracker.record(impact)  # should not error
        assert impact.validation_run_id == ""
        assert impact.benchmark_id == ""
        assert impact.mode == ""
        assert impact.execution_id == ""

    def test_evidence_store_backwards_compatible(self):
        """EvidenceStore accepts optional ExperimentContext."""
        store = EvidenceStore()
        store2 = EvidenceStore(experiment_context=None)
        ctx = ExperimentContext(
            validation_run_id="run_001",
            benchmark_id="bench",
            task_id="task",
            execution_id="exec",
            mode="COLD",
        )
        store3 = EvidenceStore(experiment_context=ctx)

    def test_evidence_store_store_evidence_context(self):
        """EvidenceStore.store_evidence() stores context fields."""
        store = EvidenceStore()
        pkg = EvidencePackage(
            node_id="n1",
            capability="code_gen",
            agent="agent_a",
            task_id="t1",
            execution_id="e1",
        )
        ctx = ExperimentContext(
            validation_run_id="run_001",
            benchmark_id="bench",
            task_id="task",
            execution_id="exec",
            mode="LEARNED",
        )
        store.store_evidence(pkg, experiment_context=ctx)
        assert pkg.validation_run_id == "run_001"
        assert pkg.benchmark_id == "bench"
        assert pkg.mode == "LEARNED"

    def test_evidence_store_store_record_context(self):
        """EvidenceStore.store_record() stores context fields."""
        store = EvidenceStore()
        record = CanonicalExecutionRecord(
            task_id="t1",
            execution_id="e1",
            workflow_name="wf1",
            nodes=[],
            capability_requirements={},
            assignments={},
            results={},
            evidence_hashes=[],
        )
        ctx = ExperimentContext(
            validation_run_id="run_001",
            benchmark_id="bench",
            task_id="task",
            execution_id="exec",
            mode="LEARNED",
        )
        store.store_record(record, experiment_context=ctx)
        assert record.validation_run_id == "run_001"
        assert record.mode == "LEARNED"

    def test_experiment_tracker_backwards_compatible(self):
        """ExperimentTracker accepts optional ExperimentContext."""
        tracker = ExperimentTracker()
        exp = ModelExperiment(
            experiment_id="e1",
            model_id="m1",
            capability="code_gen",
            task_id="t1",
            result_score=0.85,
            is_exploration=False,
        )
        ctx = ExperimentContext(
            validation_run_id="run_001",
            benchmark_id="bench",
            task_id="task",
            execution_id="exec",
            mode="LEARNED",
        )
        tracker.record(exp)  # no context
        assert exp.validation_run_id == ""
        assert exp.mode == ""

        tracker.record(exp, experiment_context=ctx)  # with context
        assert exp.validation_run_id == "run_001"
        assert exp.mode == "LEARNED"

    def test_performance_registry_backwards_compatible(self):
        """PerformanceRegistryV3 accepts optional ExperimentContext."""
        reg = PerformanceRegistryV3()
        sample = PerformanceSample(
            task_id="t1",
            model_id="m1",
            capability="code_gen",
            success=True,
            score=0.95,
            latency_ms=100.0,
            iterations=1,
        )
        ctx = ExperimentContext(
            validation_run_id="run_001",
            benchmark_id="bench",
            task_id="task",
            execution_id="exec",
            mode="LEARNED",
        )
        reg.record(sample)  # no context
        assert sample.validation_run_id == ""
        assert sample.mode == ""

        reg.record(sample, experiment_context=ctx)  # with context
        assert sample.validation_run_id == "run_001"
        assert sample.mode == "LEARNED"

    def test_performance_registry_rank_models_backwards_compatible(self):
        """PerformanceRegistryV3.rank_models() accepts optional ExperimentContext."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = PerformanceRegistryV3(storage_path=os.path.join(tmpdir, "registry.json"))
            result = reg.rank_models("code_gen")
            assert result == []
            result2 = reg.rank_models("code_gen", experiment_context=None)
            assert result2 == []

    def test_structured_failure_backwards_compatible(self):
        """StructuredFailure still works with validation_run_id and mode fields."""
        failure = StructuredFailure(
            node_id="n1",
            capability="code_gen",
            category="implementation",
            sub_category="logic_error",
            severity="major",
            error_message="test error",
        )
        assert failure.validation_run_id == ""
        assert failure.mode == ""
        # Can set them
        failure.validation_run_id = "run_001"
        failure.mode = "LEARNED"
        assert failure.validation_run_id == "run_001"
        assert failure.mode == "LEARNED"


class TestStructuredExperimentContextFields:
    """Test that experiment-scoped objects carry context fields."""

    def test_structured_experience_has_context_fields(self):
        """StructuredExperience carries validation_run_id, benchmark_id, mode, execution_id."""
        exp = StructuredExperience(
            task_id="t1",
            task_context="test",
            capabilities_used=[],
            strategy={},
            outcome={},
        )
        assert hasattr(exp, "validation_run_id")
        assert hasattr(exp, "benchmark_id")
        assert hasattr(exp, "mode")
        assert hasattr(exp, "execution_id")
        assert exp.validation_run_id == ""
        assert exp.benchmark_id == ""
        assert exp.mode == ""
        assert exp.execution_id == ""

    def test_strategy_has_context_fields(self):
        """Strategy carries validation_run_id, benchmark_id, mode."""
        strategy = Strategy(
            strategy_id="s1",
            task_class="code_gen",
            capabilities=[],
            conditions={},
            workflow={},
        )
        assert hasattr(strategy, "validation_run_id")
        assert hasattr(strategy, "benchmark_id")
        assert hasattr(strategy, "mode")
        assert strategy.validation_run_id == ""
        assert strategy.benchmark_id == ""
        assert strategy.mode == ""

    def test_decision_impact_has_context_fields(self):
        """DecisionImpact carries validation_run_id, benchmark_id, mode, execution_id."""
        impact = DecisionImpact(
            decision_id="d1",
            context_id="c1",
            decision_type="model",
            baseline_choice="a",
            actual_choice="b",
            influenced_by_learning=True,
        )
        assert hasattr(impact, "validation_run_id")
        assert hasattr(impact, "benchmark_id")
        assert hasattr(impact, "mode")
        assert hasattr(impact, "execution_id")

    def test_evidence_package_has_context_fields(self):
        """EvidencePackage carries validation_run_id, benchmark_id, mode."""
        pkg = EvidencePackage(
            node_id="n1",
            capability="c1",
            agent="a1",
            task_id="t1",
            execution_id="e1",
        )
        assert hasattr(pkg, "validation_run_id")
        assert hasattr(pkg, "benchmark_id")
        assert hasattr(pkg, "mode")

    def test_canonical_execution_record_has_context_fields(self):
        """CanonicalExecutionRecord carries validation_run_id and mode."""
        record = CanonicalExecutionRecord(
            task_id="t1",
            execution_id="e1",
            workflow_name="wf1",
            nodes=[],
            capability_requirements={},
            assignments={},
            results={},
            evidence_hashes=[],
        )
        assert hasattr(record, "validation_run_id")
        assert hasattr(record, "mode")

    def test_model_experiment_has_context_fields(self):
        """ModelExperiment carries validation_run_id and mode."""
        exp = ModelExperiment(
            experiment_id="e1",
            model_id="m1",
            capability="c1",
            task_id="t1",
            result_score=0.9,
            is_exploration=False,
        )
        assert hasattr(exp, "validation_run_id")
        assert hasattr(exp, "mode")

    def test_performance_sample_has_context_fields(self):
        """PerformanceSample carries validation_run_id and mode."""
        sample = PerformanceSample(
            task_id="t1",
            model_id="m1",
            capability="c1",
            success=True,
            score=0.9,
            latency_ms=100,
            iterations=1,
        )
        assert hasattr(sample, "validation_run_id")
        assert hasattr(sample, "mode")

    def test_structured_failure_has_context_fields(self):
        """StructuredFailure carries validation_run_id and mode."""
        failure = StructuredFailure(
            node_id="n1",
            capability="c1",
            category="implementation",
            sub_category="logic_error",
            severity="major",
            error_message="err",
        )
        assert hasattr(failure, "validation_run_id")
        assert hasattr(failure, "mode")
