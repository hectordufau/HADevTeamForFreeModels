# tests/integration/test_v31_integration.py — V3.1 Learning→Decision Coupling Tests
"""Integration tests for V3.1 Learning→Decision coupling."""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.learning.experience import (
    ExperiencePipeline, ExecutionTrace, StructuredExperience,
    ExperienceExtractorV31,
)
from harness.learning.retrieval import (
    MultiFactorRetriever, RetrievalContext, HistoricalUsefulnessTracker,
)
from harness.learning.strategy import (
    Strategy, StrategyGenerator, StrategyValidator, StrategyStore,
)
from harness.planner.v3_integration import (
    UnifiedDecisionPipeline, DecisionContext, DecisionImpact,
    DecisionImpactTracker,
)


class TestDecisionContext:
    def test_create_with_all_fields(self):
        ctx = DecisionContext(
            task={"id": "T1", "objective": "Build API"},
            capability={"required": ["backend"]},
            policy={"invariant": "free_models"},
            experience={"count": 5},
            strategy={"available": 2},
            performance={"avg_score": 0.8},
        )
        assert ctx.task["id"] == "T1"
        assert ctx.context_id.startswith("ctx_")

    def test_to_dict(self):
        ctx = DecisionContext()
        data = ctx.to_dict()
        assert "context_id" in data
        assert "timestamp" in data


class TestDecisionImpactTracker:
    def test_record_and_summary(self):
        tracker = DecisionImpactTracker(storage_dir=tempfile.mkdtemp())
        tracker.record(DecisionImpact(
            decision_id="d1", context_id="c1", decision_type="model",
            baseline_choice="model_a", actual_choice="model_b",
            influenced_by_learning=True, outcome_improved=True, score_delta=0.1,
        ))
        tracker.record(DecisionImpact(
            decision_id="d2", context_id="c2", decision_type="agent",
            baseline_choice="agent_a", actual_choice="agent_a",
            influenced_by_learning=False,
        ))

        summary = tracker.get_summary()
        assert summary["total_decisions"] == 2
        assert summary["influenced_by_learning"] == 1
        assert summary["improvements"] == 1
        assert summary["regressions"] == 0


class TestFullPipeline:
    def test_experience_to_strategy_pipeline(self):
        """End-to-end test: trace → experience → retrieval → strategy → decision."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, "exps")
            strat_dir = os.path.join(tmpdir, "strategies")

            # 1. Create experience pipeline
            pipeline = ExperiencePipeline(storage_dir=exp_dir)

            # 2. Process multiple traces
            for i in range(4):
                trace = ExecutionTrace(
                    task_id=f"api-{i}",
                    steps=[],
                    final_status="COMPLETED" if i < 3 else "FAILED",
                    final_score=0.8 + i * 0.05 if i < 3 else -0.5,
                    duration_seconds=100 + i * 20,
                    errors=[{"message": f"error-{i}", "node_id": f"node-{i}"}] if i == 3 else [],
                )
                pipeline.process_trace(
                    trace,
                    f"Build API endpoint {i}",
                    capabilities=["backend_development"],
                    strategy={"model_id": "deepseek:free", "agent_id": "coder"},
                )

            # 3. Generate strategy from experiences
            generator = StrategyGenerator(pipeline)
            strategy = generator.generate("backend_development", min_experiences=3)
            assert strategy is not None
            assert len(strategy.capabilities) > 0
            assert len(strategy.preferred_models) > 0

            # 4. Validate strategy
            validation = StrategyValidator.validate_all(strategy)
            assert isinstance(validation, dict)

            # 5. Store strategy
            store = StrategyStore(storage_dir=strat_dir)
            store.save(strategy)
            loaded = store.load(strategy.strategy_id)
            assert loaded is not None

            # 6. Use strategy in decision pipeline
            tracker = DecisionImpactTracker(storage_dir=os.path.join(tmpdir, "decisions"))
            decision_pipeline = UnifiedDecisionPipeline(
                cap_recommender=None, model_policy=None, plan_selector=None,
                replan_optimizer=None, governance=None,
                strategy_store=store, decision_tracker=tracker,
            )
            impact = DecisionImpact(
                decision_id="test-impact",
                context_id="ctx-1",
                decision_type="model",
                baseline_choice="default",
                actual_choice=strategy.preferred_models[0],
                influenced_by_learning=True,
            )
            tracker.record(impact)

            # Verify impact tracking
            summary = tracker.get_summary()
            assert summary["total_decisions"] == 1
            assert summary["influenced_by_learning"] == 1


class TestCoupling:
    def test_decision_influence_model_selection(self):
        """Test that learned strategies influence but don't force model selection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StrategyStore(storage_dir=os.path.join(tmpdir, "s"))
            s = Strategy(
                strategy_id="s-test",
                task_class="testing",
                capabilities=["testing"],
                conditions={},
                workflow={},
                preferred_models=["recommended:free"],
            )
            store.save(s)

            tracker = DecisionImpactTracker(storage_dir=os.path.join(tmpdir, "d"))
            pipeline = UnifiedDecisionPipeline(
                cap_recommender=None, model_policy=None, plan_selector=None,
                replan_optimizer=None, governance=None,
                strategy_store=store, decision_tracker=tracker,
            )

            # Verify strategy store has strategies
            strategies = store.find_by_task_class("testing")
            assert len(strategies) == 1
            assert strategies[0].preferred_models == ["recommended:free"]

    def test_retrieval_explainability(self):
        """Test that retrievals include explanations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ExperiencePipeline(storage_dir=os.path.join(tmpdir, "e"))
            trace = ExecutionTrace(
                task_id="exp-1", final_status="COMPLETED",
                final_score=0.9, duration_seconds=100,
            )
            pipeline.process_trace(trace, "Test task", capabilities=["testing"])

            retriever = MultiFactorRetriever(experience_pipeline=pipeline)
            context = RetrievalContext(query="test task", limit=5)

            results = retriever.retrieve(context)
            assert len(results) > 0
            assert len(results[0].explanations) > 0
            assert results[0].overall_score > 0
