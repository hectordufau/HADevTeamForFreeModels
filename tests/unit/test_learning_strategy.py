# tests/unit/test_learning_strategy.py — Tests for V3.1 Strategy Layer
"""Tests for Learned Strategy Layer (Phase C)."""

import os
import sys
import json
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.learning.experience import (
    StructuredExperience, ExperiencePipeline, ExecutionTrace, Evidence,
)
from harness.learning.strategy import (
    Strategy, StrategyGenerator, StrategyValidator, StrategyStore,
)


class TestStrategy:
    def test_create_strategy(self):
        s = Strategy(
            strategy_id="strat-001",
            task_class="backend_development",
            capabilities=["backend", "testing"],
            conditions={"task_type": "api", "complexity_range": "high"},
            workflow={"name": "API workflow", "steps": ["backend", "testing"], "ordering": "sequential"},
            preferred_models=["model1"],
            preferred_agents=["agent1"],
            expected_outcome={"score": 0.85, "success_rate": 0.9},
            evidence_count=10,
            confidence=0.8,
        )
        assert s.task_class == "backend_development"
        assert len(s.preferred_models) == 1
        assert s.confidence == 0.8

    def test_to_dict_roundtrip(self):
        s = Strategy(
            strategy_id="s1",
            task_class="testing",
            capabilities=["testing"],
            conditions={},
            workflow={},
        )
        data = s.to_dict()
        assert data["strategy_id"] == "s1"
        assert "preferred_models" in data


class TestStrategyValidator:
    def test_validate_policy_paid_model(self):
        s = Strategy(
            strategy_id="bad-s",
            task_class="test",
            capabilities=["test"],
            conditions={},
            workflow={},
            preferred_models=["gpt-4"],
        )

        class MockFreeEnforcer:
            def enforce(self, model_id):
                return model_id != "gpt-4"

        violations = StrategyValidator.validate_policy(
            s, free_model_enforcer=MockFreeEnforcer()
        )
        assert len(violations) == 1
        assert "paid model" in violations[0]

    def test_validate_policy_free_model(self):
        s = Strategy(
            strategy_id="good-s",
            task_class="test",
            capabilities=["test"],
            conditions={},
            workflow={},
            preferred_models=["deepseek:free"],
        )

        class MockFreeEnforcer:
            def enforce(self, model_id):
                return True

        violations = StrategyValidator.validate_policy(
            s, free_model_enforcer=MockFreeEnforcer()
        )
        assert len(violations) == 0

    def test_validate_capabilities_missing(self):
        s = Strategy(strategy_id="s1", task_class="t", capabilities=["missing_cap"], conditions={}, workflow={})

        class MockRegistry:
            def find_agents_for(self, cap):
                return []

        warnings = StrategyValidator.validate_capabilities(s, registry=MockRegistry())
        assert len(warnings) == 1
        assert "no agent provides it" in warnings[0]

    def test_validate_evidence_insufficient(self):
        s = Strategy(strategy_id="s1", task_class="t", capabilities=[], conditions={}, workflow={},
                      evidence_count=1, confidence=0.2)
        warnings = StrategyValidator.validate_evidence(s, min_evidence=3)
        assert len(warnings) == 2  # count + confidence

    def test_validate_evidence_sufficient(self):
        s = Strategy(strategy_id="s1", task_class="t", capabilities=[], conditions={}, workflow={},
                      evidence_count=5, confidence=0.8)
        warnings = StrategyValidator.validate_evidence(s, min_evidence=3)
        assert len(warnings) == 0

    def test_validate_all(self):
        s = Strategy(strategy_id="s1", task_class="t", capabilities=[], conditions={}, workflow={})
        result = StrategyValidator.validate_all(s)
        assert "policy" in result
        assert "capabilities" in result
        assert "evidence" in result


class TestStrategyGenerator:
    def test_generate_from_experiences(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ExperiencePipeline(storage_dir=os.path.join(tmpdir, "exps"))
            # Add 3 experiences with same category
            for i in range(3):
                pipeline.process_trace(
                    ExecutionTrace(
                        task_id=f"gen-{i}",
                        final_status="COMPLETED",
                        final_score=0.8 + i * 0.05,
                        duration_seconds=100,
                    ),
                    f"Backend API task {i}",
                    capabilities=["backend_development", "testing"],
                    strategy={"model_id": "deepseek:free", "agent_id": "coder"},
                )

            generator = StrategyGenerator(pipeline)
            strategy = generator.generate("backend_development", min_experiences=3)
            assert strategy is not None
            assert strategy.task_class == "backend_development"
            assert "backend_development" in strategy.capabilities
            assert len(strategy.preferred_models) > 0

    def test_generate_insufficient_experiences(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ExperiencePipeline(storage_dir=os.path.join(tmpdir, "exps"))
            pipeline.process_trace(
                ExecutionTrace(task_id="single", final_status="COMPLETED", final_score=0.9),
                "Single task",
                capabilities=["testing"],
            )

            generator = StrategyGenerator(pipeline)
            strategy = generator.generate("testing", min_experiences=3)
            assert strategy is None

    def test_generate_no_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ExperiencePipeline(storage_dir=os.path.join(tmpdir, "exps"))
            generator = StrategyGenerator(pipeline)
            strategy = generator.generate("nonexistent")
            assert strategy is None


class TestStrategyStore:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StrategyStore(storage_dir=os.path.join(tmpdir, "strategies"))
            s = Strategy(
                strategy_id="s-test",
                task_class="testing",
                capabilities=["testing"],
                conditions={},
                workflow={"name": "test"},
            )
            store.save(s)

            loaded = store.load("s-test")
            assert loaded is not None
            assert loaded.task_class == "testing"
            assert loaded.workflow["name"] == "test"

    def test_list_strategies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StrategyStore(storage_dir=os.path.join(tmpdir, "strategies"))
            assert store.list_strategies() == []

            s1 = Strategy(strategy_id="s1", task_class="a", capabilities=[], conditions={}, workflow={}, confidence=0.9)
            s2 = Strategy(strategy_id="s2", task_class="b", capabilities=[], conditions={}, workflow={}, confidence=0.5)
            store.save(s1)
            store.save(s2)

            all_s = store.list_strategies()
            assert len(all_s) == 2
            assert all_s[0].confidence == 0.9  # highest first

    def test_find_by_task_class(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StrategyStore(storage_dir=os.path.join(tmpdir, "strategies"))
            s1 = Strategy(strategy_id="s1", task_class="backend", capabilities=[], conditions={}, workflow={})
            s2 = Strategy(strategy_id="s2", task_class="frontend", capabilities=[], conditions={}, workflow={})
            store.save(s1)
            store.save(s2)

            found = store.find_by_task_class("backend")
            assert len(found) == 1
            assert found[0].strategy_id == "s1"
