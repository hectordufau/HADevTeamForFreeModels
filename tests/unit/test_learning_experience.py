# tests/unit/test_learning_experience.py — Tests for V3.1 Experience Extraction 2.0
"""Tests for Experience Extraction 2.0 (Phase A)."""

import os
import sys
import json
import tempfile
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.learning.experience import (
    StructuredExperience,
    ExperienceExtractorV31,
    ExperienceQualityScorer,
    ExecutionTrace,
    Evidence,
    ExperiencePipeline,
)


class TestExecutionTrace:
    def test_create_trace(self):
        trace = ExecutionTrace(
            task_id="test-001",
            steps=[{"step": 1, "action": "analyze"}],
            final_status="COMPLETED",
            final_score=0.85,
            duration_seconds=120.5,
        )
        assert trace.task_id == "test-001"
        assert trace.final_status == "COMPLETED"
        assert trace.final_score == 0.85
        assert trace.duration_seconds == 120.5

    def test_trace_to_dict(self):
        trace = ExecutionTrace(task_id="t1", errors=[{"message": "error1"}])
        d = trace.to_dict()
        assert d["task_id"] == "t1"
        assert len(d["errors"]) == 1
        assert d["errors"][0]["message"] == "error1"


class TestEvidence:
    def test_create_evidence(self):
        ev = Evidence(
            description="Test evidence",
            metric="score",
            value=0.9,
            source="evaluation",
        )
        assert ev.metric == "score"
        assert ev.value == 0.9

    def test_evidence_quality_scores(self):
        evaluation = Evidence(description="eval", metric="score", value=1.0, source="evaluation")
        execution = Evidence(description="exec", metric="duration", value=100, source="execution")
        review = Evidence(description="review", metric="score", value=0.8, source="review")
        unknown = Evidence(description="other", metric="score", value=0.5, source="unknown")

        assert Evidence.quality(evaluation) == 0.9
        assert Evidence.quality(execution) == 0.7
        assert Evidence.quality(review) == 0.5
        assert Evidence.quality(unknown) == 0.3


class TestStructuredExperience:
    def test_create_experience(self):
        exp = StructuredExperience(
            task_id="exp-001",
            task_context="Build a web API",
            capabilities_used=["backend_development", "testing"],
            strategy={"model_id": "model1", "agent_id": "agent1"},
            outcome={"status": "COMPLETED", "score": 0.92, "duration": 200},
            evidence=[Evidence(description="good", metric="score", value=0.92, source="evaluation")],
            lesson="Use model1 for backend tasks",
            confidence=0.8,
            category="success_pattern",
        )
        assert exp.lesson == "Use model1 for backend tasks"
        assert exp.confidence == 0.8
        assert exp.category == "success_pattern"

    def test_to_dict_roundtrip(self):
        exp = StructuredExperience(
            task_id="exp-002",
            task_context="Test",
            capabilities_used=["testing"],
            strategy={},
            outcome={"status": "COMPLETED", "score": 1.0, "duration": 10},
        )
        data = exp.to_dict()
        assert data["task_id"] == "exp-002"
        assert len(data["evidence"]) == 0

    def test_failure_experience(self):
        exp = StructuredExperience(
            task_id="exp-fail",
            task_context="Failed deployment",
            capabilities_used=["deployment"],
            strategy={"model_id": "model2"},
            outcome={"status": "FAILED", "score": -0.5, "duration": 300},
            failures=[{"message": "Deployment timeout", "node_id": "deploy-1"}],
            lesson="Increase timeout for large deployments",
            confidence=0.6,
            category="failure_lesson",
        )
        assert exp.outcome["status"] == "FAILED"
        assert len(exp.failures) == 1


class TestExperienceQualityScorer:
    def test_high_quality_score(self):
        exp = StructuredExperience(
            task_id="high",
            task_context="Test",
            capabilities_used=["test"],
            strategy={},
            outcome={"status": "COMPLETED", "score": 0.95, "duration": 50},
            evidence=[
                Evidence(description="s1", metric="score", value=0.95, source="evaluation"),
                Evidence(description="s2", metric="duration", value=50, source="execution"),
                Evidence(description="s3", metric="success", value=1.0, source="evaluation"),
            ],
            lesson="Works well",
        )
        components = ExperienceQualityScorer.score_components(exp)
        assert components["evidence_quality"] > 0.7
        assert components["outcome_quality"] > 0.7
        assert components["overall"] > 0.3

    def test_low_quality_score(self):
        exp = StructuredExperience(
            task_id="low",
            task_context="Unknown",
            capabilities_used=[],
            strategy={},
            outcome={"status": "UNKNOWN", "score": 0.0, "duration": 0},
            lesson="",
        )
        score = ExperienceQualityScorer.score(exp)
        assert score < 0.1

    def test_failure_outcome_quality(self):
        exp = StructuredExperience(
            task_id="fail",
            task_context="Test",
            capabilities_used=[],
            strategy={},
            outcome={"status": "FAILED", "score": -0.8, "duration": 100},
            lesson="Failed badly",
            evidence=[Evidence(description="err", metric="score", value=-0.8, source="evaluation")],
        )
        score = ExperienceQualityScorer.score(exp)
        assert 0 < score < 0.5  # failure should reduce quality but not to 0


class TestExperienceExtractorV31:
    def test_extract_from_trace(self):
        trace = ExecutionTrace(
            task_id="trace-001",
            steps=[{"step": 1, "action": "code"}, {"step": 2, "action": "test"}],
            final_status="COMPLETED",
            final_score=0.88,
            duration_seconds=150,
        )
        exp = ExperienceExtractorV31.extract(
            trace,
            task_objective="Build a feature",
            capabilities=["backend_development"],
            strategy={"model_id": "m1", "agent_id": "a1"},
        )
        assert exp.task_id == "trace-001"
        assert "backend_development" in exp.capabilities_used
        assert exp.strategy["model_id"] == "m1"
        assert exp.outcome["score"] == 0.88
        assert exp.category in ("success_pattern", "performance_tip")
        assert len(exp.evidence) >= 2  # score + duration

    def test_extract_with_errors(self):
        trace = ExecutionTrace(
            task_id="trace-fail",
            steps=[],
            final_status="FAILED",
            final_score=-0.3,
            duration_seconds=60,
            errors=[{"message": "Syntax error in deploy script", "node_id": "deploy"}],
        )
        exp = ExperienceExtractorV31.extract(trace, task_objective="Deploy app")
        assert exp.outcome["status"] == "FAILED"
        assert len(exp.failures) == 1
        assert "Syntax error" in exp.lesson

    def test_from_v3_experience(self):
        class MockV3Experience:
            def to_dict(self):
                return {
                    "task_id": "v3-task",
                    "task_objective": "V3 task",
                    "capabilities_used": ["testing"],
                    "result_status": "COMPLETED",
                    "score": 0.75,
                    "duration_seconds": 100,
                    "lessons": ["Failed capabilities: test-node"],
                    "timestamp": "2025-01-01T00:00:00",
                }
        exp = ExperienceExtractorV31.from_v3_experience(MockV3Experience())
        assert exp.task_id == "v3-task"
        assert "testing" in exp.capabilities_used
        assert exp.outcome["score"] == 0.75
        assert exp.confidence == 0.5  # default for converted


class TestExperiencePipeline:
    def test_process_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ExperiencePipeline(storage_dir=os.path.join(tmpdir, "exps"))
            trace = ExecutionTrace(
                task_id="pipe-001",
                final_status="COMPLETED",
                final_score=0.9,
                duration_seconds=100,
            )
            exp = pipeline.process_trace(
                trace,
                task_objective="Pipeline test",
                capabilities=["testing"],
            )
            assert exp.task_id == "pipe-001"

            retrieved = pipeline.retrieve("pipe-001")
            assert retrieved is not None
            assert retrieved.task_context == "Pipeline test"

            quality = pipeline.get_quality("pipe-001")
            assert quality is not None
            assert "overall" in quality

    def test_list_experiences(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ExperiencePipeline(storage_dir=os.path.join(tmpdir, "exps2"))
            assert pipeline.list_experiences() == []

            trace = ExecutionTrace(task_id="l1", final_status="COMPLETED", final_score=0.5)
            pipeline.process_trace(trace, "Task 1")
            trace2 = ExecutionTrace(task_id="l2", final_status="FAILED", final_score=-0.2)
            pipeline.process_trace(trace2, "Task 2")

            listing = pipeline.list_experiences()
            assert len(listing) == 2
            assert "l1" in listing
            assert "l2" in listing
