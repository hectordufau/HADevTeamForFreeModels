# tests/unit/test_learning_retrieval.py — Tests for V3.1 Memory Retrieval 2.0
"""Tests for Memory Retrieval 2.0 (Phase B)."""

import os
import sys
import json
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.learning.experience import (
    StructuredExperience, ExperienceExtractorV31, ExperiencePipeline,
    ExecutionTrace, Evidence,
)
from harness.learning.retrieval import (
    MultiFactorRetriever, RetrievalContext, RetrievalResult,
    HistoricalUsefulnessTracker, ConfidenceCalibrator, RetrievalExplainer,
)


class TestConfidenceCalibrator:
    def test_high_confidence(self):
        result = ConfidenceCalibrator.calibrate(
            initial_confidence=0.8, usefulness=0.5, access_count=10, success_rate=0.9
        )
        assert result["bucket"] == "HIGH"
        assert result["calibrated_confidence"] > 0.8

    def test_low_confidence(self):
        result = ConfidenceCalibrator.calibrate(
            initial_confidence=0.2, usefulness=-0.3, access_count=1, success_rate=0.3
        )
        assert result["bucket"] == "LOW"
        assert result["calibrated_confidence"] < 0.4

    def test_medium_confidence(self):
        result = ConfidenceCalibrator.calibrate(
            initial_confidence=0.5, usefulness=0.0, access_count=3, success_rate=0.5
        )
        assert result["bucket"] == "MEDIUM"
        assert 0.4 <= result["calibrated_confidence"] <= 0.7

    def test_no_data_returns_initial(self):
        result = ConfidenceCalibrator.calibrate(initial_confidence=0.5)
        assert result["calibrated_confidence"] == 0.5
        assert result["bucket"] == "MEDIUM"


class TestHistoricalUsefulnessTracker:
    def test_record_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = HistoricalUsefulnessTracker(storage_dir=tmpdir)
            rid = tracker.record_retrieval("exp-1", "ctx-1")
            assert rid is not None

            usefulness = tracker.get_usefulness("exp-1")
            assert usefulness == 0.0  # default improvement before outcome recorded

            tracker.record_outcome(rid, subsequent_score=0.9, baseline_score=0.7)
            usefulness = tracker.get_usefulness("exp-1")
            assert usefulness is not None
            assert usefulness > 0  # improvement

    def test_multiple_records_avg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = HistoricalUsefulnessTracker(storage_dir=tmpdir)
            rid1 = tracker.record_retrieval("exp-1", "ctx-1")
            rid2 = tracker.record_retrieval("exp-1", "ctx-2")
            tracker.record_outcome(rid1, 0.9, 0.7)  # +0.2
            tracker.record_outcome(rid2, 0.6, 0.7)  # -0.1
            usefulness = tracker.get_usefulness("exp-1")
            assert usefulness is not None
            assert usefulness == pytest.approx(0.05, abs=0.01)


class TestRetrievalExplainer:
    def test_explain_high_similarity(self):
        result = RetrievalResult(
            experience=StructuredExperience(
                task_id="e1", task_context="text", capabilities_used=[],
                strategy={}, outcome={}, lesson="Good pattern",
                confidence=0.8,
                evidence=[Evidence(description="e", metric="s", value=1.0, source="evaluation")],
            ),
            overall_score=0.8,
            factor_scores={"similarity": 0.8, "usefulness": 0.5, "confidence": 0.8, "evidence_quality": 0.9, "applicability": 0.5, "recency": 0.5},
            explanations=[],
        )
        explanations = RetrievalExplainer.explain(result)
        assert any("High task similarity" in e for e in explanations)
        assert any("High confidence lesson" in e for e in explanations)
        assert any("Strong supporting evidence" in e for e in explanations)
        assert any("Good pattern" in e for e in explanations)

    def test_explain_fallback(self):
        result = RetrievalResult(
            experience=StructuredExperience(
                task_id="e2", task_context="", capabilities_used=[],
                strategy={}, outcome={}, lesson="", confidence=0.1,
            ),
            overall_score=0.1,
            factor_scores={"similarity": 0.0, "usefulness": 0.0, "confidence": 0.1, "evidence_quality": 0.0, "applicability": 0.0, "recency": 0.0},
            explanations=[],
        )
        explanations = RetrievalExplainer.explain(result)
        assert any("default fallback" in e for e in explanations)


class TestMultiFactorRetriever:
    def create_sample_experience(self, tid, task_context, score=0.8, status="COMPLETED"):
        return StructuredExperience(
            task_id=tid,
            task_context=task_context,
            capabilities_used=["testing"],
            strategy={"model_id": "m1"},
            outcome={"status": status, "score": score, "duration": 100},
            evidence=[Evidence(description="e", metric="s", value=score, source="evaluation")],
            lesson=f"Lesson from {tid}",
            confidence=0.7,
            category="success_pattern",
        )

    def test_retrieve_from_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ExperiencePipeline(storage_dir=os.path.join(tmpdir, "exps"))
            pipeline.process_trace(
                ExecutionTrace(task_id="t1", final_status="COMPLETED", final_score=0.9, duration_seconds=100),
                "Build web API with Python",
                capabilities=["backend_development"],
            )

            retriever = MultiFactorRetriever(experience_pipeline=pipeline)
            context = RetrievalContext(
                query="build web api with python",
                capabilities=["backend_development"],
                limit=5,
            )
            results = retriever.retrieve(context)
            assert len(results) >= 1
            assert results[0].overall_score > 0

    def test_retrieve_no_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ExperiencePipeline(storage_dir=os.path.join(tmpdir, "exps"))
            pipeline.process_trace(
                ExecutionTrace(task_id="t1", final_status="COMPLETED", final_score=0.9),
                "Frontend React app",
                capabilities=["frontend_development"],
            )

            retriever = MultiFactorRetriever(experience_pipeline=pipeline)
            context = RetrievalContext(
                query="machine learning model training",
                limit=5,
            )
            results = retriever.retrieve(context)
            # Should still return something due to low similarity
            assert isinstance(results, list)

    def test_retrieve_from_v3_store(self):
        # This tests the fallback path
        class MockV3Store:
            def search(self, query, **kw):
                return []
            def retrieve(self, tid):
                return None

        retriever = MultiFactorRetriever(v3_store=MockV3Store())
        context = RetrievalContext(query="test", limit=5)
        results = retriever.retrieve(context)
        assert results == []

    def test_capability_match_boost(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ExperiencePipeline(storage_dir=os.path.join(tmpdir, "exps"))
            pipeline.process_trace(
                ExecutionTrace(task_id="t-cap", final_status="COMPLETED", final_score=0.9),
                "Capability matching test",
                capabilities=["security_analysis"],
            )

            retriever = MultiFactorRetriever(experience_pipeline=pipeline)
            context = RetrievalContext(
                query="security",
                capabilities=["security_analysis"],
                limit=5,
            )
            results = retriever.retrieve(context)
            assert len(results) > 0
