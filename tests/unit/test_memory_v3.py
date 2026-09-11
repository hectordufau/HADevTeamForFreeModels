"""Tests for Memory V3 (Phase D)."""

import sys, os, tempfile, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.memory.v3 import (
    ExperienceRecord, ExperienceExtractor, ExperienceStore,
    RelevanceScorer, ConfidenceTracker, MemoryScore,
)


class MockTask:
    def __init__(self):
        self.metadata = type('obj', (object,), {'id': 'TASK-001'})
        self.spec = type('obj', (object,), {
            'objective': 'Implement user authentication API',
        })


def test_experience_record_creation():
    exp = ExperienceRecord(
        task_id="T1", task_objective="Build API",
        capabilities_used=["backend", "testing"],
        result_status="COMPLETED", score=0.95, duration_seconds=120.0,
        lessons=["Used FastAPI successfully"],
    )
    assert exp.task_id == "T1"
    assert "backend" in exp.capabilities_used
    assert exp.score == 0.95


def test_experience_extractor():
    report = {
        "status": "COMPLETED",
        "execution": {
            "node_results": {
                "design": {"capability": "api_design", "agent": "architect"},
                "impl": {"capability": "backend_development", "agent": "coder"},
            },
        },
        "evaluation": {"score": 0.9},
    }
    task = MockTask()
    exp = ExperienceExtractor.extract(report, task)
    assert exp.task_objective == "Implement user authentication API"
    assert "api_design" in exp.capabilities_used
    assert "backend_development" in exp.capabilities_used


def test_experience_store_and_retrieve():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ExperienceStore(storage_dir=os.path.join(tmpdir, "exp"))
        exp = ExperienceRecord(
            task_id="T1", task_objective="Test", capabilities_used=["test"],
            result_status="COMPLETED", score=1.0, duration_seconds=10.0,
        )
        store.store(exp)
        retrieved = store.retrieve("T1")
        assert retrieved is not None
        assert retrieved.task_id == "T1"
        assert retrieved.result_status == "COMPLETED"


def test_relevance_scorer():
    exp = ExperienceRecord(
        task_id="T1", task_objective="Implement REST API endpoint",
        capabilities_used=["api_implementation", "testing"],
        result_status="COMPLETED", score=1.0, duration_seconds=0.0,
    )
    score = RelevanceScorer.score(exp, "API implementation")
    assert score > 0.0

    # Unrelated query
    score2 = RelevanceScorer.score(exp, "database schema design")
    assert score2 >= 0.0


def test_confidence_tracker():
    tracker = ConfidenceTracker(entry_id="test-entry")
    assert tracker.confidence == 0.5  # Neutral start

    tracker.record_access("success")
    assert tracker.confidence > 0.5  # Increased after success

    tracker.record_access("failure")
    tracker.record_access("failure")
    assert tracker.confidence < 0.6


def test_experience_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ExperienceStore(storage_dir=os.path.join(tmpdir, "exp"))

        exp1 = ExperienceRecord(
            task_id="T1", task_objective="Build REST API",
            capabilities_used=["api_implementation"],
            result_status="COMPLETED", score=0.9, duration_seconds=0.0,
        )
        exp2 = ExperienceRecord(
            task_id="T2", task_objective="Design database schema",
            capabilities_used=["database_schema"],
            result_status="FAILED", score=0.3, duration_seconds=0.0,
        )
        store.store(exp1)
        store.store(exp2)

        results = store.search("REST API", min_relevance=0.0)
        assert len(results) >= 1
