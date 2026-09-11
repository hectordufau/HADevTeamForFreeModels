"""Tests for Failure Intelligence (Phase J)."""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.planner.failure_intelligence import (
    RootCauseAnalyzer, StructuredFailure, FailureToLearningPipeline,
    ReplanningOptimizer, FAILURE_TAXONOMY,
)
from harness.memory.v3 import ExperienceStore


def test_root_cause_analysis():
    analyzer = RootCauseAnalyzer()

    # Implementation error
    failure = type('obj', (object,), {
        'node_id': 'impl', 'capability': 'backend_development',
        'message': 'TypeError: cannot concatenate str and int',
    })
    result = analyzer.analyze(failure)
    assert result.category == "implementation"
    assert result.node_id == "impl"
    assert result.capability == "backend_development"

    # Security error
    failure2 = type('obj', (object,), {
        'node_id': 'auth', 'capability': 'security_analysis',
        'message': 'SQL injection vulnerability detected in login endpoint',
    })
    result2 = analyzer.analyze(failure2)
    assert result2.category == "security"


def test_root_cause_structured_input():
    analyzer = RootCauseAnalyzer()

    sf = StructuredFailure(
        node_id="test", capability="testing", category="test",
        sub_category="assertion_failure", severity="major",
        error_message="Test assertion failed: expected 5, got 3",
    )
    result = analyzer.analyze(sf)
    # Dict input should work via _get_error
    result2 = analyzer.analyze({"node_id": "x", "capability": "y", "error_message": "config error"})
    assert result2.category == "configuration"


def test_failure_patterns():
    analyzer = RootCauseAnalyzer()
    for _ in range(3):
        f = type('obj', (object,), {
            'node_id': 'test', 'capability': 'testing',
            'message': 'Assertion error in test suite',
        })
        analyzer.analyze(f)

    patterns = analyzer.get_patterns()
    assert len(patterns) > 0
    assert "test/assertion_failure" in patterns


def test_failure_to_learning():
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_store = ExperienceStore(storage_dir=os.path.join(tmpdir, "exp"))
        pipeline = FailureToLearningPipeline(exp_store)

        failure = StructuredFailure(
            node_id="impl", capability="backend_development",
            category="test", sub_category="assertion_failure",
            severity="major", error_message="Test failed: expected 200 got 500",
        )
        result = pipeline.process(failure)
        assert "lesson" in result
        assert "AVOID" in result["lesson"]
        assert result["capability"] == "backend_development"

        # Check it was stored
        stored = exp_store.search("backend_development", min_relevance=0.0)
        assert len(stored) >= 1


def test_replanning_optimizer():
    optimizer = ReplanningOptimizer(failure_analyzer=None)

    # Implementation failures: should replan
    f1 = type('obj', (object,), {'category': 'implementation'})
    assert optimizer.should_replan("T1", f1) is True

    # Security failures: should NOT replan
    f2 = type('obj', (object,), {'category': 'security'})
    assert optimizer.should_replan("T1", f2) is False

    # Architecture failures: should NOT replan
    f3 = type('obj', (object,), {'category': 'architecture'})
    assert optimizer.should_replan("T1", f3) is False

    # After max replans, no more
    optimizer.record_attempt("T2")
    optimizer.record_attempt("T2")
    optimizer.record_attempt("T2")
    f4 = type('obj', (object,), {'category': 'implementation'})
    assert optimizer.should_replan("T2", f4) is False


def test_replan_strategies():
    optimizer = ReplanningOptimizer(failure_analyzer=None)
    assert optimizer.get_optimal_replan_strategy(
        type('obj', (object,), {'category': 'implementation'})) == "retry_with_different_agent"
    assert optimizer.get_optimal_replan_strategy(
        type('obj', (object,), {'category': 'security'})) == "escalate_to_human"
    assert optimizer.get_optimal_replan_strategy(
        type('obj', (object,), {'category': 'configuration'})) == "add_environment_setup_node"
