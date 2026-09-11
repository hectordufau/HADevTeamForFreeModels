"""Tests for Observability (Phase L)."""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.planner.observability import (
    StructuredLogger, DecisionTracer, DecisionRecord,
    ExecutionTracer, ExecutionTraceNode, LogEntry,
)


def test_structured_logger():
    logger = StructuredLogger(storage_dir=os.path.join(tempfile.mkdtemp(), "logs"))
    logger.info("test_module", "This is a test message", context={"key": "value"})
    logger.error("test_module", "Error occurred", trace_id="trace-001")

    entries = logger.get_recent()
    assert len(entries) == 2
    assert entries[0].level == "INFO"
    assert entries[0].module == "test_module"
    assert entries[1].trace_id == "trace-001"


def test_decision_tracer():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracer = DecisionTracer(storage_dir=os.path.join(tmpdir, "decisions"))

        decision = DecisionRecord(
            decision_id="d1", decision_type="model_selection",
            context={"capabilities": ["coding"]},
            alternatives=[{"model": "a"}, {"model": "b"}],
            selected="model_a", rationale="Best match",
        )
        tracer.record(decision)

        decisions = tracer.get_decisions()
        assert len(decisions) == 1
        assert decisions[0].decision_type == "model_selection"


def test_execution_tracer():
    tracer = ExecutionTracer()

    # Begin a trace
    nid = tracer.begin("planner", "plan", input_summary="task: build API")
    assert nid is not None

    # Begin subtrace
    child_id = tracer.begin("executor", "execute", input_summary="node: design")
    tracer.end(child_id, status="success", output_summary="design completed")

    tracer.end(nid, status="success", output_summary="plan completed")

    trace = tracer.get_trace()
    assert trace is not None
    assert trace.component == "planner"
    assert len(trace.children) == 1
    assert trace.children[0].component == "executor"


def test_execution_trace_tree():
    tracer = ExecutionTracer()

    # Build a tree
    root = tracer.begin("root", "workflow")
    c1 = tracer.begin("child1", "step1")
    tracer.end(c1, "success")
    c2 = tracer.begin("child2", "step2")
    gc = tracer.begin("grandchild", "substep")
    tracer.end(gc, "success")
    tracer.end(c2, "failure", error="Something went wrong")
    tracer.end(root, "success")

    trace = tracer.get_trace()
    assert trace.status == "success"
    assert trace.children[1].status == "failure"
    assert trace.children[1].error == "Something went wrong"
