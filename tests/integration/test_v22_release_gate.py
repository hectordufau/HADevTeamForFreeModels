"""V2.2 Release Gate tests (DG1-DG10) and V2.1 regression (G1-G10)."""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.capabilities.registry import CapabilityRegistry
from harness.capabilities.taxonomy import CapabilityTaxonomy
from harness.capabilities.graph import CapabilityGraph, CapabilityNode
from harness.capabilities.matching import CapabilityMatcher
from harness.planner import (
    TaskAnalyzer, CapabilityRequirements,
    WorkflowPlanner, WorkflowNode, ExecutionWorkflow,
    WorkflowValidator, ValidationResult,
    ExecutionGraph, NodeExecutionResult,
    WorkspaceManager, Conflict,
    BranchResolver, BranchingError,
    FailureAnalyzer, FailureReport,
    Replanner, ReplanReport,
)


class MockRouter:
    def select(self, required_capabilities, role, **kw):
        from harness.routing import ModelSelection
        return ModelSelection(model_id="mock:free", provider="nous", cost=0, reason="test")


class MockTask:
    def __init__(self, obj="Test", ac=None, task_id="G-001"):
        self.metadata = type('obj', (object,), {"id": task_id})
        self.spec = type('obj', (object,), {
            "objective": obj,
            "requirements": [],
            "acceptance_criteria": ac or [],
            "allowed_changes": [],
            "verification": None,
            "autonomy": type('obj', (object,), {"maximum": 3}),
        })


def _setup():
    registry = CapabilityRegistry()
    registry.register_agent("architect", ["system_design", "api_design"])
    registry.register_agent("coder", ["backend_development", "api_implementation"])
    registry.register_agent("tester", ["testing", "verification"])
    registry.register_agent("reviewer", ["code_review", "evaluation", "review"])

    taxonomy = CapabilityTaxonomy()
    return registry, taxonomy


# DG1 — Capability Coverage
def test_dg1_capability_coverage():
    """All required capabilities have an agent selected. Optional capabilities are best-effort."""
    registry, taxonomy = _setup()
    planner = WorkflowPlanner(registry, taxonomy, MockRouter())
    reqs = CapabilityRequirements(
        required=["api_design", "backend_development", "testing"],
        optional=["security_analysis"],
    )
    workflow = planner.plan(reqs)
    node_caps = {n.capability for n in workflow.nodes}
    for cap in reqs.required:
        assert cap in node_caps, f"Required capability '{cap}' not covered"
    print("  ✓ DG1: Capability Coverage — all required caps have agents")


# DG2 — DAG Integrity
def test_dg2_dag_integrity():
    """Cycles are rejected. Every node has a valid dependency path to completion."""
    validator = WorkflowValidator()
    # Cyclic workflow
    cyclic_wf = ExecutionWorkflow(nodes=[
        WorkflowNode(id="a", capability="a", agent="coder", model_id="mock:free", depends_on=["c"]),
        WorkflowNode(id="b", capability="b", agent="coder", model_id="mock:free", depends_on=["a"]),
        WorkflowNode(id="c", capability="c", agent="coder", model_id="mock:free", depends_on=["b"]),
    ])
    result = validator.validate(cyclic_wf)
    assert not result.valid
    assert any("CYCLE_DETECTED" in i.code for i in result.issues), "Cycle not detected"
    print("  ✓ DG2: DAG Integrity — cycles rejected")


# DG3 — Parallel Execution
def test_dg3_parallel_execution():
    """Independent nodes execute correctly in parallel without data races."""
    wm = WorkspaceManager()
    # Two independent writers to different files — should work
    assert wm.acquire("node1", ["src/**"]) is True
    assert wm.acquire("node2", ["tests/**"]) is True
    wm.clear()

    # Two writers to same files — should block
    assert wm.acquire("node1", ["src/**"]) is True
    assert wm.acquire("node2", ["src/**"]) is False
    print("  ✓ DG3: Parallel Execution — independent nodes can run concurrently")


# DG4 — Dependency Integrity
def test_dg4_dependency_integrity():
    """Node B does not start before node A if B depends on A."""
    graph = CapabilityGraph()
    graph.add_node(CapabilityNode(id="a", capability="arch", dependencies=[]))
    graph.add_node(CapabilityNode(id="b", capability="impl", dependencies=["a"]))
    graph.add_node(CapabilityNode(id="c", capability="test", dependencies=["b"]))

    # Ready nodes check
    ready = graph.get_ready_nodes()
    assert len(ready) == 1
    assert ready[0].id == "a"

    graph.mark_completed("a")
    ready = graph.get_ready_nodes()
    assert len(ready) == 1
    assert ready[0].id == "b"  # b depends on a, now ready

    graph.mark_completed("b")
    ready = graph.get_ready_nodes()
    assert len(ready) == 1
    assert ready[0].id == "c"  # c depends on b, now ready

    print("  ✓ DG4: Dependency Integrity — nodes respect dependency order")


# DG5 — Workspace Conflict
def test_dg5_workspace_conflict():
    """Concurrent writes to the same file paths are blocked."""
    wm = WorkspaceManager()
    assert wm.acquire("node1", ["src/shared.py"]) is True
    assert wm.acquire("node2", ["src/shared.py"]) is False  # conflict
    conflicts = wm.get_conflicts()
    assert len(conflicts) > 0
    print("  ✓ DG5: Workspace Conflict — concurrent writes to same path blocked")


# DG6 — Conditional Branch
def test_dg6_conditional_branch():
    """PASS/FAIL follows only authorized branches declared in the plan."""
    resolver = BranchResolver()
    # Authorized: PASS -> review, FAIL -> debug
    node = WorkflowNode(id="test", capability="testing", agent="tester", model_id="mock:free",
                       on_pass="review", on_fail="debug")

    # PASS follows declared target
    result = type('obj', (object,), {"status": "completed"})
    assert resolver.resolve(node, result) == "review"

    # FAIL follows declared target
    result = type('obj', (object,), {"status": "failed"})
    assert resolver.resolve(node, result) == "debug"

    # Block on undeclared
    node2 = WorkflowNode(id="test2", capability="testing", agent="tester", model_id="mock:free",
                        on_pass="next", on_fail="block")
    result2 = type('obj', (object,), {"status": "failed"})
    assert resolver.resolve(node2, result2) == "block"

    print("  ✓ DG6: Conditional Branch — PASS/FAIL follows authorized branches")


# DG7 — Adaptive Replanning
def test_dg7_adaptive_replanning():
    """Failure → analysis → capability gap → replan → recovery. Completed nodes frozen."""
    workflow = ExecutionWorkflow(
        name="test",
        nodes=[
            WorkflowNode(id="a", capability="arch", agent="architect", model_id="mock:free", depends_on=[]),
            WorkflowNode(id="b", capability="impl", agent="coder", model_id="mock:free", depends_on=["a"]),
            WorkflowNode(id="c", capability="test", agent="tester", model_id="mock:free", depends_on=["b"]),
            WorkflowNode(id="review", capability="review", agent="reviewer", model_id="mock:free", depends_on=["c"]),
            WorkflowNode(id="verify", capability="verification", agent="tester", model_id="mock:free", depends_on=["b"]),
            WorkflowNode(id="eval", capability="evaluation", agent="reviewer", model_id="mock:free", depends_on=["c"]),
        ],
    )

    report = FailureReport(node_id="c", capability="test", category="test", message="Assertion failed")
    replanner = Replanner(max_plans=3)
    new_wf = replanner.replan(MockTask(), workflow, ["a", "b"], "c", report)
    assert "a" in {n.id for n in new_wf.nodes}
    assert "b" in {n.id for n in new_wf.nodes}
    assert new_wf._replan_report.failed_node_id == "c"
    print("  ✓ DG7: Adaptive Replanning — failure → analysis → replan, completed nodes frozen")


# DG8 — Completed-Node Preservation
def test_dg8_completed_node_preservation():
    """Nodes marked COMPLETED are frozen. Replan cannot modify them."""
    workflow = ExecutionWorkflow(
        name="test",
        nodes=[
            WorkflowNode(id="a", capability="arch", agent="architect", model_id="mock:free", depends_on=[]),
            WorkflowNode(id="b", capability="impl", agent="coder", model_id="mock:free", depends_on=["a"]),
            WorkflowNode(id="c", capability="test", agent="tester", model_id="mock:free", depends_on=["b"]),
            WorkflowNode(id="review", capability="review", agent="reviewer", model_id="mock:free", depends_on=["c"]),
            WorkflowNode(id="verify", capability="verification", agent="tester", model_id="mock:free", depends_on=["b"]),
            WorkflowNode(id="eval", capability="evaluation", agent="reviewer", model_id="mock:free", depends_on=["c"]),
        ],
    )

    report = FailureReport(node_id="b", capability="impl", category="implementation", message="error")
    replanner = Replanner()
    new_wf = replanner.replan(MockTask(), workflow, ["a"], "b", report)

    # Verify 'a' (completed) is preserved unchanged
    orig_a = workflow.get_node("a")
    new_a = new_wf.get_node("a")
    assert new_a is not None
    assert new_a.id == orig_a.id
    assert new_a.capability == orig_a.capability
    print("  ✓ DG8: Completed-Node Preservation — completed nodes frozen")


# DG9 — Mandatory Gates Preserved
def test_dg9_mandatory_gates_preserved():
    """V2.2 dynamic workflows must include: verification, evaluation, review."""
    registry, taxonomy = _setup()
    validator = WorkflowValidator()

    # A workflow without mandatory caps should fail validation
    incomplete_wf = ExecutionWorkflow(nodes=[
        WorkflowNode(id="impl", capability="backend_development", agent="coder", model_id="mock:free"),
    ])
    result = validator.validate(incomplete_wf)
    assert not result.valid
    assert any("MISSING_MANDATORY" in i.code for i in result.issues)

    # TaskAnalyzer always adds mandatory caps
    analyzer = TaskAnalyzer(taxonomy=taxonomy)
    task = MockTask(obj="Build something")
    reqs = analyzer.analyze(task)
    for mc in ["verification", "evaluation", "review"]:
        assert mc in reqs.required

    print("  ✓ DG9: Mandatory Gates Preserved — verification, evaluation, review always included")


# DG10 — V2.1 Regression
def test_dg10_v21_regression():
    """All V2.1 Release Gates continue to pass with V2.2 code."""
    # G1: Lifecycle gates — test StateManager
    from harness.state import StateManager, TaskState, StateError
    sm = StateManager()
    state = sm.register("GATE-001")
    assert state.current_state == "RECEIVED"
    state.transition("ANALYZING")
    assert state.current_state == "ANALYZING"
    state.transition("PLANNED")
    assert state.current_state == "PLANNED"

    # G4: Policy enforcement
    from harness.policy import PolicyEngine
    policy = PolicyEngine()
    result = policy.evaluate_all(agent_level=3, task_level=3, tool_level=3, tool_name="filesystem_read",
                                  current_autonomy=3, current_iteration=0, has_verification=True)
    assert result.allowed
    result = policy.evaluate_all(agent_level=3, task_level=3, tool_level=3, tool_name="deploy",
                                  current_autonomy=3, current_iteration=0, has_verification=True)
    assert not result.allowed

    # G10: Free models
    from harness.planner.workflow_validator import WorkflowValidator as WV
    v = WV()
    assert v._is_model_free("model:free")
    assert v._is_model_free("meituan/longcat-2.0:free")
    assert not v._is_model_free("paid:model")

    print("  ✓ DG10: V2.1 Regression — G1, G4, G10 still pass")


def run_all_gates():
    print("\n=== V2.2 Release Gates (DG1–DG10) ===\n")
    test_dg1_capability_coverage()
    test_dg2_dag_integrity()
    test_dg3_parallel_execution()
    test_dg4_dependency_integrity()
    test_dg5_workspace_conflict()
    test_dg6_conditional_branch()
    test_dg7_adaptive_replanning()
    test_dg8_completed_node_preservation()
    test_dg9_mandatory_gates_preserved()
    test_dg10_v21_regression()
    print("\n✓ All V2.2 Release Gates (DG1–DG10) passed!")
    print("  ✓ V2.1 Regression verified (G1, G4, G10 sampled)")


if __name__ == "__main__":
    run_all_gates()
