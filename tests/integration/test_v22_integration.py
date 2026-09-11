"""Integration tests for V2.2 pipeline: TaskAnalyzer -> Planner -> Validator -> ExecGraph."""
import sys, os, tempfile, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.capabilities.registry import CapabilityRegistry
from harness.capabilities.taxonomy import CapabilityTaxonomy
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


def _setup_registry_and_taxonomy():
    """Create a fully populated registry and taxonomy for integration tests."""
    registry = CapabilityRegistry()
    registry.register_agent("architect", ["system_design", "api_design", "domain_modeling", "integration_design"])
    registry.register_agent("coder", ["backend_development", "api_implementation", "database_schema",
                                       "frontend_development", "ui_implementation", "refactoring"])
    registry.register_agent("tester", ["testing", "unit_testing", "integration_testing", "e2e_testing", "verification"])
    registry.register_agent("reviewer", ["code_review", "security_analysis", "vulnerability_assessment",
                                          "evaluation", "review"])

    taxonomy = CapabilityTaxonomy()
    taxonomy_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "capability_taxonomy.yaml")
    if os.path.exists(taxonomy_path):
        taxonomy.load(taxonomy_path)

    return registry, taxonomy


class MockRouter:
    """Mock model router for testing."""
    def select(self, required_capabilities, role, **kw):
        from harness.routing import ModelSelection
        return ModelSelection(model_id="mock:free", provider="nous", cost=0, reason="test")

    def __init__(self):
        pass


class MockTask:
    """Mock TaskContract for testing."""
    def __init__(self, objective="", requirements=None, acceptance_criteria=None, verification=None, task_id="INT-001"):
        self.metadata = type('obj', (object,), {"id": task_id})
        self.spec = type('obj', (object,), {
            "objective": objective,
            "requirements": requirements or [],
            "acceptance_criteria": acceptance_criteria or [],
            "allowed_changes": [],
            "verification": verification,
            "autonomy": type('obj', (object,), {"maximum": 3}),
        })


def test_task_analyzer_to_planner():
    """Integration: TaskAnalyzer output feeds directly into WorkflowPlanner."""
    registry, taxonomy = _setup_registry_and_taxonomy()
    router = MockRouter()

    analyzer = TaskAnalyzer(taxonomy=taxonomy)
    planner = WorkflowPlanner(registry, taxonomy, router)

    task = MockTask(
        objective="Build a REST API for user management with authentication",
        requirements=["Must have database schema", "Must have unit tests"],
        acceptance_criteria=["Tests pass", "Security review completed"],
    )

    # Analyze
    requirements = analyzer.analyze(task)
    assert requirements.is_valid(), f"Invalid requirements: {requirements.validate()}"
    assert len(requirements.required) >= 3  # at least 1 extracted + 3 mandatory
    for mc in ["verification", "evaluation", "review"]:
        assert mc in requirements.required, f"Missing mandatory: {mc}"

    # Plan
    workflow = planner.plan(requirements, task=task)
    assert isinstance(workflow, ExecutionWorkflow)
    assert len(workflow.nodes) >= 3

    # Verify each node has agent + model
    for node in workflow.nodes:
        assert node.agent, f"Node {node.id} missing agent"
        assert node.model_id, f"Node {node.id} missing model_id"

    # Verify mandatory capabilities are in the plan
    node_caps = {n.capability for n in workflow.nodes}
    for mc in ["verification", "evaluation", "review"]:
        assert mc in node_caps, f"Mandatory cap {mc} missing from plan"

    print("  ✓ TaskAnalyzer -> WorkflowPlanner integration works")


def test_planner_to_validator():
    """Integration: WorkflowPlanner output validated by WorkflowValidator."""
    registry, taxonomy = _setup_registry_and_taxonomy()
    router = MockRouter()

    planner = WorkflowPlanner(registry, taxonomy, router)
    validator = WorkflowValidator(agent_registry=registry)

    task = MockTask(objective="Create a simple CRUD API")
    requirements = TaskAnalyzer(taxonomy=taxonomy).analyze(task)
    workflow = planner.plan(requirements, task=task)

    # Validate
    result = validator.validate(workflow)
    # Should pass if planner includes all mandatory caps
    if not result.valid:
        print(f"  Issues: {[str(i) for i in result.issues]}")
    assert result.valid, f"Validation failed: {[i.code for i in result.issues]}"

    print("  ✓ WorkflowPlanner -> WorkflowValidator integration works")


def test_validator_to_execution_graph():
    """Integration: Validated workflow can be executed by ExecutionGraph."""
    registry, taxonomy = _setup_registry_and_taxonomy()
    router = MockRouter()

    planner = WorkflowPlanner(registry, taxonomy, router)
    validator = WorkflowValidator(agent_registry=registry)

    task = MockTask(objective="Implement a simple API endpoint")
    requirements = TaskAnalyzer(taxonomy=taxonomy).analyze(task)
    workflow = planner.plan(requirements, task=task)

    result = validator.validate(workflow)
    assert result.valid, f"Validation failed: {[i.code for i in result.issues]}"

    # Now run via ExecutionGraph with mock orchestrator
    class MockExecutor:
        def execute(self, **kw):
            from harness.execution import ExecutionResult
            return ExecutionResult(status="completed", changes=["file.py"], claims=["done"])

    class MockContext:
        def build_context(self, **kw):
            from harness.context import ContextPack
            return ContextPack(task_id="INT-001", agent_role=kw.get("agent_role", "coder"))

    class MockAgents:
        def load(self, role):
            from harness.agents import AgentDefinition, AgentProfile
            return AgentDefinition(
                role=role, soul_content=f"# {role}",
                profile=AgentProfile(name=role, capabilities=["coding"], autonomy={"level": 3}),
            )

    class MockOrch:
        def __init__(self):
            self.agents = MockAgents()
            self.context = MockContext()
            self.executor = MockExecutor()
            self._current_task = task

    orch = MockOrch()
    graph = ExecutionGraph(workflow, orch)

    import asyncio
    exec_result = asyncio.run(graph.execute())
    assert exec_result["status"] in ("completed", "failed")
    assert exec_result["total_nodes"] == len(workflow.nodes)

    print(f"  ✓ Validator -> ExecutionGraph integration works (executed {exec_result['total_nodes']} nodes)")


def test_failure_analysis_to_replanning():
    """Integration: FailureAnalyzer output feeds into Replanner."""
    registry, taxonomy = _setup_registry_and_taxonomy()

    analyzer = FailureAnalyzer()
    replanner = Replanner(max_plans=3)

    # Create a workflow with the planner
    router = MockRouter()
    planner = WorkflowPlanner(registry, taxonomy, router)
    task = MockTask(objective="Build an API")
    requirements = TaskAnalyzer(taxonomy=taxonomy).analyze(task)
    workflow = planner.plan(requirements, task=task)

    # Simulate a failure
    node_id = "testing" if "testing" in {n.id for n in workflow.nodes} else workflow.nodes[0].id
    node = workflow.get_node(node_id)
    result = type('obj', (object,), {"status": "failed", "error": "AssertionError: test failed"})

    # Analyze failure
    report = analyzer.analyze(node, result)
    assert isinstance(report, FailureReport)
    assert report.category == "test"

    # Replan
    completed = [n.id for n in workflow.nodes if n.id != node_id]
    new_workflow = replanner.replan(task, workflow, completed, node_id, report)

    # Completed nodes preserved
    for nid in completed:
        assert nid in {n.id for n in new_workflow.nodes}

    print(f"  ✓ FailureAnalyzer -> Replanner integration works (replanned after {report.category} failure)")


def test_workspace_with_execution_graph():
    """Integration: WorkspaceManager prevents conflicts during parallel execution."""
    wm = WorkspaceManager()

    # Simulate two parallel nodes trying to write to the same files
    assert wm.acquire("node1", ["src/**"]) is True
    assert wm.acquire("node2", ["src/**"]) is False  # conflict

    conflicts = wm.get_conflicts()
    assert len(conflicts) >= 1
    assert conflicts[0].node_a == "node2"
    assert conflicts[0].node_b == "node1"

    # Release and retry
    wm.release("node1")
    assert wm.acquire("node2", ["src/**"]) is True

    print("  ✓ WorkspaceManager integration works")


def test_full_pipeline():
    """Full integration: Task -> Analyze -> Plan -> Validate -> Execute -> Failure -> Replan."""
    registry, taxonomy = _setup_registry_and_taxonomy()
    router = MockRouter()

    analyzer = TaskAnalyzer(taxonomy=taxonomy)
    planner = WorkflowPlanner(registry, taxonomy, router)
    validator = WorkflowValidator(agent_registry=registry)

    task = MockTask(
        objective="Build a user profile API with CRUD operations",
        acceptance_criteria=["API returns valid responses", "Security review passed"],
    )

    # Step 1: Analyze
    requirements = analyzer.analyze(task)
    assert requirements.is_valid()

    # Step 2: Plan
    workflow = planner.plan(requirements, task=task)
    assert len(workflow.nodes) >= 3

    # Step 3: Validate
    validation = validator.validate(workflow)
    assert validation.valid, f"Pipeline validation failed: {[i.code for i in validation.issues]}"

    # Step 4: Execute (with mock)
    class MockExec:
        def execute(self, **kw):
            from harness.execution import ExecutionResult
            return ExecutionResult(status="completed", changes=["f.py"], claims=["done"])
    class MockCtx:
        def build_context(self, **kw):
            from harness.context import ContextPack
            return ContextPack(task_id="INT-001", agent_role=kw.get("agent_role", "coder"))
    class MockAgt:
        def load(self, role):
            from harness.agents import AgentDefinition, AgentProfile
            return AgentDefinition(role=role, soul_content=f"# {role}",
                profile=AgentProfile(name=role, capabilities=["coding"], autonomy={"level": 3}))
    class MockOrchFull:
        def __init__(self):
            self.agents = MockAgt()
            self.context = MockCtx()
            self.executor = MockExec()
            self._current_task = task

    orch = MockOrchFull()
    graph = ExecutionGraph(workflow, orch)
    import asyncio
    exec_result = asyncio.run(graph.execute())
    assert exec_result["status"] in ("completed", "failed")

    # Step 5: Handle failure (if applicable)
    if exec_result["status"] == "failed":
        failure_analyzer = FailureAnalyzer()
        replanner = Replanner()
        failed_nodes = exec_result["failed_nodes"]
        for fnid in failed_nodes:
            fn = workflow.get_node(fnid)
            if fn:
                report = failure_analyzer.analyze(
                    fn,
                    {"status": "failed", "error": exec_result["node_results"][fnid].get("error", "unknown")},
                )
                completed = exec_result["completed_nodes"]
                new_wf = replanner.replan(task, workflow, completed, fnid, report)
                assert new_wf is not None

    print(f"  ✓ Full pipeline integration works (status={exec_result['status']}, {len(workflow.nodes)} nodes)")


if __name__ == "__main__":
    print("\n=== V2.2 Integration Tests ===\n")
    test_task_analyzer_to_planner()
    test_planner_to_validator()
    test_validator_to_execution_graph()
    test_failure_analysis_to_replanning()
    test_workspace_with_execution_graph()
    test_full_pipeline()
    print("\n✓ All V2.2 integration tests passed!")
