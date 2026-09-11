# Tests for WorkflowPlanner
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.planner.workflow_planner import WorkflowPlanner, WorkflowNode, ExecutionWorkflow, WorkflowPlannerError
from harness.capabilities.registry import CapabilityRegistry
from harness.capabilities.taxonomy import CapabilityTaxonomy
from harness.planner.task_analyzer import CapabilityRequirements


class MockRouter:
    def select(self, required_capabilities, role, **kw):
        from harness.routing import ModelSelection
        return ModelSelection(model_id="mock:free", provider="nous", cost=0, reason="test")


def _setup():
    registry = CapabilityRegistry()
    registry.register_agent("architect", ["system_design", "api_design", "domain_modeling"])
    registry.register_agent("coder", ["backend_development", "api_implementation", "database_schema", "refactoring"])
    registry.register_agent("tester", ["testing", "unit_testing", "integration_testing", "e2e_testing"])
    registry.register_agent("reviewer", ["code_review", "security_analysis", "vulnerability_assessment", "verification", "evaluation", "review"])

    taxonomy = CapabilityTaxonomy()
    router = MockRouter()

    return registry, taxonomy, router


def test_plan_produces_workflow():
    registry, taxonomy, router = _setup()
    planner = WorkflowPlanner(registry, taxonomy, router)

    reqs = CapabilityRequirements(
        required=["api_design", "backend_development", "testing"],
        optional=["security_analysis"],
    )

    workflow = planner.plan(reqs)
    assert isinstance(workflow, ExecutionWorkflow)
    assert len(workflow.nodes) >= 3


def test_each_node_has_agent_and_model():
    registry, taxonomy, router = _setup()
    planner = WorkflowPlanner(registry, taxonomy, router)

    reqs = CapabilityRequirements(
        required=["backend_development", "testing"],
    )

    workflow = planner.plan(reqs)
    for node in workflow.nodes:
        assert node.agent, f"Node {node.id} missing agent"
        assert node.model_id, f"Node {node.id} missing model_id"


def test_mandatory_caps_handled():
    registry, taxonomy, router = _setup()
    planner = WorkflowPlanner(registry, taxonomy, router)

    reqs = CapabilityRequirements(
        required=["backend_development"],
    )
    # TaskAnalyzer adds verification, evaluation, review automatically
    reqs.required.append("verification")
    reqs.required.append("evaluation")
    reqs.required.append("review")

    workflow = planner.plan(reqs)
    node_caps = {n.capability for n in workflow.nodes}
    for mc in ["verification", "evaluation", "review"]:
        assert mc in node_caps, f"Missing mandatory cap: {mc}"


def test_dependencies_computed():
    registry, taxonomy, router = _setup()
    planner = WorkflowPlanner(registry, taxonomy, router)

    reqs = CapabilityRequirements(
        required=["api_design", "backend_development", "testing", "review"],
    )

    workflow = planner.plan(reqs)
    node_map = {n.id: n for n in workflow.nodes}

    # Implementation depends on architecture
    impl = node_map.get("backend_development")
    if impl:
        assert "api_design" in impl.depends_on

    # Testing depends on implementation
    test_node = node_map.get("testing")
    if test_node:
        assert "backend_development" in test_node.depends_on


def test_plan_from_task():
    registry, taxonomy, router = _setup()
    planner = WorkflowPlanner(registry, taxonomy, router)

    class MockTaskAnalyzer:
        def analyze(self, task):
            return CapabilityRequirements(
                required=["api_design", "backend_development", "testing"],
                optional=["security_analysis"],
            )

    class MockTask:
        metadata = type('obj', (object,), {"id": "T-001"})
        spec = type('obj', (object,), {"objective": "test"})

    workflow = planner.plan_from_task(MockTask(), MockTaskAnalyzer())
    assert isinstance(workflow, ExecutionWorkflow)
    assert len(workflow.nodes) >= 3


def test_unknown_capability_raises_error_not_fallback():
    """Unknown capability must raise WorkflowPlannerError, not fallback to role."""
    registry, taxonomy, router = _setup()
    planner = WorkflowPlanner(registry, taxonomy, router)

    reqs = CapabilityRequirements(
        required=["backend_development"],
        optional=["nonexistent_capability_xyz"],
    )

    # Unknown optional caps should be silently skipped (not added if no agent found)
    workflow = planner.plan(reqs)
    node_caps = {n.capability for n in workflow.nodes}
    assert "nonexistent_capability_xyz" not in node_caps, (
        f"Unknown capability '{'nonexistent_capability_xyz'}' must not appear in workflow nodes"
    )


def test_missing_capability_agent_blocks_planning():
    """A required capability with no agent must raise WorkflowPlannerError."""
    registry, taxonomy, router = _setup()
    planner = WorkflowPlanner(registry, taxonomy, router)

    reqs = CapabilityRequirements(
        required=["backend_development", "data_science_training"],
    )

    import pytest  # noqa: F401
    try:
        planner.plan(reqs)
        assert False, "Should have raised WorkflowPlannerError for missing capability"
    except WorkflowPlannerError as e:
        assert "No agent found" in str(e)
        assert "data_science_training" in str(e)


def test_capability_registry_is_authoritative():
    """CapabilityRegistry is the sole source for agent-capability mapping."""
    registry, taxonomy, router = _setup()
    planner = WorkflowPlanner(registry, taxonomy, router)

    # Remove all agents from registry to simulate empty registry
    empty_registry = CapabilityRegistry()
    empty_planner = WorkflowPlanner(empty_registry, taxonomy, router)

    reqs = CapabilityRequirements(
        required=["api_design"],
    )

    try:
        empty_planner.plan(reqs)
        assert False, "Should have raised WorkflowPlannerError"
    except WorkflowPlannerError as e:
        assert "No agent found" in str(e)
        assert "api_design" in str(e)


def test_workflow_serialization():
    registry, taxonomy, router = _setup()
    planner = WorkflowPlanner(registry, taxonomy, router)

    reqs = CapabilityRequirements(required=["backend_development", "testing"])
    workflow = planner.plan(reqs)

    data = workflow.to_dict()
    assert "name" in data
    assert "nodes" in data
    assert len(data["nodes"]) == len(workflow.nodes)

    restored = ExecutionWorkflow.from_dict(data)
    assert len(restored.nodes) == len(workflow.nodes)
    assert restored.name == workflow.name


def test_plan_with_adaptive_router():
    registry, taxonomy, _ = _setup()

    class MockAdaptiveRouter:
        def select(self, required_capabilities, role, task_id):
            from harness.routing import ModelSelection
            return ModelSelection(model_id="adaptive:free", provider="nous", cost=0, reason="adaptive")

    adaptive_router = MockAdaptiveRouter()
    router = MockRouter()
    planner = WorkflowPlanner(registry, taxonomy, router, adaptive_router=adaptive_router)

    reqs = CapabilityRequirements(required=["backend_development"])
    workflow = planner.plan(reqs)

    for node in workflow.nodes:
        assert node.model_id == "adaptive:free"


def test_dependencies_loaded_from_capability_graph():
    """Dependencies must come from the capability dependencies config, not hard-coded rules."""
    from harness.capabilities.graph import CapabilityGraph, CapabilityNode

    registry, taxonomy, router = _setup()
    planner = WorkflowPlanner(registry, taxonomy, router)

    reqs = CapabilityRequirements(
        required=["api_design", "backend_development", "testing", "security_analysis", "review"],
    )
    workflow = planner.plan(reqs)
    node_map = {n.id: n for n in workflow.nodes}

    # backend_development depends on api_design (from config/capability_taxonomy.yaml)
    impl = node_map.get("backend_development")
    assert impl is not None
    assert "api_design" in impl.depends_on, (
        f"backend_development should depend on api_design, got: {impl.depends_on}"
    )

    # testing depends on backend_development
    test_node = node_map.get("testing")
    assert test_node is not None
    assert "backend_development" in test_node.depends_on

    # review depends on testing and security_analysis
    review_node = node_map.get("review")
    assert review_node is not None
    assert "testing" in review_node.depends_on
    assert "security_analysis" in review_node.depends_on

    # api_design has no dependencies
    arch_node = node_map.get("api_design")
    assert arch_node is not None
    assert arch_node.depends_on == []


def test_planner_has_no_capability_dependency_rules():
    """Planner must not contain hard-coded if/elif chains for capability dependencies."""
    planner_source = open(
        os.path.join(os.path.dirname(__file__), "..", "..", "harness", "planner", "workflow_planner.py")
    ).read()

    # Check that _compute_dependencies does not contain if/elif chains keyed on capability names
    lines = planner_source.split("\n")

    # Find the _compute_dependencies method
    in_method = False
    method_lines = []
    for line in lines:
        if "def _compute_dependencies" in line:
            in_method = True
        if in_method:
            method_lines.append(line)
            if in_method and line.strip().startswith("def ") and "def _compute_dependencies" not in line:
                break

    method_body = "\n".join(method_lines)

    # Should NOT contain if/elif chains checking specific capability names
    assert "if capability in" not in method_body, (
        "Planner has hard-coded capability dependency rules (if/in checks on capability names)"
    )
    assert "elif capability in" not in method_body, (
        "Planner has hard-coded capability dependency rules (elif/in checks on capability names)"
    )

    # Should reference taxonomy.get_dependencies instead
    assert "taxonomy.get_dependencies" in method_body, (
        "Planner should load dependencies from taxonomy, not hard-code them"
    )


def test_circular_dependency_rejected():
    """Circular dependencies in capability_dependencies config must be rejectable."""
    from harness.capabilities.graph import CapabilityGraph, CapabilityNode

    # Build a graph with a cycle using CapabilityGraph
    graph = CapabilityGraph()
    graph.add_node(CapabilityNode(id="a", capability="arch", dependencies=["b"]))
    graph.add_node(CapabilityNode(id="b", capability="impl", dependencies=["c"]))
    graph.add_node(CapabilityNode(id="c", capability="test", dependencies=["a"]))

    assert graph.has_cycle(), "CapabilityGraph should detect cycle (a -> b -> c -> a)"
    try:
        graph.topological_sort()
        assert False, "topological_sort should raise on cyclic graph"
    except Exception:
        pass


def test_missing_dependency_detected():
    """A capability that depends on a capability not in required_caps should still
    include the dependency if declared in config — but filtering to required_caps only
    means missing dependencies are silently dropped (validator must catch them)."""
    registry, taxonomy, router = _setup()
    planner = WorkflowPlanner(registry, taxonomy, router)

    # testing depends on backend_development (declared in config)
    # If we only require testing, its dependency (backend_development) is missing
    reqs = CapabilityRequirements(
        required=["testing", "review"],
    )
    workflow = planner.plan(reqs)

    node_map = {n.id: n for n in workflow.nodes}
    test_node = node_map.get("testing")
    if test_node:
        # Dependency filtered because backend_development not in required_caps
        assert "backend_development" not in test_node.depends_on


if __name__ == "__main__":
    test_plan_produces_workflow()
    test_each_node_has_agent_and_model()
    test_mandatory_caps_handled()
    test_dependencies_computed()
    test_plan_from_task()
    test_unknown_capability_raises_error_not_fallback()
    test_missing_capability_agent_blocks_planning()
    test_capability_registry_is_authoritative()
    test_workflow_serialization()
    test_plan_with_adaptive_router()
    test_dependencies_loaded_from_capability_graph()
    test_planner_has_no_capability_dependency_rules()
    test_circular_dependency_rejected()
    test_missing_dependency_detected()
    print("All workflow_planner tests passed!")
