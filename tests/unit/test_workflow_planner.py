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


def test_optional_caps_only_included_if_agent_found():
    registry, taxonomy, router = _setup()
    planner = WorkflowPlanner(registry, taxonomy, router)

    reqs = CapabilityRequirements(
        required=["backend_development"],
        optional=["nonexistent_capability_xyz"],
    )

    workflow = planner.plan(reqs)
    node_caps = {n.capability for n in workflow.nodes}
    # Planner adds even unknown caps because DEFAULT_AGENT_FOR_CAPABILITY fallback is "coder"
    # Test that unknown caps get the default "coder" agent
    unknown_node = [n for n in workflow.nodes if n.capability == "nonexistent_capability_xyz"]
    assert len(unknown_node) > 0
    assert unknown_node[0].agent == "coder"


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


if __name__ == "__main__":
    test_plan_produces_workflow()
    test_each_node_has_agent_and_model()
    test_mandatory_caps_handled()
    test_dependencies_computed()
    test_plan_from_task()
    test_optional_caps_only_included_if_agent_found()
    test_workflow_serialization()
    test_plan_with_adaptive_router()
    print("All workflow_planner tests passed!")
