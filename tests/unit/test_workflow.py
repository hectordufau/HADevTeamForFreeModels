"""Tests for Workflow Engine."""
import sys, os, tempfile, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from harness.workflow import WorkflowEngine, WorkflowDefinition, WorkflowStep
from harness.task import TaskManager, TaskContract


class MockHarness:
    class MockAgents:
        def load(self, role):
            from harness.agents import AgentDefinition, AgentProfile
            return AgentDefinition(
                role=role,
                soul_content=f"# {role} soul",
                profile=AgentProfile(name=role, capabilities=["coding"], autonomy={"level": 3}),
            )

    class MockRouter:
        def select(self, required_capabilities, role, **kw):
            from harness.routing import ModelSelection
            return ModelSelection(model_id="mock:free", provider="nous", cost=0, reason="mock")

    class MockContext:
        def build_context(self, **kw):
            from harness.context import ContextPack
            return ContextPack(task_id="T001", agent_role=kw["agent_role"])

    class MockExecutor:
        def execute(self, **kw):
            from harness.execution import ExecutionResult
            return ExecutionResult(status="completed", changes=["file1.py"], claims=["done"])

    def __init__(self):
        self.agents = self.MockAgents()
        self.router = self.MockRouter()
        self.context = self.MockContext()
        self.executor = self.MockExecutor()


def test_workflow_definition():
    steps = [
        WorkflowStep(role="coder", required_capabilities=["coding"], on_pass="next", on_fail="feedback"),
        WorkflowStep(role="reviewer", required_capabilities=["code_review"], on_pass="complete", on_fail="block"),
    ]
    wf = WorkflowDefinition(name="test", steps=steps, max_iterations=3)
    assert len(wf.steps) == 2
    assert wf.max_iterations == 3


def test_workflow_execution():
    harness = MockHarness()
    engine = WorkflowEngine(harness)

    steps = [
        WorkflowStep(role="coder", required_capabilities=["coding"], on_pass="next", on_fail="feedback"),
        WorkflowStep(role="reviewer", required_capabilities=["code_review"], on_pass="complete", on_fail="block"),
    ]
    wf = WorkflowDefinition(name="test", steps=steps, max_iterations=3)

    mgr = TaskManager()
    task = mgr.create({
        "metadata": {"id": "TEST-001"},
        "spec": {"objective": "test", "acceptance_criteria": ["pass"]},
    })

    result = engine.execute(wf, task)
    assert result["status"] == "completed"
    assert len(result["steps"]) == 2
    assert result["steps"][0]["step"] == "coder"


def test_workflow_load_yaml():
    wf_yaml = {
        "name": "loaded",
        "steps": [
            {"role": "architect", "capabilities": ["architecture"], "on_pass": "next"},
            {"role": "coder", "capabilities": ["coding"], "on_pass": "complete"},
        ],
        "max_iterations": 2,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(wf_yaml, f)
        f.flush()
        harness = MockHarness()
        engine = WorkflowEngine(harness)
        wf = engine.load_definition(f.name)
        assert wf.name == "loaded"
        assert len(wf.steps) == 2
        assert wf.steps[0].role == "architect"
        os.unlink(f.name)


if __name__ == "__main__":
    test_workflow_definition()
    test_workflow_execution()
    test_workflow_load_yaml()
    print("All workflow tests passed!")
