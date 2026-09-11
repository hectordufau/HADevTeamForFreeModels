# Tests for ExecutionGraph
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.planner.exec_graph import ExecutionGraph, NodeExecutionResult, ExecutionGraphError
from harness.planner.workflow_planner import WorkflowNode, ExecutionWorkflow


class MockOrchestrator:
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
            return ContextPack(task_id="T001", agent_role=kw.get("agent_role", "coder"))

    class MockExecutor:
        def execute(self, **kw):
            from harness.execution import ExecutionResult
            return ExecutionResult(status="completed", changes=["file.py"], claims=["done"])

    def __init__(self):
        self.agents = self.MockAgents()
        self.router = self.MockRouter()
        self.context = self.MockContext()
        self.executor = self.MockExecutor()
        self._current_task = None


def test_execute_sequential():
    orch = MockOrchestrator()
    workflow = ExecutionWorkflow(
        name="seq",
        nodes=[
            WorkflowNode(id="a", capability="arch", agent="coder", model_id="mock:free", depends_on=[]),
            WorkflowNode(id="b", capability="impl", agent="coder", model_id="mock:free", depends_on=["a"]),
            WorkflowNode(id="c", capability="test", agent="coder", model_id="mock:free", depends_on=["b"]),
        ],
    )
    graph = ExecutionGraph(workflow, orch)
    import asyncio
    result = asyncio.run(graph.execute())
    assert result["status"] == "completed"
    assert len(result["completed_nodes"]) == 3


def test_parallel_execution():
    orch = MockOrchestrator()
    workflow = ExecutionWorkflow(
        name="par",
        nodes=[
            WorkflowNode(id="a", capability="arch", agent="coder", model_id="mock:free", depends_on=[]),
            WorkflowNode(id="b", capability="impl", agent="coder", model_id="mock:free", depends_on=[]),
            WorkflowNode(id="c", capability="test", agent="coder", model_id="mock:free", depends_on=["a", "b"]),
        ],
    )
    graph = ExecutionGraph(workflow, orch)
    import asyncio
    result = asyncio.run(graph.execute())
    assert result["status"] == "completed"
    assert len(result["completed_nodes"]) == 3


def test_node_failure_handling():
    class FailExecutor:
        def execute(self, **kw):
            from harness.execution import ExecutionResult
            return ExecutionResult(status="failed", changes=[], claims=[], error="Test failure")

    class FailOrch:
        class MockAgents:
            def load(self, role):
                from harness.agents import AgentDefinition, AgentProfile
                return AgentDefinition(
                    role=role, soul_content=f"# {role}",
                    profile=AgentProfile(name=role, capabilities=["coding"], autonomy={"level": 3}),
                )

        class MockContext:
            def build_context(self, **kw):
                from harness.context import ContextPack
                return ContextPack(task_id="T001", agent_role=kw.get("agent_role", "coder"))

        def __init__(self):
            self.agents = self.MockAgents()
            self.executor = FailExecutor()
            self.context = self.MockContext()
            self._current_task = None

    orch = FailOrch()
    workflow = ExecutionWorkflow(
        name="fail",
        nodes=[
            WorkflowNode(id="a", capability="arch", agent="coder", model_id="mock:free", depends_on=[]),
        ],
    )
    graph = ExecutionGraph(workflow, orch)
    import asyncio
    result = asyncio.run(graph.execute())
    assert result["status"] == "failed"
    assert "a" in result["failed_nodes"]


def test_result_aggregation():
    orch = MockOrchestrator()
    workflow = ExecutionWorkflow(
        name="agg",
        nodes=[
            WorkflowNode(id="a", capability="arch", agent="coder", model_id="mock:free", depends_on=[]),
            WorkflowNode(id="b", capability="impl", agent="coder", model_id="mock:free", depends_on=["a"]),
        ],
    )
    graph = ExecutionGraph(workflow, orch)
    import asyncio
    result = asyncio.run(graph.execute())
    assert "node_results" in result
    assert "a" in result["node_results"]
    assert "b" in result["node_results"]
    assert result["node_results"]["a"]["status"] == "completed"
    assert result["node_results"]["b"]["status"] == "completed"


def test_dependency_order():
    orch = MockOrchestrator()
    # Node b depends on a, node c depends on b
    workflow = ExecutionWorkflow(
        name="order",
        nodes=[
            WorkflowNode(id="b", capability="impl", agent="coder", model_id="mock:free", depends_on=["a"]),
            WorkflowNode(id="c", capability="test", agent="coder", model_id="mock:free", depends_on=["b"]),
            WorkflowNode(id="a", capability="arch", agent="coder", model_id="mock:free", depends_on=[]),
        ],
    )
    graph = ExecutionGraph(workflow, orch)
    import asyncio
    result = asyncio.run(graph.execute())
    assert result["status"] == "completed"
    assert len(result["completed_nodes"]) == 3


def test_stalled_dependency():
    class FailExecutor2:
        def execute(self, **kw):
            from harness.execution import ExecutionResult
            return ExecutionResult(status="failed", changes=[], claims=[], error="fail")

    class FailOrch2:
        class MockAgents:
            def load(self, role):
                from harness.agents import AgentDefinition, AgentProfile
                return AgentDefinition(
                    role=role, soul_content=f"# {role}",
                    profile=AgentProfile(name=role, capabilities=["coding"], autonomy={"level": 3}),
                )

        class MockContext:
            def build_context(self, **kw):
                from harness.context import ContextPack
                return ContextPack(task_id="T001", agent_role=kw.get("agent_role", "coder"))

        def __init__(self):
            self.agents = self.MockAgents()
            self.executor = FailExecutor2()
            self.context = self.MockContext()
            self._current_task = None

    orch = FailOrch2()
    workflow = ExecutionWorkflow(
        name="stall",
        nodes=[
            WorkflowNode(id="a", capability="arch", agent="coder", model_id="mock:free", depends_on=[]),
            WorkflowNode(id="b", capability="impl", agent="coder", model_id="mock:free", depends_on=["a"]),
        ],
    )
    graph = ExecutionGraph(workflow, orch)
    import asyncio
    result = asyncio.run(graph.execute())
    assert result["status"] == "failed"


if __name__ == "__main__":
    test_execute_sequential()
    test_parallel_execution()
    test_node_failure_handling()
    test_result_aggregation()
    test_dependency_order()
    test_stalled_dependency()
    print("All exec_graph tests passed!")
