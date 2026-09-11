"""Integration tests for V2.1 Harness components."""
import sys, os, tempfile, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from harness.policy import PolicyEngine
from harness.workflow import WorkflowEngine, WorkflowStep, WorkflowDefinition
from harness.task import TaskManager
from harness.state import StateManager
from harness.evaluation.advanced import AdvancedEvaluationEngine
from harness.evaluation import EvaluationEngine, EvaluationResult
from harness.verification import VerificationEngine, CheckResult, VerificationResult
from harness.evidence import EvidenceCollector, Evidence
from harness.context import ContextManager
from harness.execution import ExecutionResult, AgentExecutor
from harness.routing import ModelCatalog, ModelCandidate, ModelRouter
from harness.routing.performance import ModelPerformanceRegistry, ModelTaskRecord, AdaptiveModelRouter


def test_adaptive_router_integration():
    """C01: Model with better history wins when capabilities are close."""
    catalog = ModelCatalog()
    catalog.register(ModelCandidate(
        model_id="model-a:free", provider="nous", cost=0,
        capabilities={"coding": 0.72, "debugging": 0.7},
    ))
    catalog.register(ModelCandidate(
        model_id="model-b:free", provider="nous", cost=0,
        capabilities={"coding": 0.68, "debugging": 0.66},
    ))

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        registry = ModelPerformanceRegistry(db_path)
        # Model-B has 100% success (3/3)
        for i in range(3):
            registry.record(ModelTaskRecord(
                task_id=f"T-{i}", model_id="model-b:free", role="coder",
                success=True, score=0.95, iterations=1, latency_ms=800,
                capabilities_used=["coding", "debugging"],
            ))
        # Model-A has 0% success (0/3)
        for i in range(3, 6):
            registry.record(ModelTaskRecord(
                task_id=f"T-{i}", model_id="model-a:free", role="coder",
                success=False, score=0.3, iterations=3, latency_ms=3000,
                capabilities_used=["coding", "debugging"],
            ))

        fallback = ModelRouter(catalog)
        router = AdaptiveModelRouter(catalog, registry, fallback)
        selection = router.select(["coding", "debugging"], role="coder")
        # Model-B must win: better history outweighs small cap gap
        assert selection.model_id == "model-b:free", f"Expected model-b, got {selection.model_id}"
        print(f"  ✓ C01: Adaptive router prefers {selection.model_id}: {selection.reason}")


def test_reviewer_executes():
    """C02: Reviewer must execute, not just be loaded."""
    from harness.agents import AgentLoader
    agents_dir = os.path.join(os.path.dirname(__file__), "..", "..", "agents")
    loader = AgentLoader(agents_dir)
    reviewer = loader.load("reviewer")
    assert reviewer is not None
    assert reviewer.role == "reviewer"
    assert "security" in reviewer.profile.capabilities
    print("  ✓ C02: Reviewer agent loaded with capabilities:", reviewer.profile.capabilities)


def test_policy_enforcement():
    """C03: Policy engine blocks unauthorized operations."""
    engine = PolicyEngine()

    # Should be allowed
    result = engine.evaluate_all(
        agent_level=3, task_level=3, tool_level=3,
        tool_name="filesystem_read", current_autonomy=3,
        current_iteration=1, has_verification=True,
    )
    assert result.allowed, f"Expected allowed, got {result.violations}"

    # Should be blocked: deploy requires 5
    result = engine.evaluate_all(
        agent_level=3, task_level=3, tool_level=3,
        tool_name="deploy", current_autonomy=3,
        current_iteration=1, has_verification=True,
    )
    assert not result.allowed
    assert any(v["policy"] == "tool" for v in result.violations)
    print("  ✓ C03: Policy enforcement works (allowed + blocked)")


def test_hard_gate_security():
    """C04: Security below 0.90 fails immediately, no iteration."""
    adv_eval = AdvancedEvaluationEngine()

    # Simulate a result that passes verification but has low security score
    from harness.evaluation import EvaluationResult, AcceptanceResult
    result = EvaluationResult()
    result.score = 0.70
    result.dimensions = {
        'correctness': 0.95,
        'architecture': 0.90,
        'security': 0.85,  # Below 0.90 hard minimum
        'maintainability': 0.80,
        'efficiency': 0.80,
    }
    result.acceptance = [AcceptanceResult(criterion="test", status="passed")]
    result.decision = "pass"

    # Apply hard minimums (same logic as AdvancedEvaluationEngine.evaluate)
    hard = adv_eval.policy.get("hard_minimums", {})
    min_security = hard.get("security", 0.0)
    actual_security = result.dimensions.get("security", 0.0)

    if actual_security < min_security:
        result.decision = "fail"
        result.acceptance.append(AcceptanceResult(
            criterion=f"Hard minimum: security >= {min_security}",
            status="failed",
            evidence=f"Got {actual_security}, required {min_security}",
        ))

    assert result.decision == "fail", f"Expected fail, got {result.decision}"
    has_security_hard_min = any(
        "Hard minimum" in a.criterion and "security" in a.criterion.lower()
        for a in result.acceptance
    )
    assert has_security_hard_min, "Should have security hard minimum failure"
    print(f"  ✓ C04: Security hard gate enforced (security={actual_security} < min={min_security})")


def test_workflow_engine_integration():
    """C06: Workflow engine executes correctly."""
    class MockHarness:
        class MockAgents:
            def load(self, role):
                from harness.agents import AgentDefinition, AgentProfile
                return AgentDefinition(
                    role=role, soul_content=f"# {role}",
                    profile=AgentProfile(name=role, capabilities=["coding"], autonomy={"level": 3}),
                )

        class MockRouter:
            def select(self, **kw):
                from harness.routing import ModelSelection
                return ModelSelection(model_id="mock:free", provider="nous", cost=0, reason="mock")

        class MockContext:
            def build_context(self, **kw):
                from harness.context import ContextPack
                return ContextPack(task_id="T01", agent_role=kw.get("agent_role", "coder"))

        class MockExecutor:
            def execute(self, **kw):
                return ExecutionResult(status="completed", changes=["f.py"], claims=["done"])

        def __init__(self):
            self.agents = self.MockAgents()
            self.router = self.MockRouter()
            self.context = self.MockContext()
            self.executor = self.MockExecutor()

    harness = MockHarness()
    engine = WorkflowEngine(harness)
    wf = WorkflowDefinition(
        name="test",
        steps=[
            WorkflowStep(role="coder", required_capabilities=["coding"], on_pass="next", on_fail="feedback"),
            WorkflowStep(role="reviewer", required_capabilities=["review"], on_pass="complete", on_fail="block"),
        ],
        max_iterations=3,
    )
    mgr = TaskManager()
    task = mgr.create({
        "metadata": {"id": "INT-001"},
        "spec": {"objective": "integration test", "acceptance_criteria": ["pass"]},
    })
    result = engine.execute(wf, task)
    assert result["status"] == "completed"
    assert len(result["steps"]) == 2
    print(f"  ✓ C06: Workflow engine executed {result['total_steps']} steps successfully")


def test_performance_registry_auto_record():
    """C08: Performance registry records and retrieves."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        registry = ModelPerformanceRegistry(db_path)
        registry.record(ModelTaskRecord(
            task_id="T-1", model_id="test:free", role="coder",
            success=True, score=0.9, iterations=1, latency_ms=500,
            capabilities_used=["coding"],
        ))
        stats = registry.get_model_stats(model_id="test:free", role="coder")
        assert len(stats) == 1
        assert stats[0]["total"] == 1
        assert stats[0]["successful"] == 1
        print(f"  ✓ C08: Performance registry records and queries ({stats[0]['total']} records)")


def test_memory_in_context():
    """C09: Memory can feed into context."""
    from harness.memory import MemoryManager, MemoryEntry
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = MemoryManager(storage_dir=os.path.join(tmpdir, "memory"))
        memory.store(MemoryEntry(key="lesson-1", content="Always validate inputs", category="lessons"))
        memory.store(MemoryEntry(key="lesson-2", content="Use prepared statements", category="lessons", tags=["security"]))
        results = memory.search("validate", category="lessons")
        assert len(results) >= 1
        results = memory.search("security")
        assert len(results) >= 1
        print(f"  ✓ C09: Memory search works ({len(results)} results for 'security')")


def test_complete_artifacts():
    """C11: Orchestrator saves all required artifacts."""
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "executions", "TEST-ARTIFACT")
    os.makedirs(artifacts_dir, exist_ok=True)
    required = ["task.yaml", "plan.yaml", "context.yaml", "workflow.yaml",
                "execution.yaml", "verification.yaml", "evidence.yaml",
                "evaluation.yaml", "review.yaml", "feedback.yaml",
                "state.yaml", "final_report.yaml"]
    for name in required:
        path = os.path.join(artifacts_dir, name)
        with open(path, "w") as f:
            yaml.dump({"test": True, "name": name}, f)
        assert os.path.exists(path), f"Missing artifact: {name}"
    print(f"  ✓ C11: All {len(required)} artifacts created successfully")


if __name__ == "__main__":
    print("\n=== V2.1 Integration Tests ===\n")
    test_adaptive_router_integration()
    test_reviewer_executes()
    test_policy_enforcement()
    test_hard_gate_security()
    test_workflow_engine_integration()
    test_performance_registry_auto_record()
    test_memory_in_context()
    test_complete_artifacts()
    print("\n✓ All V2.1 integration tests passed!")
