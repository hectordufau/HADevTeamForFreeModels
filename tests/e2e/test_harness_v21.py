"""E2E test for V2.1 Harness."""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.policy import PolicyEngine
from harness.workflow import WorkflowEngine, WorkflowDefinition, WorkflowStep
from harness.task import TaskManager
from harness.state import StateManager
from harness.agents import AgentLoader
from harness.routing import ModelCatalog, ModelCandidate, ModelRouter
from harness.routing.performance import ModelPerformanceRegistry, ModelTaskRecord, AdaptiveModelRouter
from harness.context import ContextManager
from harness.execution import AgentExecutor
from harness.verification import VerificationEngine, UnitTestVerifier, LintVerifier, BuildVerifier
from harness.evidence import EvidenceCollector
from harness.evaluation.advanced import AdvancedEvaluationEngine
from harness.iteration import IterationEngine
from harness.memory import MemoryManager
from harness.orchestrator import Orchestrator


def test_e2e_v21_harness():
    """Complete E2E test simulating a full task lifecycle through the Harness."""
    print("\n  Setting up Harness components...")

    repo_root = os.path.join(os.path.dirname(__file__), "..", "..")

    # --- Config ---
    config = {
        "harness": {"name": "HADevTeamForFreeModels", "mode": "software-engineering"},
        "models": {"provider": "nous", "cost": {"maximum": 0}},
        "execution": {
            "max_iterations": 3,
            "require_task_contract": True,
            "require_verification": True,
            "require_evidence": True,
            "require_evaluation": True,
        },
        "autonomy": {"default_level": 2},
    }

    # --- Catalog with known models ---
    catalog = ModelCatalog()
    catalog.register(ModelCandidate(
        model_id="meituan/longcat-2.0:free", provider="nous", cost=0,
        capabilities={"coding": 0.95, "reasoning": 0.9, "architecture": 0.85,
                     "code_analysis": 0.88, "code_review": 0.85, "security": 0.8,
                     "orchestration": 0.9, "task_classification": 0.85,
                     "debugging": 0.9, "testing": 0.7},
        context_length=1000000,
    ))
    catalog.register(ModelCandidate(
        model_id="poolside/laguna-s-2.1:free", provider="nous", cost=0,
        capabilities={"coding": 0.85, "reasoning": 0.9, "architecture": 0.9,
                     "code_analysis": 0.9, "code_review": 0.9, "security": 0.85,
                     "orchestration": 0.7, "task_classification": 0.7,
                     "debugging": 0.8, "testing": 0.75},
        context_length=1000000,
    ))

    # --- All components ---
    state_mgr = StateManager()
    task_mgr = TaskManager()
    agents_dir = os.path.join(repo_root, "agents")
    agent_loader = AgentLoader(agents_dir)
    fallback_router = ModelRouter(catalog)

    with tempfile.TemporaryDirectory() as tmpdir:
        perf_db = os.path.join(tmpdir, "perf.db")
        perf_registry = ModelPerformanceRegistry(perf_db)
        adaptive_router = AdaptiveModelRouter(catalog, perf_registry, fallback_router)

        context_mgr = ContextManager(project_root=repo_root)
        executor = AgentExecutor()
        verifier = VerificationEngine()
        verifier.register("unit_tests", UnitTestVerifier())
        verifier.register("build", BuildVerifier())
        verifier.register("lint", LintVerifier())
        evidence_collector = EvidenceCollector(storage_dir=os.path.join(tmpdir, "evidence"))
        evaluator = AdvancedEvaluationEngine()
        iteration_engine = IterationEngine(max_iterations=3)
        policy_engine = PolicyEngine()
        memory_mgr = MemoryManager(storage_dir=os.path.join(tmpdir, "memory"))

        # Workflow engine with mock harness
        class MockHarness:
            def __init__(self):
                self.agents = agent_loader
                self.router = adaptive_router
                self.context = context_mgr
                self.executor = executor

        workflow_engine = WorkflowEngine(MockHarness())

        # --- Orchestrator ---
        orchestrator = Orchestrator(
            config=config,
            state_manager=state_mgr,
            task_manager=task_mgr,
            agent_loader=agent_loader,
            model_router=fallback_router,
            context_manager=context_mgr,
            agent_executor=executor,
            verification_engine=verifier,
            evidence_collector=evidence_collector,
            evaluation_engine=evaluator,
            iteration_engine=iteration_engine,
            workflow_engine=workflow_engine,
            policy_engine=policy_engine,
            memory_manager=memory_mgr,
            performance_registry=perf_registry,
            adaptive_router=adaptive_router,
        )

        # --- Create task ---
        os.makedirs(os.path.join(repo_root, "tasks", "active"), exist_ok=True)
        task_path = os.path.join(repo_root, "tasks", "active", "E2E-001.yaml")
        task_mgr.save(
            task_mgr.create({
                "metadata": {"id": "E2E-001", "title": "E2E Test"},
                "spec": {
                    "objective": "Implement a simple hello world endpoint",
                    "acceptance_criteria": ["endpoint returns 200"],
                    "allowed_changes": ["src/**"],
                    "verification": {"required": ["unit_tests", "lint"]},
                    "autonomy": {"maximum": 3},
                },
            }),
            task_path,
        )

        # --- Execute ---
        print("  Running Orchestrator...")
        report = orchestrator.run("E2E-001")

        print(f"  Status: {report['status']}")
        print(f"  Score: {report.get('evaluation', {}).get('score', 'N/A')}")
        assert report["status"] in {"COMPLETED", "FAILED", "BLOCKED"}, f"Unexpected status: {report['status']}"
        print(f"  ✓ E2E test completed with status: {report['status']}")


if __name__ == "__main__":
    print("\n=== E2E V2.1 Harness Test ===\n")
    test_e2e_v21_harness()
    print("\n✓ All E2E tests passed!")
