"""Tests for the Benchmark Engine module (Phase B - V3.0)."""

import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.benchmark.engine import BenchmarkEngine, BenchmarkScenario, BenchmarkResult, BenchmarkError
from harness.benchmark.metrics import ExecutionMetrics, MetricsCollector
from harness.benchmark.report import BenchmarkReportGenerator


class MockOrchestrator:
    """Mock orchestrator that returns deterministic results for testing."""

    def __init__(self, should_pass=True):
        self.should_pass = should_pass

    def run(self, task_id: str) -> dict:
        return {
            "status": "COMPLETED" if self.should_pass else "FAILED",
            "task_id": task_id,
            "execution": {
                "status": "COMPLETED" if self.should_pass else "FAILED",
                "node_results": {
                    "design": {"capability": "api_design", "agent": "architect", "status": "completed"},
                    "implement": {"capability": "backend_development", "agent": "coder", "status": "completed"},
                    "test": {"capability": "testing", "agent": "tester", "status": "completed"},
                    "review": {"capability": "review", "agent": "reviewer", "status": "completed"},
                },
            },
        }


class MockTaskManager:
    """Mock task manager that stores/retrieves tasks."""

    def __init__(self):
        self._tasks = {}

    def create(self, data: dict) -> object:
        class MockTask:
            def __init__(self, d):
                self.metadata = type('obj', (object,), {'id': d['metadata']['id']})
                self.spec = type('obj', (object,), d['spec'])
        task = MockTask(data)
        self._tasks[data['metadata']['id']] = task
        return task

    def save(self, task, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({"id": task.metadata.id}, f)


def test_benchmark_scenario_creation():
    """Test creating a benchmark scenario."""
    scenario = BenchmarkScenario(
        id="TEST-001",
        name="Test Scenario",
        description="A test",
        task_objective="Implement a feature",
        acceptance_criteria=["tests pass", "review done"],
        allowed_changes=["src/**"],
        verification_required=["unit_tests"],
        autonomy_level=3,
        expected_capabilities=["backend_development", "testing"],
        tags=["test"],
    )
    assert scenario.id == "TEST-001"
    assert "backend_development" in scenario.expected_capabilities
    assert "tests pass" in scenario.acceptance_criteria


def test_benchmark_engine_registration():
    """Test registering scenarios with the engine."""
    engine = BenchmarkEngine(orchestrator=None, task_manager=None)
    s1 = BenchmarkScenario(id="S1", name="Scenario 1", description="", task_objective="Do stuff",
                           acceptance_criteria=["done"])
    engine.register_scenario(s1)
    assert "S1" in engine._scenarios

    # Duplicate registration
    try:
        engine.register_scenario(s1)
        assert False, "Should have raised"
    except BenchmarkError:
        pass


def test_benchmark_execution_pass():
    """Test executing a scenario that passes."""
    orch = MockOrchestrator(should_pass=True)
    task_mgr = MockTaskManager()
    engine = BenchmarkEngine(orchestrator=orch, task_manager=task_mgr)

    s1 = BenchmarkScenario(id="BM-PASS", name="Passing", description="",
                           task_objective="Do stuff", acceptance_criteria=["done"])
    engine.register_scenario(s1)
    result = engine.run_scenario("BM-PASS")

    assert result.status == "PASSED"
    assert result.execution_status == "COMPLETED"
    assert len(result.capabilities_used) == 4
    assert result.duration_seconds >= 0


def test_benchmark_execution_fail():
    """Test executing a scenario that fails."""
    orch = MockOrchestrator(should_pass=False)
    task_mgr = MockTaskManager()
    engine = BenchmarkEngine(orchestrator=orch, task_manager=task_mgr)

    s1 = BenchmarkScenario(id="BM-FAIL", name="Failing", description="",
                           task_objective="Do stuff", acceptance_criteria=["done"])
    engine.register_scenario(s1)
    result = engine.run_scenario("BM-FAIL")

    assert result.status == "FAILED"
    assert result.execution_status == "FAILED"


def test_benchmark_execution_error():
    """Test handling of execution errors."""
    class FailingOrchestrator:
        def run(self, task_id):
            raise RuntimeError("Orchestrator crashed")

    engine = BenchmarkEngine(orchestrator=FailingOrchestrator(), task_manager=MockTaskManager())
    s1 = BenchmarkScenario(id="BM-ERR", name="Error", description="",
                           task_objective="Do stuff", acceptance_criteria=["done"])
    engine.register_scenario(s1)
    result = engine.run_scenario("BM-ERR")

    assert result.status == "ERROR"
    assert "Orchestrator crashed" in (result.error or "")


def test_benchmark_run_all():
    """Test running multiple scenarios."""
    orch = MockOrchestrator(should_pass=True)
    task_mgr = MockTaskManager()
    engine = BenchmarkEngine(orchestrator=orch, task_manager=task_mgr)

    for i in range(3):
        s = BenchmarkScenario(id=f"BM-ALL-{i}", name=f"Scenario {i}", description="",
                              task_objective="Do stuff", acceptance_criteria=["done"])
        engine.register_scenario(s)

    results = engine.run_all()
    assert len(results) == 3
    assert all(r.status == "PASSED" for r in results)


def test_benchmark_summary():
    """Test benchmark summary generation."""
    engine = BenchmarkEngine(orchestrator=MockOrchestrator(should_pass=True), task_manager=MockTaskManager())
    for i in range(5):
        s = BenchmarkScenario(id=f"BM-SUM-{i}", name=f"S{i}", description="",
                              task_objective="Do stuff", acceptance_criteria=["done"])
        engine.register_scenario(s)

    engine.run_all()
    summary = engine.get_summary()
    assert summary["total_scenarios"] == 5
    assert summary["passed"] == 5
    assert summary["pass_rate"] == 100.0


def test_metrics_collector():
    """Test metrics collection and aggregation."""
    collector = MetricsCollector()

    results = [
        BenchmarkResult(scenario_id="A", scenario_name="A", status="PASSED",
                        duration_seconds=1.0, execution_status="completed",
                        capabilities_used=["cap1", "cap2"],
                        agent_assignments={"cap1": "agent1"}),
        BenchmarkResult(scenario_id="B", scenario_name="B", status="FAILED",
                        duration_seconds=2.0, execution_status="failed",
                        capabilities_used=["cap1"],
                        agent_assignments={"cap1": "agent2"}),
        BenchmarkResult(scenario_id="C", scenario_name="C", status="PASSED",
                        duration_seconds=3.0, execution_status="completed",
                        capabilities_used=["cap2", "cap3"],
                        agent_assignments={"cap2": "agent1"}),
    ]

    for r in results:
        collector.record_result(r)

    metrics = collector.get_metrics()
    assert metrics.execution_count == 3
    assert metrics.success_count == 2
    assert metrics.failure_count == 1
    assert metrics.avg_duration == 2.0
    assert metrics.capabilities_used == {"cap1": 2, "cap2": 2, "cap3": 1}
    assert metrics.agent_usage == {"agent1": 2, "agent2": 1}


def test_metrics_collector_empty():
    """Test metrics collector with no data."""
    collector = MetricsCollector()
    metrics = collector.get_metrics()
    assert metrics.execution_count == 0


def test_report_generation():
    """Test benchmark report generation."""
    generator = BenchmarkReportGenerator(output_dir="/tmp/bm_test_reports")
    results = [
        BenchmarkResult(scenario_id="A", scenario_name="Alpha", status="PASSED",
                        duration_seconds=1.5, execution_status="completed",
                        capabilities_used=["api_design", "backend_development"],
                        agent_assignments={"api_design": "architect", "backend_development": "coder"}),
        BenchmarkResult(scenario_id="B", scenario_name="Beta", status="FAILED",
                        duration_seconds=2.5, execution_status="failed",
                        capabilities_used=["testing"],
                        agent_assignments={"testing": "tester"},
                        error="Tests failed"),
    ]
    summary = {"total_scenarios": 2, "passed": 1, "failed": 1, "pass_rate": 50.0,
               "avg_duration_seconds": 2.0, "total_duration_seconds": 4.0}

    report = generator.generate(results, summary)
    assert "# Engineering Benchmark Report" in report
    assert "Alpha" in report
    assert "Beta" in report
    assert "50.0%" in report
    assert "api_design" in report

    path = generator.save(report, "test_report.md")
    assert os.path.exists(path)
    os.unlink(path)


def test_load_scenarios():
    """Test loading scenarios from JSON file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([
            {"id": "L1", "name": "Loaded 1", "description": "", "task_objective": "Test",
             "acceptance_criteria": ["done"]},
            {"id": "L2", "name": "Loaded 2", "description": "", "task_objective": "Test",
             "acceptance_criteria": ["done"]},
        ], f)
        fname = f.name

    engine = BenchmarkEngine(orchestrator=None, task_manager=None)
    count = engine.load_scenarios(fname)
    assert count == 2
    assert "L1" in engine._scenarios
    assert "L2" in engine._scenarios
    os.unlink(fname)
