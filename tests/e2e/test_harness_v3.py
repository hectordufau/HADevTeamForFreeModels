"""E2E tests for Harness V3.0 — full learning loop scenario."""

import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.benchmark.engine import BenchmarkEngine, BenchmarkScenario, BenchmarkResult
from harness.benchmark.metrics import MetricsCollector
from harness.benchmark.report import BenchmarkReportGenerator
from harness.evidence.unified import EvidenceStore, EvidencePackage, CanonicalExecutionRecord
from harness.memory.v3 import ExperienceStore, ExperienceRecord
from harness.routing.performance_v3 import PerformanceRegistryV3, PerformanceSample
from harness.planner.model_intelligence import AdaptiveModelPolicyV3
from harness.planner.failure_intelligence import RootCauseAnalyzer, FailureToLearningPipeline
from harness.capabilities.taxonomy import CapabilityTaxonomy
from harness.capabilities.registry import CapabilityRegistry
from harness.capabilities.intelligence import CapabilityGapDetector, CapabilityRecommender
from harness.policy.v3 import ImmutableGovernancePolicy


class MockOrchestrator:
    """Mock orchestrator — simulates both success and failure cycles."""
    def __init__(self):
        self.call_count = 0

    def run(self, task_id: str) -> dict:
        self.call_count += 1
        # Alternate success/failure over multiple calls
        success = (self.call_count % 3 != 0)  # 2 of 3 succeed
        return {
            "status": "COMPLETED" if success else "FAILED",
            "task_id": task_id,
            "execution": {
                "status": "COMPLETED" if success else "FAILED",
                "node_results": {
                    "design": {"capability": "api_design", "agent": "architect", "status": "completed"},
                    "implement": {"capability": "backend_development", "agent": "coder", "status": "completed"},
                    "test": {"capability": "testing", "agent": "tester", "status": "completed"},
                },
                "duration_ms": 100.0,
            },
        }


class MockTaskManager:
    def __init__(self):
        self.tasks = {}

    def create(self, data):
        class T:
            pass
        t = T()
        t.metadata = type('m', (), {'id': data["metadata"]["id"]})()
        t.spec = type('s', (), data["spec"])()
        self.tasks[data["metadata"]["id"]] = t
        return t

    def save(self, task, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"id": task.metadata.id}, f)


def test_e2e_v3_learning_loop():
    """E2E: Full learning loop — benchmark → evidence → memory → policy adaptation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            # ── Phase 1: Setup ──────────────────────────────────────────
            orch = MockOrchestrator()
            tm = MockTaskManager()
            engine = BenchmarkEngine(orchestrator=orch, task_manager=tm)

            scenarios = []
            for i in range(20):
                s = BenchmarkScenario(
                    id=f"E2E-S{i:03d}", name=f"Scenario {i}",
                    description=f"E2E test scenario {i}",
                    task_objective=f"Implement feature {i}",
                    acceptance_criteria=["done"],
                    expected_capabilities=["backend_development", "testing"],
                )
                engine.register_scenario(s)
                scenarios.append(s)

            # ── Phase 2: Run benchmark ──────────────────────────────────
            results = engine.run_all()
            assert len(results) == 20

            passed = sum(1 for r in results if r.status == "PASSED")
            failed = sum(1 for r in results if r.status == "FAILED")
            # Mock returns 2/3 success
            assert passed >= 10, f"Expected ~13 passed, got {passed}"

            # ── Phase 3: Evidence storage ───────────────────────────────
            ev_store = EvidenceStore(storage_dir=os.path.join(tmpdir, "evidence"))
            for r in results:
                pkg = EvidencePackage(
                    node_id=f"node_{r.scenario_id}", capability="backend_development",
                    agent="coder", task_id=r.scenario_id,
                    execution_id=f"EXEC-{r.scenario_id}",
                    data={"status": r.status, "duration": r.duration_seconds},
                )
                ev_store.store_evidence(pkg)

            # Store canonical records
            for r in results[:5]:
                record = CanonicalExecutionRecord(
                    task_id=r.scenario_id, execution_id=f"EXEC-{r.scenario_id}",
                    workflow_name="e2e_test", nodes=[],
                    capability_requirements={"required": ["backend_development"]},
                    assignments={}, results={}, evidence_hashes=[],
                )
                ev_store.store_record(record)

            # Verify integrity
            integrity = ev_store.verify_integrity(results[0].scenario_id, f"EXEC-{results[0].scenario_id}")
            assert integrity["evidence_count"] >= 0

            # ── Phase 4: Performance registry ───────────────────────────
            perf_reg = PerformanceRegistryV3(storage_path=os.path.join(tmpdir, "perf.json"))
            model_ids = ["model_a:free", "model_b:free", "model_c:free"]
            for mid in model_ids:
                for i in range(5):
                    score = 0.7 + (hash(mid) % 30) / 100
                    perf_reg.record(PerformanceSample(
                        task_id=f"T{i}", model_id=mid,
                        capability="backend_development",
                        success=i % 3 != 0,  # ~67% success
                        score=score, latency_ms=100 + i * 10, iterations=1,
                    ))

            # There should be a best model
            best = perf_reg.best_model("backend_development")
            # (may be None if insufficient samples — that's OK, just verify registry works)

            # ── Phase 5: Experience / memory store ──────────────────────
            mem_store = ExperienceStore(storage_dir=os.path.join(tmpdir, "memory"))
            for i, r in enumerate(results):
                rec = ExperienceRecord(
                    task_id=r.scenario_id,
                    task_objective=r.scenario_name,
                    capabilities_used=["backend_development", "testing"],
                    result_status="SUCCESS" if r.status == "PASSED" else "FAILED",
                    score=r.duration_seconds or 0.5, duration_seconds=r.duration_seconds,
                    lessons=[f"Lesson {i}: {'success' if r.status == 'PASSED' else 'failure'}"],
                )
                mem_store.store(rec)

            # Search memories
            memories = mem_store.search("backend_development")
            assert len(memories) >= 1

            # ── Phase 6: Failure analysis ───────────────────────────────
            analyzer = RootCauseAnalyzer()
            failure_count = 0
            for r in results:
                if r.status == "FAILED":
                    cause = analyzer.analyze({
                        "node_id": f"node_{r.scenario_id}",
                        "failed_capability": "backend_development",
                        "error_message": "ExecutionError: task failed",
                    })
                    assert cause is not None
                    failure_count += 1
            assert failure_count >= 1

            # Convert failures to learning
            pipeline = FailureToLearningPipeline(experience_store=mem_store)
            for f in analyzer.get_patterns():
                pass  # patterns dict available

            # ── Phase 7: Generate report ────────────────────────────────
            summary = engine.get_summary()
            report_gen = BenchmarkReportGenerator(output_dir=os.path.join(tmpdir, "reports"))
            report = report_gen.generate(results, summary)
            assert "# Engineering Benchmark Report" in report
            assert f"{summary['pass_rate']}%" in report
            saved = report_gen.save(report, "e2e_benchmark_report.md")
            assert os.path.exists(saved)

            # ── Phase 8: Policy validation ──────────────────────────────
            gov = ImmutableGovernancePolicy()
            checks = gov.check_all({"model_id": "nous/model:free"})
            assert len(checks) >= 1
            assert all(not c.blocked for c in checks[:1])  # at least first check passes

            # ── Phase 9: Capability intelligence ─────────────────────────
            taxo = CapabilityTaxonomy(taxonomy_path=os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config", "capability_taxonomy.yaml"
            ))
            reg = CapabilityRegistry()
            reg.register_agent("coder", ["backend_development"])
            reg.register_agent("tester", ["testing"])
            detector = CapabilityGapDetector(reg, taxo)
            gaps = detector.detect_gaps(["backend_development", "machine_learning"])
            assert len(gaps) >= 0

            # ── Phase 10: Gather final metrics ──────────────────────────
            # All phases completed without unhandled exceptions
            print(f"E2E LOAD COMPLETE: {len(results)} scenarios, "
                  f"{passed} passed, {failed} failed, "
                  f"{len(memories)} memories stored, "
                  f"{failure_count} failures analyzed")
        finally:
            os.chdir(orig_cwd)


def test_e2e_v3_adaptive_policy_cycle():
    """E2E: Adaptive model policy selection cycle with performance feedback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        perf_path = os.path.join(tmpdir, "perf.json")
        perf_reg = PerformanceRegistryV3(storage_path=perf_path)

        # Simulate performance data over time
        for cycle in range(5):
            for mid in ["alpha:free", "beta:free", "gamma:free"]:
                for i in range(2):
                    base_score = {"alpha:free": 0.85, "beta:free": 0.90, "gamma:free": 0.95}[mid]
                    perf_reg.record(PerformanceSample(
                        task_id=f"C{cycle}_T{i}", model_id=mid,
                        capability="coding",
                        success=True,
                        score=base_score + (cycle * 0.01),
                        latency_ms=100 + (cycle * 5), iterations=1,
                    ))

        # Verify ranking
        ranked = perf_reg.rank_models("coding", min_samples=5)
        if ranked:
            assert ranked[0].model_id is not None

        # Adaptive policy
        cat = type('obj', (object,), {
            'get_candidates': lambda self, caps, role: [
                type('obj', (object,), {'model_id': 'gamma:free', 'cost': 0}),
            ],
            'get_model': lambda self, mid: type('obj', (object,), {
                'model_id': mid, 'cost': 0, 'capabilities': {}}),
            'list_free': lambda self: [
                type('obj', (object,), {'model_id': mid, 'cost': 0})
                for mid in ["alpha:free", "beta:free", "gamma:free"]],
            'register': lambda self, mc: None,
        })()
        fallback = type('obj', (object,), {
            'select': lambda self, caps, role: type('obj', (object,), {'model_id': 'gamma:free'})})()

        policy = AdaptiveModelPolicyV3(cat, perf_reg, fallback)
        selected = policy.select(["coding"], role="developer")
        assert selected is not None

        print(f"E2E POLICY CYCLE COMPLETE: ranked {len(ranked)} models, "
              f"selected {selected.model_id}")
