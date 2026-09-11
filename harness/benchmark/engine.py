# harness/benchmark/engine.py — BenchmarkEngine (V3.0)
"""
Runs benchmark scenarios against the harness to measure capability-driven performance.
Each scenario is a structured task that exercises specific capabilities.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time
import json
import os


class BenchmarkError(Exception):
    """Raised on benchmark errors."""


@dataclass
class BenchmarkScenario:
    """A single benchmark scenario with structured inputs and expected outcomes."""
    id: str
    name: str
    description: str
    task_objective: str
    acceptance_criteria: List[str]
    allowed_changes: List[str] = field(default_factory=lambda: ["src/**", "tests/**"])
    verification_required: List[str] = field(default_factory=lambda: ["unit_tests", "lint"])
    autonomy_level: int = 3
    expected_capabilities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    timeout_seconds: int = 300


@dataclass
class BenchmarkResult:
    """Result of executing a single benchmark scenario."""
    scenario_id: str
    scenario_name: str
    status: str  # PASSED, FAILED, ERROR, TIMEOUT
    duration_seconds: float
    execution_status: str
    capabilities_used: List[str]
    agent_assignments: Dict[str, str]
    scoring: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "execution_status": self.execution_status,
            "capabilities_used": self.capabilities_used,
            "agent_assignments": self.agent_assignments,
            "scoring": self.scoring,
            "error": self.error,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class BenchmarkEngine:
    """Executes benchmark scenarios against a configured Orchestrator."""

    def __init__(self, orchestrator: Any, task_manager: Any,
                 scenarios_dir: str = ""):
        self.orchestrator = orchestrator
        self.task_manager = task_manager
        self.scenarios_dir = scenarios_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "config", "benchmark_scenarios"
        )
        self._scenarios: Dict[str, BenchmarkScenario] = {}
        self._results: List[BenchmarkResult] = []

    def register_scenario(self, scenario: BenchmarkScenario):
        """Register a benchmark scenario."""
        if scenario.id in self._scenarios:
            raise BenchmarkError(f"Scenario '{scenario.id}' already registered")
        self._scenarios[scenario.id] = scenario

    def load_scenarios(self, path: Optional[str] = None) -> int:
        """Load benchmark scenarios from a JSON file."""
        path = path or os.path.join(self.scenarios_dir, "scenarios.json")
        if not os.path.exists(path):
            return 0
        with open(path) as f:
            data = json.load(f)
        count = 0
        for item in data:
            scenario = BenchmarkScenario(
                id=item["id"],
                name=item["name"],
                description=item.get("description", ""),
                task_objective=item["task_objective"],
                acceptance_criteria=item.get("acceptance_criteria", []),
                allowed_changes=item.get("allowed_changes", ["src/**", "tests/**"]),
                verification_required=item.get("verification_required", ["unit_tests", "lint"]),
                autonomy_level=item.get("autonomy_level", 3),
                expected_capabilities=item.get("expected_capabilities", []),
                tags=item.get("tags", []),
                timeout_seconds=item.get("timeout_seconds", 300),
            )
            self.register_scenario(scenario)
            count += 1
        return count

    def run_scenario(self, scenario_id: str) -> BenchmarkResult:
        """Execute a single benchmark scenario and return the result."""
        if scenario_id not in self._scenarios:
            raise BenchmarkError(f"Scenario '{scenario_id}' not found")

        scenario = self._scenarios[scenario_id]
        task_id = f"BM-{scenario_id}"

        # Create task from scenario
        task = self.task_manager.create({
            "metadata": {"id": task_id, "title": scenario.name},
            "spec": {
                "objective": scenario.task_objective,
                "acceptance_criteria": scenario.acceptance_criteria,
                "allowed_changes": scenario.allowed_changes,
                "verification": {"required": scenario.verification_required},
                "autonomy": {"maximum": scenario.autonomy_level},
            },
        })

        # Save task
        task_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "tasks", "active", f"{task_id}.yaml"
        )
        os.makedirs(os.path.dirname(task_path), exist_ok=True)
        self.task_manager.save(task, task_path)

        # Execute with timeout
        start_time = time.time()
        try:
            report = self.orchestrator.run(task_id)
            duration = time.time() - start_time

            # Extract capabilities used
            caps_used = []
            agent_assignments = {}
            if "execution" in report:
                node_results = report["execution"].get("node_results", {})
                for nid, ndata in node_results.items():
                    if ndata.get("capability"):
                        caps_used.append(ndata["capability"])
                    if ndata.get("agent"):
                        agent_assignments[ndata["capability"]] = ndata["agent"]

            status = "PASSED" if str(report.get("status", "")).upper() == "COMPLETED" else "FAILED"

            result = BenchmarkResult(
                scenario_id=scenario_id,
                scenario_name=scenario.name,
                status=status,
                duration_seconds=duration,
                execution_status=report.get("status", "UNKNOWN"),
                capabilities_used=caps_used,
                agent_assignments=agent_assignments,
                scoring={
                    "coverage": self._score_capability_coverage(caps_used, scenario.expected_capabilities),
                },
                details={"report": report},
            )

        except Exception as e:
            duration = time.time() - start_time
            result = BenchmarkResult(
                scenario_id=scenario_id,
                scenario_name=scenario.name,
                status="ERROR",
                duration_seconds=duration,
                execution_status="ERROR",
                capabilities_used=[],
                agent_assignments={},
                error=str(e),
            )

        self._results.append(result)
        return result

    def run_all(self, scenario_ids: Optional[List[str]] = None) -> List[BenchmarkResult]:
        """Run all or selected benchmark scenarios."""
        ids = scenario_ids or list(self._scenarios.keys())
        results = []
        for sid in ids:
            result = self.run_scenario(sid)
            results.append(result)
        return results

    def get_results(self) -> List[BenchmarkResult]:
        """Get all benchmark results."""
        return list(self._results)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all benchmark results."""
        total = len(self._results)
        passed = sum(1 for r in self._results if r.status == "PASSED")
        failed = sum(1 for r in self._results if r.status in ("FAILED", "ERROR"))
        avg_duration = sum(r.duration_seconds for r in self._results) / max(total, 1)

        return {
            "total_scenarios": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / max(total, 1) * 100,
            "avg_duration_seconds": round(avg_duration, 2),
            "total_duration_seconds": round(sum(r.duration_seconds for r in self._results), 2),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _score_capability_coverage(self, used: List[str], expected: List[str]) -> float:
        """Score how well used capabilities cover expected ones."""
        if not expected:
            return 1.0
        covered = sum(1 for e in expected if e in used)
        return covered / len(expected)
