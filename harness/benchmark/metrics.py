# harness/benchmark/metrics.py — Execution Metrics Collection (V3.0)
"""
Collects and aggregates execution metrics from benchmark runs.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import statistics


@dataclass
class ExecutionMetrics:
    """Metrics from a single execution or aggregated across runs."""
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_duration_seconds: float = 0.0
    min_duration: float = 0.0
    max_duration: float = 0.0
    avg_duration: float = 0.0
    median_duration: float = 0.0
    p95_duration: float = 0.0
    capabilities_used: Dict[str, int] = field(default_factory=dict)
    agent_usage: Dict[str, int] = field(default_factory=dict)
    model_usage: Dict[str, int] = field(default_factory=dict)
    rejection_reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_count / max(self.execution_count, 1) * 100,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "avg_duration": round(self.avg_duration, 2),
            "median_duration": round(self.median_duration, 2),
            "p95_duration": round(self.p95_duration, 2),
            "capabilities_used": self.capabilities_used,
            "agent_usage": self.agent_usage,
            "model_usage": self.model_usage,
            "rejection_reasons": self.rejection_reasons,
        }


class MetricsCollector:
    """Collects and aggregates execution metrics across benchmark runs."""

    def __init__(self):
        self._durations: List[float] = []
        self._capabilities_count: Dict[str, int] = {}
        self._agent_count: Dict[str, int] = {}
        self._model_count: Dict[str, int] = {}
        self._rejection_count: Dict[str, int] = {}
        self._successes = 0
        self._failures = 0

    def record_result(self, result: Any):
        """Record a benchmark result into metrics."""
        self._durations.append(result.duration_seconds)

        if result.status == "PASSED":
            self._successes += 1
        else:
            self._failures += 1

        for cap in result.capabilities_used:
            self._capabilities_count[cap] = self._capabilities_count.get(cap, 0) + 1

        for agent in result.agent_assignments.values():
            self._agent_count[agent] = self._agent_count.get(agent, 0) + 1

        if result.error:
            reason = result.error[:100]
            self._rejection_count[reason] = self._rejection_count.get(reason, 0) + 1

    def get_metrics(self) -> ExecutionMetrics:
        """Aggregate all recorded metrics."""
        total = self._successes + self._failures
        if not self._durations:
            return ExecutionMetrics()

        sorted_durations = sorted(self._durations)
        n = len(sorted_durations)
        p95_idx = max(0, min(n - 1, int(n * 0.95)))

        return ExecutionMetrics(
            execution_count=total,
            success_count=self._successes,
            failure_count=self._failures,
            total_duration_seconds=sum(self._durations),
            min_duration=min(self._durations),
            max_duration=max(self._durations),
            avg_duration=statistics.mean(self._durations) if self._durations else 0.0,
            median_duration=statistics.median(self._durations) if self._durations else 0.0,
            p95_duration=sorted_durations[p95_idx],
            capabilities_used=dict(self._capabilities_count),
            agent_usage=dict(self._agent_count),
            model_usage=dict(self._model_count),
            rejection_reasons=dict(self._rejection_count),
        )

    def reset(self):
        """Reset all collected metrics."""
        self._durations.clear()
        self._capabilities_count.clear()
        self._agent_count.clear()
        self._model_count.clear()
        self._rejection_count.clear()
        self._successes = 0
        self._failures = 0
