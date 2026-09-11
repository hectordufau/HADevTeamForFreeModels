# harness/planner/efficiency.py — Performance and Efficiency V3 (Phase P)
"""
Execution cost model, efficiency scoring, and parallelism optimizer.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json


class EfficiencyError(Exception):
    """Raised on efficiency scoring errors."""


@dataclass
class ExecutionCost:
    """Cost breakdown of an execution."""
    model_calls: int
    total_duration_ms: float
    agent_executions: int
    retry_count: int
    tokens_used: int = 0
    cost_score: float = 0.0  # 0.0 (free) to 1.0 (expensive)

    def to_dict(self) -> dict:
        return {
            "model_calls": self.model_calls,
            "total_duration_ms": self.total_duration_ms,
            "agent_executions": self.agent_executions,
            "retry_count": self.retry_count,
            "tokens_used": self.tokens_used,
            "cost_score": self.cost_score,
        }


@dataclass
class EfficiencyScore:
    """Composite efficiency score for an execution."""
    overall: float  # 0.0 to 1.0
    time_efficiency: float  # duration-based
    resource_efficiency: float  # model calls / tokens
    cost_efficiency: float  # always 1.0 for free models
    quality_efficiency: float  # score per unit time
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "overall": round(self.overall, 3),
            "time_efficiency": round(self.time_efficiency, 3),
            "resource_efficiency": round(self.resource_efficiency, 3),
            "cost_efficiency": round(self.cost_efficiency, 3),
            "quality_efficiency": round(self.quality_efficiency, 3),
            "reasoning": self.reasoning,
        }


class ExecutionCostModel:
    """Models the cost of execution for efficiency analysis."""

    @staticmethod
    def calculate(report: Dict[str, Any]) -> ExecutionCost:
        """Calculate execution cost from a report."""
        execution = report.get("execution", {})
        node_results = execution.get("node_results", {})

        model_calls = len(node_results)
        agent_executions = len(set(
            nd.get("agent", "") for nd in node_results.values()
        ))

        # Estimate retries from node error count
        retry_count = sum(
            1 for nd in node_results.values()
            if nd.get("error")
        )

        return ExecutionCost(
            model_calls=model_calls,
            total_duration_ms=0.0,  # would come from timing
            agent_executions=agent_executions,
            retry_count=retry_count,
            cost_score=0.0,  # all free models = 0 cost
        )


class EfficiencyScorer:
    """Scores execution efficiency with quality as the primary dimension."""

    MIN_DURATION_SECONDS = 1.0
    MAX_DURATION_SECONDS = 300.0

    @classmethod
    def score(cls, report: Dict[str, Any]) -> EfficiencyScore:
        """Score efficiency of an execution report.

        Quality dominates the score: a slow but correct execution
        scores higher than a fast but wrong one.
        """
        execution = report.get("execution", {})
        evaluation = report.get("evaluation", {})

        quality_score = evaluation.get("score", 0.0) if isinstance(evaluation, dict) else 0.0

        # Time efficiency (lower is better, but quality is more important)
        duration = execution.get("duration", 0.0) if isinstance(execution, dict) else 0.0
        if duration <= 0:
            time_eff = 0.5
        else:
            time_eff = max(0.0, 1.0 - (duration - cls.MIN_DURATION_SECONDS) /
                          (cls.MAX_DURATION_SECONDS - cls.MIN_DURATION_SECONDS))

        # Resource efficiency (fewer model calls = better)
        node_results = (execution.get("node_results", {})
                       if isinstance(execution, dict) else {})
        node_count = len(node_results)
        resource_eff = max(0.0, 1.0 - (node_count - 3) / 10.0) if node_count > 0 else 0.5

        # Cost efficiency (always 1.0 for free models)
        cost_eff = 1.0

        # Quality efficiency (score per unit time)
        if duration > 0:
            quality_eff = min(1.0, quality_score / max(duration / 60, 1))
        else:
            quality_eff = quality_score

        # Overall: quality dominates (50%), time (20%), resources (20%), cost (10%)
        overall = (
            quality_score * 0.50 +
            time_eff * 0.20 +
            resource_eff * 0.20 +
            cost_eff * 0.10
        )

        return EfficiencyScore(
            overall=round(overall, 3),
            time_efficiency=round(time_eff, 3),
            resource_efficiency=round(resource_eff, 3),
            cost_efficiency=round(cost_eff, 3),
            quality_efficiency=round(quality_eff, 3),
            reasoning=f"Quality score {quality_score:.2f} dominates at 50% weight",
        )


class ParallelismOptimizer:
    """Optimizes parallelism in workflow execution for efficiency."""

    @staticmethod
    def optimize(workflow: Any) -> List[str]:
        """Identify nodes that can safely run in parallel.

        Returns list of node group IDs that can execute concurrently.
        """
        nodes = workflow.nodes if hasattr(workflow, 'nodes') else workflow.get("nodes", [])
        if not nodes:
            return []

        # Build dependency map
        dep_map = {}
        for node in nodes:
            nid = node.id if hasattr(node, 'id') else node.get("id", "")
            deps = node.depends_on if hasattr(node, 'depends_on') else node.get("depends_on", [])
            dep_map[nid] = set(deps)

        # Find independent nodes (no dependencies)
        independent = [nid for nid, deps in dep_map.items() if not deps]

        return independent

    @staticmethod
    def estimate_speedup(workflow: Any, parallel_count: int) -> float:
        """Estimate speedup from parallel execution (Amdahl's Law)."""
        nodes = workflow.nodes if hasattr(workflow, 'nodes') else workflow.get("nodes", [])
        total = len(nodes) if nodes else 1

        if parallel_count <= 1 or total <= 1:
            return 1.0

        # Fraction that can be parallelized
        parallelizable = parallel_count / total
        serial = 1.0 - parallelizable

        # Amdahl's Law: 1 / (serial + parallelizable / N)
        speedup = 1.0 / (serial + parallelizable / min(parallel_count, total))
        return round(speedup, 2)
