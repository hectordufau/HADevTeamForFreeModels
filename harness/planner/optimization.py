# harness/planner/optimization.py — Optimization Engine (Phase M)
"""
Optimization Engine for workflow parameters, targets, constraints,
simulation, and audit trail.
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


class OptimizationError(Exception):
    """Raised on optimization errors."""


@dataclass
class OptimizationTarget:
    """A measurable target for optimization."""
    name: str
    metric: str  # duration, cost, score, iterations
    direction: str  # minimize, maximize
    weight: float = 1.0  # importance weight
    current_value: float = 0.0
    target_value: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "metric": self.metric,
            "direction": self.direction,
            "weight": self.weight,
            "current_value": self.current_value,
            "target_value": self.target_value,
        }


@dataclass
class OptimizationConstraint:
    """A constraint that optimization must respect."""
    name: str
    condition: str  # less_than, greater_than, equals
    value: float
    current_value: float = 0.0
    violated: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "condition": self.condition,
            "value": self.value,
            "current_value": self.current_value,
            "violated": self.violated,
        }


@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    target_name: str
    original_value: float
    optimized_value: float
    improvement_pct: float
    constraints_met: bool
    iterations: int
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "target_name": self.target_name,
            "original_value": self.original_value,
            "optimized_value": self.optimized_value,
            "improvement_pct": round(self.improvement_pct, 2),
            "constraints_met": self.constraints_met,
            "iterations": self.iterations,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class OptimizationAuditEntry:
    """A single entry in the optimization audit trail."""
    optimization_id: str
    action: str
    target: str
    parameters: Dict[str, Any]
    result: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "optimization_id": self.optimization_id,
            "action": self.action,
            "target": self.target,
            "parameters": self.parameters,
            "result": self.result,
            "timestamp": self.timestamp,
        }


class OptimizationEngine:
    """Optimization Engine for workflow parameters.

    Features:
    - Multiple optimization targets
    - Constraints enforcement
    - Simulation capability (what-if analysis)
    - Full audit trail
    """

    def __init__(self):
        self._targets: List[OptimizationTarget] = []
        self._constraints: List[OptimizationConstraint] = []
        self._audit_trail: List[OptimizationAuditEntry] = []
        self._audit_file: str = ""

    def add_target(self, target: OptimizationTarget):
        """Add an optimization target."""
        self._targets.append(target)

    def add_constraint(self, constraint: OptimizationConstraint):
        """Add an optimization constraint."""
        self._constraints.append(constraint)

    def optimize_workflow(self, workflow: Any, metrics: Dict[str, float]) -> OptimizationResult:
        """Optimize workflow based on current metrics and targets.

        This is a simulation-based optimizer. It doesn't modify the workflow
        directly but computes what the optimal parameters would be.
        """
        if not self._targets:
            raise OptimizationError("No optimization targets configured")

        # Find the primary target
        primary = max(self._targets, key=lambda t: t.weight)

        # Get current value
        current = metrics.get(primary.metric, 0.0)

        # Simulate optimized value (in practice, this would try different configurations)
        optimized = self._simulate_optimization(primary, current, metrics)

        # Check constraints
        constraints_met = self._check_constraints(metrics)

        # Calculate improvement
        if current != 0:
            improvement = (optimized - current) / abs(current) * 100
        else:
            improvement = 0.0

        # Adjust direction
        if primary.direction == "minimize":
            improvement = -improvement

        result = OptimizationResult(
            target_name=primary.name,
            original_value=current,
            optimized_value=optimized,
            improvement_pct=improvement,
            constraints_met=constraints_met,
            iterations=1,
            details={
                "targets": [t.to_dict() for t in self._targets],
                "constraints": [c.to_dict() for c in self._constraints],
                "metrics": metrics,
            },
        )

        self._audit(result)
        return result

    def _simulate_optimization(self, target: OptimizationTarget,
                               current: float, metrics: Dict[str, float]) -> float:
        """Simulate an optimized value for a target."""
        if target.direction == "minimize":
            # Assume 20% improvement for simulation
            return current * 0.8
        else:
            # Assume 20% improvement for simulation
            return current * 1.2

    def _check_constraints(self, metrics: Dict[str, float]) -> bool:
        """Check all constraints against current metrics."""
        all_met = True
        for constraint in self._constraints:
            current = metrics.get(constraint.name, 0.0)
            constraint.current_value = current

            if constraint.condition == "less_than":
                constraint.violated = current >= constraint.value
            elif constraint.condition == "greater_than":
                constraint.violated = current <= constraint.value
            elif constraint.condition == "equals":
                constraint.violated = abs(current - constraint.value) > 0.001

            if constraint.violated:
                all_met = False

        return all_met

    def _audit(self, result: OptimizationResult):
        """Record an optimization audit entry."""
        entry = OptimizationAuditEntry(
            optimization_id=f"opt_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            action="optimize",
            target=result.target_name,
            parameters={"targets": [t.to_dict() for t in self._targets]},
            result=result.to_dict(),
        )
        self._audit_trail.append(entry)

    def get_audit_trail(self, limit: int = 100) -> List[OptimizationAuditEntry]:
        """Get the audit trail."""
        return self._audit_trail[-limit:]

    def get_targets(self) -> List[OptimizationTarget]:
        """Get configured optimization targets."""
        return list(self._targets)

    def get_constraints(self) -> List[OptimizationConstraint]:
        """Get configured optimization constraints."""
        return list(self._constraints)

    def clear(self):
        """Clear all targets and constraints."""
        self._targets.clear()
        self._constraints.clear()
