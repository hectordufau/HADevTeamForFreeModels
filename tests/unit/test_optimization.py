"""Tests for Optimization Engine (Phase M)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.planner.optimization import (
    OptimizationEngine, OptimizationTarget, OptimizationConstraint,
    OptimizationResult, OptimizationAuditEntry, OptimizationError,
)


def test_optimization_target():
    target = OptimizationTarget(
        name="Minimize duration", metric="duration",
        direction="minimize", weight=1.0,
        current_value=120.0, target_value=60.0,
    )
    assert target.name == "Minimize duration"
    assert target.direction == "minimize"


def test_optimization_constraint():
    constraint = OptimizationConstraint(
        name="max_cost", condition="less_than",
        value=0.0, current_value=0.0,
    )
    assert constraint.condition == "less_than"
    assert constraint.violated is False


def test_optimization_engine():
    engine = OptimizationEngine()
    engine.add_target(OptimizationTarget(
        name="Improve score", metric="score",
        direction="maximize", weight=1.0,
    ))
    engine.add_constraint(OptimizationConstraint(
        name="max_duration", condition="less_than",
        value=300.0,
    ))

    result = engine.optimize_workflow(
        workflow=None,
        metrics={"score": 0.7, "max_duration": 150.0},
    )
    assert isinstance(result, OptimizationResult)
    assert result.original_value == 0.7
    assert result.constraints_met is True
    assert result.improvement_pct > 0


def test_optimization_constraint_violation():
    engine = OptimizationEngine()
    engine.add_target(OptimizationTarget(
        name="Reduce duration", metric="duration",
        direction="minimize", weight=1.0,
    ))
    engine.add_constraint(OptimizationConstraint(
        name="max_cost", condition="equals",
        value=0.0,
    ))

    result = engine.optimize_workflow(
        workflow=None,
        metrics={"duration": 100.0, "max_cost": 0.1},
    )
    assert result.constraints_met is False


def test_audit_trail():
    engine = OptimizationEngine()
    engine.add_target(OptimizationTarget(
        name="Test", metric="score",
        direction="maximize", weight=1.0,
    ))

    engine.optimize_workflow(None, {"score": 0.5})
    engine.optimize_workflow(None, {"score": 0.8})

    trail = engine.get_audit_trail()
    assert len(trail) == 2
    assert all(isinstance(e, OptimizationAuditEntry) for e in trail)


def test_no_targets():
    engine = OptimizationEngine()
    try:
        engine.optimize_workflow(None, {})
        assert False, "Should have raised"
    except OptimizationError:
        pass
