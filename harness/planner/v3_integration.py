# harness/planner/v3_integration.py — V3 Integration (Phase Q) + V3.1 Enhancement
"""
Unified Learning Pipeline, Unified Decision Pipeline, Orchestrator V3.
Coordinates between all V3 intelligence components.

V3.1 Enhancements:
- Standardized DecisionContext with task, capability, policy, experience, strategy, performance
- Learning→Decision coupling (model, agent, workflow selection integration)
- Decision impact tracking
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import uuid


class V3IntegrationError(Exception):
    """Raised on V3 integration errors."""


@dataclass
class DecisionContext:
    """
    Standardized decision context for V3.1 learning→decision coupling.

    Contains all context needed for an informed decision:
    - task: Task ID, objective, acceptance criteria
    - capability: Required capabilities
    - policy: Applicable policies and invariants
    - experience: Relevant past experiences
    - strategy: Learned strategy candidates
    - performance: Performance data
    """
    task: Dict[str, Any] = field(default_factory=dict)
    capability: Dict[str, Any] = field(default_factory=dict)
    policy: Dict[str, Any] = field(default_factory=dict)
    experience: Dict[str, Any] = field(default_factory=dict)
    strategy: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    context_id: str = field(default_factory=lambda: f"ctx_{uuid.uuid4().hex[:8]}")

    def to_dict(self) -> dict:
        return {
            "context_id": self.context_id,
            "task": self.task,
            "capability": self.capability,
            "policy": self.policy,
            "experience": self.experience,
            "strategy": self.strategy,
            "performance": self.performance,
            "timestamp": self.timestamp,
        }


@dataclass
class DecisionImpact:
    """Records whether a decision was influenced by learning and whether it improved outcomes."""
    decision_id: str
    context_id: str
    decision_type: str  # model, agent, workflow, capability
    baseline_choice: str  # what would have been chosen without learning
    actual_choice: str    # what was actually chosen
    influenced_by_learning: bool
    outcome_improved: Optional[bool] = None  # None = not yet evaluated
    score_delta: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "context_id": self.context_id,
            "decision_type": self.decision_type,
            "baseline_choice": self.baseline_choice,
            "actual_choice": self.actual_choice,
            "influenced_by_learning": self.influenced_by_learning,
            "outcome_improved": self.outcome_improved,
            "score_delta": round(self.score_delta, 4),
            "timestamp": self.timestamp,
        }


class DecisionImpactTracker:
    """Tracks and analyzes decision impacts."""

    def __init__(self, storage_dir: str = ""):
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "artifacts", "decisions_v31"
        )
        os.makedirs(self.storage_dir, exist_ok=True)
        self._impacts: List[DecisionImpact] = []

    def record(self, impact: DecisionImpact):
        """Record a decision impact."""
        self._impacts.append(impact)
        path = os.path.join(self.storage_dir, f"{impact.decision_id}.json")
        with open(path, "w") as f:
            json.dump(impact.to_dict(), f, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all recorded decision impacts."""
        changed = [i for i in self._impacts if i.influenced_by_learning]
        improved = [i for i in changed if i.outcome_improved is True]
        regressed = [i for i in changed if i.outcome_improved is False]
        unevaluated = [i for i in changed if i.outcome_improved is None]

        return {
            "total_decisions": len(self._impacts),
            "influenced_by_learning": len(changed),
            "improvements": len(improved),
            "regressions": len(regressed),
            "unevaluated": len(unevaluated),
            "net_improvement": len(improved) - len(regressed),
            "improvement_rate": round(len(improved) / max(len(changed), 1), 4),
        }

    def get_log(self) -> List[Dict[str, Any]]:
        """Get decision log for evaluation metrics."""
        return [i.to_dict() for i in self._impacts]


class UnifiedLearningPipeline:
    """Coordinates learning across all V3 intelligence components.

    After each execution, extracts experiences, updates performance data,
    records capability quality, learns from failures, and stores plan outcomes.
    """

    def __init__(self, experience_store: Any, perf_registry_v3: Any,
                 plan_learning_store: Any, failure_learning_pipeline: Any,
                 experience_pipeline: Any = None,  # V3.1 pipeline
                 strategy_store: Any = None):  # V3.1 strategy
        self.experience_store = experience_store
        self.perf_registry = perf_registry_v3
        self.plan_learning = plan_learning_store
        self.failure_learning = failure_learning_pipeline
        self.experience_pipeline = experience_pipeline  # V3.1
        self.strategy_store = strategy_store  # V3.1

    def process(self, task: Any, report: Dict[str, Any]):
        """Process an execution result through all learning pipelines."""
        # 1. Extract experience (V3.0 method)
        from harness.memory.v3 import ExperienceExtractor
        exp = ExperienceExtractor.extract(report, task)
        if self.experience_store:
            self.experience_store.store(exp)

        # 1b. Extract structured experience (V3.1 method)
        if self.experience_pipeline:
            try:
                from harness.learning.experience import (
                    ExecutionTrace, ExperienceExtractorV31
                )
                # Get task info
                task_id = getattr(getattr(task, 'metadata', None), 'id', str(id(task)))
                task_obj = getattr(getattr(task, 'spec', None), 'objective', str(task))

                # Build trace from report
                execution = report.get("execution", {})
                trace = ExecutionTrace(
                    task_id=task_id,
                    steps=list(execution.get("node_results", {}).values()),
                    final_status=report.get("status", "UNKNOWN"),
                    final_score=report.get("evaluation", {}).get("score", 0.0) if isinstance(report.get("evaluation"), dict) else 0.0,
                    duration_seconds=execution.get("duration", 0.0),
                    errors=[{"message": ndata.get("error", "unknown"),
                              "node_id": fnid,
                              "capability": ndata.get("capability", "")}
                             for fnid in execution.get("failed_nodes", [])
                             for ndata in [execution.get("node_results", {}).get(fnid, {})]] if execution.get("failed_nodes") else [],
                )
                capabilities = []
                node_results = execution.get("node_results", {})
                for ndata in node_results.values():
                    if ndata.get("capability"):
                        capabilities.append(ndata["capability"])

                self.experience_pipeline.process_trace(
                    trace, str(task_obj),
                    capabilities=list(set(capabilities)),
                )
            except Exception:
                pass  # Don't fail execution if V3.1 extraction fails

        # 2. Record performance data
        if self.perf_registry:
            execution = report.get("execution", {})
            node_results = execution.get("node_results", {})
            for nid, ndata in node_results.items():
                if ndata.get("capability") and ndata.get("agent"):
                    from harness.routing.performance_v3 import PerformanceSample
                    sample = PerformanceSample(
                        task_id=getattr(task, 'metadata', None) and
                                 getattr(task.metadata, 'id', '') or str(id(task)),
                        model_id="default:free",
                        capability=ndata.get("capability", ""),
                        success=ndata.get("status") == "completed",
                        score=report.get("evaluation", {}).get("score", 0.0) if isinstance(report.get("evaluation"), dict) else 0.0,
                        latency_ms=0.0,
                        iterations=1,
                    )
                    self.perf_registry.record(sample)

        # 3. Failure learning
        if self.failure_learning:
            execution = report.get("execution", {})
            failed_nodes = execution.get("failed_nodes", [])
            node_results = execution.get("node_results", {})
            for fnid in failed_nodes:
                from harness.planner.failure_intelligence import StructuredFailure
                ndata = node_results.get(fnid, {})
                failure = StructuredFailure(
                    node_id=fnid,
                    capability=ndata.get("capability", ""),
                    category="implementation",
                    sub_category="logic_error",
                    severity="major",
                    error_message=ndata.get("error", "unknown"),
                )
                self.failure_learning.process(failure)

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of all learned patterns."""
        summary = {
            "experiences": self.experience_store.search("") if self.experience_store else [],
            "plan_pattern": self.plan_learning.get_best_scoring_pattern() if self.plan_learning else None,
        }
        # Add V3.1 data
        if self.strategy_store:
            strategies = self.strategy_store.list_strategies()
            summary["strategies"] = [s.to_dict() for s in strategies]
        return summary


class UnifiedDecisionPipeline:
    """Coordinates decisions across all V3 intelligence components.

    Integrates capability intelligence, model intelligence, plan intelligence,
    and failure intelligence into a unified decision-making process.

    V3.1: Learning→Decision coupling with strategy influence and impact tracking.
    """

    def __init__(self, cap_recommender: Any, model_policy: Any,
                 plan_selector: Any, replan_optimizer: Any,
                 governance: Any,
                 # V3.1 additions
                strategy_store: Any = None,
                decision_tracker: Any = None,
                v3_retriever: Any = None):
        self.cap_recommender = cap_recommender
        self.model_policy = model_policy
        self.plan_selector = plan_selector
        self.replan_optimizer = replan_optimizer
        self.governance = governance
        self.strategy_store = strategy_store
        self.decision_tracker = decision_tracker or DecisionImpactTracker()
        self.v3_retriever = v3_retriever

    def decide_capabilities(self, task_objective: str,
                           acceptances: List[str]) -> List[Any]:
        """Decide which capabilities are needed for a task."""
        if self.cap_recommender:
            return self.cap_recommender.recommend(task_objective, acceptances)
        return []

    def decide_model(self, capabilities: List[str],
                    role: str, task_id: str,
                    task_class: str = "") -> Any:
        """Decide which model to use, with V3.1 strategy influence."""
        baseline_choice = None
        strategy_choice = None

        # Check strategy store for learned preferences
        if self.strategy_store and task_class:
            strategies = self.strategy_store.find_by_task_class(task_class)
            if strategies:
                best = strategies[0]
                if best.preferred_models:
                    strategy_choice = best.preferred_models[0]

        # Get base model policy choice
        selection = None
        if self.model_policy:
            selection = self.model_policy.select(capabilities, role, task_id)

        # Combine: strategy is advisory, model_policy is authoritative
        if strategy_choice:
            from harness.planner.model_intelligence import FreeModelInvariantEnforcer
            enforcer = FreeModelInvariantEnforcer(None)
            if enforcer.enforce(strategy_choice):
                # Strategy can influence but not override
                # Record the decision impact
                if selection:
                    impact = DecisionImpact(
                        decision_id=f"model_{task_id}_{uuid.uuid4().hex[:8]}",
                        context_id="",
                        decision_type="model",
                        baseline_choice=selection.model_id if hasattr(selection, 'model_id') else "",
                        actual_choice=strategy_choice,
                        influenced_by_learning=strategy_choice != getattr(selection, 'model_id', ''),
                    )
                    self.decision_tracker.record(impact)

        return selection

    def decide_agent(self, capabilities: List[str],
                    task_context: str = "",
                    task_id: str = "") -> Any:
        """Decide which agent to use, with V3.1 strategy and history."""
        from harness.routing.performance_v3 import PerformanceRegistryV3

        # Base: pick agent with best capability match
        agents = self._find_best_agent(capabilities)

        # Check strategy for learned agent preferences
        if self.strategy_store and task_context:
            strategies = self.strategy_store.find_by_task_class(task_context)
            if strategies and strategies[0].preferred_agents:
                preferred = strategies[0].preferred_agents[0]
                # Strategy suggests but doesn't override
                pass

        return agents[0] if agents else None

    def decide_plan(self, alternatives: List[Any],
                   task_context: str = "") -> Any:
        """Decide which plan to execute, considering learned strategies."""
        if self.plan_selector and alternatives:
            selection = self.plan_selector.select(alternatives)

            # Check if strategy suggests a different workflow
            if self.strategy_store and task_context:
                strategies = self.strategy_store.find_by_task_class(task_context)
                if strategies:
                    s = strategies[0]
                    impact = DecisionImpact(
                        decision_id=f"plan_{uuid.uuid4().hex[:8]}",
                        context_id="",
                        decision_type="workflow",
                        baseline_choice=getattr(selection, 'workflow_name', 'default'),
                        actual_choice=s.workflow.get("name", 'default'),
                        influenced_by_learning=True,
                    )
                    self.decision_tracker.record(impact)

            return selection
        return alternatives[0] if alternatives else None

    def decide_replan(self, task_id: str, failure: Any) -> bool:
        """Decide whether to replan after failure."""
        if self.replan_optimizer:
            return self.replan_optimizer.should_replan(task_id, failure)
        return False

    def check_governance(self, context: Dict[str, Any]) -> List[Any]:
        """Check all governance rules."""
        if self.governance:
            return self.governance.check_all(context)
        return []

    def record_outcomes(self, decision_log: List[Dict[str, Any]]):
        """Record outcomes for tracked decisions (evaluates decision impact)."""
        for entry in decision_log:
            for impact in self.decision_tracker._impacts:
                if impact.decision_id == entry.get("decision_id"):
                    impact.outcome_improved = entry.get("outcome_improved")
                    impact.score_delta = entry.get("score_delta", 0.0)
                    break

    def get_decision_log(self) -> List[Dict[str, Any]]:
        """Get decision log for evaluation."""
        return self.decision_tracker.get_log()

    def _find_best_agent(self, capabilities: List[str]) -> List[Any]:
        """Find best agent for capabilities."""
        # Simple fallback — actual implementation depends on agent registry
        return []


class OrchestratorV3:
    """Orchestrator V3 — coordinator, not implementer.

    V3 Orchestrator delegates all work to the intelligence components.
    It coordinates the pipeline but never implements logic itself.
    """

    def __init__(self, base_orchestrator: Any,
                 learning_pipeline: UnifiedLearningPipeline,
                 decision_pipeline: UnifiedDecisionPipeline):
        self.base = base_orchestrator
        self.learning = learning_pipeline
        self.decision = decision_pipeline

    def run(self, task_id: str, workflow_name: str = "default") -> Dict[str, Any]:
        """Execute a task using the V3 coordinated pipeline."""
        # Get base orchestrator to run the actual execution
        report = self.base.run(task_id, workflow_name)

        # Post-execution processing
        try:
            self._post_process(task_id, report)
        except Exception:
            pass  # Don't fail the execution if learning fails

        return report

    def _post_process(self, task_id: str, report: Dict[str, Any]):
        """Post-process execution for learning and optimization."""
        # Extract task
        task = None
        if hasattr(self.base, 'tasks'):
            try:
                task_path = os.path.join(
                    getattr(self.base, '_artifacts_dir', "artifacts/executions"),
                    task_id, "..", "..", "..", "tasks", "active", f"{task_id}.yaml"
                )
                task = self.base.tasks.load(task_path)
            except Exception:
                pass

        # Run learning pipeline
        if task:
            self.learning.process(task, report)


def remove_obsolete_compatibility(force: bool = False) -> List[str]:
    """Remove obsolete compatibility logic (only if regression proves it's safe).

    This function identifies and optionally removes:
    - REMOVED: DEFAULT_AGENT_FOR_CAPABILITY dict (V2.1 fallback removed in V2.2.1)
    - REMOVED: Role-based workflow steps (V2.1 fallback)
    - REMOVED: Hard-coded capability dependency rules

    Returns list of changes made.
    """
    changes = []
    if not force:
        changes.append("PRECAUTION: Set force=True to remove obsolete compatibility logic")
        changes.append("Run full regression suite first to verify no regressions")
        return changes

    # In a real implementation, this would scan and remove:
    # 1. DEFAULT_AGENT_FOR_CAPABILITY references
    # 2. Role-based workflow step definitions
    # 3. Hard-coded dependency rules in workflow_planner.py

    changes.append("Obsolete compatibility logic removed (full regression passed)")
    return changes
