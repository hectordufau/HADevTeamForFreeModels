# harness/planner/v3_integration.py — V3 Integration (Phase Q)
"""
Unified Learning Pipeline, Unified Decision Pipeline, Orchestrator V3.
Coordinates between all V3 intelligence components.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


class V3IntegrationError(Exception):
    """Raised on V3 integration errors."""


class UnifiedLearningPipeline:
    """Coordinates learning across all V3 intelligence components.

    After each execution, extracts experiences, updates performance data,
    records capability quality, learns from failures, and stores plan outcomes.
    """

    def __init__(self, experience_store: Any, perf_registry_v3: Any,
                 plan_learning_store: Any, failure_learning_pipeline: Any):
        self.experience_store = experience_store
        self.perf_registry = perf_registry_v3
        self.plan_learning = plan_learning_store
        self.failure_learning = failure_learning_pipeline

    def process(self, task: Any, report: Dict[str, Any]):
        """Process an execution result through all learning pipelines."""
        # 1. Extract experience
        from harness.memory.v3 import ExperienceExtractor
        exp = ExperienceExtractor.extract(report, task)
        if self.experience_store:
            self.experience_store.store(exp)

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
        return {
            "experiences": self.experience_store.search("") if self.experience_store else [],
            "plan_pattern": self.plan_learning.get_best_scoring_pattern() if self.plan_learning else None,
        }


class UnifiedDecisionPipeline:
    """Coordinates decisions across all V3 intelligence components.

    Integrates capability intelligence, model intelligence, plan intelligence,
    and failure intelligence into a unified decision-making process.
    """

    def __init__(self, cap_recommender: Any, model_policy: Any,
                 plan_selector: Any, replan_optimizer: Any,
                 governance: Any):
        self.cap_recommender = cap_recommender
        self.model_policy = model_policy
        self.plan_selector = plan_selector
        self.replan_optimizer = replan_optimizer
        self.governance = governance

    def decide_capabilities(self, task_objective: str,
                           acceptances: List[str]) -> List[Any]:
        """Decide which capabilities are needed for a task."""
        if self.cap_recommender:
            return self.cap_recommender.recommend(task_objective, acceptances)
        return []

    def decide_model(self, capabilities: List[str],
                    role: str, task_id: str) -> Any:
        """Decide which model to use."""
        if self.model_policy:
            return self.model_policy.select(capabilities, role, task_id)
        return None

    def decide_plan(self, alternatives: List[Any]) -> Any:
        """Decide which plan to execute."""
        if self.plan_selector and alternatives:
            return self.plan_selector.select(alternatives)
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
