# harness/learning/strategy.py — Phase C: Learned Strategy Layer
"""
Strategy representation, generation from accumulated experiences,
and validation (policy, capability, evidence checks).
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import uuid

from .experience import StructuredExperience
from .context import ExperimentContext
from .isolation import check_test_protection, namespace_path


class StrategyError(Exception):
    """Raised on strategy errors."""


@dataclass
class Strategy:
    """
    A learned strategy derived from accumulated experiences.

    A strategy captures successful patterns: which models, agents, workflows,
    and capabilities work best for a given task class.
    """
    strategy_id: str
    task_class: str
    capabilities: List[str]
    conditions: Dict[str, Any]  # {context_pattern, task_type, complexity_range}
    workflow: Dict[str, Any]    # {name, steps, ordering}
    preferred_models: List[str] = field(default_factory=list)
    preferred_agents: List[str] = field(default_factory=list)
    expected_outcome: Dict[str, float] = field(default_factory=dict)  # {score, success_rate}
    evidence_count: int = 0
    confidence: float = 0.5
    created: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    # V3.2 ExperimentContext fields (optional for backward compatibility)
    validation_run_id: str = ""
    benchmark_id: str = ""
    mode: str = ""

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "task_class": self.task_class,
            "capabilities": self.capabilities,
            "conditions": self.conditions,
            "workflow": self.workflow,
            "preferred_models": self.preferred_models,
            "preferred_agents": self.preferred_agents,
            "expected_outcome": self.expected_outcome,
            "evidence_count": self.evidence_count,
            "confidence": round(self.confidence, 4),
            "created": self.created,
            "updated": self.updated,
            "validation_run_id": self.validation_run_id,
            "benchmark_id": self.benchmark_id,
            "mode": self.mode,
        }


class StrategyValidator:
    """
    Validates strategies against policies, capabilities, and evidence.

    Learning is advisory to governance — it never overrides.
    """

    @staticmethod
    def validate_policy(strategy: Strategy,
                        governance: Optional[Any] = None,
                        free_model_enforcer: Optional[Any] = None) -> List[str]:
        """Check strategy against governance policies.

        Returns list of policy violations (empty = valid).
        """
        violations = []

        # Free model invariant check
        if free_model_enforcer:
            for model in strategy.preferred_models:
                if not free_model_enforcer.enforce(model):
                    violations.append(
                        f"Strategy '{strategy.strategy_id}' uses paid model '{model}'"
                    )

        # Governance check
        if governance:
            violations.extend(governance.check_all({
                "action": "apply_strategy",
                "strategy_id": strategy.strategy_id,
                "task_class": strategy.task_class,
            }))

        return violations

    @staticmethod
    def validate_capabilities(strategy: Strategy,
                              registry: Optional[Any] = None) -> List[str]:
        """Check that required capabilities are available.

        Returns list of missing capability warnings (empty = valid).
        """
        missing = []
        if registry:
            for cap in strategy.capabilities:
                agents = registry.find_agents_for(cap) if hasattr(registry, 'find_agents_for') else []
                if not agents:
                    missing.append(
                        f"Strategy requires capability '{cap}' but no agent provides it"
                    )
        return missing

    @staticmethod
    def validate_evidence(strategy: Strategy,
                          min_evidence: int = 3) -> List[str]:
        """Check that strategy has sufficient evidence.

        Returns list of evidence warnings (empty = valid).
        """
        warnings = []
        if strategy.evidence_count < min_evidence:
            warnings.append(
                f"Strategy has only {strategy.evidence_count} evidence items "
                f"(minimum {min_evidence} recommended)"
            )
        if strategy.confidence < 0.3:
            warnings.append(
                f"Strategy confidence ({strategy.confidence:.2f}) is below threshold (0.3)"
            )
        return warnings

    @staticmethod
    def validate_all(strategy: Strategy,
                     governance: Optional[Any] = None,
                     free_model_enforcer: Optional[Any] = None,
                     registry: Optional[Any] = None,
                     min_evidence: int = 3) -> Dict[str, List[str]]:
        """Run all validations and return categorized results."""
        return {
            "policy": StrategyValidator.validate_policy(strategy, governance, free_model_enforcer),
            "capabilities": StrategyValidator.validate_capabilities(strategy, registry),
            "evidence": StrategyValidator.validate_evidence(strategy, min_evidence),
        }


class StrategyGenerator:
    """
    Generates strategies from accumulated experiences.

    Clusters similar experiences and extracts common patterns.
    """

    def __init__(self, experience_pipeline: Any):
        self.pipeline = experience_pipeline

    def generate(self, task_class: str,
                 min_experiences: int = 3) -> Optional[Strategy]:
        """Generate a strategy for a task class from accumulated experiences.

        Args:
            task_class: The class of task to generate a strategy for
            min_experiences: Minimum number of experiences required

        Returns:
            A Strategy object, or None if insufficient experiences
        """
        relevant = self._get_relevant_experiences(task_class)

        if len(relevant) < min_experiences:
            return None

        # Extract common patterns
        all_caps = set()
        all_models = set()
        all_agents = set()
        scores = []
        successes = 0
        total_evidence = 0

        for exp in relevant:
            all_caps.update(exp.capabilities_used)
            if exp.strategy.get("model_id"):
                all_models.add(exp.strategy["model_id"])
            if exp.strategy.get("agent_id"):
                all_agents.add(exp.strategy["agent_id"])
            scores.append(exp.outcome.get("score", 0))
            if exp.outcome.get("status") == "COMPLETED":
                successes += 1
            total_evidence += len(exp.evidence)

        # Compute expected outcome
        avg_score = sum(scores) / max(len(scores), 1)
        success_rate = successes / max(len(relevant), 1)

        # Build conditions from common task contexts
        contexts = [exp.task_context for exp in relevant if exp.task_context]
        conditions = {
            "task_type": task_class,
            "complexity_range": "medium",  # default
        }

        # Build workflow from most common pattern
        workflow = {
            "name": f"Strategy for {task_class}",
            "steps": list(all_caps),
            "ordering": "sequential",
        }

        # Confidence based on evidence volume and consistency
        confidence = min(0.95, (total_evidence / 20) * 0.5 + success_rate * 0.5)

        return Strategy(
            strategy_id=f"strategy_{task_class}_{uuid.uuid4().hex[:8]}",
            task_class=task_class,
            capabilities=list(all_caps),
            conditions=conditions,
            workflow=workflow,
            preferred_models=list(all_models),
            preferred_agents=list(all_agents),
            expected_outcome={"score": round(avg_score, 4), "success_rate": round(success_rate, 4)},
            evidence_count=total_evidence,
            confidence=round(confidence, 4),
        )

    def _get_relevant_experiences(self, task_class: str) -> List[StructuredExperience]:
        """Get experiences relevant to a task class."""
        relevant = []
        for tid in self.pipeline.list_experiences():
            exp = self.pipeline.retrieve(tid)
            if exp and self._matches_task_class(exp, task_class):
                relevant.append(exp)
        return relevant

    @staticmethod
    def _matches_task_class(exp: StructuredExperience, task_class: str) -> bool:
        """Check if an experience matches a task class."""
        # Match on category or task context
        if exp.category and task_class.lower() in exp.category.lower():
            return True
        if task_class.lower() in exp.task_context.lower():
            return True
        # Match on capabilities
        if any(task_class.lower() in cap.lower() for cap in exp.capabilities_used):
            return True
        return False


class StrategyStore:
    """Persistent store for strategies with validation."""

    def __init__(self, storage_dir: str = "",
                 experiment_context: Optional[ExperimentContext] = None):
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "artifacts", "strategies"
        )
        os.makedirs(self.storage_dir, exist_ok=True)
        self.experiment_context = experiment_context

    def save(self, strategy: Strategy,
             experiment_context: Optional[ExperimentContext] = None):
        """Save a strategy to namespaced directory."""
        if experiment_context is None:
            experiment_context = self.experiment_context
        check_test_protection(experiment_context, "store")

        store_dir = namespace_path(
            "strategies",
            experiment_context=experiment_context
        ) if experiment_context else self.storage_dir
        path = os.path.join(store_dir, f"{strategy.strategy_id}.json")
        with open(path, "w") as f:
            json.dump(strategy.to_dict(), f, indent=2)

    def load(self, strategy_id: str,
             experiment_context: Optional[ExperimentContext] = None) -> Optional[Strategy]:
        """Load a strategy by ID, filtered by context."""
        if experiment_context is None:
            experiment_context = self.experiment_context

        store_dir = namespace_path(
            "strategies",
            experiment_context=experiment_context
        ) if experiment_context else self.storage_dir

        path = os.path.join(store_dir, f"{strategy_id}.json")
        if not os.path.exists(path):
            # Fallback: try V3.1 flat directory if no context
            if experiment_context is None:
                path = os.path.join(self.storage_dir, f"{strategy_id}.json")
                if not os.path.exists(path):
                    return None
            else:
                return None
        with open(path) as f:
            data = json.load(f)
        return Strategy(**data)

    def list_strategies(self,
                        experiment_context: Optional[ExperimentContext] = None) -> List[Strategy]:
        """List all stored strategies, filtered by context."""
        if experiment_context is None:
            experiment_context = self.experiment_context

        store_dir = namespace_path(
            "strategies",
            experiment_context=experiment_context
        ) if experiment_context else self.storage_dir

        if not os.path.exists(store_dir):
            return []

        strategies = []
        for fname in os.listdir(store_dir):
            if fname.endswith(".json"):
                with open(os.path.join(store_dir, fname)) as f:
                    data = json.load(f)
                strategies.append(Strategy(**data))
        return sorted(strategies, key=lambda s: s.confidence, reverse=True)

    def find_by_task_class(self, task_class: str,
                           experiment_context: Optional[ExperimentContext] = None) -> List[Strategy]:
        """Find strategies for a specific task class, filtered by context."""
        all_strategies = self.list_strategies(experiment_context=experiment_context)
        return [s for s in all_strategies if s.task_class == task_class]
