# harness/planner/plan_intelligence.py — Plan Intelligence V3 (Phase I)
"""
Alternative workflow generation, plan scoring, plan selection, and plan learning.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

from ..learning.context import ExperimentContext
from ..learning.isolation import check_test_protection, namespace_path


class PlanIntelligenceError(Exception):
    """Raised on plan intelligence errors."""


@dataclass
class PlanScore:
    """Score for a workflow plan across multiple dimensions."""
    overall: float  # 0.0 to 1.0
    capability_coverage: float
    agent_fit: float
    model_quality: float
    efficiency: float
    risk: float  # 0.0 (safe) to 1.0 (risky)
    reasoning: str = ""


@dataclass
class WorkflowAlternative:
    """An alternative workflow plan with scoring."""
    id: str
    nodes: List[Dict[str, Any]]
    score: PlanScore
    generated_by: str  # planner, mutation, llm
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "nodes": self.nodes,
            "score": {
                "overall": self.score.overall,
                "capability_coverage": self.score.capability_coverage,
                "agent_fit": self.score.agent_fit,
                "model_quality": self.score.model_quality,
                "efficiency": self.score.efficiency,
                "risk": self.score.risk,
                "reasoning": self.score.reasoning,
            },
            "generated_by": self.generated_by,
            "timestamp": self.timestamp,
        }


class PlanScorer:
    """Scores workflow plans across multiple dimensions."""

    def __init__(self, registry: Any, taxonomy: Any):
        self.registry = registry
        self.taxonomy = taxonomy

    def score(self, workflow: Any) -> PlanScore:
        """Score a workflow plan."""
        nodes = workflow.nodes if hasattr(workflow, 'nodes') else workflow.get("nodes", [])
        if not nodes:
            return PlanScore(overall=0.0, capability_coverage=0.0, agent_fit=0.0,
                             model_quality=0.0, efficiency=0.0, risk=1.0,
                             reasoning="Empty workflow")

        caps = [n.capability if hasattr(n, 'capability') else n.get("capability", "")
                for n in nodes]
        agents = [n.agent if hasattr(n, 'agent') else n.get("agent", "") for n in nodes]
        models = [n.model_id if hasattr(n, 'model_id') else n.get("model_id", "") for n in nodes]

        cap_coverage = self._score_capability_coverage(caps)
        agent_fit = self._score_agent_fit(caps, agents)
        model_quality = self._score_model_quality(models)
        efficiency = self._score_efficiency(nodes)
        risk = self._score_risk(nodes)

        overall = (
            cap_coverage * 0.30 +
            agent_fit * 0.25 +
            model_quality * 0.20 +
            efficiency * 0.15 +
            (1.0 - risk) * 0.10
        )

        return PlanScore(
            overall=round(overall, 3),
            capability_coverage=round(cap_coverage, 3),
            agent_fit=round(agent_fit, 3),
            model_quality=round(model_quality, 3),
            efficiency=round(efficiency, 3),
            risk=round(risk, 3),
            reasoning=f"Scored plan with {len(nodes)} nodes",
        )

    def _score_capability_coverage(self, caps: List[str]) -> float:
        """Score how well capabilities cover the full taxonomy."""
        if not caps or not self.taxonomy:
            return 0.5
        all_caps = self.taxonomy.get_all_capabilities()
        unique = set(caps)
        return min(1.0, len(unique) / max(len(all_caps), 1) * 2)

    def _score_agent_fit(self, caps: List[str], agents: List[str]) -> float:
        """Score how well agents match their assigned capabilities."""
        if not caps or not agents:
            return 0.0
        scores = []
        for cap, agent in zip(caps, agents):
            if self.registry:
                try:
                    agent_caps = self.registry.get_capabilities(agent)
                    scores.append(1.0 if cap in agent_caps else 0.0)
                except Exception:
                    scores.append(0.0)
        return sum(scores) / len(scores) if scores else 0.0

    def _score_model_quality(self, models: List[str]) -> float:
        """Score model quality (free models that support capabilities)."""
        return 0.7  # All free models are considered good

    def _score_efficiency(self, nodes: List[Any]) -> float:
        """Score based on number of parallel-execution opportunities."""
        if len(nodes) <= 1:
            return 0.5
        # More parallel nodes = better efficiency
        return min(1.0, len(nodes) / 5.0)

    def _score_risk(self, nodes: List[Any]) -> float:
        """Score risk based on node count and dependencies."""
        if not nodes:
            return 1.0
        # More nodes = more risk
        return min(1.0, len(nodes) / 10.0)


class AlternativeGenerator:
    """Generates alternative workflow plans for comparison."""

    def __init__(self, planner: Any, registry: Any, taxonomy: Any):
        self.planner = planner
        self.registry = registry
        self.taxonomy = taxonomy
        self._scorer = PlanScorer(registry, taxonomy)

    def generate(self, requirements: Any, count: int = 3) -> List[WorkflowAlternative]:
        """Generate alternative plans from the same requirements."""
        alternatives = []

        # Primary plan (standard ordering)
        primary = self.planner.plan(requirements)
        alternatives.append(WorkflowAlternative(
            id="primary",
            nodes=primary.to_dict().get("nodes", []) if hasattr(primary, 'to_dict') else [],
            score=self._scorer.score(primary),
            generated_by="planner",
        ))

        # Alternative 1: Reversed dependency ordering
        if count > 1:
            alt1 = self.planner.plan(requirements)
            if hasattr(alt1, 'nodes'):
                alt1.nodes = list(reversed(alt1.nodes))
            alternatives.append(WorkflowAlternative(
                id="alt_reversed",
                nodes=alt1.to_dict().get("nodes", []) if hasattr(alt1, 'to_dict') else [],
                score=self._scorer.score(alt1),
                generated_by="mutation",
            ))

        # Alternative 2: Consolidated (fewer nodes)
        if count > 2:
            alt2 = self.planner.plan(requirements)
            # Simplify by merging sequential implementation nodes
            if hasattr(alt2, 'nodes'):
                alt2 = self._merge_similar_nodes(alt2)
            alternatives.append(WorkflowAlternative(
                id="alt_consolidated",
                nodes=alt2.to_dict().get("nodes", []) if hasattr(alt2, 'to_dict') else [],
                score=self._scorer.score(alt2),
                generated_by="mutation",
            ))

        return alternatives

    def _merge_similar_nodes(self, workflow: Any) -> Any:
        """Merge similar consecutive nodes (simplification)."""
        if not hasattr(workflow, 'nodes') or len(workflow.nodes) < 2:
            return workflow
        merged = [workflow.nodes[0]]
        for node in workflow.nodes[1:]:
            last = merged[-1]
            if (node.agent == last.agent and
                node.capability.split('_')[0] == last.capability.split('_')[0]):
                # Merge: keep dependencies from both
                merged[-1].depends_on = list(set(last.depends_on + node.depends_on))
            else:
                merged.append(node)
        workflow.nodes = merged
        return workflow


class PlanSelector:
    """Selects the best plan from alternatives based on scores."""

    def __init__(self, scorer: PlanScorer):
        self.scorer = scorer

    def select(self, alternatives: List[WorkflowAlternative]) -> WorkflowAlternative:
        """Select the best plan based on overall score."""
        if not alternatives:
            raise PlanIntelligenceError("No alternatives to select from")
        return max(alternatives, key=lambda a: a.score.overall)


class PlanLearningStore:
    """Stores plan scores and selection outcomes for learning."""

    def __init__(self, storage_dir: str = "",
                 experiment_context: Optional[ExperimentContext] = None):
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "artifacts", "plan_learning"
        )
        os.makedirs(self.storage_dir, exist_ok=True)
        self.experiment_context = experiment_context

    def record_outcome(self, plan_id: str, score: PlanScore,
                       result_status: str,
                       experiment_context: Optional[ExperimentContext] = None):
        """Record a plan's execution outcome in namespaced directory."""
        if experiment_context is None:
            experiment_context = self.experiment_context

        entry = {
            "plan_id": plan_id,
            "score": {
                "overall": score.overall,
                "capability_coverage": score.capability_coverage,
                "agent_fit": score.agent_fit,
                "model_quality": score.model_quality,
                "efficiency": score.efficiency,
                "risk": score.risk,
            },
            "result_status": result_status,
            "validation_run_id": experiment_context.validation_run_id if experiment_context else "",
            "mode": experiment_context.mode if experiment_context else "",
            "timestamp": datetime.utcnow().isoformat(),
        }

        store_dir = namespace_path(
            "plan_learning",
            experiment_context=experiment_context
        ) if experiment_context else self.storage_dir
        path = os.path.join(store_dir, f"{plan_id}.json")
        with open(path, "w") as f:
            json.dump(entry, f, indent=2)

    def get_best_scoring_pattern(self,
                                 experiment_context: Optional[ExperimentContext] = None) -> Optional[Dict[str, Any]]:
        """Get the best-scoring plan pattern from history, filtered by context."""
        if experiment_context is None:
            experiment_context = self.experiment_context

        store_dir = namespace_path(
            "plan_learning",
            experiment_context=experiment_context
        ) if experiment_context else self.storage_dir

        if not os.path.exists(store_dir):
            return None

        best = None
        best_score = -1.0
        for fname in os.listdir(store_dir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(store_dir, fname)) as f:
                data = json.load(f)
            score = data.get("score", {}).get("overall", 0)
            if score > best_score:
                best_score = score
                best = data
        return best
