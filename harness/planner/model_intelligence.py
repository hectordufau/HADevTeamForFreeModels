# harness/planner/model_intelligence.py — Model Intelligence V3 (Phase F)
"""
Model specialization profiles, adaptive model policy V3, exploration vs exploitation,
model experiment tracking, and free model invariant enforcement.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import random


class ModelIntelligenceError(Exception):
    """Raised on model intelligence errors."""


@dataclass
class ModelSpecializationProfile:
    """Specialization profile for a model across capabilities."""
    model_id: str
    capabilities: Dict[str, float]  # capability -> score 0.0-1.0
    specialization_score: float  # 0.0=generalist, 1.0=specialist
    top_capabilities: List[str] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "capabilities": self.capabilities,
            "specialization_score": self.specialization_score,
            "top_capabilities": self.top_capabilities,
            "updated_at": self.updated_at,
        }


@dataclass
class ModelExperiment:
    """Record of a model experiment for tracking purposes."""
    experiment_id: str
    model_id: str
    capability: str
    task_id: str
    result_score: float
    is_exploration: bool
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "model_id": self.model_id,
            "capability": self.capability,
            "task_id": self.task_id,
            "result_score": self.result_score,
            "is_exploration": self.is_exploration,
            "timestamp": self.timestamp,
        }


class ModelProfileStore:
    """Persistent store for model specialization profiles."""

    def __init__(self, storage_dir: str = ""):
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "artifacts", "model_profiles"
        )
        os.makedirs(self.storage_dir, exist_ok=True)

    def save_profile(self, profile: ModelSpecializationProfile):
        """Save a model profile."""
        path = os.path.join(self.storage_dir, f"{profile.model_id.replace('/', '_')}.json")
        with open(path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)

    def load_profile(self, model_id: str) -> Optional[ModelSpecializationProfile]:
        """Load a model profile."""
        path = os.path.join(self.storage_dir, f"{model_id.replace('/', '_')}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        return ModelSpecializationProfile(**data)

    def list_profiles(self) -> List[str]:
        """List all stored model profile IDs."""
        profiles = []
        for fname in os.listdir(self.storage_dir):
            if fname.endswith(".json"):
                profiles.append(fname[:-5].replace("_", "/"))
        return profiles


class AdaptiveModelPolicyV3:
    """V3 adaptive model selection with exploration/exploitation tradeoff.

    Features:
    - 90/10 exploration vs exploitation split
    - Confidence-aware exploitation
    - Fallback to V2 behavior for unknown capabilities
    """

    EXPLORATION_RATE = 0.10  # 10% exploration

    def __init__(self, catalog: Any, performance_registry_v3: Any,
                 fallback_router: Any, profile_store: Optional[ModelProfileStore] = None):
        self.catalog = catalog
        self.perf_registry = performance_registry_v3
        self.fallback_router = fallback_router
        self.profile_store = profile_store or ModelProfileStore()

    def select(self, required_capabilities: List[str],
               role: str = "", task_id: str = "",
               force_exploit: bool = False) -> Any:
        """Select a model using adaptive policy V3.

        Args:
            required_capabilities: Capabilities needed.
            role: Agent role for logging.
            task_id: Task ID for tracking.
            force_exploit: Skip exploration, always exploit.

        Returns:
            A ModelSelection (or similar) with the chosen model.
        """
        # Exploration: 10% of the time, try a random free model
        if not force_exploit and random.random() < self.EXPLORATION_RATE:
            return self._explore(required_capabilities, role, task_id)

        # Exploitation: use best known model
        return self._exploit(required_capabilities, role, task_id)

    def _explore(self, caps: List[str], role: str, task_id: str) -> Any:
        """Explore: select a random free model for data collection."""
        free_models = self.catalog.list_free() if self.catalog else []
        if not free_models:
            return self.fallback_router.select(caps, role=role)

        # Pick a random free model
        model = random.choice(free_models)

        # Create a simple ModelSelection-like result
        return self._make_selection(model.model_id, role, caps, is_exploration=True)

    def _exploit(self, caps: List[str], role: str, task_id: str) -> Any:
        """Exploit: use best known model from performance registry."""
        best = None

        # Try each capability to find a high-confidence best model
        for cap in caps:
            models = self.perf_registry.rank_models(cap) if self.perf_registry else []
            if models:
                best = models[0]
                break

        if best and best.min_samples_met:
            return self._make_selection(best.model_id, role, caps, is_exploration=False)

        # Fallback to standard router
        selection = self.fallback_router.select(caps, role=role)
        return selection

    def _make_selection(self, model_id: str, role: str,
                        caps: List[str], is_exploration: bool) -> Any:
        """Create a selection result compatible with ModelSelection."""
        class Selection:
            def __init__(self, **kw):
                self.__dict__.update(kw)
        return Selection(
            model_id=model_id,
            provider="nous",
            cost=0.0,
            reason=f"AdaptiveModelPolicyV3: {'exploration' if is_exploration else 'exploitation'} for role '{role}'",
            fallback_used=False,
            is_exploration=is_exploration,
        )


class ExperimentTracker:
    """Tracks model experiments for analysis."""

    def __init__(self, storage_dir: str = ""):
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "artifacts", "experiments"
        )
        os.makedirs(self.storage_dir, exist_ok=True)

    def record(self, experiment: ModelExperiment):
        """Record an experiment."""
        path = os.path.join(self.storage_dir, f"{experiment.experiment_id}.json")
        with open(path, "w") as f:
            json.dump(experiment.to_dict(), f, indent=2)

    def get_results(self, model_id: str, capability: str) -> List[ModelExperiment]:
        """Get all experiments for a (model, capability) pair."""
        results = []
        for fname in os.listdir(self.storage_dir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(self.storage_dir, fname)) as f:
                data = json.load(f)
            exp = ModelExperiment(**data)
            if exp.model_id == model_id and exp.capability == capability:
                results.append(exp)
        return sorted(results, key=lambda e: e.timestamp)


class FreeModelInvariantEnforcer:
    """Enforces the free model invariant: all selected models must be free (cost=0).

    This gate runs after model selection to prevent paid model usage.
    """

    def __init__(self, catalog: Any):
        self.catalog = catalog

    def enforce(self, model_id: str) -> bool:
        """Check if a model is free. Returns True if free, False if paid."""
        model = self.catalog.get(model_id) if self.catalog else None
        if model:
            return model.cost == 0
        # If model not in catalog, check by naming convention
        return model_id.endswith(":free") if model_id else False

    def validate_workflow(self, workflow: Any) -> List[str]:
        """Validate all models in a workflow are free. Returns list of violations."""
        violations = []
        for node in workflow.nodes if hasattr(workflow, 'nodes') else workflow.get("nodes", []):
            model_id = node.model_id if hasattr(node, 'model_id') else node.get("model_id", "")
            if not self.enforce(model_id):
                violations.append(f"Node '{node.id if hasattr(node, 'id') else node.get('id', '')}' uses paid model '{model_id}'")
        return violations
