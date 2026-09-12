# harness/planner/model_intelligence.py — Model Intelligence V3 (Phase F) + V3.1 Adaptive Exploration
"""
Model specialization profiles, adaptive model policy V3, exploration vs exploitation,
model experiment tracking, and free model invariant enforcement.

V3.1 Enhancement: Adaptive exploration rate based on confidence, purposeful candidate
selection, and exploration result learning.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import random
import math

from ..learning.context import ExperimentContext
from ..learning.isolation import check_test_protection, namespace_path


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
    # V3.2 ExperimentContext fields (optional for backward compatibility)
    validation_run_id: str = ""
    mode: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "model_id": self.model_id,
            "capability": self.capability,
            "task_id": self.task_id,
            "result_score": self.result_score,
            "is_exploration": self.is_exploration,
            "validation_run_id": self.validation_run_id,
            "mode": self.mode,
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


class ExplorationResultLearner:
    """
    V3.1: Learns from exploration results to improve future exploration.

    Tracks which exploration choices yielded useful data and which didn't,
    adjusting the exploration strategy over time.
    """

    def __init__(self, storage_dir: str = ""):
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "artifacts", "exploration_results"
        )
        os.makedirs(self.storage_dir, exist_ok=True)
        self._results: List[Dict[str, Any]] = []
        self._load()

    def record(self, model_id: str, capability: str,
               result_score: float, exploration_choice: str):
        """Record an exploration result."""
        entry = {
            "experiment_id": f"explore_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            "model_id": model_id,
            "capability": capability,
            "result_score": result_score,
            "exploration_choice": exploration_choice,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._results.append(entry)
        self._save()

    def get_best_exploration_for(self, capability: str) -> Optional[str]:
        """Get the best previously-explored model for a capability."""
        candidates = [
            r for r in self._results
            if r.get("capability") == capability and r.get("result_score", 0) > 0.5
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r["result_score"]).get("model_id")

    def get_exploration_quality(self, capability: str) -> float:
        """Get the average result score for exploration in a capability area."""
        scores = [
            r["result_score"] for r in self._results
            if r.get("capability") == capability
        ]
        if not scores:
            return 0.5  # neutral
        return sum(scores) / len(scores)

    def _load(self):
        """Load exploration results from disk."""
        for fname in os.listdir(self.storage_dir):
            if fname.endswith(".json"):
                with open(os.path.join(self.storage_dir, fname)) as f:
                    self._results.append(json.load(f))

    def _save(self):
        """Save exploration results to disk."""
        for entry in self._results:
            path = os.path.join(self.storage_dir, f"{entry['experiment_id']}.json")
            with open(path, "w") as f:
                json.dump(entry, f, indent=2)


class AdaptiveModelPolicyV3:
    """V3 adaptive model selection with exploration/exploitation tradeoff.

    V3.1 Enhancement:
    - Adaptive exploration rate based on confidence in current best model
    - Purposeful candidate selection (maximizing information gain + potential improvement)
    - Exploration result learning

    Features:
    - 90/10 exploration vs exploitation split
    - Confidence-aware exploitation
    - Fallback to V2 behavior for unknown capabilities
    """

    EXPLORATION_RATE = 0.10  # base 10% exploration

    def __init__(self, catalog: Any, performance_registry_v3: Any,
                 fallback_router: Any, profile_store: Optional[ModelProfileStore] = None,
                 exploration_learner: Optional[ExplorationResultLearner] = None,
                 # V3.2 Phase 6 — deterministic exploration wiring (optional,
                 # backward compatible). When experiment_context is provided the
                 # exploration path uses a local random.Random derived from the
                 # context (never the global module RNG).
                 experiment_context: Optional[ExperimentContext] = None,
                 exploration_enabled: Optional[bool] = None,
                 exploration_rate: Optional[float] = None):
        self.catalog = catalog
        self.perf_registry = performance_registry_v3
        self.fallback_router = fallback_router
        self.profile_store = profile_store or ModelProfileStore()
        self.exploration_learner = exploration_learner or ExplorationResultLearner()
        self.experiment_context = experiment_context
        self.exploration_enabled = exploration_enabled
        self.exploration_rate = exploration_rate

    # -- deterministic exploration helpers (V3.2 Phase 6) ------------------
    def _exploration_active(self) -> bool:
        """Whether exploration may run.

        Never mutate the caller -- pure query. Gated by mode (LEARNED_NO_
        EXPLORATION / COLD forbid it) and by the explicit flag when set.
        """
        if self.experiment_context is not None:
            if self.experiment_context.exploration_is_disabled():
                return False
            if not self.experiment_context.exploration_is_enabled():
                return False
        elif self.exploration_enabled is False:
            return False
        return True

    def _effective_exploration_rate(self) -> float:
        if self.experiment_context is not None:
            return self.experiment_context.effective_exploration_rate(self.EXPLORATION_RATE)
        if self.exploration_rate is not None:
            return float(self.exploration_rate)
        return self.EXPLORATION_RATE

    def _exploration_rng(self, decision_scope: str = "model") -> "random.Random":
        """A local seeded RNG derived from the experiment context (Phase 6).

        Uses the deterministic seed derivation (SHA-256 over the context +
        decision scope) — never the global module RNG and never Python hash().
        When there is no context a stable default seed is used so behavior stays
        reproducible even for legacy callers.
        """
        import random as _random
        from ..learning.determinism import (
            context_derived_seed,
            derive_exploration_seed,
        )
        if self.experiment_context is not None:
            seed = context_derived_seed(self.experiment_context, decision_scope)
        else:
            seed = derive_exploration_seed(0, "none", "none", "none", decision_scope)
        return _random.Random(seed)

    def select(self, required_capabilities: List[str],
               role: str = "", task_id: str = "",
               force_exploit: bool = False) -> Any:
        """Select a model using adaptive policy V3.

        V3.1: Uses adaptive exploration rate based on confidence.

        Args:
            required_capabilities: Capabilities needed.
            role: Agent role for logging.
            task_id: Task ID for tracking.
            force_exploit: Skip exploration, always exploit.

        Returns:
            A ModelSelection (or similar) with the chosen model.
        """
        # Compute adaptive exploration rate (V3.1 confidence-based), bounded by
        # the configured/effective rate when Phase 6 config is present.
        adaptive_rate = self._compute_exploration_rate(required_capabilities)
        rate = min(adaptive_rate, self._effective_exploration_rate())

        # V3.2 Phase 6: exploration is gated by mode/flag and driven by a local
        # seeded RNG derived from the ExperimentContext — never the global RNG.
        explore = (not force_exploit
                   and self._exploration_active()
                   and self._exploration_rng().random() < rate)
        if explore:
            return self._explore(required_capabilities, role, task_id)

        # Exploitation: use best known model
        return self._exploit(required_capabilities, role, task_id)

    def _compute_exploration_rate(self, caps: List[str]) -> float:
        """
        V3.1: Compute adaptive exploration rate based on confidence.

        Lower rate when confidence is high (exploit known strong models).
        Higher rate when confidence is low (need to discover better options).
        """
        if not caps:
            return self.EXPLORATION_RATE  # default

        # Get confidence in best model for each capability
        confidences = []
        for cap in caps:
            if self.perf_registry:
                models = self.perf_registry.rank_models(cap)
                if models:
                    confidences.append(models[0].confidence_score())
                else:
                    confidences.append(0.0)  # no data -> high exploration
            else:
                confidences.append(0.5)

        avg_confidence = sum(confidences) / max(len(confidences), 1)

        # Map: no data (0.0) -> 20% exploration, high confidence (1.0) -> 2% exploration
        rate = 0.20 - avg_confidence * 0.18
        return max(0.02, min(0.20, rate))

    def _explore(self, caps: List[str], role: str, task_id: str) -> Any:
        """
        V3.1: Purposeful exploration — maximize information gain + improvement potential.

        Instead of purely random exploration, favor models that:
        1. Have limited data (high information gain potential)
        2. Show promise in related capabilities
        """
        free_models = self.catalog.list_free() if self.catalog else []
        if not free_models:
            return self.fallback_router.select(caps, role=role)

        # Score each candidate model for exploration
        candidates = []
        for model in free_models:
            score = 0.0

            # Information gain: prefer models with less data
            for cap in caps:
                count = self.perf_registry.get_sample_count(model.model_id, cap) if self.perf_registry else 0
                info_gain = max(0, 5 - count) / 5.0  # 0 data -> 1.0, 5+ data -> 0.0
                score += info_gain * 0.5

                # Potential improvement: check if related model did well
                best = self.exploration_learner.get_best_exploration_for(cap)
                if best and best != model.model_id:
                    score += 0.2  # room for improvement

                # Previous exploration result
                prev_quality = self.exploration_learner.get_exploration_quality(cap)
                if prev_quality > 0.3:
                    score += prev_quality * 0.3

            candidates.append((model, score))

        # Sort by score, pick from top candidates probabilistically
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Weighted random selection from top 3 (or fewer) using the
        # experiment-scoped seeded RNG (V3.2 Phase 6) — never the global RNG.
        top_n = min(3, len(candidates))
        if top_n == 0:
            return self.fallback_router.select(caps, role=role)

        weights = [max(0.1, 1.0 - i * 0.3) for i in range(top_n)]
        total = sum(weights)
        weights = [w / total for w in weights]

        rng = self._exploration_rng(decision_scope="model_explore")
        pick = rng.random() * total
        acc = 0.0
        chosen_idx = 0
        for i, w in enumerate(weights):
            acc += w
            if pick <= acc:
                chosen_idx = i
                break
        chosen_model = candidates[chosen_idx][0]

        # Record the exploration choice
        for cap in caps:
            self.exploration_learner.record(
                model_id=chosen_model.model_id,
                capability=cap,
                result_score=0.0,  # will be updated when result comes in
                exploration_choice="purposeful",
            )

        return self._make_selection(chosen_model.model_id, role, caps, is_exploration=True)

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

    def __init__(self, storage_dir: str = "",
                 experiment_context: Optional[ExperimentContext] = None):
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "artifacts", "experiments"
        )
        os.makedirs(self.storage_dir, exist_ok=True)
        self.experiment_context = experiment_context

    def record(self, experiment: ModelExperiment,
               experiment_context: Optional[ExperimentContext] = None):
        """Record an experiment in namespaced directory."""
        if experiment_context is None:
            experiment_context = self.experiment_context
        # Apply experiment context fields if provided
        if experiment_context:
            experiment.validation_run_id = experiment_context.validation_run_id
            experiment.mode = experiment_context.mode

        store_dir = namespace_path(
            "experiments",
            experiment_context=experiment_context
        ) if experiment_context else self.storage_dir
        path = os.path.join(store_dir, f"{experiment.experiment_id}.json")
        with open(path, "w") as f:
            json.dump(experiment.to_dict(), f, indent=2)

    def get_results(self, model_id: str, capability: str,
                    experiment_context: Optional[ExperimentContext] = None) -> List[ModelExperiment]:
        """Get all experiments for a (model, capability) pair, filtered by context."""
        if experiment_context is None:
            experiment_context = self.experiment_context

        store_dir = namespace_path(
            "experiments",
            experiment_context=experiment_context
        ) if experiment_context else self.storage_dir

        if not os.path.exists(store_dir):
            return []

        results = []
        for fname in os.listdir(store_dir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(store_dir, fname)) as f:
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
