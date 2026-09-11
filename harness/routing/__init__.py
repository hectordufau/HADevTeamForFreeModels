# harness/routing/__init__.py — Model Catalog and Model Router
"""
Dynamic model selection by capability, enforcing provider=nous, cost=0.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


class RouterError(Exception):
    """Raised on model routing failures."""


@dataclass
class ModelCandidate:
    model_id: str
    provider: str = "nous"
    cost: float = 0.0
    capabilities: Dict[str, float] = field(default_factory=dict)
    context_length: int = 128000
    available: bool = True


@dataclass
class ModelSelection:
    model_id: str
    provider: str
    cost: float
    reason: str
    fallback_used: bool = False


class ModelCatalog:
    """Registry of available models and their capabilities."""

    def __init__(self):
        self._models: Dict[str, ModelCandidate] = {}

    def register(self, candidate: ModelCandidate):
        self._models[candidate.model_id] = candidate

    def register_many(self, candidates: List[ModelCandidate]):
        for c in candidates:
            self.register(c)

    def get(self, model_id: str) -> Optional[ModelCandidate]:
        return self._models.get(model_id)

    def list_available(self) -> List[ModelCandidate]:
        return [m for m in self._models.values() if m.available]

    def list_free(self) -> List[ModelCandidate]:
        return [m for m in self._models.values() if m.available and m.cost == 0]


class ModelRouter:
    """Selects the best model for a task based on required capabilities."""

    def __init__(self, catalog: ModelCatalog):
        self.catalog = catalog

    def select(
        self,
        required_capabilities: List[str],
        role: str = "",
        prefer_fallback: bool = False,
    ) -> ModelSelection:
        """
        Select the best available free model for the required capabilities.

        Args:
            required_capabilities: List of capability names needed.
            role: Agent role name (for logging/explanation).
            prefer_fallback: If True, skip the top candidate and use the next best.

        Returns:
            A ModelSelection with the chosen model and rationale.

        Raises:
            RouterError: If no suitable free model is found.
        """
        candidates = self.catalog.list_free()
        if not candidates:
            raise RouterError("No free models available in catalog")

        # Score each model by cumulative capability match
        scored: List[Tuple[float, ModelCandidate]] = []
        for model in candidates:
            score = 0.0
            matched = []
            for cap in required_capabilities:
                cap_score = model.capabilities.get(cap, 0.0)
                score += cap_score
                if cap_score > 0:
                    matched.append(cap)
            if matched:  # must match at least one capability
                scored.append((score, model))

        if not scored:
            raise RouterError(
                f"No free model matches required capabilities: {required_capabilities}"
            )

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)

        idx = 1 if prefer_fallback and len(scored) > 1 else 0
        score, best = scored[idx]

        return ModelSelection(
            model_id=best.model_id,
            provider=best.provider,
            cost=best.cost,
            reason=(
                f"Selected {best.model_id} for role '{role}' "
                f"with capability match score {score:.2f}"
                f"{' (fallback)' if idx > 0 else ''}"
            ),
            fallback_used=idx > 0,
        )
