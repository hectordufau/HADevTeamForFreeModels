# harness/learning/context.py — V3.2 Experiment Context (Phase 1)
"""
Experiment-scoped context for reproducible, auditable learning experiments.

Defines ExperimentContext and documents the experiment-scoped vs global
artifact boundary.

GLOBAL artifacts (static, shared across experiments):
- Capability taxonomy, agent registry, model catalog, governance policies

EXPERIMENT-SCOPED artifacts (isolated per experiment):
- Benchmark executions, experiences, strategies, decision impacts,
  evidence packages, experiment records, execution artifacts,
  validation results, evaluation results
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid


class ExperimentContextError(Exception):
    """Raised on ExperimentContext validation errors."""


# V3.2 Phase 6 — modes extended.
# * COLD  — no learning, no exploration.
# * LEARNED (legacy alias) — with learning; readable for backward compatibility.
#          It is a *superset*: it may or may not have explored historically. We do
#          NOT silently reinterpret it as either no-exploration or exploration.
# * LEARNED_NO_EXPLORATION — full learning, exploration explicitly disabled
#          (greedy best-known selection, deterministic).
# * LEARNED_EXPLORATION    — full learning, exploration enabled with a
#          deterministic experiment-scoped seed.
# * TEST  — test/verification mode, never training.
VALID_MODES = {
    "COLD",
    "LEARNED",
    "LEARNED_NO_EXPLORATION",
    "LEARNED_EXPLORATION",
    "TEST",
}

# Canonical mode groups (semantic groupings; kept separate from VALID_MODES so
# legacy LEARNED artifacts remain individually addressable, not reinterpreted).
LEARNED_MODES = {"LEARNED", "LEARNED_NO_EXPLORATION", "LEARNED_EXPLORATION"}

# Modes where exploration is explicitly forbidden (greedy best-known only).
GREEDY_NO_EXPLORATION_MODES = {"COLD", "LEARNED_NO_EXPLORATION"}

# Modes where exploration is explicitly enabled (experiment-scoped seeded RNG).
EXPLORATION_ENABLED_MODES = {"LEARNED_EXPLORATION"}


@dataclass
class ExperimentContext:
    """
    Context for a single learning experiment execution.

    Carries the identifiers needed to trace every artifact back to its
    originating experiment, enabling isolation and reproducibility.

    Fields:
        validation_run_id: Unique run identifier for the validation session
        benchmark_id: The benchmark being executed
        task_id: The task within the benchmark
        execution_id: Unique identifier for this execution
        mode: Execution mode — "COLD" (no learning), "LEARNED" (with learning),
              or "TEST" (test/verification mode)
        reproducibility_seed: Random seed for deterministic execution (default 0)
        timestamp: ISO-8601 timestamp of context creation
        context_id: Auto-generated unique identifier for this context instance
    """

    validation_run_id: str
    benchmark_id: str
    task_id: str
    execution_id: str
    mode: str  # "COLD" | "LEARNED" | "LEARNED_NO_EXPLORATION" | "LEARNED_EXPLORATION" | "TEST"
    reproducibility_seed: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    context_id: str = field(default_factory=lambda: f"expctx_{uuid.uuid4().hex[:12]}")
    # V3.2 Phase 6 — optional explicit exploration configuration. When omitted
    # (None) exploration falls back to mode-based defaults, preserving backward
    # compatibility with legacy callers / artifacts.
    exploration_enabled: Optional[bool] = None
    exploration_rate: Optional[float] = None

    def __post_init__(self):
        """Validate required identifiers and mode."""
        if not self.validation_run_id or not isinstance(self.validation_run_id, str):
            raise ValueError("validation_run_id must be a non-empty string")
        if not self.benchmark_id or not isinstance(self.benchmark_id, str):
            raise ValueError("benchmark_id must be a non-empty string")
        if not self.task_id or not isinstance(self.task_id, str):
            raise ValueError("task_id must be a non-empty string")
        if not self.execution_id or not isinstance(self.execution_id, str):
            raise ValueError("execution_id must be a non-empty string")
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Must be one of: {', '.join(sorted(VALID_MODES))}"
            )

    def to_dict(self) -> dict:
        """Serialize to dictionary (deterministic key order)."""
        return {
            "context_id": self.context_id,
            "validation_run_id": self.validation_run_id,
            "benchmark_id": self.benchmark_id,
            "task_id": self.task_id,
            "execution_id": self.execution_id,
            "mode": self.mode,
            "reproducibility_seed": self.reproducibility_seed,
            "exploration_enabled": self.exploration_enabled,
            "exploration_rate": self.exploration_rate,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentContext":
        """Create an ExperimentContext from a dictionary (requires all fields)."""
        return cls(
            validation_run_id=data["validation_run_id"],
            benchmark_id=data["benchmark_id"],
            task_id=data["task_id"],
            execution_id=data["execution_id"],
            mode=data["mode"],
            reproducibility_seed=data.get("reproducibility_seed", 0),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            exploration_enabled=data.get("exploration_enabled"),
            exploration_rate=data.get("exploration_rate"),
        )

    # ------------------------------------------------------------------
    # V3.2 Phase 6 — exploration-mode semantics (deterministic helpers)
    # ------------------------------------------------------------------
    def exploration_is_disabled(self) -> bool:
        """True when this context forbids exploration (greedy best-known only).

        LEARNED_NO_EXPLORATION always forbids exploration regardless of any
        explicit exploration_enabled flag — the mode itself is the authority for
        the no-exploration branch, and no RNG outcome may influence its decision.
        """
        if self.mode in GREEDY_NO_EXPLORATION_MODES:
            return True
        if self.exploration_enabled is False:
            return True
        return False

    def exploration_is_enabled(self) -> bool:
        """True when this context permits exploration via seeded RNG."""
        if self.exploration_is_disabled():
            return False
        if self.exploration_enabled is True:
            return True
        if self.mode in EXPLORATION_ENABLED_MODES:
            return True
        # Legacy LEARNED / TEST / COLD default: not enabled by default; callers
        # that explicitly opt into exploration pass exploration_enabled=True.
        return False

    def is_learned_mode(self) -> bool:
        """True when the mode carries learning (incl. legacy LEARNED)."""
        return self.mode in LEARNED_MODES

    def effective_exploration_rate(self, default_rate: float = 0.10) -> float:
        """The configured exploration rate, falling back to the caller default.

        Only meaningful when exploration is enabled; returns default_rate for
        disabled/greedy modes (rate is not used there).
        """
        if self.exploration_rate is not None:
            return float(self.exploration_rate)
        return float(default_rate)

    def __eq__(self, other: object) -> bool:
        """Two contexts are equal if all fields match (deterministic comparison)."""
        if not isinstance(other, ExperimentContext):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        """Hash based on deterministic dict representation."""
        return hash(json.dumps(self.to_dict(), sort_keys=True))
