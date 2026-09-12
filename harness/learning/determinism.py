# harness/learning/determinism.py — V3.2 Phase 6: Reproducibility & Deterministic Exploration
"""
Deterministic, experiment-scoped exploration for reproducible learning.

Phase 6 goal (spec §10): separate learned execution into
COLD / LEARNED_NO_EXPLORATION (greedy best-known) / LEARNED_EXPLORATION
(deterministic seeded), so that:

    same task + same learning state + same seed + same config
        => same exploration decision
        => same selected strategy/model/policy
        => same execution plan

Rules implemented here
-----------------------
1. NO direct uncontrolled global-RNG use. `random.random()/choice()/shuffle()/
   uniform()` on the module-global Random() are forbidden in the learning /
   exploration path. All randomness flows through a local `random.Random(seed)`.

2. Seed derivation (spec §2): the PRNG seed is derived with SHA-256 over

       reproducibility_seed + validation_run_id + benchmark_id + task_id
       + decision_scope

   captured as:  sha256(f"{reproducibility_seed}::{validation_run_id}::
   {benchmark_id}::{task_id}::{decision_scope}").  We NEVER use Python's built-in
   `hash()` (it is process-randomized by default and therefore not reproducible
   across process boundaries). The derived seed is persisted with the decision
   evidence.

3. LEARNED_NO_EXPLORATION (greedy, spec §4): select the best-known eligible
   decision deterministically. Tie-breaking is fully deterministic:
   primary score DESC, confidence DESC, historical utility DESC, stable id ASC.
   No dict/set iteration order is relied upon.

4. LEARNED_EXPLORATION (spec §5): use the experiment-scoped deterministic RNG
   to decide exploration (compare rng.random() against the configured rate) and,
   when exploring, to perform a weighted choice over top candidates. This
   preserves V3.1/V3.2 exploration semantics (weighted draw over top-N) while
   making it reproducible. No new optimization algorithm is introduced.

5. exploration.enabled=false (spec §6): exploration_probability=0,
   exploration_taken=false, greedy best-known selection. No RNG outcome
   influences the decision.

Security/governance (spec §13): exploration is advisory. Consumers (model /
strategy / plan selection layers) MUST still reject a candidate that violates
policy regardless of what exploration returns. This module records the decision
but never authorizes bypassing PolicyEngine / autonomy limits / security gates.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .context import ExperimentContext

# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------


def derive_exploration_seed(
    reproducibility_seed: int,
    validation_run_id: str,
    benchmark_id: str,
    task_id: str,
    decision_scope: str,
) -> int:
    """Derive a deterministic PRNG seed from experiment identity + decision scope.

    SHA-256 over the five components joined by '::'. Deterministic across
    process boundaries (never Python hash()); a different seed/run/benchmark/
    task/scope yields a different (still deterministic) seed.
    """
    payload = (
        f"{int(reproducibility_seed)}::{validation_run_id}::{benchmark_id}"
        f"::{task_id}::{decision_scope}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    # 8 bytes => seed in [0, 2**64): ample range, stable across platforms.
    return int(digest[:16], 16)


def context_derived_seed(context: ExperimentContext, decision_scope: str) -> int:
    """Derive the exploration seed for an ExperimentContext + decision scope."""
    return derive_exploration_seed(
        reproducibility_seed=context.reproducibility_seed,
        validation_run_id=context.validation_run_id,
        benchmark_id=context.benchmark_id,
        task_id=context.task_id,
        decision_scope=decision_scope,
    )


# ---------------------------------------------------------------------------
# Candidate model (lightweight, policy-agnostic view used for selection)
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """A deterministic, sortable exploration/selection candidate."""

    stable_id: str
    primary_score: float = 0.0
    confidence: float = 0.0
    historical_utility: float = 0.0
    payload: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stable_id": self.stable_id,
            "primary_score": round(float(self.primary_score), 6),
            "confidence": round(float(self.confidence), 6),
            "historical_utility": round(float(self.historical_utility), 6),
        }


def _sort_key(candidate: Candidate) -> Tuple:
    """Deterministic ascending tie-break key (spec §4), best candidate first.

    Order: primary_score DESC, confidence DESC, historical_utility DESC,
    stable_id ASC. The negatives make higher values sort *earlier* under an
    ascending sort, so ``min()``/``sorted()`` (both ascending) yield the best
    candidate first. The tuple is a total order that never depends on
    dict/set iteration order, and is stable for ties regardless of insertion
    order (R6).
    """
    return (
        -float(candidate.primary_score),
        -float(candidate.confidence),
        -float(candidate.historical_utility),
        str(candidate.stable_id),
    )


def select_greedy(candidates: Sequence[Candidate]) -> Optional[Candidate]:
    """Deterministic best-known selection (LEARNED_NO_EXPLORATION).

    Returns the single best candidate under the deterministic tie-break key
    (min over an ascending key = best first), or None when there are no
    candidates. The result depends only on the candidate set's contents — never
    on insertion/iteration order (R6).
    """
    if not candidates:
        return None
    return min(candidates, key=_sort_key)


def sorted_candidates(candidates: Sequence[Candidate]) -> List[Candidate]:
    """Return candidates sorted by the deterministic tie-break key (best first)."""
    return sorted(candidates, key=_sort_key)


def weighted_choice(
    candidates: Sequence[Candidate],
    weights: Sequence[float],
    rng: random.Random,
) -> Candidate:
    """Seeded weighted choice over candidates (exploration branch).

    Uses only the provided local Random — never the global module RNG. When
    weights are all zero (or the total is not positive), falls back to a
    deterministic (first-by-tie-break) pick so no candidate is silently chosen
    via an uncontrolled global RNG.
    """
    if not candidates:
        raise ValueError("weighted_choice called with no candidates")
    total = sum(float(w) for w in weights)
    if total <= 0.0:
        # Degenerate weights: deterministic first-best candidate.
        return sorted_candidates(candidates)[0]
    pick = rng.random() * total
    acc = 0.0
    # Search deterministically in tie-break order so ties resolve identically.
    for cand, w in zip(sorted_candidates(candidates), weights):
        acc += float(w)
        if pick <= acc:
            return cand
    return sorted_candidates(candidates)[-1]


# ---------------------------------------------------------------------------
# Decision evidence
# ---------------------------------------------------------------------------


@dataclass
class ExplorationDecision:
    """Persistable record of one learning-influenced (exploration) decision."""

    task_id: str
    validation_run_id: str
    benchmark_id: str
    execution_id: str
    mode: str
    decision_scope: str
    reproducibility_seed: int
    derived_seed: int
    exploration_enabled: bool
    configured_rate: float
    exploration_taken: bool
    candidate_ids: List[str] = field(default_factory=list)
    selected_candidate_id: Optional[str] = None
    selection_reason: str = ""
    # Where computable, the greedy best-known and the exploratory pick:
    greedy_candidate_id: Optional[str] = None
    exploratory_candidate_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exploration": {
                "mode": self.mode,
                "enabled": self.exploration_enabled,
                "configured_rate": self.configured_rate,
                "exploration_taken": self.exploration_taken,
                "reproducibility_seed": self.reproducibility_seed,
                "derived_seed": self.derived_seed,
                "decision_scope": self.decision_scope,
                "candidate_ids": list(self.candidate_ids),
                "selected_candidate_id": self.selected_candidate_id,
                "selection_reason": self.selection_reason,
                "greedy_candidate_id": self.greedy_candidate_id,
                "exploratory_candidate_id": self.exploratory_candidate_id,
            },
            "identity": {
                "task_id": self.task_id,
                "validation_run_id": self.validation_run_id,
                "benchmark_id": self.benchmark_id,
                "execution_id": self.execution_id,
            },
        }


# ---------------------------------------------------------------------------
# Deterministic exploration policy
# ---------------------------------------------------------------------------


class DeterministicExplorationPolicy:
    """Seeded, reproducible exploration policy.

    Decision flow (single source of truth for Phase 6 semantics):

    1. If exploration is disabled (mode COLD / LEARNED_NO_EXPLORATION, or
       exploration_enabled=False): `deterministic_decision` returns
       exploration_taken=False and the greedy best-known candidate. The RNG is
       constructed but its output NEVER influences the decision (R4: any seed
       yields the same result).

    2. If exploration is enabled (LEARNED_EXPLORATION or explicit flag):
       - derive the experiment-scoped seed from the context + decision_scope;
       - draw `rng.random()`; if `draw < rate` -> explore: choose an
         exploratory candidate via seeded weighted choice (R1/R2 same seed);
         exploration_taken=True. Otherwise exploit -> greedy best-known.
    """

    DEFAULT_RATE = 0.10

    def __init__(
        self,
        rate: float = DEFAULT_RATE,
        *,
        experiment_context: Optional[ExperimentContext] = None,
        exploration_enabled: Optional[bool] = None,
        decision_scope: str = "model",
    ):
        self.rate = float(rate)
        self.experiment_context = experiment_context
        self.explicit_enabled = exploration_enabled
        self.decision_scope = decision_scope

    # -- effective configuration ------------------------------------------
    def _effective_enabled(self) -> bool:
        if self.explicit_enabled is not None:
            if self.explicit_enabled is False:
                return False
        if self.experiment_context is not None:
            if self.experiment_context.exploration_is_disabled():
                return False
        # default: enabled only for LEARNED_EXPLORATION or explicit True
        return bool(self.experiment_context and
                    self.experiment_context.exploration_is_enabled())

    def _effective_rate(self) -> float:
        if self.experiment_context is not None:
            return self.experiment_context.effective_exploration_rate(self.rate)
        return self.rate

    def _seed(self) -> int:
        if self.experiment_context is not None:
            return context_derived_seed(self.experiment_context, self.decision_scope)
        # No context: deterministic default seed (still reproducible, never hash()).
        return derive_exploration_seed(
            reproducibility_seed=0,
            validation_run_id="none",
            benchmark_id="none",
            task_id="none",
            decision_scope=self.decision_scope,
        )

    # -- decision ----------------------------------------------------------
    def deterministic_decision(
        self,
        candidates: Sequence[Candidate],
        *,
        task_id: str = "",
        execution_id: str = "",
        greedy: Optional[Candidate] = None,
    ) -> ExplorationDecision:
        """Return the deterministic decision for a candidate set.

        Also computes the counterfactual greedy candidate when possible. When
        computing the greedy candidate requires information not present here,
        greedy_candidate_id stays None rather than fabricated.
        """
        enabled = self._effective_enabled()
        rate = self._effective_rate()
        seed = self._seed()
        rng = random.Random(seed)

        cands = sorted_candidates(list(candidates))
        ids = [c.stable_id for c in cands]
        greedy_cand = select_greedy(cands)

        ctx = self.experiment_context
        run_id = getattr(ctx, "validation_run_id", "") or ""
        bench = getattr(ctx, "benchmark_id", "") or ""
        mode = getattr(ctx, "mode", "") or ""

        if not enabled or not cands:
            # Greedy / disabled branch: no RNG outcome influences the decision.
            chosen = greedy_cand
            taken = False
            reason = (
                "exploration disabled (LEARNED_NO_EXPLORATION/COLD/flag): "
                "deterministic greedy best-known selection, no RNG influence"
                if not enabled
                else "no candidates available"
            )
        else:
            draw = rng.random()
            if draw < rate:
                # Explore: deterministic weighted choice over top candidates.
                top = cands[: min(3, len(cands))]
                if len(top) < 1:
                    chosen = greedy_cand
                    taken = False
                    reason = "no exploratory candidates; fell back to greedy"
                else:
                    weights = [
                        max(0.1, 1.0 - i * 0.3) for i in range(len(top))
                    ]
                    exploratory = weighted_choice(top, weights, rng)
                    chosen = exploratory
                    taken = True
                    reason = (
                        "LEARNED_EXPLORATION seeded draw "
                        f"(draw={draw:.6f} < rate={rate:.3f}): exploratory "
                        "weighted choice over top candidates"
                    )
            else:
                chosen = greedy_cand
                taken = False
                reason = (
                    "LEARNED_EXPLORATION seeded draw "
                    f"(draw={draw:.6f} >= rate={rate:.3f}): exploit greedy "
                    "best-known"
                )

        return ExplorationDecision(
            task_id=task_id or getattr(ctx, "task_id", "") or "",
            validation_run_id=run_id,
            benchmark_id=bench,
            execution_id=execution_id or getattr(ctx, "execution_id", "") or "",
            mode=mode,
            decision_scope=self.decision_scope,
            reproducibility_seed=(
                getattr(ctx, "reproducibility_seed", 0)
                if ctx is not None else 0
            ),
            derived_seed=seed,
            exploration_enabled=enabled,
            configured_rate=rate,
            exploration_taken=taken,
            candidate_ids=ids,
            selected_candidate_id=chosen.stable_id if chosen else None,
            selection_reason=reason,
            greedy_candidate_id=greedy_cand.stable_id if greedy_cand else None,
            exploratory_candidate_id=(
                (chosen.stable_id if chosen else None) if taken else None
            ),
        )


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_exploration_config(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Extract exploration settings from a config dict (learning.exploration).

    Supported keys: enabled (bool), rate (float), seed (int, advisory — the
    reproducible per-decision seed is derived by derivation, not taken verbatim
    as the PRNG seed for exploration decisions).
    """
    if not config:
        return {"enabled": True, "rate": DeterministicExplorationPolicy.DEFAULT_RATE,
                "seed": None}
    exploration = config.get("learning", {}).get("exploration", {}) or {}
    if not isinstance(exploration, dict):
        return {"enabled": True, "rate": DeterministicExplorationPolicy.DEFAULT_RATE,
                "seed": None}
    return {
        "enabled": bool(exploration.get("enabled", True)),
        "rate": float(exploration.get("rate", DeterministicExplorationPolicy.DEFAULT_RATE)),
        "seed": exploration.get("seed"),
    }


# ---------------------------------------------------------------------------
# Exploration evidence persistence (spec §7)
# ---------------------------------------------------------------------------


def persist_exploration_decision(
    decision: ExplorationDecision,
    storage_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """Persist an ExplorationDecision to JSON, returning the written path.

    By default writes into the namespaced v3.2 tree:
        artifacts/v3.2/<validation_run_id>/<mode>/exploration/<task_id>__<scope>.json

    This is what lets an audit answer: "would Harness have selected differently
    if exploration were disabled?" — the record always carries the greedy
    candidate id alongside the actually-selected id and whether exploration was
    actually taken, without fabricating a counterfactual beyond what is
    computable.
    """
    import json
    import os as _os
    if storage_dir is None:
        base = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.dirname(
                _os.path.abspath(__file__)))),
            "artifacts", "v3.2",
            decision.validation_run_id or "nornid",
            decision.mode or "nomode",
            "exploration",
        )
    else:
        base = storage_dir
    _os.makedirs(base, exist_ok=True)
    if filename is None:
        filename = (
            f"{decision.task_id or 'notask'}__{decision.decision_scope}"
            f"__{decision.execution_id or 'noexec'}.json"
        )
    path = _os.path.join(base, filename)
    with open(path, "w") as f:
        json.dump(decision.to_dict(), f, indent=2, sort_keys=True)
    return path


def would_selection_differ(decision: ExplorationDecision) -> Dict[str, Any]:
    """Deterministic answer to the disabled-exploration counterfactual.

    - If exploration was NOT taken: the selected candidate equals the greedy
      candidate, so disabling exploration would NOT change the selection.
    - If exploration WAS taken, the greedy candidate is available in the record
      (we always compute it), so we can report exactly what greedy would have
      picked; a selection change is real, not fabricated.
    Returns a small dict with the answer, the two candidates, and whether the
    counterfactual is computable.
    """
    computable = decision.greedy_candidate_id is not None
    if not decision.exploration_taken:
        return {
            "would_differ": False,
            "selected": decision.selected_candidate_id,
            "greedy": decision.greedy_candidate_id,
            "computable": computable,
            "note": "exploration not taken; selection is the greedy best-known",
        }
    if not computable:
        return {
            "would_differ": None,  # cannot be computed, not fabricated
            "selected": decision.selected_candidate_id,
            "greedy": None,
            "computable": False,
            "note": "greedy candidate not computable from available evidence",
        }
    differ = decision.selected_candidate_id != decision.greedy_candidate_id
    return {
        "would_differ": differ,
        "selected": decision.selected_candidate_id,
        "greedy": decision.greedy_candidate_id,
        "computable": True,
        "note": (
            "disabled-exploration counterfactual: greedy would pick "
            f"{decision.greedy_candidate_id}"
        ),
    }

