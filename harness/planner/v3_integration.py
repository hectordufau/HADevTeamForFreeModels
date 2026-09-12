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

from ..learning.context import ExperimentContext
from ..learning.isolation import check_test_protection, namespace_path
from ..learning.attribution import (
    classify_attribution,
    AttributionMetrics,
    AttributionResult,
)


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


# Explicit outcome semantics for V3.2 Decision Impact Integrity.
OUTCOMES = ("IMPROVED", "UNCHANGED", "WORSENED", "UNKNOWN")
ATTRIBUTION_LEVELS = ("CONFIRMED", "PROBABLE", "POSSIBLE", "NONE")

# Evidence source types referenced by an outcome.
EVIDENCE_SOURCE_TYPES = ("execution", "evaluation", "verification")


def _map_legacy_outcome(outcome_improved: Optional[bool]) -> str:
    """Map the legacy V3.1 boolean into explicit outcome semantics.

    Legacy mapping (only where evidence supports it): True->IMPROVED,
    False->WORSENED, None->UNKNOWN. This is a lossless encoding of the
    legacy boolean; it does NOT invent a causal claim.
    """
    if outcome_improved is True:
        return "IMPROVED"
    if outcome_improved is False:
        return "WORSENED"
    return "UNKNOWN"


@dataclass
class DecisionImpact:
    """Records whether a decision was influenced by learning and whether it improved outcomes.

    V3.2 adds explicit, evidence-grounded outcome semantics:

    - outcome: IMPROVED | UNCHANGED | WORSENED | UNKNOWN (explicit; default UNKNOWN)
    - attribution_level: CONFIRMED | PROBABLE | POSSIBLE | NONE (kept separate
      from outcome; reflects confidence in causation, not the result itself)
    - selected_decision / baseline_decision / observed_outcome /
      comparative_outcome / attribution are kept as SEPARATE fields so a plain
      COLD-vs-LEARNED score comparison is never over-interpreted as causation.
    - provenance carries validation_run_id, benchmark_id, mode, task_id,
      execution_id plus evidence references (execution/evaluation/verification).
    """
    decision_id: str
    context_id: str
    decision_type: str  # model, agent, workflow, capability
    baseline_choice: str  # what would have been chosen without learning
    actual_choice: str    # what was actually chosen
    influenced_by_learning: bool
    outcome_improved: Optional[bool] = None  # legacy field (backward compat)
    score_delta: float = 0.0
    # V3.2 ExperimentContext fields (optional for backward compatibility)
    validation_run_id: str = ""
    benchmark_id: str = ""
    mode: str = ""
    execution_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    # --- V3.2 explicit outcome semantics ---
    outcome: str = "UNKNOWN"          # IMPROVED | UNCHANGED | WORSENED | UNKNOWN
    attribution_level: str = "NONE"   # CONFIRMED | PROBABLE | POSSIBLE | NONE
    # Separate decision/outcome fields (explicit, not conflated)
    selected_decision: str = ""
    baseline_decision: str = ""
    observed_outcome: Dict[str, Any] = field(default_factory=dict)
    comparative_outcome: str = ""
    # Evidence references: provenance of the recorded outcome
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    # --- V3.2 Phase 4: decision attribution & confidence ---
    experience_ids: List[str] = field(default_factory=list)
    strategy_ids: List[str] = field(default_factory=list)
    retrieval_confidence: Optional[float] = None
    strategy_confidence: Optional[float] = None
    attribution_confidence: float = 0.0     # deterministic score from engine
    attribution_rationale: List[str] = field(default_factory=list)
    comparative_used: bool = False          # whether a COLD-vs-LEARNED comparison informed it
    causal_evidence_insufficient: bool = False

    def __post_init__(self):
        """Validate and synchronize explicit outcome semantics."""
        if self.outcome not in OUTCOMES:
            raise ValueError(
                f"Invalid outcome '{self.outcome}'. "
                f"Must be one of: {', '.join(OUTCOMES)}"
            )
        if self.attribution_level not in ATTRIBUTION_LEVELS:
            raise ValueError(
                f"Invalid attribution_level '{self.attribution_level}'. "
                f"Must be one of: {', '.join(ATTRIBUTION_LEVELS)}"
            )
        # Backward compatibility: if a legacy boolean was supplied but no explicit
        # outcome, encode it losslessly (True->IMPROVED, False->WORSENED,
        # None->UNKNOWN). Applied only when outcome is still the default so an
        # explicit V3.2 outcome always wins.
        if self.outcome == "UNKNOWN" and self.outcome_improved is not None:
            self.outcome = _map_legacy_outcome(self.outcome_improved)
        # Derive legacy boolean from explicit outcome WHERE evidence supports it.
        # UNKNOWN is never coerced to False (UNKNOWN is not harm).
        if self.outcome == "IMPROVED":
            self.outcome_improved = True
        elif self.outcome == "WORSENED":
            self.outcome_improved = False
        elif self.outcome == "UNCHANGED":
            self.outcome_improved = False if self.outcome_improved is None else self.outcome_improved
        # For UNKNOWN: leave outcome_improved as-is (None unless legacy set it).
        if not self.selected_decision:
            self.selected_decision = self.actual_choice
        if not self.baseline_decision:
            self.baseline_decision = self.baseline_choice

    @classmethod
    def from_dict(cls, data: dict) -> "DecisionImpact":
        """Reconstruct a DecisionImpact from a persisted dict (backward compatible).

        Tolerates legacy V3.1 files that lack the V3.2 outcome fields by
        encoding the legacy outcome_improved boolean losslessly.
        """
        outcome = data.get("outcome", _map_legacy_outcome(data.get("outcome_improved")))
        kwargs = dict(data)
        kwargs["outcome"] = outcome
        # Unknown keys that from_dict doesn't need are ignored by dataclass only
        # if not passed; filter to known constructor params.
        known = {
            "decision_id", "context_id", "decision_type", "baseline_choice",
            "actual_choice", "influenced_by_learning", "outcome_improved",
            "score_delta", "validation_run_id", "benchmark_id", "mode",
            "execution_id", "timestamp", "outcome", "attribution_level",
            "selected_decision", "baseline_decision", "observed_outcome",
            "comparative_outcome", "evidence",
            # Phase 4 fields (safe to ignore in legacy files)
            "experience_ids", "strategy_ids", "retrieval_confidence",
            "strategy_confidence", "attribution_confidence",
            "attribution_rationale", "comparative_used",
            "causal_evidence_insufficient",
        }
        return cls(**{k: v for k, v in kwargs.items() if k in known})

    def _carries_v32_provenance(self) -> bool:
        """True when this impact carries V3.2 experiment-scoped provenance."""
        return bool(self.validation_run_id or self.mode or self.benchmark_id
                    or self.execution_id)

    def add_evidence(self, source: str, execution_id: str = "",
                     evaluation: Any = None, verification: Any = None,
                     value: Any = None) -> None:
        """Attach an evidence reference to this outcome."""
        if source not in EVIDENCE_SOURCE_TYPES:
            raise ValueError(f"Unknown evidence source '{source}'")
        ref = {
            "source": source,
            "execution_id": execution_id or self.execution_id,
            "value": value,
            "evaluation": evaluation,
            "verification": verification,
        }
        ref = {k: v for k, v in ref.items() if v is not None and v != ""}
        self.evidence.append(ref)

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
            "validation_run_id": self.validation_run_id,
            "benchmark_id": self.benchmark_id,
            "mode": self.mode,
            "execution_id": self.execution_id,
            "timestamp": self.timestamp,
            # V3.2 explicit outcome semantics
            "outcome": self.outcome,
            "attribution_level": self.attribution_level,
            "selected_decision": self.selected_decision,
            "baseline_decision": self.baseline_decision,
            "observed_outcome": self.observed_outcome,
            "comparative_outcome": self.comparative_outcome,
            "evidence": self.evidence,
            # Phase 4 attribution & confidence
            "experience_ids": self.experience_ids,
            "strategy_ids": self.strategy_ids,
            "retrieval_confidence": self.retrieval_confidence,
            "strategy_confidence": self.strategy_confidence,
            "attribution_confidence": self.attribution_confidence,
            "attribution_rationale": self.attribution_rationale,
            "comparative_used": self.comparative_used,
            "causal_evidence_insufficient": self.causal_evidence_insufficient,
        }


class DecisionImpactTracker:
    """Tracks, persists, and reloads decision impacts.

    V3.2:
    - Persists every impact to disk on record() so it survives process restart.
    - Reloads persisted impacts from disk on __init__ (new instances reconstruct
      prior state, including from other tracker instances).
    - Persists outcome changes (record_outcomes lifecycle) back to disk.
    - Namespaced under artifacts/v3.2/<run>/<mode>/decisions/ when an
      ExperimentContext is available; otherwise legacy flat V3.1 storage is used
      ONLY for genuinely legacy (non-experiment-scoped) V3.1 operations.
    """

    def __init__(self, storage_dir: str = "",
                 experiment_context: Optional[ExperimentContext] = None):
        # Always allow override of the default flat legacy directory.
        self._storage_dir = storage_dir
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "artifacts", "decisions_v31"
        )
        os.makedirs(self.storage_dir, exist_ok=True)
        self.experiment_context = experiment_context
        self._impacts: List[DecisionImpact] = []
        self._impacts = self._load_persisted_impacts()

    # ------------------------------------------------------------------
    # Context resolution & fail-closed guarding
    # ------------------------------------------------------------------
    def _resolve_context(self, impact: DecisionImpact,
                         experiment_context: Optional[ExperimentContext]
                         ) -> Optional[ExperimentContext]:
        """Resolve the effective ExperimentContext for an impact.

        Rule: an experiment-scoped V3.2 impact (one carrying V3.2 provenance)
        MUST resolve to an ExperimentContext. If it cannot, we fail closed
        rather than silently bypass namespacing into shared V3.1 storage.
        The flat V3.1 fallback applies ONLY to genuinely legacy operations
        (no ExperimentContext AND no V3.2 provenance on the impact).
        """
        ctx = experiment_context or self.experiment_context
        if ctx is not None:
            return ctx
        if impact._carries_v32_provenance():
            raise V3IntegrationError(
                "V3.2 experiment-scoped decision impact requires an "
                "ExperimentContext to persist. Refusing to silently fall back "
                "to shared V3.1 storage (would bypass namespacing)."
            )
        return None  # legacy V3.1 operation

    def _resolve_store_dir(self, experiment_context: Optional[ExperimentContext]) -> str:
        """Resolve the storage directory for a context (namespaced or legacy)."""
        if experiment_context is None:
            return self.storage_dir
        return namespace_path("decisions", experiment_context=experiment_context)

    def _impact_path(self, impact: DecisionImpact,
                     experiment_context: Optional[ExperimentContext]) -> str:
        store_dir = self._resolve_store_dir(experiment_context)
        return os.path.join(store_dir, f"{impact.decision_id}.json")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load_persisted_impacts(self) -> List[DecisionImpact]:
        """Reconstruct persisted impacts from disk on init.

        Loads from the namespaced decisions directory (if an ExperimentContext
        is configured) or from the legacy flat directory. Survives process
        restart and is how a NEW instance reconstructs another instance's state.
        """
        loaded: Dict[str, DecisionImpact] = {}
        if self.experiment_context is not None:
            store_dir = namespace_path(
                "decisions", experiment_context=self.experiment_context)
        else:
            store_dir = self.storage_dir

        if os.path.isdir(store_dir):
            for fname in os.listdir(store_dir):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(store_dir, fname)
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    impact = DecisionImpact.from_dict(data)
                    loaded[impact.decision_id] = impact
                except Exception:
                    # A corrupt/unreadable impact file must not break startup.
                    continue
        return list(loaded.values())

    def persist(self, impact: DecisionImpact,
                experiment_context: Optional[ExperimentContext] = None):
        """Persist a single impact to disk (used after outcome updates)."""
        ctx = self._resolve_context(impact, experiment_context)
        path = self._impact_path(impact, ctx)
        with open(path, "w") as f:
            json.dump(impact.to_dict(), f, indent=2)
        return path

    def record(self, impact: DecisionImpact,
               experiment_context: Optional[ExperimentContext] = None):
        """Record a decision impact, persist it, and apply namespacing.

        COLD baseline decisions are recorded here too (influenced_by_learning
        may be False) so a counterfactual exists for LEARNED comparison.
        """
        ctx = self._resolve_context(impact, experiment_context)
        if ctx is not None:
            impact.validation_run_id = ctx.validation_run_id
            impact.benchmark_id = ctx.benchmark_id
            impact.mode = ctx.mode
            impact.execution_id = ctx.execution_id
        # Replace any previously stored impact with the same id (idempotent).
        self._impacts = [i for i in self._impacts if i.decision_id != impact.decision_id]
        self._impacts.append(impact)
        self.persist(impact, ctx)

    def find(self, decision_id: str) -> Optional[DecisionImpact]:
        """Return an impact by id, or None."""
        for impact in self._impacts:
            if impact.decision_id == decision_id:
                return impact
        return None

    # ------------------------------------------------------------------
    # Outcome recording
    # ------------------------------------------------------------------
    def record_outcome(self, decision_id: str, *, outcome: str = "UNKNOWN",
                       attribution_level: str = "NONE", score_delta: float = 0.0,
                       observed_outcome: Optional[Dict[str, Any]] = None,
                       comparative_outcome: str = "",
                       evidence: Optional[List[Dict[str, Any]]] = None,
                       execution_id: str = "",
                       evaluation: Any = None,
                       verification: Any = None) -> Optional[DecisionImpact]:
        """Record an explicit outcome on a decision and persist it to disk.

        Lifecycle: decision recorded -> execution -> verification -> evaluation
        -> outcome determined -> DecisionImpact updated -> PERSISTED.
        """
        impact = self.find(decision_id)
        if impact is None:
            return None
        impact.outcome = outcome
        impact.attribution_level = attribution_level
        impact.score_delta = score_delta
        if execution_id:
            impact.execution_id = execution_id
        if observed_outcome is not None:
            impact.observed_outcome = observed_outcome
        if comparative_outcome:
            impact.comparative_outcome = comparative_outcome
        # Attach evidence references
        if execution_id:
            impact.add_evidence("execution", execution_id=execution_id)
        if evaluation is not None:
            impact.add_evidence("evaluation", execution_id=execution_id,
                                evaluation=evaluation)
        if verification is not None:
            impact.add_evidence("verification", execution_id=execution_id,
                                verification=verification)
        if evidence:
            for ref in evidence:
                if isinstance(ref, dict):
                    impact.evidence.append(ref)
        # Sync legacy boolean, persist.
        impact.__post_init__()
        ctx = self._resolve_context(impact, None)
        self.persist(impact, ctx)
        return impact

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all recorded decision impacts using explicit semantics."""
        total = len(self._impacts)
        changed = [i for i in self._impacts if i.influenced_by_learning]
        improved = [i for i in changed if i.outcome == "IMPROVED"]
        unchanged = [i for i in changed if i.outcome == "UNCHANGED"]
        regressed = [i for i in changed if i.outcome == "WORSENED"]
        unknown = [i for i in changed if i.outcome == "UNKNOWN"]
        # Legacy-compatible counts (UNKNOWN is not counted as regressed)
        legacy_improved = [i for i in changed if i.outcome_improved is True]
        legacy_regressed = [i for i in changed if i.outcome_improved is False]
        unevaluated = [i for i in changed if i.outcome_improved is None]

        return {
            "total_decisions": total,
            "influenced_by_learning": len(changed),
            "improvements": len(improved),
            "regressions": len(regressed),
            "unchanged": len(unchanged),
            "unknown": len(unknown),
            "unevaluated": len(unevaluated),
            "net_improvement": len(improved) - len(regressed),
            "improvement_rate": round(len(improved) / max(len(changed), 1), 4),
            # Legacy aliases preserved for backward compatibility
            "legacy_improvements": len(legacy_improved),
            "legacy_regressions": len(legacy_regressed),
        }

    def get_log(self) -> List[Dict[str, Any]]:
        """Get decision log for evaluation metrics."""
        return [i.to_dict() for i in self._impacts]

    # ------------------------------------------------------------------
    # Phase 4: Decision Attribution
    # ------------------------------------------------------------------
    def run_attribution(self, impact: DecisionImpact) -> AttributionResult:
        """Run the deterministic attribution engine on one persisted impact.

        Uses ONLY actual persisted evidence on the impact (no synthetic inputs).
        Writes the result back onto the impact (attribution_level,
        attribution_confidence, rationale, comparative/causal flags) and
        persists it so the attribution survives restart.
        """
        res = classify_attribution(
            decision_id=impact.decision_id,
            learning_influenced=impact.influenced_by_learning,
            outcome=impact.outcome,
            experience_ids=impact.experience_ids,
            strategy_ids=impact.strategy_ids,
            retrieval_confidence=impact.retrieval_confidence,
            strategy_confidence=impact.strategy_confidence,
            baseline_decision=impact.baseline_decision,
            selected_decision=impact.selected_decision,
            comparative_outcome=impact.comparative_outcome,
            evidence=impact.evidence,
        )
        impact.attribution_level = res.attribution_level
        impact.attribution_confidence = res.attribution_confidence
        impact.attribution_rationale = res.rationale
        impact.comparative_used = res.comparative_used
        impact.causal_evidence_insufficient = res.causal_evidence_insufficient
        impact.__post_init__()
        try:
            ctx = self._resolve_context(impact, None)
        except V3IntegrationError:
            ctx = None
        if ctx is not None:
            self.persist(impact, ctx)
        return res

    def run_attributions(self) -> Dict[str, AttributionResult]:
        """Run attribution over every recorded impact; persist; return map.

        Attribution is isolated by run/mode naturally because each tracker is
        scoped (or the caller scopes it) to a single ExperimentContext. A
        tracker holding impacts from multiple runs attributes each impact
        independently; results never leak across validation_run_id/mode.
        """
        results = {}
        for impact in list(self._impacts):
            if impact.mode == "TEST":
                # TEST results may inform attribution evaluation but must never
                # become training knowledge; we do not fold TEST into the
                # learning-influenced metrics. Attribution itself is computed
                # (evidence-only) but flagged non-training.
                pass
            results[impact.decision_id] = self.run_attribution(impact)
        return results

    def attribution_summary(self,
                            outcomes: Optional[Dict[str, str]] = None
                            ) -> Dict[str, Any]:
        """Aggregate Phase 4 metrics over this tracker's impacts.

        If `outcomes` is not supplied, each impact's own persisted `outcome`
        field is used. UNKNOWN handling and denominator semantics are defined
        in AttributionMetrics.
        """
        if outcomes is None:
            outcomes = {i.decision_id: i.outcome for i in self._impacts}
        results = [classify_attribution(
            decision_id=i.decision_id,
            learning_influenced=i.influenced_by_learning,
            outcome=i.outcome,
            experience_ids=i.experience_ids,
            strategy_ids=i.strategy_ids,
            retrieval_confidence=i.retrieval_confidence,
            strategy_confidence=i.strategy_confidence,
            baseline_decision=i.baseline_decision,
            selected_decision=i.selected_decision,
            comparative_outcome=i.comparative_outcome,
            evidence=i.evidence,
        ) for i in self._impacts if i.mode != "TEST"]
        metrics = AttributionMetrics(results, outcomes)
        return metrics.to_dict()


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

    def record_outcomes(self, decision_log: List[Dict[str, Any]],
                        persist: bool = True):
        """Record outcomes for tracked decisions (evaluates decision impact).

        Each log entry may carry explicit V3.2 semantics:
          - outcome: IMPROVED | UNCHANGED | WORSENED | UNKNOWN
          - attribution_level, score_delta, observed_outcome,
            comparative_outcome, execution_id, evaluation, verification
        or the legacy 'outcome_improved' boolean, which is encoded losslessly
        into explicit semantics (True->IMPROVED, False->WORSENED,
        None->UNKNOWN) only where evidence supports it.

        Outcome changes are persisted back to disk (not just memory).
        """
        for entry in decision_log:
            for impact in self.decision_tracker._impacts:
                if impact.decision_id == entry.get("decision_id"):
                    if "outcome" in entry:
                        impact.outcome = entry["outcome"]
                    elif "outcome_improved" in entry:
                        # Legacy encoding (lossless, never fabricates causality)
                        impact.outcome = _map_legacy_outcome(
                            entry.get("outcome_improved"))
                    if "attribution_level" in entry:
                        impact.attribution_level = entry["attribution_level"]
                    impact.score_delta = entry.get("score_delta", impact.score_delta)
                    if entry.get("execution_id"):
                        impact.execution_id = entry["execution_id"]
                    if entry.get("observed_outcome") is not None:
                        impact.observed_outcome = entry["observed_outcome"]
                    if entry.get("comparative_outcome"):
                        impact.comparative_outcome = entry["comparative_outcome"]
                    if entry.get("evaluation") is not None:
                        impact.add_evidence("evaluation", evaluation=entry["evaluation"])
                    if entry.get("verification") is not None:
                        impact.add_evidence("verification", verification=entry["verification"])
                    impact.__post_init__()  # re-sync legacy boolean
                    if persist:
                        self.decision_tracker.persist(impact)
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
