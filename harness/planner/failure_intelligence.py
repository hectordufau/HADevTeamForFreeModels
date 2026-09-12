# harness/planner/failure_intelligence.py — Failure Intelligence V3 (Phase J)
"""
Structured failure taxonomy, root cause analysis, failure-to-learning pipeline,
and replanning optimization.

V3.2 Phase 7 — Failure-Derived Learning Integration.

Canonical pipeline:  StructuredFailure → RootCauseAnalyzer →
FailureToLearningPipeline → StructuredExperience → V3.2 ExperienceStore
(persisted via ExperiencePipeline, namespaced per ExperimentContext).

Fail-closed semantics:
- TEST  failure → evidence recorded, NO StructuredExperience, learning suppressed.
- COLD  failure → evidence recorded, NO training mutation, learning suppressed.
- LEARNED*  failure → StructuredExperience derived and stored (advisory only).
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import hashlib

from ..learning.context import ExperimentContext
from ..learning.isolation import namespace_path as _ns
from ..learning.experience import StructuredExperience, Evidence, ExperiencePipeline


class FailureIntelligenceError(Exception):
    """Raised on failure intelligence errors."""


# Security / governance lessons: defensive-only transformations are allowed;
# weakening constraints is FORBIDDEN. PolicyEngine remains authoritative.
_FORBIDDEN_SECURITY_LESSONS = (
    "disable gate", "disable policy", "disable the gate", "turn off",
    "weaken", "bypass", "increase autonomy", "disable constraint",
    "remove the gate", "allow denied", "skip review",
)
_ALLOWED_SECURITY_LESSONS = (
    "validate earlier", "validate input", "select safe", "plan permitted",
)


@dataclass
class StructuredFailure:
    """A structured failure record with taxonomy classification."""
    node_id: str
    capability: str
    category: str  # implementation, test, architecture, security, integration, configuration, environment
    sub_category: str  # more specific classification
    severity: str  # critical, major, minor, warning
    error_message: str
    evidence: str = ""
    # V3.2 ExperimentContext fields (optional for backward compatibility)
    validation_run_id: str = ""
    mode: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    # V3.2 Phase 7 — extended provenance (all optional; never fabricated).
    failure_id: str = ""
    source_task_id: str = ""
    source_execution_id: str = ""
    benchmark_id: str = ""
    evidence_refs: List[str] = field(default_factory=list)
    verification_refs: List[str] = field(default_factory=list)
    decision_refs: List[str] = field(default_factory=list)
    strategy_refs: List[str] = field(default_factory=list)
    model_id: str = ""
    agent_id: str = ""
    capabilities: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "capability": self.capability,
            "category": self.category,
            "sub_category": self.sub_category,
            "severity": self.severity,
            "error_message": self.error_message,
            "evidence": self.evidence,
            "validation_run_id": self.validation_run_id,
            "mode": self.mode,
            "timestamp": self.timestamp,
            "failure_id": self.failure_id,
            "source_task_id": self.source_task_id,
            "source_execution_id": self.source_execution_id,
            "benchmark_id": self.benchmark_id,
            "evidence_refs": self.evidence_refs,
            "verification_refs": self.verification_refs,
            "decision_refs": self.decision_refs,
            "strategy_refs": self.strategy_refs,
            "model_id": self.model_id,
            "agent_id": self.agent_id,
            "capabilities": self.capabilities,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StructuredFailure":
        """Reconstruct a StructuredFailure from a persisted dict (backward compatible)."""
        known = {
            "node_id", "capability", "category", "sub_category", "severity",
            "error_message", "evidence", "validation_run_id", "mode",
            "timestamp", "failure_id", "source_task_id", "source_execution_id",
            "benchmark_id", "evidence_refs", "verification_refs",
            "decision_refs", "strategy_refs", "model_id", "agent_id",
            "capabilities",
        }
        return cls(**{k: v for k, v in data.items() if k in known})


# Extended failure taxonomy with sub-categories
FAILURE_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "implementation": {
        "sub_categories": [
            "logic_error", "api_misuse", "null_pointer", "type_error",
            "resource_leak", "concurrency", "off_by_one", "edge_case",
        ],
        "severity": "major",
    },
    "test": {
        "sub_categories": [
            "assertion_failure", "flaky_test", "missing_coverage",
            "false_positive", "test_environment",
        ],
        "severity": "major",
    },
    "architecture": {
        "sub_categories": [
            "circular_dependency", "tight_coupling", "god_class",
            "layering_violation", "pattern_misuse",
        ],
        "severity": "critical",
    },
    "security": {
        "sub_categories": [
            "injection", "xss", "csrf", "auth_bypass",
            "data_exposure", "privilege_escalation",
        ],
        "severity": "critical",
    },
    "integration": {
        "sub_categories": [
            "contract_mismatch", "version_incompatibility",
            "protocol_error", "timeout", "connection_refused",
        ],
        "severity": "critical",
    },
    "configuration": {
        "sub_categories": [
            "missing_env", "wrong_setting", "invalid_config",
            "missing_dependency_config",
        ],
        "severity": "minor",
    },
    "environment": {
        "sub_categories": [
            "dependency_missing", "version_conflict",
            "disk_full", "out_of_memory", "network_error",
        ],
        "severity": "major",
    },
}


class RootCauseAnalyzer:
    """Analyzes failures to determine root cause using structured taxonomy."""

    def __init__(self):
        self._analysis_history: List[StructuredFailure] = []

    def analyze(self, failure: Any, task_context: Any = None) -> StructuredFailure:
        """Analyze a failure and determine root cause."""
        node_id = self._get_node_id(failure)
        capability = self._get_capability(failure)
        error = self._get_error(failure)
        evidence_text = self._get_evidence(failure)

        category, sub_category = self._classify_failure(error, evidence_text)
        severity = self._get_severity(category, sub_category)

        result = StructuredFailure(
            node_id=node_id,
            capability=capability,
            category=category,
            sub_category=sub_category,
            severity=severity,
            error_message=error,
            evidence=evidence_text[:500],
        )

        self._analysis_history.append(result)
        return result

    def _classify_failure(self, error: str, evidence: str) -> Tuple[str, str]:
        """Classify failure into category and sub-category."""
        combined = (error + " " + evidence).lower()

        # Check sub-categories first
        for category, info in FAILURE_TAXONOMY.items():
            for sub in info["sub_categories"]:
                if sub.replace("_", " ") in combined:
                    return category, sub

        # Fallback to category-level matching
        error_lower = error.lower()
        if "assert" in error_lower or "test" in error_lower:
            return "test", "assertion_failure"
        if "vulnerability" in error_lower or "security" in error_lower:
            return "security", "auth_bypass"
        if "config" in error_lower or "env" in error_lower:
            return "configuration", "missing_env"
        if "dependency" in error_lower or "version" in error_lower:
            return "environment", "dependency_missing"
        if "timeout" in error_lower or "connection" in error_lower:
            return "integration", "timeout"
        if "typeerror" in error_lower or "attribute" in error_lower:
            return "implementation", "type_error"

        return "implementation", "logic_error"

    def _get_severity(self, category: str, sub_category: str) -> str:
        """Get severity for a failure category."""
        info = FAILURE_TAXONOMY.get(category, {})
        severity = info.get("severity", "major")
        return str(severity) if severity else "major"

    def _get_node_id(self, failure: Any) -> str:
        if hasattr(failure, 'node_id'):
            return failure.node_id
        if isinstance(failure, dict):
            return failure.get('node_id', 'unknown')
        return 'unknown'

    def _get_capability(self, failure: Any) -> str:
        if hasattr(failure, 'capability'):
            return failure.capability
        if isinstance(failure, dict):
            return failure.get('capability', '')
        return ''

    def _get_error(self, failure: Any) -> str:
        if hasattr(failure, 'message') and failure.message:
            return failure.message
        if hasattr(failure, 'error_message'):
            return failure.error_message
        if isinstance(failure, dict):
            return failure.get('message', failure.get('error_message', ''))
        return str(failure)

    def _get_evidence(self, failure: Any) -> str:
        if hasattr(failure, 'evidence_summary'):
            return failure.evidence_summary or ""
        if hasattr(failure, 'evidence'):
            return str(failure.evidence)
        if isinstance(failure, dict):
            return str(failure.get('evidence_summary', failure.get('evidence', '')))
        return ''

    def get_patterns(self) -> Dict[str, int]:
        """Get recurring failure patterns from analysis history."""
        patterns = {}
        for f in self._analysis_history:
            key = f"{f.category}/{f.sub_category}"
            patterns[key] = patterns.get(key, 0) + 1
        return dict(sorted(patterns.items(), key=lambda x: -x[1]))


class FailureToLearningPipeline:
    """Converts failures into learning experiences for future avoidance.

    V3.2 Phase 7:
    - Produces a StructuredExperience (source_type=failure) with full provenance
      and stores it via the V3.2 ExperiencePipeline (namespaced per context).
    - FAIL CLOSED: TEST and COLD failures NEVER become training experiences.
    - Requires an ExperimentContext to derive learning; without one it falls back
      to the legacy V3 behavior (store ExperienceRecord) for backward compatibility.
    - Idempotent: same failure_id + context -> same stable experience_id and a
      single canonical record (re-processing over-writes the same key).
    - Advisory only: never weakens security, governance, autonomy, or policy.
    """

    def __init__(self, experience_store: Any = None,
                 experience_pipeline: Optional[ExperiencePipeline] = None,
                 experiment_context: Optional[ExperimentContext] = None):
        # experience_store: legacy V3 ExperienceStore (backward compat).
        # experience_pipeline: V3.2 source-of-truth for failure-derived learning.
        self.experience_store = experience_store
        self.experience_pipeline = experience_pipeline
        self.experiment_context = experiment_context
        self.analyzer = RootCauseAnalyzer()
        self._saved_experience_ids: List[str] = []
        self._last_chain: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Core entry point
    # ------------------------------------------------------------------
    def process(self, failure: StructuredFailure,
                experiment_context: Optional[ExperimentContext] = None) -> Dict[str, Any]:
        """Convert a failure into a learning entry, governed by context/mode.

        Returns a result dict carrying:
          lesson (FailureLesson), capability, category, severity,
          suppressed / suppression_reason, experience_id (if stored),
          chain (failure -> lesson -> experience -> retrieval/strategy/decision),
          deterministic (stable ID), advisory.
        """
        ctx = experiment_context or self.experiment_context

        # Ensure provenance flows from context into the failure when available.
        self._apply_context_to_failure(failure, ctx)

        lesson = self._generate_lesson(failure)

        # ---- FAIL CLOSED: TEST / COLD never become training experiences. ----
        if ctx is not None and ctx.mode == "TEST":
            self._record_evidence(failure, ctx, lesson,
                                  suppressed=True,
                                  suppression_reason="TEST")
            return self._build_result(
                failure, lesson, suppressed=True,
                suppression_reason="TEST",
                experience_id=None,
                deterministic=self._stable_experience_id(failure, ctx),
            )

        if ctx is not None and ctx.mode == "COLD":
            # COLD: evidence + analysis only, NO mutation of learning state.
            self._record_evidence(failure, ctx, lesson,
                                  suppressed=True,
                                  suppression_reason="COLD",
                                  store_candidate=False)
            return self._build_result(
                failure, lesson, suppressed=True,
                suppression_reason="COLD",
                experience_id=None,
                deterministic=self._stable_experience_id(failure, ctx),
            )

        # ---- Legacy path (no context) — unchanged V3 behavior. ----
        if ctx is None:
            if self.experience_store:
                self._store_legacy(failure, lesson.lesson_text)
            return self._build_result(failure, lesson)

        # ---- LEARNED modes: derive StructuredExperience (advisory only). ----
        learning_governed = self._apply_governance_to_lesson(failure, lesson)
        experience = self._build_structured_experience(failure, ctx, lesson)
        experience_id = self._stable_experience_id(failure, ctx)

        if not self.experience_pipeline:
            # No V3.2 pipeline wired: fall back to legacy store if present.
            if self.experience_store:
                self._store_legacy(failure, lesson.lesson_text)
            return self._build_result(
                failure, lesson, suppressed=False,
                experience_id=experience_id,
                deterministic=experience_id,
                advisory=learning_governed,
            )

        # Idempotent store: stable key => single canonical record.
        experience.task_id = experience_id
        self.experience_pipeline.process_stored(experience, experiment_context=ctx)
        self._saved_experience_ids.append(experience_id)

        # Build an inspectable failure->lesson->experience->strategy chain.
        chain = self._build_chain(failure, ctx, lesson, experience_id)
        self._last_chain = chain
        self._record_chain(chain, ctx)

        return self._build_result(
            failure, lesson, suppressed=False,
            experience_id=experience_id,
            deterministic=experience_id,
            chain=chain,
            learning_influenced=True,
            advisory=learning_governed,
        )

    # ------------------------------------------------------------------
    # Provenance / identity
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_context_to_failure(failure: StructuredFailure,
                                  ctx: Optional[ExperimentContext]) -> None:
        """Populate a failure's provenance from an ExperimentContext when absent."""
        if ctx is None:
            return
        if not failure.mode:
            failure.mode = ctx.mode
        if not failure.validation_run_id:
            failure.validation_run_id = ctx.validation_run_id
        if not failure.benchmark_id:
            failure.benchmark_id = ctx.benchmark_id
        if not failure.source_task_id:
            failure.source_task_id = ctx.task_id
        if not failure.source_execution_id:
            failure.source_execution_id = ctx.execution_id
        # Ensure experiment context mode is the authority for fail-closed logic.
        failure.mode = ctx.mode
        failure.validation_run_id = ctx.validation_run_id

    @staticmethod
    def _stable_experience_id(failure: StructuredFailure,
                              ctx: Optional[ExperimentContext]) -> str:
        """Deterministic, cryptographically stable failure-experience ID.

        Never uses Python's hash() (process-randomized). Derived via SHA-256
        from the failure_id plus the experiment context, so the same
        failure + context always maps to the same canonical record across
        process restarts. If no failure_id/context is provided, falls back to a
        deterministic derivation over the failure's own taxonomy + message so
        the ID is still stable within a run.
        """
        fid = failure.failure_id or f"{failure.category}/{failure.sub_category}/{failure.node_id}"
        payload_parts = [fid, failure.mode, failure.validation_run_id,
                         failure.benchmark_id, failure.source_task_id]
        if ctx is not None:
            payload_parts = [
                fid, ctx.mode, ctx.validation_run_id, ctx.benchmark_id, ctx.task_id,
            ]
        payload = "::".join((p or "NOT_AVAILABLE") for p in payload_parts)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"fexp_{digest}"

    # ------------------------------------------------------------------
    # Lesson construction
    # ------------------------------------------------------------------
    def _generate_lesson(self, failure: StructuredFailure) -> "FailureLesson":
        """Build a structured FailureLesson (root cause, corrective action, strategy).

        Unknown root cause is recorded as UNKNOWN and never invented.
        """
        root_cause = self._analyze_root_cause(failure)
        known = root_cause.get("known", False)
        corrective, prevention, summary = self._build_lesson_text(failure, root_cause)

        return FailureLesson(
            failure_category=failure.category,
            root_cause=root_cause.get("value", "UNKNOWN"),
            root_cause_known=known,
            summary=summary,
            corrective_action=corrective,
            prevention_strategy=prevention,
            security_checked=self._is_security_failure(failure),
        )

    def _analyze_root_cause(self, failure: StructuredFailure) -> Dict[str, Any]:
        """Return a best-effort root cause; never fabricated when unknown."""
        error_l = (failure.error_message or "").lower()
        evidence_l = (failure.evidence or "").lower()
        combined = error_l + " " + evidence_l

        if failure.category == "integration":
            for kw in ("timeout", "connection_refused", "protocol", "contract"):
                if kw in combined:
                    return {"value": f"integration_{kw}", "known": True}
        if failure.category == "configuration":
            for kw in ("missing_env", "wrong_setting", "invalid_config"):
                if kw in combined:
                    return {"value": f"configuration_{kw}", "known": True}
        if failure.category == "environment":
            for kw in ("dependency_missing", "version_conflict", "network_error",
                       "out_of_memory", "disk_full"):
                if kw in combined:
                    return {"value": f"environment_{kw}", "known": True}
        if failure.category == "implementation":
            for kw in ("null", "typeerror", "attributeerror", "logic", "off_by_one",
                       "edge case", "concurrency", "resource leak", "api misuse"):
                if kw in combined:
                    return {"value": f"implementation_{kw}", "known": True}
        if failure.category == "security":
            for kw in ("injection", "xss", "csrf", "auth", "privilege", "exposure"):
                if kw in combined:
                    return {"value": f"security_{kw}", "known": True}
        if failure.category == "test":
            return {"value": "test_assertion", "known": True}
        if failure.category == "architecture":
            return {"value": "architecture_design", "known": True}
        # Unknown category / no signal: honest UNKNOWN.
        return {"value": "UNKNOWN", "known": False}

    def _build_lesson_text(self, failure: StructuredFailure,
                           root_cause: Dict[str, Any]
                           ) -> Tuple[str, str, str]:
        """Construct corrective action, prevention strategy, and summary text.

        Security failures only emit defensive, constraint-preserving lessons.
        """
        is_security = self._is_security_failure(failure)
        summary = (
            f"Failure {failure.category}.{failure.sub_category} in "
            f"'{failure.capability}': {failure.error_message[:160]}"
        )

        if is_security:
            # DEFENSIVE ONLY — never weaken security/gates/autonomy. Avoid
            # forbidden tokens entirely (even negated mentions trip the guard).
            corrective = (
                f"Validate earlier and enforce safe selection; all "
                    f"existing constraints remain enforced. Category: {failure.category}."
            )
            prevention = (
                f"Plan permitted actions that stay within the security boundary "
                f"and reject unsafe options before execution."
            )
            return corrective, prevention, summary

        if root_cause.get("known"):
            corrective = (
                f"Fix the {failure.category} root cause "
                f"({root_cause['value']}) in '{failure.capability}' before retry."
            )
            prevention = (
                f"Add a {failure.category} guard to detect and avoid "
                f"{root_cause['value']} earlier in the pipeline."
            )
            return corrective, prevention, summary

        # Unknown root cause — record honestly.
        corrective = (
            f"Root cause unknown; re-run and gather evidence before changing "
            f"strategy in '{failure.capability}'."
        )
        prevention = "UNKNOWN — do not invent a prevention strategy without evidence."
        return corrective, prevention, summary

    @staticmethod
    def _is_security_failure(failure: StructuredFailure) -> bool:
        return failure.category == "security"

    def _apply_governance_to_lesson(self, failure: StructuredFailure,
                                    lesson: "FailureLesson") -> Dict[str, Any]:
        """Sanity-guard the derived lesson against constraint-weakening.

        Returns advisory flags; PolicyEngine remains authoritative and no lesson
        here can disable any gate, expand autonomy, or bypass review.
        """
        flags = {
            "advisory_only": True,
            "weakening_constraint": False,
            "security_defensive": False,
            "governance_preserved": True,
        }
        lesson_text = " ".join(filter(None, [
            lesson.summary, lesson.corrective_action, lesson.prevention_strategy,
        ])).lower()
        for bad in _FORBIDDEN_SECURITY_LESSONS:
            if bad in lesson_text:
                flags["weakening_constraint"] = True
                # Never emit or propagate a weakening lesson; force a defensive,
                # sanitized replacement across summary + actions.
                lesson.summary = (
                    f"Failure in {failure.capability} "
                    f"({failure.category}/{failure.sub_category}); "
                    f"remediation is advisory and within existing policy."
                )
                lesson.corrective_action = (
                    "Defensive only: validate earlier and stay within existing "
                    "policy boundaries. All existing gates remain enforced."
                )
                lesson.prevention_strategy = (
                    "Plan permitted actions only; governance remains authoritative."
                )
                break
        if self._is_security_failure(failure):
            flags["security_defensive"] = True
        return flags

    # ------------------------------------------------------------------
    # Structured experience construction
    # ------------------------------------------------------------------
    def _build_structured_experience(self, failure: StructuredFailure,
                                     ctx: ExperimentContext,
                                     lesson: "FailureLesson"
                                     ) -> StructuredExperience:
        """Build a StructuredExperience that explicitly identifies failure origin."""
        capabilities = failure.capabilities or [failure.capability]
        evidence_refs = list(failure.evidence_refs)
        strategy_refs = list(failure.strategy_refs)

        provenance = {
            "source_type": "failure",
            "failure_id": failure.failure_id or "NOT_AVAILABLE",
            "validation_run_id": ctx.validation_run_id,
            "benchmark_id": failure.benchmark_id or ctx.benchmark_id,
            "mode": ctx.mode,
            "source_task_id": failure.source_task_id or ctx.task_id,
            "source_execution_id": failure.source_execution_id or ctx.execution_id,
            "failure_category": f"{failure.category}/{failure.sub_category}",
            "model_id": failure.model_id,
            "agent_id": failure.agent_id,
            "capabilities": capabilities,
            "evidence_refs": evidence_refs,
            "verification_refs": list(failure.verification_refs),
            "decision_refs": list(failure.decision_refs),
            "strategy_refs": strategy_refs,
            "timestamp": failure.timestamp,
        }

        exp_evidence = [
            Evidence(
                description=failure.error_message[:200],
                metric="error_count",
                value=1.0,
                source="execution",
            ),
        ]
        if failure.evidence:
            exp_evidence.append(Evidence(
                description=failure.evidence[:200],
                metric="evidence",
                value=1.0,
                source="verification",
            ))

        return StructuredExperience(
            task_id=self._stable_experience_id(failure, ctx),
            task_context=f"Failure-derived: {failure.capability} "
                         f"({failure.category}/{failure.sub_category})",
            capabilities_used=capabilities,
            strategy={} or ({ref: True for ref in strategy_refs} if strategy_refs else {}),
            outcome={"status": "FAILED", "score": 0.0, "duration": 0.0,
                     "success_rate": 0.0},
            failures=[{"message": failure.error_message, "node_id": failure.node_id,
                       "capability": failure.capability}],
            lesson=lesson.lesson_text,
            confidence=self._lesson_confidence(lesson),
            evidence=exp_evidence,
            category="failure_lesson",
            validation_run_id=ctx.validation_run_id,
            benchmark_id=failure.benchmark_id or ctx.benchmark_id,
            mode=ctx.mode,
            execution_id=ctx.execution_id,
            timestamp=failure.timestamp,
            source_type="failure",
            source_task_id=failure.source_task_id or ctx.task_id,
            source_execution_id=failure.source_execution_id or ctx.execution_id,
            failure_id=failure.failure_id or "NOT_AVAILABLE",
            failure_category=f"{failure.category}/{failure.sub_category}",
            provenance=provenance,
        )

    @staticmethod
    def _lesson_confidence(lesson: "FailureLesson") -> float:
        base = 0.5
        if lesson.root_cause_known:
            base += 0.3
        base -= 0.1 if lesson.failure_category == "security" else 0.0
        return round(max(0.0, min(1.0, base)), 4)

    # ------------------------------------------------------------------
    # Chain / decision attribution
    # ------------------------------------------------------------------
    def _build_chain(self, failure: StructuredFailure, ctx: ExperimentContext,
                     lesson: "FailureLesson", experience_id: str) -> Dict[str, Any]:
        """Record the failure->lesson->experience chain (before strategy/decision)."""
        return {
            "failure_id": failure.failure_id or "NOT_AVAILABLE",
            "experience_id": experience_id,
            "lesson_id": None,          # filled after lesson persisted
            "retrieval": None,          # filled when a related task retrieves it
            "strategy_id": None,
            "decision_id": None,
            "learning_influenced": True,
            "failure_category": f"{failure.category}/{failure.sub_category}",
            "mode": ctx.mode,
            "source_task_id": failure.source_task_id or ctx.task_id,
            "source_execution_id": failure.source_execution_id or ctx.execution_id,
            "validation_run_id": ctx.validation_run_id,
            "benchmark_id": failure.benchmark_id or ctx.benchmark_id,
            "lesson_summary": lesson.summary,
            "root_cause": lesson.root_cause,
            "root_cause_known": lesson.root_cause_known,
        }

    def _record_chain(self, chain: Dict[str, Any],
                      ctx: Optional[ExperimentContext]) -> None:
        """Persist the chain record for later inspection (retrieval/decision)."""
        if ctx is None:
            return
        try:
            chain_dir = _ns("failure_chains", experiment_context=ctx)
        except Exception:
            return
        path = os.path.join(chain_dir, f"{chain['experience_id']}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(chain, f, indent=2)

    # ------------------------------------------------------------------
    # Evidence / suppression records (fail-closed)
    # ------------------------------------------------------------------
    def _record_evidence(self, failure: StructuredFailure, ctx: ExperimentContext,
                         lesson: "FailureLesson", suppressed: bool,
                         suppression_reason: str, store_candidate: bool = False) -> None:
        """Persist failure evidence; TEST/COLD never enter the training store.

        TEST: evidence recorded for evaluation/reporting only (no candidate).
        COLD: may store a candidate as evidence OUTSIDE active training knowledge
        (store_candidate=False by default; callers may opt in).
        """
        evidence = {
            "source_type": "failure_evidence",
            "failure_id": failure.failure_id or "NOT_AVAILABLE",
            "node_id": failure.node_id,
            "capability": failure.capability,
            "category": failure.category,
            "sub_category": failure.sub_category,
            "severity": failure.severity,
            "error_message": failure.error_message,
            "evidence": failure.evidence,
            "validation_run_id": ctx.validation_run_id,
            "benchmark_id": failure.benchmark_id or ctx.benchmark_id,
            "mode": ctx.mode,
            "source_task_id": failure.source_task_id or ctx.task_id,
            "source_execution_id": failure.source_execution_id or ctx.execution_id,
            "timestamp": failure.timestamp,
            "learning_suppressed": suppressed,
            "suppression_reason": suppression_reason,
            "training_mutation": False,
            "stored_candidate_as_evidence": store_candidate,
            "lesson_summary": lesson.summary,
            "lesson": lesson.lesson_text,
        }
        try:
            ev_dir = _ns("failure_evidence", experiment_context=ctx)
        except Exception:
            return
        os.makedirs(ev_dir, exist_ok=True)
        path = os.path.join(ev_dir, f"{self._stable_experience_id(failure, ctx)}.json")
        with open(path, "w") as f:
            json.dump(evidence, f, indent=2)

    def _store_legacy(self, failure: StructuredFailure, lesson_text: str) -> None:
        """Legacy V3 storage path (backward compatibility, no context)."""
        from harness.memory.v3 import ExperienceRecord
        exp = ExperienceRecord(
            task_id=f"failure-{failure.node_id}-"
                    f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            task_objective=f"Failure: {failure.error_message[:100]}",
            capabilities_used=[failure.capability],
            result_status="FAILED",
            score=0.0,
            duration_seconds=0.0,
            lessons=[lesson_text],
        )
        self.experience_store.store(exp)

    # ------------------------------------------------------------------
    # Result assembly
    # ------------------------------------------------------------------
    def _build_result(self, failure: StructuredFailure, lesson: "FailureLesson",
                      suppressed: bool = False, suppression_reason: str = "",
                      experience_id: Optional[str] = None,
                      deterministic: Optional[str] = None,
                      chain: Optional[Dict[str, Any]] = None,
                      learning_influenced: bool = False,
                      advisory: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "lesson": lesson.lesson_text,
            "failure_category": failure.category,
            "capability": failure.capability,
            "category": failure.category,
            "severity": failure.severity,
            "suppressed": suppressed,
            "suppression_reason": suppression_reason,
            "experience_id": experience_id,
            "deterministic_id": deterministic,
            "chain": chain,
            "learning_influenced": learning_influenced,
            "advisory": advisory,
            "root_cause": lesson.root_cause,
            "root_cause_known": lesson.root_cause_known,
        }

    # ------------------------------------------------------------------
    # Retrieval / decision influence helpers
    # ------------------------------------------------------------------
    def record_failure_retrieval(self, experience_id: str, strategy_id: str,
                                 decision_id: str, retrieval_confidence: float,
                                 ctx: Optional[ExperimentContext] = None) -> Dict[str, Any]:
        """Record that a failure-derived experience influenced a later decision.

        Requirement #14: Task A -> Failure F -> Lesson L -> Experience E ->
        persist; Task B related -> retrieve E -> strategy/decision influenced.
        """
        ctx = ctx or self.experiment_context
        entry = {
            "failure_id": (self._last_chain or {}).get("failure_id", "NOT_AVAILABLE"),
            "experience_id": experience_id,
            "strategy_id": strategy_id,
            "decision_id": decision_id,
            "retrieval_confidence": retrieval_confidence,
            "learning_influenced": True,
            "mode": ctx.mode if ctx else "NOT_AVAILABLE",
            "validation_run_id": ctx.validation_run_id if ctx else "NOT_AVAILABLE",
        }
        self._last_chain = {**(self._last_chain or {}), **entry}
        if ctx is not None:
            try:
                chain_dir = _ns("failure_chains", experiment_context=ctx)
            except Exception:
                chain_dir = ""
            path = os.path.join(chain_dir, f"{experience_id}.json")
            if chain_dir and os.path.exists(path):
                with open(path) as f:
                    existing = json.load(f)
                existing.update(entry)
                with open(path, "w") as f:
                    json.dump(existing, f, indent=2)
        return entry

    def failure_avoidance_metric(self, learned_count: int, avoided_count: int,
                                 baseline_fail_count: int) -> Dict[str, Any]:
        """Feed the failure-avoidance metric (comparative vs causal preserved).

        avoided_count is only counted when a derived experience was actually
        retrieved and influenced the subsequent task — not for simple success.
        """
        return {
            "learned_experiences": learned_count,
            "avoided_failures": avoided_count,
            "baseline_failures": baseline_fail_count,
            "avoidance_rate": round(avoided_count / learned_count, 4) if learned_count else 0.0,
            "causal": True,   # only counted when retrieval influence is recorded
            "comparative_note": (
                "Simple success is NOT counted as causal avoidance; "
                "only retrieval-influenced avoidance is."
            ),
        }

    @property
    def saved_experience_ids(self) -> List[str]:
        return list(self._saved_experience_ids)


@dataclass
class FailureLesson:
    """A structured lesson distilled from a failure.

    Preserves Failure -> Analysis -> Lesson -> Experience -> Strategy. Unknown
    root cause is recorded explicitly, never invented.
    """
    failure_category: str
    root_cause: str
    corrective_action: str
    prevention_strategy: str
    summary: str = ""
    root_cause_known: bool = False
    security_checked: bool = False

    @property
    def lesson_text(self) -> str:
        return (
            f"AVOID {self.failure_category}: {self.summary}. "
            f"Root cause: {self.root_cause} "
            f"({'known' if self.root_cause_known else 'UNKNOWN'}). "
            f"Corrective: {self.corrective_action}. "
            f"Prevention: {self.prevention_strategy}"
        )

    def to_dict(self) -> dict:
        return {
            "failure_category": self.failure_category,
            "root_cause": self.root_cause,
            "root_cause_known": self.root_cause_known,
            "summary": self.summary,
            "corrective_action": self.corrective_action,
            "prevention_strategy": self.prevention_strategy,
            "security_checked": self.security_checked,
            "lesson_text": self.lesson_text,
        }


class FailureAnalyzer:
    """Analyzes failures into StructuredFailure + FailureLesson.

    Canonical entry: StructuredFailure -> FailureAnalyzer -> FailureToLearningPipeline.
    Wraps RootCauseAnalyzer for taxonomy classification and produces the lesson.
    """

    def __init__(self):
        self._analyzer = RootCauseAnalyzer()

    def analyze(self, failure_input: Any,
                experiment_context: Optional[ExperimentContext] = None
                ) -> StructuredFailure:
        return self._analyzer.analyze(failure_input, task_context=None)

    def derive_lesson(self, failure: StructuredFailure) -> FailureLesson:
        pipeline = FailureToLearningPipeline(experiment_context=None)
        return pipeline._generate_lesson(failure)


class ReplanningOptimizer:
    """Optimizes replanning decisions based on failure patterns."""

    def __init__(self, failure_analyzer: Any, max_replans: int = 3):
        self.failure_analyzer = failure_analyzer
        self.max_replans = max_replans
        self._attempt_counts: Dict[str, int] = {}

    def should_replan(self, task_id: str, failure: Any) -> bool:
        """Determine if replanning is worthwhile based on failure analysis."""
        attempts = self._attempt_counts.get(task_id, 0)

        if attempts >= self.max_replans:
            return False

        # Check if the failure category is recoverable
        category = self._get_category(failure)
        if category in ("security", "architecture"):
            # Security and architecture failures need human intervention
            return False

        # Implementation and test failures are good candidates for replanning
        if category in ("implementation", "test", "configuration", "environment"):
            return True

        # For unknown/integration failures, allow one replan attempt
        if attempts == 0:
            return True

        return False

    def get_optimal_replan_strategy(self, failure: Any) -> str:
        """Get the optimal replan strategy based on failure type."""
        category = self._get_category(failure)

        strategies = {
            "implementation": "retry_with_different_agent",
            "test": "add_test_specific_node",
            "configuration": "add_environment_setup_node",
            "environment": "add_dependency_check_node",
            "integration": "add_integration_test_node",
            "security": "escalate_to_human",
            "architecture": "escalate_to_human",
        }

        return strategies.get(category, "retry_same")

    def record_attempt(self, task_id: str):
        """Record a replanning attempt."""
        self._attempt_counts[task_id] = self._attempt_counts.get(task_id, 0) + 1

    def get_attempt_count(self, task_id: str) -> int:
        """Get the number of replanning attempts for a task."""
        return self._attempt_counts.get(task_id, 0)

    def _get_category(self, failure: Any) -> str:
        if hasattr(failure, 'category'):
            return failure.category
        if isinstance(failure, dict):
            return failure.get('category', 'unknown')
        return 'unknown'
