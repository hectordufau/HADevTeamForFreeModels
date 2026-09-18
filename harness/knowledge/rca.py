# harness/knowledge/rca.py — V3.3 Phase 7: RCA Record & Failure Intelligence Integration
"""
Root Cause Analysis Record (RCA) — canonical Engineering Knowledge for failure analysis.

Phase 7 Implementation — TASK-057 + TASK-058.

RCA is an Engineering Record (persistent knowledge), NOT the analyzer (mechanism).
- RootCauseAnalyzer = analysis mechanism (service)
- RCA = persistent Engineering Record (knowledge)

RCA != FailureLesson. A single RCA may generate 0..N FailureLessons.
Derivation is explicit with stable provenance authority separation.

Architecture:
- RCA inherits from EngineeringRecord base (Phase 1)
- RCA references failures/evidence indirectly via stable IDs (no embedding)
- Agent-generated RCA always starts as PROPOSED (never auto-ACCEPTED)
- Corrective vs Preventive actions are distinguished
- Security precedence: RCA-derived lessons cannot override SEC

Three-domain separation:
- RCA belongs to Engineering Knowledge (does NOT become Learning, NOT Evidence)
- FailureLesson belongs to Learning Knowledge (references RCA by ID)
- StructuredFailure and Evidence remain externally owned (referenced by ID)
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json

from .records import (
    EngineeringRecord,
    RecordRelationship,
    RecordError,
    RECORD_ID_PATTERN,
    VALID_RELATIONSHIP_TYPES,
    RELATIONSHIP_CAUSED_BY,
    RELATIONSHIP_RESOLVED_BY,
    RELATIONSHIP_VERIFIED_BY,
    RELATIONSHIP_SUPPORTED_BY,
    RELATIONSHIP_CONTRADICTS,
    RELATIONSHIP_RELATES_TO,
    RELATIONSHIP_DERIVED_FROM,
    RELATIONSHIP_CONSTRAINED_BY,
    RELATIONSHIP_INTRODUCES,
    RELATIONSHIP_MITIGATES,
    RELATIONSHIP_SUPERSEDES,
    RELATIONSHIP_REQUIRES,
    RELATIONSHIP_SATISFIES,
    RELATIONSHIP_DECIDED_BY,
)
from .provenance import (
    Provenance,
    AUTHORITY_PROPOSED,
    AUTHORITY_ACCEPTED,
    AUTHORITY_DEPRECATED,
    AUTHORITY_ARCHIVED,
)
from .lifecycle import (
    get_initial_state,
    get_lifecycle_definition,
    is_valid_transition,
    is_terminal_state,
)

# ──────────────────────────────────────────────────────────────────────
# RCA Failure Taxonomy — maps V3.2 failure categories to RCA root cause categories
# ──────────────────────────────────────────────────────────────────────

RCA_CAUSE_CATEGORY_IMPLEMENTATION = "implementation"
RCA_CAUSE_CATEGORY_TEST = "test"
RCA_CAUSE_CATEGORY_ARCHITECTURE = "architecture"
RCA_CAUSE_CATEGORY_SECURITY = "security"
RCA_CAUSE_CATEGORY_INTEGRATION = "integration"
RCA_CAUSE_CATEGORY_CONFIGURATION = "configuration"
RCA_CAUSE_CATEGORY_ENVIRONMENT = "environment"
RCA_CAUSE_CATEGORY_UNKNOWN = "unknown"

VALID_RCA_CAUSE_CATEGORIES = {
    RCA_CAUSE_CATEGORY_IMPLEMENTATION,
    RCA_CAUSE_CATEGORY_TEST,
    RCA_CAUSE_CATEGORY_ARCHITECTURE,
    RCA_CAUSE_CATEGORY_SECURITY,
    RCA_CAUSE_CATEGORY_INTEGRATION,
    RCA_CAUSE_CATEGORY_CONFIGURATION,
    RCA_CAUSE_CATEGORY_ENVIRONMENT,
    RCA_CAUSE_CATEGORY_UNKNOWN,
}

# ──────────────────────────────────────────────────────────────────────
# RCA Lifecycle States
# ──────────────────────────────────────────────────────────────────────

RCA_STATE_DRAFT = "draft"
RCA_STATE_ANALYSIS = "analysis"
RCA_STATE_CORRECTIVE_ACTION = "corrective_action"
RCA_STATE_PREVENTIVE_ACTION = "preventive_action"
RCA_STATE_CLOSED = "closed"

VALID_RCA_STATES = {
    RCA_STATE_DRAFT,
    RCA_STATE_ANALYSIS,
    RCA_STATE_CORRECTIVE_ACTION,
    RCA_STATE_PREVENTIVE_ACTION,
    RCA_STATE_CLOSED,
}

# ──────────────────────────────────────────────────────────────────────
# Failure Knowledge Coverage — deterministic completeness for RCA
# ──────────────────────────────────────────────────────────────────────


@dataclass
class RCACompleteness:
    """Deterministic completeness check for an RCA."""
    has_failure_ref: bool = False
    has_summary: bool = False
    has_symptoms: bool = False
    has_root_causes: bool = False
    has_impact: bool = False
    has_corrective_actions: bool = False
    has_preventive_actions: bool = False
    has_evidence: bool = False
    has_provenance: bool = False
    is_complete: bool = False


@dataclass
class FailureKnowledgeCoverage:
    """Deterministic coverage: Failure → RCA → (Evidence, Actions) + Learning."""
    failure_id: str = ""
    has_rca: bool = False
    rca_id: Optional[str] = None
    rca_has_evidence: bool = False
    rca_has_corrective_action: bool = False
    rca_has_preventive_action: bool = False
    has_failure_lesson: bool = False
    failure_lesson_id: Optional[str] = None
    has_experience: bool = False
    experience_id: Optional[str] = None
    has_strategy: bool = False
    strategy_id: Optional[str] = None
    coverage_complete: bool = False  # RCA + Learning both present


# ──────────────────────────────────────────────────────────────────────
# RootCauseAnalysisRecord (RCA) — Engineering Knowledge
# ──────────────────────────────────────────────────────────────────────


@dataclass
class RootCauseAnalysisRecord(EngineeringRecord):
    """
    Root Cause Analysis Record — persistent Engineering Knowledge.

    RCA is an Engineering Record (knowledge), NOT the analyzer (mechanism).
    A single RCA answers: What happened? Why? What contributed?
    What corrected it? What should engineering change?

    Lifecycle: draft → analysis → corrective_action → preventive_action → closed

    Authority:
    - Agent-generated RCA always starts as PROPOSED (never auto-ACCEPTED)
    - Analysis confidence != Engineering Knowledge authority
    - RCA ACCEPTED → requires explicit governance approval

    Semantic boundary:
    - RCA != FailureLesson (RCA is knowledge, lesson is learning)
    - RCA != StructuredFailure (RCA is analysis, failure is evidence)
    - RCA != Strategy (RCA informs, strategy decides)

    Version vs Supersession:
    - Version = same investigation evolves (same RCA ID, version++)
    - Supersession = different RCA replaces earlier (new RCA + SUPERSEDES)
    """

    record_type: str = "RCA"
    title: str = ""
    description: str = ""  # Full analysis description
    status: str = RCA_STATE_DRAFT
    authority: str = AUTHORITY_PROPOSED

    # RCA-specific fields
    summary: str = ""                    # Executive summary
    failure_ref: str = ""                # Stable failure identity (failure_id)
    failure_type: str = ""               # From FAILURE_TAXONOMY
    symptoms: List[str] = field(default_factory=list)        # Observed symptoms
    root_causes: List[str] = field(default_factory=list)     # Root causes (multiple)
    contributing_factors: List[str] = field(default_factory=list)  # Contributing factors
    impact: str = ""                     # Impact description
    resolution: str = ""                 # Resolution summary
    corrective_actions: List[str] = field(default_factory=list)   # Fixes applied
    preventive_actions: List[str] = field(default_factory=list)   # Prevention steps
    incident_id: str = ""                # Associated incident ID

    # Cross-domain references (by ID, no embedding)
    evidence_refs: List[str] = field(default_factory=list)        # External evidence IDs
    related_failures: List[str] = field(default_factory=list)     # Related failure IDs
    related_decisions: List[str] = field(default_factory=list)     # ADR/DR IDs

    def __post_init__(self):
        if not self.title:
            self.title = self.record_id
        if not self.description:
            self.description = self.title
        # Establish canonical relationships
        for ev_id in self.evidence_refs:
            self.add_relationship(ev_id, RELATIONSHIP_SUPPORTED_BY)
        for adr_id in self.related_decisions:
            self.add_relationship(adr_id, RELATIONSHIP_RELATES_TO)
        for fail_id in self.related_failures:
            self.add_relationship(fail_id, RELATIONSHIP_RELATES_TO)
        super().__post_init__()

    def validate(self) -> None:
        """Validate RCA-specific fields."""
        super().validate()

        # Validate RCA ID prefix
        prefix = self.record_id.split("-")[0]
        if prefix != "RCA":
            raise RecordError(f"RCA record_id prefix must be 'RCA', got '{prefix}'")

        # Validate failure_ref (required for completeness)
        if not isinstance(self.failure_ref, str) or not self.failure_ref.strip():
            raise RecordError("failure_ref is required (RCA must reference a stable failure identity)")

        # Validate at least one root cause (required for completeness)
        if not isinstance(self.root_causes, list):
            raise RecordError("root_causes must be a list")
        if len(self.root_causes) == 0:
            raise RecordError("at least one root cause is required")

        # Validate symptoms
        if not isinstance(self.symptoms, list):
            raise RecordError("symptoms must be a list")

        # Validate contributing_factors
        if not isinstance(self.contributing_factors, list):
            raise RecordError("contributing_factors must be a list")

        # Validate corrective_actions
        if not isinstance(self.corrective_actions, list):
            raise RecordError("corrective_actions must be a list")
        if len(self.corrective_actions) == 0:
            raise RecordError("at least one corrective action is required")

        # Validate preventive_actions
        if not isinstance(self.preventive_actions, list):
            raise RecordError("preventive_actions must be a list")
        if len(self.preventive_actions) == 0:
            raise RecordError("at least one preventive action is required")

        # Validate evidence_refs (external references)
        if not isinstance(self.evidence_refs, list):
            raise RecordError("evidence_refs must be a list")
        for ev_id in self.evidence_refs:
            if not isinstance(ev_id, str) or not ev_id.strip():
                raise RecordError("evidence_refs must contain non-empty strings")

        # Validate related_failures (external references)
        if not isinstance(self.related_failures, list):
            raise RecordError("related_failures must be a list")
        for fail_id in self.related_failures:
            if not isinstance(fail_id, str) or not fail_id.strip():
                raise RecordError("related_failures must contain non-empty strings")

        # Validate related_decisions (ADR/DR IDs)
        if not isinstance(self.related_decisions, list):
            raise RecordError("related_decisions must be a list")
        for adr_id in self.related_decisions:
            if not isinstance(adr_id, str):
                raise RecordError("related_decisions must contain strings")
            if not (adr_id.startswith("ADR-") or adr_id.startswith("DR-")):
                raise RecordError(
                    f"related_decisions must reference ADR/DR IDs, got '{adr_id}'"
                )

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["summary"] = self.summary
        base["failure_ref"] = self.failure_ref
        base["failure_type"] = self.failure_type
        base["symptoms"] = list(self.symptoms)
        base["root_causes"] = list(self.root_causes)
        base["contributing_factors"] = list(self.contributing_factors)
        base["impact"] = self.impact
        base["resolution"] = self.resolution
        base["corrective_actions"] = list(self.corrective_actions)
        base["preventive_actions"] = list(self.preventive_actions)
        base["incident_id"] = self.incident_id
        base["evidence_refs"] = list(self.evidence_refs)
        base["related_failures"] = list(self.related_failures)
        base["related_decisions"] = list(self.related_decisions)
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "RootCauseAnalysisRecord":
        record = cls(
            record_id=data["record_id"],
            record_type="RCA",
            title=data["title"],
            description=data["description"],
            summary=data.get("summary", ""),
            failure_ref=data.get("failure_ref", ""),
            failure_type=data.get("failure_type", ""),
            symptoms=list(data.get("symptoms", [])),
            root_causes=list(data.get("root_causes", [])),
            contributing_factors=list(data.get("contributing_factors", [])),
            impact=data.get("impact", ""),
            resolution=data.get("resolution", ""),
            corrective_actions=list(data.get("corrective_actions", [])),
            preventive_actions=list(data.get("preventive_actions", [])),
            incident_id=data.get("incident_id", ""),
            evidence_refs=list(data.get("evidence_refs", [])),
            related_failures=list(data.get("related_failures", [])),
            related_decisions=list(data.get("related_decisions", [])),
            status=data.get("status", RCA_STATE_DRAFT),
            authority=data.get("authority", AUTHORITY_PROPOSED),
            provenance=Provenance.from_dict(data["provenance"]),
            tags=list(data.get("tags", [])),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
            version=data.get("version", 1),
            superseded_by=data.get("superseded_by"),
            experiment_context=data.get("experiment_context"),
        )
        record._relationships = [
            RecordRelationship.from_dict(r) for r in data.get("relationships", [])
        ]
        return record

    def link_adr(self, adr_id: str) -> None:
        """Link an ADR/DR that is related to this RCA."""
        if not (adr_id.startswith("ADR-") or adr_id.startswith("DR-")):
            raise RecordError(f"Must reference an ADR/DR ID, got '{adr_id}'")
        if adr_id not in self.related_decisions:
            self.related_decisions.append(adr_id)
            self.updated_at = datetime.utcnow().isoformat()
            self.add_relationship(adr_id, RELATIONSHIP_RELATES_TO)

    def link_failure(self, failure_id: str) -> None:
        """Link a related failure (by stable ID)."""
        if not isinstance(failure_id, str) or not failure_id.strip():
            raise RecordError("failure_id must be a non-empty string")
        if failure_id not in self.related_failures:
            self.related_failures.append(failure_id)
            self.updated_at = datetime.utcnow().isoformat()
            self.add_relationship(failure_id, RELATIONSHIP_RELATES_TO)

    def add_evidence_ref(self, evidence_id: str) -> None:
        """Add an evidence reference (by ID, no embedding)."""
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise RecordError("evidence_id must be a non-empty string")
        if evidence_id not in self.evidence_refs:
            self.evidence_refs.append(evidence_id)
            self.updated_at = datetime.utcnow().isoformat()
            self.add_relationship(evidence_id, RELATIONSHIP_SUPPORTED_BY)

    def can_transition_to(self, new_status: str) -> bool:
        """Check if transition to new_status is valid."""
        return is_valid_transition("RCA", self.status, new_status)

    def get_completeness(self) -> RCACompleteness:
        """Deterministic completeness check for an RCA."""
        completeness = RCACompleteness(
            has_failure_ref=bool(self.failure_ref and self.failure_ref.strip()),
            has_summary=bool(self.summary and self.summary.strip()),
            has_symptoms=len(self.symptoms) > 0,
            has_root_causes=len(self.root_causes) > 0,
            has_impact=bool(self.impact and self.impact.strip()),
            has_corrective_actions=len(self.corrective_actions) > 0,
            has_preventive_actions=len(self.preventive_actions) > 0,
            has_evidence=len(self.evidence_refs) > 0,
            has_provenance=self.provenance is not None and self.provenance.is_complete(),
        )
        completeness.is_complete = all([
            completeness.has_failure_ref,
            completeness.has_summary,
            completeness.has_root_causes,
            completeness.has_corrective_actions,
            completeness.has_preventive_actions,
            completeness.has_provenance,
        ])
        return completeness

    def is_complete(self) -> bool:
        """Check if the RCA is complete."""
        return self.get_completeness().is_complete

    # ──────────────────────────────────────────────────────────────────
    # Cross-domain reference helpers
    # ──────────────────────────────────────────────────────────────────

    def references_failure(self, failure_id: str) -> bool:
        """Check if this RCA references a given failure ID."""
        return failure_id == self.failure_ref or failure_id in self.related_failures

    def references_decision(self, decision_id: str) -> bool:
        """Check if this RCA references a given ADR/DR ID."""
        return decision_id in self.related_decisions

    def references_evidence(self, evidence_id: str) -> bool:
        """Check if this RCA references a given evidence ID."""
        return evidence_id in self.evidence_refs

    # ──────────────────────────────────────────────────────────────────
    # RCA → FailureLesson Derivation (Learning bridge)
    # ──────────────────────────────────────────────────────────────────

    def derive_failure_lesson(self) -> Dict[str, Any]:
        """
        Derive a FailureLesson proposal from this RCA.

        Returns a dict with lesson fields and provenance.
        The returned lesson is NOT stored — it's a proposal that
        must be reviewed and stored via FailureToLearningPipeline.

        Derivation is explicit with stable provenance:
        - source_rca_id: this RCA's record_id
        - source_failure_id: this RCA's failure_ref

        Authority separation:
        - RCA ACCEPTED does NOT automatically mean FailureLesson AUTHORITATIVE
        - High-confidence FailureLesson does NOT upgrade RCA authority
        """
        return {
            "source_rca_id": self.record_id,
            "source_failure_id": self.failure_ref,
            "failure_category": self.failure_type,
            "root_cause": "; ".join(self.root_causes) if self.root_causes else "UNKNOWN",
            "root_cause_known": len(self.root_causes) > 0,
            "summary": self.summary,
            "corrective_action": "; ".join(self.corrective_actions) if self.corrective_actions else "",
            "prevention_strategy": "; ".join(self.preventive_actions) if self.preventive_actions else "",
            "security_checked": self.failure_type == "security",
            "authority": "proposed",  # Derived lessons start as proposed
            "derivation_provenance": {
                "derived_from": "RCA",
                "source_rca_id": self.record_id,
                "source_failure_id": self.failure_ref,
            },
        }

    # ──────────────────────────────────────────────────────────────────
    # Version vs Supersession
    # ──────────────────────────────────────────────────────────────────

    def create_new_version(self, new_record_id: str) -> "RootCauseAnalysisRecord":
        """
        Create a new version of this RCA (same investigation evolves).

        Version = same logical investigation, new findings.
        """
        if self.superseded_by is not None:
            raise RecordError(
                f"RCA {self.record_id} is already superseded by {self.superseded_by}"
            )

        new_rca = RootCauseAnalysisRecord(
            record_id=new_record_id,
            record_type="RCA",
            title=self.title,
            description=self.description,
            summary=self.summary,
            failure_ref=self.failure_ref,
            failure_type=self.failure_type,
            symptoms=list(self.symptoms),
            root_causes=list(self.root_causes),
            contributing_factors=list(self.contributing_factors),
            impact=self.impact,
            resolution=self.resolution,
            corrective_actions=list(self.corrective_actions),
            preventive_actions=list(self.preventive_actions),
            incident_id=self.incident_id,
            evidence_refs=list(self.evidence_refs),
            related_failures=list(self.related_failures),
            related_decisions=list(self.related_decisions),
            status=RCA_STATE_DRAFT,
            authority=AUTHORITY_PROPOSED,
            provenance=Provenance(
                author=self.provenance.author,
                source=self.provenance.source,
                parent_record=self.record_id,
            ),
            tags=list(self.tags),
            version=self.version + 1,
        )
        self.superseded_by = new_record_id
        return new_rca


# Type alias for clarity in imports
RCA = RootCauseAnalysisRecord


# ──────────────────────────────────────────────────────────────────────
# RCAAnalyzer — creates RCA records from failures (does NOT replace RootCauseAnalyzer)
# ──────────────────────────────────────────────────────────────────────


class RCAAnalyzer:
    """
    Creates RCA Engineering Records from failure analysis.

    This is a mechanism that produces RCA records. It does NOT replace
    RootCauseAnalyzer — it uses it internally.

    RootCauseAnalyzer = analysis mechanism (service) → StructuredFailure
    RCAAnalyzer = RCA creation mechanism (service) → RCA (Engineering Knowledge)

    Architecture:
    - RCAAnalyzer uses RootCauseAnalyzer for taxonomy classification
    - Does NOT merge RootCauseAnalyzer into RCA
    - Does NOT store RCA as FailureLesson
    """

    def __init__(self):
        self._analyzer = None  # Lazy import to avoid circular deps
        self._rca_history: List[RootCauseAnalysisRecord] = []

    @property
    def analyzer(self):
        """Lazy-load RootCauseAnalyzer to avoid circular imports."""
        if self._analyzer is None:
            from ..planner.failure_intelligence import RootCauseAnalyzer
            self._analyzer = RootCauseAnalyzer()
        return self._analyzer

    def analyze_to_rca(self, failure: Any,
                       symptoms: Optional[List[str]] = None,
                       root_causes: Optional[List[str]] = None,
                       contributing_factors: Optional[List[str]] = None,
                       impact: str = "",
                       corrective_actions: Optional[List[str]] = None,
                       preventive_actions: Optional[List[str]] = None,
                       evidence_refs: Optional[List[str]] = None,
                       related_decisions: Optional[List[str]] = None,
                       incident_id: str = "",
                       rca_id: str = "") -> RootCauseAnalysisRecord:
        """
        Analyze a failure and create an RCA record.

        Uses RootCauseAnalyzer internally for classification, then produces
        a canonical RCA Engineering Record.

        Args:
            failure: The failure to analyze (StructuredFailure or raw failure)
            symptoms: Observed symptoms
            root_causes: Identified root causes (at least one required)
            contributing_factors: Contributing factors
            impact: Impact description
            corrective_actions: Fixes applied (at least one required)
            preventive_actions: Prevention steps (at least one required)
            evidence_refs: External evidence IDs
            related_decisions: Related ADR/DR IDs
            incident_id: Associated incident ID
            rca_id: RCA record ID (auto-generated if empty)

        Returns:
            RootCauseAnalysisRecord (RCA) Engineering Knowledge
        """
        # Get structured failure analysis
        structured = self.analyzer.analyze(failure)

        # Build failure reference (stable identity)
        failure_ref = self._get_failure_id(failure, structured)

        # Derive root causes if not provided
        if root_causes is None:
            root_causes = self._derive_root_causes(structured)

        # Derive corrective actions if not provided
        if corrective_actions is None:
            corrective_actions = self._derive_corrective_actions(structured)

        # Derive preventive actions if not provided
        if preventive_actions is None:
            preventive_actions = self._derive_preventive_actions(structured)

        # Generate RCA ID if not provided
        if not rca_id:
            rca_id = self._generate_rca_id(failure_ref)

        # Determine failure type for categorization
        # Use original failure's category if available (analyzer may re-classify)
        if hasattr(failure, 'category') and failure.category:
            failure_type = failure.category
        else:
            failure_type = structured.category if hasattr(structured, 'category') else "unknown"

        rca = RootCauseAnalysisRecord(
            record_id=rca_id,
            failure_ref=failure_ref,
            failure_type=failure_type,
            symptoms=symptoms or [structured.error_message] if hasattr(structured, 'error_message') else [],
            root_causes=root_causes,
            contributing_factors=contributing_factors or [],
            impact=impact,
            resolution="",
            corrective_actions=corrective_actions,
            preventive_actions=preventive_actions,
            evidence_refs=evidence_refs or [],
            related_decisions=related_decisions or [],
            incident_id=incident_id,
            title=f"RCA: {structured.error_message[:80] if hasattr(structured, 'error_message') else 'Failure'}",
            description=f"Root cause analysis for failure {failure_ref}",
            summary=f"Analysis of {failure_type} failure: {structured.error_message[:200] if hasattr(structured, 'error_message') else 'N/A'}",
            status=RCA_STATE_DRAFT,
            authority=AUTHORITY_PROPOSED,  # Agent-generated RCA always starts as PROPOSED
            provenance=Provenance(
                author="RCAAnalyzer",
                source="generated",
                evidence_refs=evidence_refs or [],
            ),
            tags=["rca", failure_type],
        )

        self._rca_history.append(rca)
        return rca

    def _get_failure_id(self, failure: Any, structured: Any) -> str:
        """Extract stable failure identity (survives process restart)."""
        if hasattr(structured, 'failure_id') and structured.failure_id:
            return structured.failure_id
        if hasattr(failure, 'failure_id') and failure.failure_id:
            return failure.failure_id
        # Deterministic fallback
        node_id = getattr(structured, 'node_id', 'unknown')
        error = getattr(structured, 'error_message', '')[:60]
        payload = f"{node_id}::{error}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return f"failure-{digest}"

    def _derive_root_causes(self, structured: Any) -> List[str]:
        """Derive root causes from structured failure analysis."""
        causes = []
        category = getattr(structured, 'category', 'unknown')
        sub_category = getattr(structured, 'sub_category', 'unknown')
        error = getattr(structured, 'error_message', 'unknown')

        if category == "integration":
            causes.append(f"Integration failure: {sub_category} — {error[:100]}")
        elif category == "configuration":
            causes.append(f"Configuration error: {sub_category} — {error[:100]}")
        elif category == "environment":
            causes.append(f"Environment issue: {sub_category} — {error[:100]}")
        elif category == "implementation":
            causes.append(f"Implementation defect: {sub_category} — {error[:100]}")
        elif category == "security":
            causes.append(f"Security failure: {sub_category} — {error[:100]}")
        elif category == "test":
            causes.append(f"Test failure: {sub_category} — {error[:100]}")
        elif category == "architecture":
            causes.append(f"Architectural issue: {sub_category} — {error[:100]}")
        else:
            causes.append(f"Root cause: {sub_category} — {error[:100]}")

        return causes

    def _derive_corrective_actions(self, structured: Any) -> List[str]:
        """Derive corrective actions from structured failure analysis."""
        category = getattr(structured, 'category', 'unknown')
        sub_category = getattr(structured, 'sub_category', 'unknown')

        return [
            f"Fix {category} root cause ({sub_category}) in affected component",
            f"Verify fix resolves {sub_category} before proceeding",
        ]

    def _derive_preventive_actions(self, structured: Any) -> List[str]:
        """Derive preventive actions from structured failure analysis."""
        category = getattr(structured, 'category', 'unknown')

        return [
            f"Add {category} validation guard to detect recurrence earlier",
            f"Document {category} failure pattern for future reference",
        ]

    def _generate_rca_id(self, failure_ref: str) -> str:
        """Generate deterministic RCA ID from failure reference."""
        # RCA-001, RCA-002, ... based on existing count
        existing = len(self._rca_history) + 1
        return f"RCA-{existing:03d}"

    def get_rca_history(self) -> List[RootCauseAnalysisRecord]:
        """Get all RCA records created by this analyzer."""
        return list(self._rca_history)

    def get_patterns(self) -> Dict[str, int]:
        """Get recurring RCA root cause patterns."""
        patterns: Dict[str, int] = {}
        for rca in self._rca_history:
            for cause in rca.root_causes:
                patterns[cause] = patterns.get(cause, 0) + 1
        return dict(sorted(patterns.items(), key=lambda x: -x[1]))


# ──────────────────────────────────────────────────────────────────────
# FailureKnowledgeBridge — connects RCA to FailureToLearningPipeline
# ──────────────────────────────────────────────────────────────────────


class FailureKnowledgeBridge:
    """
    Bridges RCA (Engineering Knowledge) with FailureToLearningPipeline (Learning).

    Provides:
    - Failure → RCA → KnowledgeStore → EngineeringKnowledgeGraph (Engineering path)
    - Failure → FailureToLearningPipeline → FailureLesson / StructuredExperience (Learning path)
    - RCA → FailureLesson derivation (with stable provenance, authority separation)

    The two paths may share source failure/evidence.
    They do NOT share canonical ownership.

    FailureLesson should contain/reference:
    - source_rca_id (if derived from RCA)
    - source_failure_id

    Strategy provenance:
    Strategy → DERIVED_FROM → FailureLesson → DERIVED_FROM → RCA → DERIVED_FROM → Failure
    """

    def __init__(self, rca_analyzer: Optional[RCAAnalyzer] = None,
                 experience_pipeline: Any = None,
                 experiment_context: Any = None):
        self.rca_analyzer = rca_analyzer or RCAAnalyzer()
        self.experience_pipeline = experience_pipeline
        self.experiment_context = experiment_context

    def process_failure(self, failure: Any,
                       rca_symptoms: Optional[List[str]] = None,
                       rca_root_causes: Optional[List[str]] = None,
                       rca_contributing_factors: Optional[List[str]] = None,
                       rca_impact: str = "",
                       rca_corrective: Optional[List[str]] = None,
                       rca_preventive: Optional[List[str]] = None,
                       rca_evidence_refs: Optional[List[str]] = None,
                       rca_related_decisions: Optional[List[str]] = None,
                       rca_incident_id: str = "",
                       rca_id: str = "") -> Dict[str, Any]:
        """
        Process a failure through both paths.

        Returns dict with:
        - rca: RCA record (Engineering Knowledge)
        - lesson: FailureLesson proposal (Learning)
        - structured_failure: StructuredFailure (from RootCauseAnalyzer)
        - derivation_provenance: source_rca_id, source_failure_id
        """
        # 1. Create RCA (Engineering Knowledge path)
        rca = self.rca_analyzer.analyze_to_rca(
            failure=failure,
            symptoms=rca_symptoms,
            root_causes=rca_root_causes,
            contributing_factors=rca_contributing_factors,
            impact=rca_impact,
            corrective_actions=rca_corrective,
            preventive_actions=rca_preventive,
            evidence_refs=rca_evidence_refs,
            related_decisions=rca_related_decisions,
            incident_id=rca_incident_id,
            rca_id=rca_id,
        )

        # 2. Derive FailureLesson (Learning path)
        lesson_proposal = rca.derive_failure_lesson()

        # 3. Get structured failure for learning pipeline
        structured = self.rca_analyzer.analyzer.analyze(failure)

        return {
            "rca": rca,
            "lesson_proposal": lesson_proposal,
            "structured_failure": structured,
            "derivation_provenance": {
                "source_rca_id": rca.record_id,
                "source_failure_id": rca.failure_ref,
                "derivation_path": "Failure → RCA → FailureLesson",
            },
        }


# ──────────────────────────────────────────────────────────────────────
# RCA Traceability Metrics
# ──────────────────────────────────────────────────────────────────────


def compute_rca_traceability_metrics(failures: List[Any],
                                     rcas: List[RootCauseAnalysisRecord],
                                     experiences: List[Any]) -> Dict[str, Any]:
    """
    Compute deterministic RCA traceability metrics.

    Metrics:
    - failures_with_rca / eligible_failures
    - rcas_with_root_cause / total_rcas
    - rcas_with_evidence / total_rcas
    - rcas_with_corrective_action / total_rcas
    - rcas_with_failure_lesson / eligible_rcas
    """
    failure_ids = set()
    for f in failures:
        fid = getattr(f, 'failure_id', None) or getattr(f, 'node_id', str(f))
        failure_ids.add(fid)

    rca_failure_refs = set()
    rcas_with_root_cause = 0
    rcas_with_evidence = 0
    rcas_with_corrective_action = 0
    rcas_with_failure_lesson = 0

    for rca in rcas:
        if rca.failure_ref:
            rca_failure_refs.add(rca.failure_ref)
        if rca.root_causes:
            rcas_with_root_cause += 1
        if rca.evidence_refs:
            rcas_with_evidence += 1
        if rca.corrective_actions:
            rcas_with_corrective_action += 1
        # Check if RCA has associated lesson (by source_rca_id)
        for exp in experiences:
            if getattr(exp, 'provenance', None) and isinstance(exp.provenance, dict):
                if exp.provenance.get('source_rca_id') == rca.record_id:
                    rcas_with_failure_lesson += 1
                    break

    total_rcas = len(rcas)
    eligible_failures = len(failure_ids)
    failures_with_rca = len(failure_ids & rca_failure_refs)

    return {
        "failures_with_rca": failures_with_rca,
        "eligible_failures": eligible_failures,
        "failures_with_rca_ratio": round(failures_with_rca / eligible_failures, 4) if eligible_failures else 0.0,
        "rcas_with_root_cause": rcas_with_root_cause,
        "total_rcas": total_rcas,
        "rcas_with_root_cause_ratio": round(rcas_with_root_cause / total_rcas, 4) if total_rcas else 0.0,
        "rcas_with_evidence": rcas_with_evidence,
        "rcas_with_evidence_ratio": round(rcas_with_evidence / total_rcas, 4) if total_rcas else 0.0,
        "rcas_with_corrective_action": rcas_with_corrective_action,
        "rcas_with_corrective_action_ratio": round(rcas_with_corrective_action / total_rcas, 4) if total_rcas else 0.0,
        "rcas_with_failure_lesson": rcas_with_failure_lesson,
        "eligible_rcas": total_rcas,
        "rcas_with_failure_lesson_ratio": round(rcas_with_failure_lesson / total_rcas, 4) if total_rcas else 0.0,
    }
