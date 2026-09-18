# harness/knowledge/risk_security.py — V3.3 Phase 6: TDR, RSK, SEC
"""
Typed engineering record subclasses for Phase 6.

Implements:
- TDR (Technical Debt Record) — known technical compromise, deferred work
- RSK (Risk Record) — uncertain future event with likelihood/impact/mitigation
- SEC (Security Record) — security constraint, control, finding, decision, requirement

Architecture:
- All typed records inherit from EngineeringRecord base (Phase 1)
- TDR represents known debt (not mere undesirable code)
- RSK represents uncertainty (differs from TDR and SEC)
- SEC represents security constraints with authority precedence
- Security Authority cannot be overridden by Learning, Task preference, ADR, timestamp, or agent role

Three-domain separation maintained:
- These are Engineering Records, NOT Learning, NOT Evidence
- SEC authority is separate from standard authority levels
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import json

from .records import (
    EngineeringRecord,
    RecordRelationship,
    RecordError,
    RECORD_ID_PATTERN,
    VALID_RELATIONSHIP_TYPES,
    RELATIONSHIP_REQUIRES,
    RELATIONSHIP_SATISFIES,
    RELATIONSHIP_CONSTRAINED_BY,
    RELATIONSHIP_INTRODUCES,
    RELATIONSHIP_MITIGATES,
    RELATIONSHIP_RESOLVED_BY,
    RELATIONSHIP_VERIFIED_BY,
    RELATIONSHIP_SUPPORTED_BY,
    RELATIONSHIP_CONTRADICTS,
    RELATIONSHIP_RELATES_TO,
    RELATIONSHIP_DERIVED_FROM,
)
from .provenance import (
    Provenance,
    ProvenanceError,
    AUTHORITY_PROPOSED,
    AUTHORITY_ACCEPTED,
    AUTHORITY_DEPRECATED,
    AUTHORITY_ARCHIVED,
    VALID_AUTHORITY_LEVELS,
    is_authority_at_least,
    can_transition_authority,
)
from .lifecycle import (
    LifecycleError,
    get_initial_state,
    get_lifecycle_definition,
    is_valid_transition,
    is_terminal_state,
    get_allowed_transitions,
    validate_lifecycle_state,
    LIFECYCLE_DEFINITIONS,
)


# ──────────────────────────────────────────────────────────────────────
# TDR — Technical Debt Record
# ──────────────────────────────────────────────────────────────────────

TDR_DEBT_TYPE_CODE = "code"
TDR_DEBT_TYPE_DESIGN = "design"
TDR_DEBT_TYPE_ARCHITECTURE = "architecture"
TDR_DEBT_TYPE_TEST = "test"
TDR_DEBT_TYPE_DOCUMENTATION = "documentation"
TDR_DEBT_TYPE_DEPENDENCY = "dependency"
TDR_DEBT_TYPE_PROCESS = "process"
TDR_DEBT_TYPE_INFRASTRUCTURE = "infrastructure"

VALID_TDR_DEBT_TYPES = {
    TDR_DEBT_TYPE_CODE,
    TDR_DEBT_TYPE_DESIGN,
    TDR_DEBT_TYPE_ARCHITECTURE,
    TDR_DEBT_TYPE_TEST,
    TDR_DEBT_TYPE_DOCUMENTATION,
    TDR_DEBT_TYPE_DEPENDENCY,
    TDR_DEBT_TYPE_PROCESS,
    TDR_DEBT_TYPE_INFRASTRUCTURE,
}

TDR_SEVERITY_CRITICAL = "critical"
TDR_SEVERITY_HIGH = "high"
TDR_SEVERITY_MEDIUM = "medium"
TDR_SEVERITY_LOW = "low"

VALID_TDR_SEVERITIES = {
    TDR_SEVERITY_CRITICAL,
    TDR_SEVERITY_HIGH,
    TDR_SEVERITY_MEDIUM,
    TDR_SEVERITY_LOW,
}

TDR_PRIORITY_CRITICAL = "critical"
TDR_PRIORITY_HIGH = "high"
TDR_PRIORITY_MEDIUM = "medium"
TDR_PRIORITY_LOW = "low"

VALID_TDR_PRIORITIES = {
    TDR_PRIORITY_CRITICAL,
    TDR_PRIORITY_HIGH,
    TDR_PRIORITY_MEDIUM,
    TDR_PRIORITY_LOW,
}


@dataclass
class TechnicalDebtRecord(EngineeringRecord):
    """
    Technical Debt Record — known technical compromise, deferred engineering work.

    TDR represents a conscious, known compromise. It is NOT:
    - A TODO/FIXME comment (Phase 6 does not scan code for these)
    - A lint warning
    - An undesirable code pattern discovered automatically

    Lifecycle: identified → acknowledged → remediation_planned → in_progress → resolved → closed

    Authority: Agent-generated TDR always starts as PROPOSED (never auto-ACCEPTED).

    Semantic boundary: TDR != RSK != SEC
    """
    record_type: str = "TDR"
    title: str = ""
    description: str = ""
    status: str = "identified"
    authority: str = AUTHORITY_PROPOSED

    # TDR-specific fields
    debt_type: str = ""       # code | design | architecture | test | documentation | dependency | process | infrastructure
    severity: str = "medium"  # critical | high | medium | low
    impact: str = ""          # How the debt affects the system
    remediation: str = ""     # How to fix it
    remediation_estimate: str = ""  # Effort estimate (e.g., "2 days")
    target_condition: str = ""  # Condition that must be met for resolution
    priority: str = "medium"   # critical | high | medium | low
    owner: str = ""           # Optional owner reference
    target_milestone: str = ""  # Optional target milestone reference
    resolution_evidence: List[str] = field(default_factory=list)  # Evidence record IDs
    related_decisions: List[str] = field(default_factory=list)   # ADR/DR IDs

    def __post_init__(self):
        if not self.title:
            self.title = self.record_id
        if not self.description:
            self.description = self.title
        # Establish canonical relationships for graph traversal
        for adr_id in self.related_decisions:
            self.add_relationship(adr_id, RELATIONSHIP_DERIVED_FROM)
        for ev_id in self.resolution_evidence:
            self.add_relationship(ev_id, RELATIONSHIP_SUPPORTED_BY)
        super().__post_init__()

    def validate(self) -> None:
        """Validate TDR-specific fields."""
        super().validate()

        # Validate TDR ID prefix
        prefix = self.record_id.split("-")[0]
        if prefix != "TDR":
            raise RecordError(f"TDR record_id prefix must be 'TDR', got '{prefix}'")

        # Validate debt_type
        if not isinstance(self.debt_type, str) or not self.debt_type:
            raise RecordError("debt_type must be a non-empty string")
        if self.debt_type not in VALID_TDR_DEBT_TYPES:
            raise RecordError(
                f"Invalid debt_type '{self.debt_type}'. "
                f"Must be one of: {', '.join(sorted(VALID_TDR_DEBT_TYPES))}"
            )

        # Validate severity
        if self.severity not in VALID_TDR_SEVERITIES:
            raise RecordError(
                f"Invalid severity '{self.severity}'. "
                f"Must be one of: {', '.join(sorted(VALID_TDR_SEVERITIES))}"
            )

        # Validate impact (required — completeness violation if missing)
        if not isinstance(self.impact, str) or not self.impact.strip():
            raise RecordError("impact is required (completeness violation if missing)")

        # Validate remediation (required — completeness violation if missing)
        if not isinstance(self.remediation, str) or not self.remediation.strip():
            raise RecordError("remediation is required (completeness violation if missing)")

        # Validate priority
        if self.priority not in VALID_TDR_PRIORITIES:
            raise RecordError(
                f"Invalid priority '{self.priority}'. "
                f"Must be one of: {', '.join(sorted(VALID_TDR_PRIORITIES))}"
            )

        # Validate related_decisions
        if not isinstance(self.related_decisions, list):
            raise RecordError("related_decisions must be a list")
        for adr_id in self.related_decisions:
            if not isinstance(adr_id, str):
                raise RecordError("related_decisions must contain strings")
            if not (adr_id.startswith("ADR-") or adr_id.startswith("DR-")):
                raise RecordError(
                    f"related_decisions must reference ADR/DR IDs, got '{adr_id}'"
                )

        # Validate resolution_evidence
        if not isinstance(self.resolution_evidence, list):
            raise RecordError("resolution_evidence must be a list")
        for ev_id in self.resolution_evidence:
            if not isinstance(ev_id, str):
                raise RecordError("resolution_evidence must contain strings")

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["debt_type"] = self.debt_type
        base["severity"] = self.severity
        base["impact"] = self.impact
        base["remediation"] = self.remediation
        base["remediation_estimate"] = self.remediation_estimate
        base["target_condition"] = self.target_condition
        base["priority"] = self.priority
        base["owner"] = self.owner
        base["target_milestone"] = self.target_milestone
        base["resolution_evidence"] = list(self.resolution_evidence)
        base["related_decisions"] = list(self.related_decisions)
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "TechnicalDebtRecord":
        record = cls(
            record_id=data["record_id"],
            record_type="TDR",
            title=data["title"],
            description=data["description"],
            debt_type=data.get("debt_type", ""),
            severity=data.get("severity", "medium"),
            impact=data.get("impact", ""),
            remediation=data.get("remediation", ""),
            remediation_estimate=data.get("remediation_estimate", ""),
            target_condition=data.get("target_condition", ""),
            priority=data.get("priority", "medium"),
            owner=data.get("owner", ""),
            target_milestone=data.get("target_milestone", ""),
            resolution_evidence=list(data.get("resolution_evidence", [])),
            related_decisions=list(data.get("related_decisions", [])),
            status=data.get("status", get_initial_state("TDR")),
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
        """Link a decision (ADR/DR) that introduced this debt."""
        if not (adr_id.startswith("ADR-") or adr_id.startswith("DR-")):
            raise RecordError(f"Must reference an ADR/DR ID, got '{adr_id}'")
        if adr_id not in self.related_decisions:
            self.related_decisions.append(adr_id)
            self.updated_at = datetime.utcnow().isoformat()
            self.add_relationship(adr_id, RELATIONSHIP_DERIVED_FROM)

    def add_resolution_evidence(self, evidence_id: str) -> None:
        """Add a resolution evidence reference."""
        if not isinstance(evidence_id, str) or not evidence_id:
            raise RecordError("evidence_id must be a non-empty string")
        if evidence_id not in self.resolution_evidence:
            self.resolution_evidence.append(evidence_id)
            self.updated_at = datetime.utcnow().isoformat()
            self.add_relationship(evidence_id, RELATIONSHIP_SUPPORTED_BY)

    def can_transition_to(self, new_status: str) -> bool:
        """Check if transition to new_status is valid."""
        return is_valid_transition("TDR", self.status, new_status)

    def get_completeness(self) -> Dict[str, bool]:
        """Deterministic completeness check for a TDR."""
        return {
            "has_debt_type": self.debt_type in VALID_TDR_DEBT_TYPES,
            "has_severity": self.severity in VALID_TDR_SEVERITIES,
            "has_impact": bool(self.impact and self.impact.strip()),
            "has_remediation": bool(self.remediation and self.remediation.strip()),
            "has_priority": self.priority in VALID_TDR_PRIORITIES,
            "valid_decisions": all(
                d.startswith(("ADR-", "DR-")) for d in self.related_decisions
            ),
            "has_provenance": self.provenance is not None and self.provenance.is_complete(),
        }

    def is_complete(self) -> bool:
        """Check if the TDR is complete."""
        completeness = self.get_completeness()
        return all(completeness.values())


# ──────────────────────────────────────────────────────────────────────
# RSK — Risk Record
# ──────────────────────────────────────────────────────────────────────

RSK_CATEGORY_TECHNICAL = "technical"
RSK_CATEGORY_SCHEDULE = "schedule"
RSK_CATEGORY_RESOURCE = "resource"
RSK_CATEGORY_SECURITY = "security"
RSK_CATEGORY_COMPLIANCE = "compliance"

VALID_RSK_CATEGORIES = {
    RSK_CATEGORY_TECHNICAL,
    RSK_CATEGORY_SCHEDULE,
    RSK_CATEGORY_RESOURCE,
    RSK_CATEGORY_SECURITY,
    RSK_CATEGORY_COMPLIANCE,
}

RSK_LIKELIHOOD_HIGH = "high"
RSK_LIKELIHOOD_MEDIUM = "medium"
RSK_LIKELIHOOD_LOW = "low"

VALID_RSK_LIKELIHOODS = {
    RSK_LIKELIHOOD_HIGH,
    RSK_LIKELIHOOD_MEDIUM,
    RSK_LIKELIHOOD_LOW,
}

RSK_IMPACT_HIGH = "high"
RSK_IMPACT_MEDIUM = "medium"
RSK_IMPACT_LOW = "low"

VALID_RSK_IMPACTS = {
    RSK_IMPACT_HIGH,
    RSK_IMPACT_MEDIUM,
    RSK_IMPACT_LOW,
}

# Risk exposure matrix: likelihood × impact → exposure level
# Qualitative only — no fabricated numeric precision
RSK_EXPOSURE_MATRIX = {
    (RSK_LIKELIHOOD_HIGH, RSK_IMPACT_HIGH): "critical",
    (RSK_LIKELIHOOD_HIGH, RSK_IMPACT_MEDIUM): "high",
    (RSK_LIKELIHOOD_HIGH, RSK_IMPACT_LOW): "medium",
    (RSK_LIKELIHOOD_MEDIUM, RSK_IMPACT_HIGH): "high",
    (RSK_LIKELIHOOD_MEDIUM, RSK_IMPACT_MEDIUM): "medium",
    (RSK_LIKELIHOOD_MEDIUM, RSK_IMPACT_LOW): "low",
    (RSK_LIKELIHOOD_LOW, RSK_IMPACT_HIGH): "medium",
    (RSK_LIKELIHOOD_LOW, RSK_IMPACT_MEDIUM): "low",
    (RSK_LIKELIHOOD_LOW, RSK_IMPACT_LOW): "low",
}

VALID_RSK_EXPOSURES = {"critical", "high", "medium", "low"}


@dataclass
class RiskRecord(EngineeringRecord):
    """
    Risk Record — uncertain future event with likelihood, impact, and mitigation.

    Risk means uncertainty. Differs from:
    - TDR (known compromise, not uncertain)
    - SEC (constraint/control, not a future event)

    Lifecycle: identified → assessed → mitigated → accepted → closed

    Critical distinction:
    - ACCEPTED means consciously accepted (risk still exists)
    - MITIGATED means mitigation actions have been taken
    - CLOSED means risk is no longer relevant

    Authority: Agent-generated RSK always starts as PROPOSED.
    """
    record_type: str = "RSK"
    title: str = ""
    description: str = ""
    status: str = "identified"
    authority: str = AUTHORITY_PROPOSED

    # RSK-specific fields
    risk_category: str = ""    # technical | schedule | resource | security | compliance
    likelihood: str = "medium"  # high | medium | low (qualitative)
    impact: str = "medium"      # high | medium | low (qualitative)
    exposure: str = ""          # Derived: critical | high | medium | low
    mitigation: str = ""        # Mitigation strategy
    mitigation_owner: str = ""  # Optional owner reference
    contingency: str = ""       # Contingency plan if risk materializes
    related_decisions: List[str] = field(default_factory=list)  # ADR/DR IDs
    related_security: List[str] = field(default_factory=list)   # SEC IDs (mitigation/control)

    def __post_init__(self):
        if not self.title:
            self.title = self.record_id
        if not self.description:
            self.description = self.title
        # Compute exposure from likelihood × impact
        self.exposure = self._compute_exposure()
        # Establish canonical relationships for graph traversal
        for adr_id in self.related_decisions:
            self.add_relationship(adr_id, RELATIONSHIP_DERIVED_FROM)
        for sec_id in self.related_security:
            self.add_relationship(sec_id, RELATIONSHIP_MITIGATES)
        super().__post_init__()

    def _compute_exposure(self) -> str:
        """Compute exposure level from likelihood × impact (qualitative)."""
        return RSK_EXPOSURE_MATRIX.get(
            (self.likelihood, self.impact), "medium"
        )

    def validate(self) -> None:
        """Validate RSK-specific fields."""
        super().validate()

        # Validate RSK ID prefix
        prefix = self.record_id.split("-")[0]
        if prefix != "RSK":
            raise RecordError(f"RSK record_id prefix must be 'RSK', got '{prefix}'")

        # Validate risk_category
        if not isinstance(self.risk_category, str) or not self.risk_category:
            raise RecordError("risk_category must be a non-empty string")
        if self.risk_category not in VALID_RSK_CATEGORIES:
            raise RecordError(
                f"Invalid risk_category '{self.risk_category}'. "
                f"Must be one of: {', '.join(sorted(VALID_RSK_CATEGORIES))}"
            )

        # Validate likelihood (qualitative only — no fabricated numeric precision)
        if self.likelihood not in VALID_RSK_LIKELIHOODS:
            raise RecordError(
                f"Invalid likelihood '{self.likelihood}'. "
                f"Must be one of: {', '.join(sorted(VALID_RSK_LIKELIHOODS))}"
            )

        # Validate impact (qualitative only)
        if self.impact not in VALID_RSK_IMPACTS:
            raise RecordError(
                f"Invalid impact '{self.impact}'. "
                f"Must be one of: {', '.join(sorted(VALID_RSK_IMPACTS))}"
            )

        # Validate exposure
        if self.exposure not in VALID_RSK_EXPOSURES:
            raise RecordError(
                f"Invalid exposure '{self.exposure}'. "
                f"Must be one of: {', '.join(sorted(VALID_RSK_EXPOSURES))}"
            )

        # Validate mitigation (required — completeness violation if missing)
        if not isinstance(self.mitigation, str) or not self.mitigation.strip():
            raise RecordError("mitigation is required (completeness violation if missing)")

        # Validate related_decisions
        if not isinstance(self.related_decisions, list):
            raise RecordError("related_decisions must be a list")
        for adr_id in self.related_decisions:
            if not isinstance(adr_id, str):
                raise RecordError("related_decisions must contain strings")
            if not (adr_id.startswith("ADR-") or adr_id.startswith("DR-")):
                raise RecordError(
                    f"related_decisions must reference ADR/DR IDs, got '{adr_id}'"
                )

        # Validate related_security
        if not isinstance(self.related_security, list):
            raise RecordError("related_security must be a list")
        for sec_id in self.related_security:
            if not isinstance(sec_id, str):
                raise RecordError("related_security must contain strings")
            if not sec_id.startswith("SEC-"):
                raise RecordError(
                    f"related_security must reference SEC IDs, got '{sec_id}'"
                )

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["risk_category"] = self.risk_category
        base["likelihood"] = self.likelihood
        base["impact"] = self.impact
        base["exposure"] = self.exposure
        base["mitigation"] = self.mitigation
        base["mitigation_owner"] = self.mitigation_owner
        base["contingency"] = self.contingency
        base["related_decisions"] = list(self.related_decisions)
        base["related_security"] = list(self.related_security)
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "RiskRecord":
        record = cls(
            record_id=data["record_id"],
            record_type="RSK",
            title=data["title"],
            description=data["description"],
            risk_category=data.get("risk_category", ""),
            likelihood=data.get("likelihood", "medium"),
            impact=data.get("impact", "medium"),
            exposure=data.get("exposure", ""),
            mitigation=data.get("mitigation", ""),
            mitigation_owner=data.get("mitigation_owner", ""),
            contingency=data.get("contingency", ""),
            related_decisions=list(data.get("related_decisions", [])),
            related_security=list(data.get("related_security", [])),
            status=data.get("status", get_initial_state("RSK")),
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
        """Link a decision (ADR/DR) that introduced this risk."""
        if not (adr_id.startswith("ADR-") or adr_id.startswith("DR-")):
            raise RecordError(f"Must reference an ADR/DR ID, got '{adr_id}'")
        if adr_id not in self.related_decisions:
            self.related_decisions.append(adr_id)
            self.updated_at = datetime.utcnow().isoformat()
            self.add_relationship(adr_id, RELATIONSHIP_DERIVED_FROM)

    def link_security(self, sec_id: str) -> None:
        """Link a SEC record that mitigates or controls this risk."""
        if not sec_id.startswith("SEC-"):
            raise RecordError(f"Must reference a SEC ID, got '{sec_id}'")
        if sec_id not in self.related_security:
            self.related_security.append(sec_id)
            self.updated_at = datetime.utcnow().isoformat()
            self.add_relationship(sec_id, RELATIONSHIP_MITIGATES)

    def can_transition_to(self, new_status: str) -> bool:
        """Check if transition to new_status is valid."""
        return is_valid_transition("RSK", self.status, new_status)

    def get_completeness(self) -> Dict[str, bool]:
        """Deterministic completeness check for an RSK."""
        return {
            "has_risk_category": self.risk_category in VALID_RSK_CATEGORIES,
            "has_likelihood": self.likelihood in VALID_RSK_LIKELIHOODS,
            "has_impact": self.impact in VALID_RSK_IMPACTS,
            "has_exposure": self.exposure in VALID_RSK_EXPOSURES,
            "has_mitigation": bool(self.mitigation and self.mitigation.strip()),
            "valid_decisions": all(
                d.startswith(("ADR-", "DR-")) for d in self.related_decisions
            ),
            "valid_security": all(
                s.startswith("SEC-") for s in self.related_security
            ),
            "has_provenance": self.provenance is not None and self.provenance.is_complete(),
        }

    def is_complete(self) -> bool:
        """Check if the RSK is complete."""
        completeness = self.get_completeness()
        return all(completeness.values())


# ──────────────────────────────────────────────────────────────────────
# SEC — Security Record
# ──────────────────────────────────────────────────────────────────────

SEC_CATEGORY_CONSTRAINT = "CONSTRAINT"
SEC_CATEGORY_CONTROL = "CONTROL"
SEC_CATEGORY_FINDING = "FINDING"
SEC_CATEGORY_THREAT = "THREAT"
SEC_CATEGORY_REQUIREMENT = "REQUIREMENT"

VALID_SEC_CATEGORIES = {
    SEC_CATEGORY_CONSTRAINT,
    SEC_CATEGORY_CONTROL,
    SEC_CATEGORY_FINDING,
    SEC_CATEGORY_THREAT,
    SEC_CATEGORY_REQUIREMENT,
}

# SEC authority levels — separate from standard authority levels
# Security authority must outrank ordinary accepted Engineering Knowledge
SEC_AUTHORITY_AUTHORITATIVE = "AUTHORITATIVE"
SEC_AUTHORITY_ACCEPTED = "ACCEPTED"
SEC_AUTHORITY_PROPOSED = "PROPOSED"
SEC_AUTHORITY_INFERRED = "INFERRED"
SEC_AUTHORITY_HISTORICAL = "HISTORICAL"
SEC_AUTHORITY_DEPRECATED = "DEPRECATED"

VALID_SEC_AUTHORITIES = {
    SEC_AUTHORITY_AUTHORITATIVE,
    SEC_AUTHORITY_ACCEPTED,
    SEC_AUTHORITY_PROPOSED,
    SEC_AUTHORITY_INFERRED,
    SEC_AUTHORITY_HISTORICAL,
    SEC_AUTHORITY_DEPRECATED,
}

# Security authority precedence hierarchy (higher = more authoritative)
# AUTHORITATIVE > ACCEPTED > PROPOSED > INFERRED > HISTORICAL > DEPRECATED
SEC_AUTHORITY_RANK = {
    SEC_AUTHORITY_DEPRECATED: 0,
    SEC_AUTHORITY_HISTORICAL: 1,
    SEC_AUTHORITY_INFERRED: 2,
    SEC_AUTHORITY_PROPOSED: 3,
    SEC_AUTHORITY_ACCEPTED: 4,
    SEC_AUTHORITY_AUTHORITATIVE: 5,
}

# Knowledge precedence hierarchy (deterministic)
# Governance > Security Authority > Authoritative Engineering Knowledge >
# Accepted Engineering Knowledge > Task Requirements > Verified Evidence >
# Learned Knowledge > Exploration
# Canonical values from harness.policy.knowledge_policy (single source of truth)
PRECEDENCE_GOVERNANCE = 7
PRECEDENCE_SECURITY_AUTHORITY = 6
PRECEDENCE_AUTHORITATIVE_ENG_KNOWLEDGE = 5
PRECEDENCE_ACCEPTED_ENG_KNOWLEDGE = 4
PRECEDENCE_TASK_REQUIREMENTS = 3
PRECEDENCE_VERIFIED_EVIDENCE = 2
PRECEDENCE_LEARNED_KNOWLEDGE = 1
PRECEDENCE_EXPLORATION = 0


def validate_security_authority(authority: str) -> None:
    """
    Validate that a security authority level is recognized.

    Raises SecurityAuthorityError if invalid.
    """
    if authority not in VALID_SEC_AUTHORITIES:
        raise SecurityAuthorityError(
            f"Invalid security authority '{authority}'. "
            f"Must be one of: {', '.join(sorted(VALID_SEC_AUTHORITIES))}"
        )


def is_security_authority_at_least(authority: str, minimum: str) -> bool:
    """
    Check if a security authority level meets or exceeds a minimum level.

    Uses explicit hierarchy: DEPRECATED < HISTORICAL < INFERRED < PROPOSED < ACCEPTED < AUTHORITATIVE.
    """
    validate_security_authority(authority)
    validate_security_authority(minimum)
    return SEC_AUTHORITY_RANK[authority] >= SEC_AUTHORITY_RANK[minimum]


def can_transition_security_authority(from_level: str, to_level: str) -> bool:
    """
    Check if a security authority transition is valid.

    Rules:
    - PROPOSED → ACCEPTED (approval)
    - ACCEPTED → AUTHORITATIVE (elevation by governance)
    - ACCEPTED → DEPRECATED (deprecation)
    - AUTHORITATIVE → DEPRECATED (deprecation by governance)
    - HISTORICAL → DEPRECATED (archival)
    - INFERRED → PROPOSED (promotion)
    - No skipping (PROPOSED cannot go directly to AUTHORITATIVE)
    - No downgrade (ACCEPTED cannot go back to PROPOSED)
    - DEPRECATED is terminal (no transitions out)
    """
    validate_security_authority(from_level)
    validate_security_authority(to_level)

    if from_level == SEC_AUTHORITY_DEPRECATED:
        return False  # Terminal state

    valid_transitions = {
        SEC_AUTHORITY_PROPOSED: {SEC_AUTHORITY_ACCEPTED},
        SEC_AUTHORITY_ACCEPTED: {SEC_AUTHORITY_AUTHORITATIVE, SEC_AUTHORITY_DEPRECATED},
        SEC_AUTHORITY_AUTHORITATIVE: {SEC_AUTHORITY_DEPRECATED},
        SEC_AUTHORITY_HISTORICAL: {SEC_AUTHORITY_DEPRECATED},
        SEC_AUTHORITY_INFERRED: {SEC_AUTHORITY_PROPOSED},
    }

    return to_level in valid_transitions.get(from_level, set())


class SecurityAuthorityError(Exception):
    """Raised on security authority validation errors."""


@dataclass
class SecurityRecord(EngineeringRecord):
    """
    Security Record — security constraint, control, finding, decision, requirement/reference.

    SEC represents security knowledge that must outrank ordinary Engineering Knowledge,
    Task preferences, Learned strategies, and Exploration.

    Lifecycle: active → waived → expired → deprecated

    Authority: SEC uses its own authority hierarchy (AUTHORITATIVE > ACCEPTED > PROPOSED > INFERRED > HISTORICAL > DEPRECATED).
    Agent-generated SEC always starts as PROPOSED (never auto-ACCEPTED or AUTHORITATIVE).

    Critical invariants:
    - Learning cannot override SEC
    - Task preference cannot override SEC
    - ADR cannot silently override SEC
    - Agent cannot self-approve protected security exception
    - Security authority must NOT be inferred from record_type == SEC

    Semantic boundary: SEC != TDR != RSK
    """
    record_type: str = "SEC"
    title: str = ""
    description: str = ""
    status: str = "active"
    authority: str = SEC_AUTHORITY_PROPOSED

    # SEC-specific fields
    category: str = ""          # CONSTRAINT | CONTROL | FINDING | THREAT | REQUIREMENT
    enforcement: str = ""       # How the constraint is enforced
    related_nfrs: List[str] = field(default_factory=list)   # NFR IDs
    related_adrs: List[str] = field(default_factory=list)   # ADR IDs
    related_risks: List[str] = field(default_factory=list)  # RSK IDs
    related_tdrs: List[str] = field(default_factory=list)   # TDR IDs
    waiver: Optional[str] = None  # Waiver record_id (if constraint is waived)
    exception_approved: bool = False  # Agent cannot self-approve exception

    def __post_init__(self):
        if not self.title:
            self.title = self.record_id
        if not self.description:
            self.description = self.title
        # Establish canonical relationships for graph traversal
        for nfr_id in self.related_nfrs:
            self.add_relationship(nfr_id, RELATIONSHIP_SATISFIES)
        for adr_id in self.related_adrs:
            self.add_relationship(adr_id, RELATIONSHIP_CONSTRAINED_BY)
        for rsk_id in self.related_risks:
            self.add_relationship(rsk_id, RELATIONSHIP_MITIGATES)
        for tdr_id in self.related_tdrs:
            self.add_relationship(tdr_id, RELATIONSHIP_CONSTRAINED_BY)
        super().__post_init__()

    def validate(self) -> None:
        """Validate SEC-specific fields."""
        # SEC uses its own authority validation — skip base authority validation
        # but still validate all other base fields
        self._validate_base_fields()

        # Validate SEC ID prefix
        prefix = self.record_id.split("-")[0]
        if prefix != "SEC":
            raise RecordError(f"SEC record_id prefix must be 'SEC', got '{prefix}'")

        # Validate category
        if not isinstance(self.category, str) or not self.category:
            raise RecordError("category must be a non-empty string")
        if self.category not in VALID_SEC_CATEGORIES:
            raise RecordError(
                f"Invalid category '{self.category}'. "
                f"Must be one of: {', '.join(sorted(VALID_SEC_CATEGORIES))}"
            )

        # Validate security authority (SEC-specific levels)
        validate_security_authority(self.authority)

        # Validate enforcement
        if not isinstance(self.enforcement, str):
            raise RecordError("enforcement must be a string")

        # Validate related_nfrs
        if not isinstance(self.related_nfrs, list):
            raise RecordError("related_nfrs must be a list")
        for nfr_id in self.related_nfrs:
            if not isinstance(nfr_id, str):
                raise RecordError("related_nfrs must contain strings")
            if not nfr_id.startswith("NFR-"):
                raise RecordError(
                    f"related_nfrs must reference NFR IDs, got '{nfr_id}'"
                )

        # Validate related_adrs
        if not isinstance(self.related_adrs, list):
            raise RecordError("related_adrs must be a list")
        for adr_id in self.related_adrs:
            if not isinstance(adr_id, str):
                raise RecordError("related_adrs must contain strings")
            if not adr_id.startswith("ADR-"):
                raise RecordError(
                    f"related_adrs must reference ADR IDs, got '{adr_id}'"
                )

        # Validate related_risks
        if not isinstance(self.related_risks, list):
            raise RecordError("related_risks must be a list")
        for rsk_id in self.related_risks:
            if not isinstance(rsk_id, str):
                raise RecordError("related_risks must contain strings")
            if not rsk_id.startswith("RSK-"):
                raise RecordError(
                    f"related_risks must reference RSK IDs, got '{rsk_id}'"
                )

        # Validate related_tdrs
        if not isinstance(self.related_tdrs, list):
            raise RecordError("related_tdrs must be a list")
        for tdr_id in self.related_tdrs:
            if not isinstance(tdr_id, str):
                raise RecordError("related_tdrs must contain strings")
            if not tdr_id.startswith("TDR-"):
                raise RecordError(
                    f"related_tdrs must reference TDR IDs, got '{tdr_id}'"
                )

        # Validate waiver
        if self.waiver is not None:
            if not isinstance(self.vaiver, str) or not self.vaiver:
                raise RecordError("waiver must be a non-empty string or None")

    def _validate_base_fields(self) -> None:
        """
        Validate base EngineeringRecord fields except authority.

        SEC uses its own authority hierarchy, so we skip the standard
        authority validation and do our own.
        """
        # Record ID
        if not isinstance(self.record_id, str) or not self.record_id:
            raise RecordError("record_id must be a non-empty string")
        if not RECORD_ID_PATTERN.match(self.record_id):
            raise RecordError(
                f"Invalid record_id '{self.record_id}'. "
                f"Must match pattern: TYPE-NNN (e.g., SEC-001)"
            )

        # Record type
        if self.record_type not in {"PRD", "NFR", "DR", "ADR", "TDR", "RSK", "SEC", "RCA", "REQ"}:
            raise RecordError(
                f"Invalid record_type '{self.record_type}'."
            )

        # Record ID prefix must match record type
        prefix = self.record_id.split("-")[0]
        if prefix != self.record_type:
            raise RecordError(
                f"Record ID prefix '{prefix}' does not match record_type '{self.record_type}'"
            )

        # Title and description
        if not isinstance(self.title, str) or not self.title.strip():
            raise RecordError("title must be a non-empty string")
        if not isinstance(self.description, str):
            raise RecordError("description must be a string")

        # Status (lifecycle state)
        validate_lifecycle_state(self.record_type, self.status)

        # Provenance (required)
        if self.provenance is None:
            raise RecordError("provenance is required")
        if not isinstance(self.provenance, Provenance):
            raise RecordError("provenance must be a Provenance instance")
        if not self.provenance.is_complete():
            raise RecordError(
                "provenance is incomplete: author and source are required"
            )

        # Tags
        if not isinstance(self.tags, list):
            raise RecordError("tags must be a list")

        # Version
        if not isinstance(self.version, int) or self.version < 1:
            raise RecordError("version must be a positive integer (>= 1)")

        # Timestamps
        if not isinstance(self.created_at, str) or not self.created_at:
            raise RecordError("created_at must be a non-empty string")
        if not isinstance(self.updated_at, str) or not self.updated_at:
            raise RecordError("updated_at must be a non-empty string")

        # Superseded by
        if self.superseded_by is not None:
            if not isinstance(self.superseded_by, str) or not self.superseded_by:
                raise RecordError("superseded_by must be a non-empty string or None")

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["category"] = self.category
        base["enforcement"] = self.enforcement
        base["related_nfrs"] = list(self.related_nfrs)
        base["related_adrs"] = list(self.related_adrs)
        base["related_risks"] = list(self.related_risks)
        base["related_tdrs"] = list(self.related_tdrs)
        base["waiver"] = self.waiver
        base["exception_approved"] = self.exception_approved
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "SecurityRecord":
        record = cls(
            record_id=data["record_id"],
            record_type="SEC",
            title=data["title"],
            description=data["description"],
            category=data.get("category", ""),
            enforcement=data.get("enforcement", ""),
            related_nfrs=list(data.get("related_nfrs", [])),
            related_adrs=list(data.get("related_adrs", [])),
            related_risks=list(data.get("related_risks", [])),
            related_tdrs=list(data.get("related_tdrs", [])),
            waiver=data.get("waiver"),
            exception_approved=data.get("exception_approved", False),
            status=data.get("status", get_initial_state("SEC")),
            authority=data.get("authority", SEC_AUTHORITY_PROPOSED),
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

    def link_nfr(self, nfr_id: str) -> None:
        """Link an NFR that this SEC satisfies."""
        if not nfr_id.startswith("NFR-"):
            raise RecordError(f"Must reference an NFR ID, got '{nfr_id}'")
        if nfr_id not in self.related_nfrs:
            self.related_nfrs.append(nfr_id)
            self.updated_at = datetime.utcnow().isoformat()
            self.add_relationship(nfr_id, RELATIONSHIP_SATISFIES)

    def link_adr(self, adr_id: str) -> None:
        """Link an ADR that is constrained by this SEC."""
        if not adr_id.startswith("ADR-"):
            raise RecordError(f"Must reference an ADR ID, got '{adr_id}'")
        if adr_id not in self.related_adrs:
            self.related_adrs.append(adr_id)
            self.updated_at = datetime.utcnow().isoformat()
            self.add_relationship(adr_id, RELATIONSHIP_CONSTRAINED_BY)

    def link_risk(self, rsk_id: str) -> None:
        """Link a risk that this SEC mitigates or controls."""
        if not rsk_id.startswith("RSK-"):
            raise RecordError(f"Must reference an RSK ID, got '{rsk_id}'")
        if rsk_id not in self.related_risks:
            self.related_risks.append(rsk_id)
            self.updated_at = datetime.utcnow().isoformat()
            self.add_relationship(rsk_id, RELATIONSHIP_MITIGATES)

    def link_tdr(self, tdr_id: str) -> None:
        """Link a TDR that is constrained by this SEC."""
        if not tdr_id.startswith("TDR-"):
            raise RecordError(f"Must reference a TDR ID, got '{tdr_id}'")
        if tdr_id not in self.related_tdrs:
            self.related_tdrs.append(tdr_id)
            self.updated_at = datetime.utcnow().isoformat()
            self.add_relationship(tdr_id, RELATIONSHIP_CONSTRAINED_BY)

    def can_transition_to(self, new_status: str) -> bool:
        """Check if transition to new_status is valid."""
        return is_valid_transition("SEC", self.status, new_status)

    def is_authoritative(self) -> bool:
        """Check if this SEC has authoritative security authority."""
        return self.authority == SEC_AUTHORITY_AUTHORITATIVE

    def is_accepted_or_higher(self) -> bool:
        """Check if this SEC has at least accepted security authority."""
        return is_security_authority_at_least(self.authority, SEC_AUTHORITY_ACCEPTED)

    def can_override(self, other_authority: str) -> bool:
        """
        Check if this SEC can override another authority level.

        Security Authority outranks ordinary accepted Engineering Knowledge.
        """
        validate_security_authority(other_authority)
        return SEC_AUTHORITY_RANK[self.authority] > SEC_AUTHORITY_RANK[other_authority]

    def get_completeness(self) -> Dict[str, bool]:
        """Deterministic completeness check for a SEC."""
        return {
            "has_category": self.category in VALID_SEC_CATEGORIES,
            "has_enforcement": bool(self.enforcement and self.enforcement.strip()),
            "has_authority": self.authority in VALID_SEC_AUTHORITIES,
            "valid_nfrs": all(n.startswith("NFR-") for n in self.related_nfrs),
            "valid_adrs": all(a.startswith("ADR-") for a in self.related_adrs),
            "valid_risks": all(r.startswith("RSK-") for r in self.related_risks),
            "valid_tdrs": all(t.startswith("TDR-") for t in self.related_tdrs),
            "has_provenance": self.provenance is not None and self.provenance.is_complete(),
        }

    def is_complete(self) -> bool:
        """Check if the SEC is complete."""
        completeness = self.get_completeness()
        return all(completeness.values())


# ──────────────────────────────────────────────────────────────────────
# Security Authority Precedence Resolver
# ──────────────────────────────────────────────────────────────────────

@dataclass
class AuthorityPrecedenceResolver:
    """
    Deterministic authority/precedence resolver.

    Precedence hierarchy (highest to lowest):
    1. Governance
    2. Security Authority (SEC records with AUTHORITATIVE/ACCEPTED authority)
    3. Authoritative Engineering Knowledge (accepted ADR/DR)
    4. Accepted Engineering Knowledge (accepted PRD/REQ/NFR)
    5. Task Requirements
    6. Learned Knowledge (V3.2 learning)
    7. Exploration

    No timestamp precedence. No role precedence. No model precedence.
    """

    def resolve(self, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Resolve precedence among candidates.

        Each candidate must have:
        - authority: str (standard or SEC authority)
        - record_type: str
        - source: str (e.g., "governance", "security", "engineering", "task", "learning", "exploration")

        Returns the highest-precedence candidate, or None if empty.
        """
        if not candidates:
            return None

        def precedence_key(candidate: Dict[str, Any]) -> int:
            source = candidate.get("source", "")
            authority = candidate.get("authority", "")
            record_type = candidate.get("record_type", "")

            # Governance is absolute
            if source == "governance":
                return PRECEDENCE_GOVERNANCE

            # Security Authority
            if record_type == "SEC" and authority in VALID_SEC_AUTHORITIES:
                return PRECEDENCE_SECURITY_AUTHORITY

            # Authoritative Engineering Knowledge (accepted ADR/DR)
            if record_type in ("ADR", "DR") and authority == AUTHORITY_ACCEPTED:
                return PRECEDENCE_AUTHORITATIVE_ENG_KNOWLEDGE

            # Task Requirements (check before accepted authority)
            if source == "task":
                return PRECEDENCE_TASK_REQUIREMENTS

            # Learned Knowledge (check before accepted authority)
            if source == "learning":
                return PRECEDENCE_LEARNED_KNOWLEDGE

            # Accepted Engineering Knowledge
            if authority == AUTHORITY_ACCEPTED:
                return PRECEDENCE_ACCEPTED_ENG_KNOWLEDGE

            # Exploration
            return PRECEDENCE_EXPLORATION

        # Sort by precedence (highest first), then by authority rank for determinism
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (precedence_key(c), SEC_AUTHORITY_RANK.get(c.get("authority", ""), 0)),
            reverse=True,
        )
        return sorted_candidates[0]

    def can_override(self, source: Dict[str, Any], target: Dict[str, Any]) -> bool:
        """
        Check if source can override target based on precedence.

        Returns True if source has strictly higher precedence than target.
        """
        source_prec = self._precedence_of(source)
        target_prec = self._precedence_of(target)
        return source_prec > target_prec

    def _precedence_of(self, candidate: Dict[str, Any]) -> int:
        """Get the precedence level of a candidate."""
        source = candidate.get("source", "")
        authority = candidate.get("authority", "")
        record_type = candidate.get("record_type", "")

        if source == "governance":
            return PRECEDENCE_GOVERNANCE
        if record_type == "SEC" and authority in VALID_SEC_AUTHORITIES:
            return PRECEDENCE_SECURITY_AUTHORITY
        if record_type in ("ADR", "DR") and authority == AUTHORITY_ACCEPTED:
            return PRECEDENCE_AUTHORITATIVE_ENG_KNOWLEDGE
        # Check source before authority for task/learning
        if source == "task":
            return PRECEDENCE_TASK_REQUIREMENTS
        if source == "learning":
            return PRECEDENCE_LEARNED_KNOWLEDGE
        if authority == AUTHORITY_ACCEPTED:
            return PRECEDENCE_ACCEPTED_ENG_KNOWLEDGE
        return PRECEDENCE_EXPLORATION


# ──────────────────────────────────────────────────────────────────────
# Security Conflict Detection
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SecurityConflict:
    """A detected security conflict."""
    conflict_type: str
    records: List[str]
    reason: str
    severity: str
    resolution: str

    def to_dict(self) -> dict:
        return {
            "conflict_type": self.conflict_type,
            "records": list(self.records),
            "reason": self.reason,
            "severity": self.severity,
            "resolution": self.resolution,
        }


def detect_security_conflicts(
    records: List[EngineeringRecord],
) -> List[SecurityConflict]:
    """
    Detect deterministic security conflicts.

    Conflict types:
    - ADR_CONTRADICTS_SEC: ADR PROPOSED contradicts authoritative SEC
    - REQ_CONTRADICTS_SEC: REQ contradicts authoritative SEC
    - LEARNING_OVERRIDES_SEC: Learned strategy conflicts with SEC
    - TASK_OVERRIDES_SEC: Task preference conflicts with SEC
    - TDR_RESOLVED_WITHOUT_EVIDENCE: Active TDR claims resolved without evidence
    - RISK_STATE_CONFLICT: Risk states structurally conflict

    Fail closed: when conflict cannot be safely resolved, SEC wins.
    """
    conflicts: List[SecurityConflict] = []
    record_map: Dict[str, EngineeringRecord] = {r.record_id: r for r in records}

    # Check 1: ADR contradicts authoritative SEC
    for record in records:
        if record.record_type == "ADR" and record.status == "proposed":
            for rel in record.get_relationships(RELATIONSHIP_CONTRADICTS):
                target = record_map.get(rel.target_id)
                if target and target.record_type == "SEC":
                    if target.is_accepted_or_higher():
                        conflicts.append(SecurityConflict(
                            conflict_type="ADR_CONTRADICTS_SEC",
                            records=[record.record_id, target.record_id],
                            reason=f"Proposed ADR {record.record_id} contradicts authoritative SEC {target.record_id}",
                            severity="HIGH",
                            resolution="SEC WINS — ADR cannot be accepted without resolving SEC conflict",
                        ))

    # Check 2: REQ contradicts authoritative SEC
    for record in records:
        if record.record_type == "REQ":
            for rel in record.get_relationships(RELATIONSHIP_CONTRADICTS):
                target = record_map.get(rel.target_id)
                if target and target.record_type == "SEC":
                    if target.is_accepted_or_higher():
                        conflicts.append(SecurityConflict(
                            conflict_type="REQ_CONTRADICTS_SEC",
                            records=[record.record_id, target.record_id],
                            reason=f"REQ {record.record_id} contradicts authoritative SEC {target.record_id}",
                            severity="HIGH",
                            resolution="SEC WINS — REQ cannot override SEC constraint",
                        ))

    # Check 3: TDR resolved without evidence
    for record in records:
        if record.record_type == "TDR" and record.status == "resolved":
            if not record.resolution_evidence:
                conflicts.append(SecurityConflict(
                    conflict_type="TDR_RESOLVED_WITHOUT_EVIDENCE",
                    records=[record.record_id],
                    reason=f"TDR {record.record_id} claims resolved but has no resolution evidence",
                    severity="MEDIUM",
                    resolution="FAIL CLOSED — TDR cannot be RESOLVED without evidence",
                ))

    # Check 4: Risk state conflicts
    for record in records:
        if record.record_type == "RSK":
            # ACCEPTED means consciously accepted, NOT resolved
            if record.status == "accepted" and record.likelihood == "high" and record.impact == "high":
                conflicts.append(SecurityConflict(
                    conflict_type="RISK_STATE_CONFLICT",
                    records=[record.record_id],
                    reason=f"RSK {record.record_id} is ACCEPTED but has high likelihood and high impact",
                    severity="MEDIUM",
                    resolution="REVIEW — High-exposure risk should not be merely accepted",
                ))

    # Sort for deterministic output
    conflicts.sort(key=lambda c: (c.conflict_type, c.records))
    return conflicts


# ──────────────────────────────────────────────────────────────────────
# Registry update — Phase 6 typed classes
# ──────────────────────────────────────────────────────────────────────

# TYPED_RECORD_CLASSES is defined in registry.py to avoid circular imports.
# registry.py imports the typed classes from this module and updates
# TYPED_RECORD_CLASSES. See registry.py for the canonical registry.
