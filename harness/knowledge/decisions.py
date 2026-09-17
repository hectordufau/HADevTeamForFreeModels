# harness/knowledge/decisions.py — V3.3 Phase 5: DR / ADR Decision Intelligence
"""
Typed engineering record subclasses for Phase 5.

Implements:
- DecisionRecord (DR) — context, decision, alternatives, rationale, consequences
- ArchitectureDecisionRecord (ADR) — specialization of DR with architecture-specific fields

Architecture:
- All typed records inherit from EngineeringRecord base (Phase 1)
- ADR is a specialization of DR (not a separate hierarchy)
- DR/ADR typed fields are additive only
- Agent-generated ADR/DR always starts as PROPOSED (never auto-ACCEPTED)
- Version vs Supersession: version = same logical decision evolved;
  supersession = new decision replaces previous (new record + SUPERSEDES edge)

Three-domain separation maintained:
- These are Engineering Records, NOT Learning (NOT StructuredExperience, etc.)
- NOT Evidence (no EvidencePackage duplication)
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import re

from .records import (
    EngineeringRecord,
    RecordRelationship,
    RecordError,
    RECORD_ID_PATTERN,
    VALID_RELATIONSHIP_TYPES,
    RELATIONSHIP_REQUIRES,
    RELATIONSHIP_SATISFIES,
    RELATIONSHIP_DECIDED_BY,
    RELATIONSHIP_CONSTRAINED_BY,
    RELATIONSHIP_INTRODUCES,
    RELATIONSHIP_MITIGATES,
    RELATIONSHIP_SUPERSEDES,
    RELATIONSHIP_CAUSED_BY,
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
# Decision types
# ──────────────────────────────────────────────────────────────────────

DECISION_TYPE_ARCHITECTURE = "ARCHITECTURE"
DECISION_TYPE_TECHNICAL = "TECHNICAL"
DECISION_TYPE_SECURITY = "SECURITY"
DECISION_TYPE_PRODUCT = "PRODUCT"
DECISION_TYPE_OPERATIONAL = "OPERATIONAL"
DECISION_TYPE_DATA = "DATA"
DECISION_TYPE_INTEGRATION = "INTEGRATION"

VALID_DECISION_TYPES = {
    DECISION_TYPE_ARCHITECTURE,
    DECISION_TYPE_TECHNICAL,
    DECISION_TYPE_SECURITY,
    DECISION_TYPE_PRODUCT,
    DECISION_TYPE_OPERATIONAL,
    DECISION_TYPE_DATA,
    DECISION_TYPE_INTEGRATION,
}


# ──────────────────────────────────────────────────────────────────────
# ADR architecture domains
# ──────────────────────────────────────────────────────────────────────

ADR_DOMAIN_DATA = "data"
ADR_DOMAIN_API = "api"
ADR_DOMAIN_SECURITY = "security"
ADR_DOMAIN_DEPLOYMENT = "deployment"
ADR_DOMAIN_MESSAGING = "messaging"
ADR_DOMAIN_UI = "ui"
ADR_DOMAIN_INFRASTRUCTURE = "infrastructure"

VALID_ADR_DOMAINS = {
    ADR_DOMAIN_DATA,
    ADR_DOMAIN_API,
    ADR_DOMAIN_SECURITY,
    ADR_DOMAIN_DEPLOYMENT,
    ADR_DOMAIN_MESSAGING,
    ADR_DOMAIN_UI,
    ADR_DOMAIN_INFRASTRUCTURE,
}


# ──────────────────────────────────────────────────────────────────────
# Disposition types for alternatives
# ──────────────────────────────────────────────────────────────────────

DISPOSITION_SELECTED = "SELECTED"
DISPOSITION_REJECTED = "REJECTED"

VALID_DISPOSITIONS = {
    DISPOSITION_SELECTED,
    DISPOSITION_REJECTED,
}


# ──────────────────────────────────────────────────────────────────────
# Alternative — a structured alternative evaluated during decision-making
# ──────────────────────────────────────────────────────────────────────

ALT_ID_PATTERN = re.compile(r"^ALT-\d{3,}$")


@dataclass
class Alternative:
    """
    A structured alternative evaluated during decision-making.

    Distinguishes:
    - No alternatives recorded (valid if only one option was evaluated)
    - Alternatives evaluated and rejected (structured, with disposition)

    No fabrication: if only one option was actually evaluated, alternatives
    must be empty (not padded with fake rejected alternatives).
    """
    alt_id: str
    description: str
    disposition: str = DISPOSITION_REJECTED
    reason: str = ""

    def __post_init__(self):
        if not isinstance(self.alt_id, str) or not self.alt_id:
            raise RecordError("alt_id must be a non-empty string")
        if not ALT_ID_PATTERN.match(self.alt_id):
            raise RecordError(
                f"Invalid alt_id '{self.alt_id}'. "
                f"Must match pattern: ALT-NNN (e.g., ALT-001)"
            )
        if not isinstance(self.description, str) or not self.description.strip():
            raise RecordError("description must be a non-empty string")
        if self.disposition not in VALID_DISPOSITIONS:
            raise RecordError(
                f"Invalid disposition '{self.disposition}'. "
                f"Must be one of: {', '.join(sorted(VALID_DISPOSITIONS))}"
            )

    def to_dict(self) -> dict:
        return {
            "alt_id": self.alt_id,
            "description": self.description,
            "disposition": self.disposition,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Alternative":
        return cls(
            alt_id=data["alt_id"],
            description=data["description"],
            disposition=data.get("disposition", DISPOSITION_REJECTED),
            reason=data.get("reason", ""),
        )


# ──────────────────────────────────────────────────────────────────────
# Consequences — structured consequences of a decision
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Consequences:
    """
    Structured consequences of a decision.

    All three lists are explicit:
    - positive: beneficial outcomes
    - negative: adverse outcomes
    - risks: potential future risks

    Empty lists are valid — they mean "none recorded" (not "none exist").
    """
    positive: List[str] = field(default_factory=list)
    negative: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)

    def validate(self) -> None:
        """Validate consequence structure."""
        if not isinstance(self.positive, list):
            raise RecordError("consequences.positive must be a list")
        if not isinstance(self.negative, list):
            raise RecordError("consequences.negative must be a list")
        if not isinstance(self.risks, list):
            raise RecordError("consequences.risks must be a list")
        for item in self.positive:
            if not isinstance(item, str):
                raise RecordError("consequences.positive must contain strings")
        for item in self.negative:
            if not isinstance(item, str):
                raise RecordError("consequences.negative must contain strings")
        for item in self.risks:
            if not isinstance(item, str):
                raise RecordError("consequences.risks must contain strings")

    def to_dict(self) -> dict:
        return {
            "positive": list(self.positive),
            "negative": list(self.negative),
            "risks": list(self.risks),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Consequences":
        return cls(
            positive=list(data.get("positive", [])),
            negative=list(data.get("negative", [])),
            risks=list(data.get("risks", [])),
        )


# ──────────────────────────────────────────────────────────────────────
# DecisionRecord (DR) — typed EngineeringRecord
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DecisionRecord(EngineeringRecord):
    """
    Decision Record — a structured record of an engineering decision.

    Captures the full decision context:
    - context: the situation requiring a decision
    - decision: the decision made
    - alternatives: structured alternatives evaluated (empty if only one option)
    - rationale: why this decision (required — completeness violation if missing)
    - consequences: structured positive/negative/risks
    - related_requirements: REQ/PRD/NFR record IDs this decision addresses
    - related_decisions: other DR/ADR record IDs this relates to

    Lifecycle: proposed → accepted → deprecated → superseded → archived
    Authority: proposed → accepted → deprecated

    Invariant: Agent-generated DR always starts as PROPOSED.
    """
    record_type: str = "DR"
    title: str = ""
    description: str = ""
    status: str = "proposed"
    authority: str = AUTHORITY_PROPOSED

    # DR-specific fields
    decision_type: str = ""  # ARCHITECTURE | TECHNICAL | SECURITY | PRODUCT | OPERATIONAL | DATA | INTEGRATION
    context: str = ""
    decision: str = ""
    alternatives: List[Alternative] = field(default_factory=list)
    rationale: str = ""
    consequences: Consequences = field(default_factory=Consequences)
    related_requirements: List[str] = field(default_factory=list)  # REQ/NFR/PRD IDs
    related_decisions: List[str] = field(default_factory=list)  # DR/ADR IDs

    def __post_init__(self):
        if not self.title:
            self.title = self.record_id
        if not self.description:
            self.description = self.title
        # Establish canonical relationships for graph traversal
        for req_id in self.related_requirements:
            self.add_relationship(req_id, RELATIONSHIP_DECIDED_BY)
        for dec_id in self.related_decisions:
            self.add_relationship(dec_id, RELATIONSHIP_RELATES_TO)
        # Call base __post_init__ AFTER establishing relationships
        super().__post_init__()

    def validate(self) -> None:
        """Validate DR-specific fields."""
        super().validate()

        # Validate prefix matches record_type (base class also checks this,
        # but we repeat here for clarity in the typed subclass)
        prefix = self.record_id.split("-")[0]
        if prefix != self.record_type:
            raise RecordError(
                f"Record ID prefix '{prefix}' does not match record_type '{self.record_type}'"
            )

        # Decision type (must be valid)
        if not isinstance(self.decision_type, str) or not self.decision_type:
            raise RecordError("decision_type must be a non-empty string")
        if self.decision_type not in VALID_DECISION_TYPES:
            raise RecordError(
                f"Invalid decision_type '{self.decision_type}'. "
                f"Must be one of: {', '.join(sorted(VALID_DECISION_TYPES))}"
            )

        # Context (required — must be non-empty)
        if not isinstance(self.context, str) or not self.context.strip():
            raise RecordError("context must be a non-empty string")

        # Decision statement (required — must be non-empty)
        if not isinstance(self.decision, str) or not self.decision.strip():
            raise RecordError("decision must be a non-empty string")

        # Alternatives (validate structure)
        if not isinstance(self.alternatives, list):
            raise RecordError("alternatives must be a list")
        for alt in self.alternatives:
            if isinstance(alt, dict):
                alt = Alternative.from_dict(alt)
            if not isinstance(alt, Alternative):
                raise RecordError("alternatives must contain Alternative instances")

        # Rationale (required — completeness violation if missing)
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise RecordError("rationale is required (completeness violation if missing)")

        # Consequences (validate structure)
        if isinstance(self.consequences, dict):
            self.consequences = Consequences.from_dict(self.consequences)
        if not isinstance(self.consequences, Consequences):
            raise RecordError("consequences must be a Consequences instance")
        self.consequences.validate()

        # Related requirements (must reference REQ/NFR/PRD IDs)
        if not isinstance(self.related_requirements, list):
            raise RecordError("related_requirements must be a list")
        for req_id in self.related_requirements:
            if not isinstance(req_id, str):
                raise RecordError("related_requirements must contain strings")
            if not (req_id.startswith("REQ-") or req_id.startswith("NFR-") or req_id.startswith("PRD-")):
                raise RecordError(
                    f"related_requirements must reference REQ/NFR/PRD IDs, got '{req_id}'"
                )

        # Related decisions (must reference DR/ADR IDs)
        if not isinstance(self.related_decisions, list):
            raise RecordError("related_decisions must be a list")
        for dec_id in self.related_decisions:
            if not isinstance(dec_id, str):
                raise RecordError("related_decisions must contain strings")
            if not (dec_id.startswith("DR-") or dec_id.startswith("ADR-")):
                raise RecordError(
                    f"related_decisions must reference DR/ADR IDs, got '{dec_id}'"
                )

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["decision_type"] = self.decision_type
        base["context"] = self.context
        base["decision"] = self.decision
        base["alternatives"] = [a.to_dict() for a in self.alternatives]
        base["rationale"] = self.rationale
        base["consequences"] = self.consequences.to_dict()
        base["related_requirements"] = list(self.related_requirements)
        base["related_decisions"] = list(self.related_decisions)
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "DecisionRecord":
        """Reconstruct a DecisionRecord from a dictionary."""
        # Reconstruct alternatives
        alternatives = []
        for alt_data in data.get("alternatives", []):
            if isinstance(alt_data, dict):
                alternatives.append(Alternative.from_dict(alt_data))
            else:
                alternatives.append(alt_data)

        # Reconstruct consequences
        consequences = Consequences.from_dict(data.get("consequences", {}))

        record = cls(
            record_id=data["record_id"],
            record_type="DR",
            title=data["title"],
            description=data["description"],
            status=data.get("status", get_initial_state("DR")),
            authority=data.get("authority", AUTHORITY_PROPOSED),
            decision_type=data.get("decision_type", ""),
            context=data.get("context", ""),
            decision=data.get("decision", ""),
            alternatives=alternatives,
            rationale=data.get("rationale", ""),
            consequences=consequences,
            related_requirements=list(data.get("related_requirements", [])),
            related_decisions=list(data.get("related_decisions", [])),
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

    # ──────────────────────────────────────────────────────────────────
    # DR-specific helpers
    # ──────────────────────────────────────────────────────────────────

    def add_alternative(self, alt_id: str, description: str,
                       disposition: str = DISPOSITION_REJECTED,
                       reason: str = "") -> Alternative:
        """Add a structured alternative."""
        alt = Alternative(
            alt_id=alt_id,
            description=description,
            disposition=disposition,
            reason=reason,
        )
        self.alternatives.append(alt)
        self.updated_at = datetime.utcnow().isoformat()
        return alt

    def link_requirement(self, req_id: str) -> None:
        """Link a requirement (REQ/NFR/PRD) to this decision."""
        if not (req_id.startswith("REQ-") or req_id.startswith("NFR-") or req_id.startswith("PRD-")):
            raise RecordError(
                f"Must reference a REQ/NFR/PRD ID, got '{req_id}'"
            )
        if req_id not in self.related_requirements:
            self.related_requirements.append(req_id)
            self.updated_at = datetime.utcnow().isoformat()
            self.add_relationship(req_id, RELATIONSHIP_DECIDED_BY)

    def link_decision(self, decision_id: str) -> None:
        """Link another decision (DR/ADR) to this decision."""
        if not (decision_id.startswith("DR-") or decision_id.startswith("ADR-")):
            raise RecordError(
                f"Must reference a DR/ADR ID, got '{decision_id}'"
            )
        if decision_id not in self.related_decisions:
            self.related_decisions.append(decision_id)
            self.updated_at = datetime.utcnow().isoformat()
            self.add_relationship(decision_id, RELATIONSHIP_RELATES_TO)

    def supersede(self, new_record_id: str) -> None:
        """
        Mark this record as superseded by a new decision record.

        This is a SUPERSEDES relationship (not a version bump):
        - Same logical decision evolved → version bump (same record_id)
        - New decision replaces → new record + SUPERSEDES edge
        """
        if self.superseded_by is not None:
            raise RecordError(
                f"Record {self.record_id} is already superseded by {self.superseded_by}"
            )
        self.superseded_by = new_record_id
        self.add_relationship(new_record_id, RELATIONSHIP_SUPERSEDES)
        # Transition status to superseded
        if is_valid_transition("DR", self.status, "superseded"):
            self.status = "superseded"
        self.updated_at = datetime.utcnow().isoformat()

    def get_completeness(self) -> Dict[str, bool]:
        """Deterministic completeness check for a DR."""
        return {
            "has_context": bool(self.context and self.context.strip()),
            "has_decision": bool(self.decision and self.decision.strip()),
            "has_rationale": bool(self.rationale and self.rationale.strip()),
            "has_decision_type": self.decision_type in VALID_DECISION_TYPES,
            "has_provenance": self.provenance is not None and self.provenance.is_complete(),
            "valid_requirements": all(
                r.startswith(("REQ-", "NFR-", "PRD-"))
                for r in self.related_requirements
            ),
            "valid_decisions": all(
                d.startswith(("DR-", "ADR-"))
                for d in self.related_decisions
            ),
        }

    def is_complete(self) -> bool:
        """Check if the DR is complete."""
        completeness = self.get_completeness()
        return all(completeness.values())


# ──────────────────────────────────────────────────────────────────────
# ArchitectureDecisionRecord (ADR) — specialization of DR
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ArchitectureDecisionRecord(DecisionRecord):
    """
    Architecture Decision Record — a specialization of DecisionRecord.

    Adds architecture-specific fields:
    - architecture_domain: data | api | security | deployment | messaging | ui | infrastructure
    - patterns_considered: list of architectural patterns evaluated
    - pattern_selected: the chosen pattern
    - trade_offs: trade-offs of the selected pattern
    - impact: architectural impact statement

    ADR Lifecycle: proposed → accepted → deprecated → superseded → archived

    Critical invariant: Agent-generated ADR always starts as PROPOSED.
    Never automatically ACCEPTED — acceptance requires human/governance review.
    """
    record_type: str = "ADR"
    status: str = "proposed"
    authority: str = AUTHORITY_PROPOSED

    # ADR-specific fields
    architecture_domain: str = ""
    patterns_considered: List[str] = field(default_factory=list)
    pattern_selected: str = ""
    trade_offs: str = ""
    impact: str = ""

    def __post_init__(self):
        # New ADRs default to PROPOSED (field default), but reconstructed
        # records (from_dict) must preserve their stored status.
        # Only set defaults for newly created records (empty status).
        if not self.status:
            self.status = "proposed"
        if not self.authority:
            self.authority = AUTHORITY_PROPOSED
        if not self.decision_type:
            self.decision_type = DECISION_TYPE_ARCHITECTURE
        super().__post_init__()

    def validate(self) -> None:
        """Validate ADR-specific fields."""
        super().validate()

        # Validate ADR ID prefix
        prefix = self.record_id.split("-")[0]
        if prefix != "ADR":
            raise RecordError(f"ADR record_id prefix must be 'ADR', got '{prefix}'")

        # Architecture domain
        if not isinstance(self.architecture_domain, str) or not self.architecture_domain:
            raise RecordError("architecture_domain must be a non-empty string")
        if self.architecture_domain not in VALID_ADR_DOMAINS:
            raise RecordError(
                f"Invalid architecture_domain '{self.architecture_domain}'. "
                f"Must be one of: {', '.join(sorted(VALID_ADR_DOMAINS))}"
            )

        # Patterns considered
        if not isinstance(self.patterns_considered, list):
            raise RecordError("patterns_considered must be a list")
        for p in self.patterns_considered:
            if not isinstance(p, str):
                raise RecordError("patterns_considered must contain strings")

        # Pattern selected (required if patterns were considered)
        if self.patterns_considered and not self.pattern_selected:
            raise RecordError(
                "pattern_selected is required when patterns_considered is non-empty"
            )
        if not isinstance(self.pattern_selected, str):
            raise RecordError("pattern_selected must be a string")

        # Trade-offs
        if not isinstance(self.trade_offs, str):
            raise RecordError("trade_offs must be a string")

        # Impact
        if not isinstance(self.impact, str):
            raise RecordError("impact must be a string")

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["architecture_domain"] = self.architecture_domain
        base["patterns_considered"] = list(self.patterns_considered)
        base["pattern_selected"] = self.pattern_selected
        base["trade_offs"] = self.trade_offs
        base["impact"] = self.impact
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "ArchitectureDecisionRecord":
        """Reconstruct an ArchitectureDecisionRecord from a dictionary."""
        # Reconstruct alternatives
        alternatives = []
        for alt_data in data.get("alternatives", []):
            if isinstance(alt_data, dict):
                alternatives.append(Alternative.from_dict(alt_data))
            else:
                alternatives.append(alt_data)

        # Reconstruct consequences
        consequences = Consequences.from_dict(data.get("consequences", {}))

        record = cls(
            record_id=data["record_id"],
            record_type="ADR",
            title=data["title"],
            description=data["description"],
            status=data.get("status", "proposed"),
            authority=data.get("authority", AUTHORITY_PROPOSED),
            decision_type=data.get("decision_type", DECISION_TYPE_ARCHITECTURE),
            context=data.get("context", ""),
            decision=data.get("decision", ""),
            alternatives=alternatives,
            rationale=data.get("rationale", ""),
            consequences=consequences,
            related_requirements=list(data.get("related_requirements", [])),
            related_decisions=list(data.get("related_decisions", [])),
            architecture_domain=data.get("architecture_domain", ""),
            patterns_considered=list(data.get("patterns_considered", [])),
            pattern_selected=data.get("pattern_selected", ""),
            trade_offs=data.get("trade_offs", ""),
            impact=data.get("impact", ""),
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

    # ──────────────────────────────────────────────────────────────────
    # ADR-specific helpers
    # ──────────────────────────────────────────────────────────────────

    def add_pattern(self, pattern: str) -> None:
        """Add an architectural pattern to those considered."""
        if not isinstance(pattern, str) or not pattern.strip():
            raise RecordError("pattern must be a non-empty string")
        if pattern not in self.patterns_considered:
            self.patterns_considered.append(pattern)
            self.updated_at = datetime.utcnow().isoformat()

    def select_pattern(self, pattern: str, trade_offs: str = "", impact: str = "") -> None:
        """
        Select the chosen pattern.

        This does NOT auto-accept the ADR — the ADR remains PROPOSED
        until human/governance review transitions it to ACCEPTED.
        """
        if pattern not in self.patterns_considered:
            self.patterns_considered.append(pattern)
        self.pattern_selected = pattern
        if trade_offs:
            self.trade_offs = trade_offs
        if impact:
            self.impact = impact
        self.updated_at = datetime.utcnow().isoformat()

    def get_completeness(self) -> Dict[str, bool]:
        """Deterministic completeness check for an ADR."""
        base = super().get_completeness()
        base.update({
            "has_architecture_domain": self.architecture_domain in VALID_ADR_DOMAINS,
            "has_patterns": len(self.patterns_considered) > 0,
            "has_pattern_selected": bool(self.pattern_selected and self.pattern_selected.strip()),
            "has_trade_offs": bool(self.trade_offs and self.trade_offs.strip()),
            "has_impact": bool(self.impact and self.impact.strip()),
        })
        return base

    def is_complete(self) -> bool:
        """Check if the ADR is complete."""
        completeness = self.get_completeness()
        return all(completeness.values())


# ──────────────────────────────────────────────────────────────────────
# Supersession utilities
# ──────────────────────────────────────────────────────────────────────

def find_supersession_conflicts(
    records: List[EngineeringRecord],
) -> List[Dict[str, Any]]:
    """
    Detect supersession conflicts in a collection of records.

    Conflict types:
    - MUTUAL_SUPERSESSION: A supersedes B AND B supersedes A
    - AMBIGUOUS_SUCCESSOR: A supersedes multiple records
    - CYCLE: supersession chain forms a cycle

    Returns list of conflict dicts (deterministic).
    """
    conflicts: List[Dict[str, Any]] = []
    record_map: Dict[str, EngineeringRecord] = {
        r.record_id: r for r in records
    }

    # Build supersession graph from both superseded_by field and relationships
    supersession_edges: Dict[str, List[str]] = {}
    for r in records:
        # Check superseded_by field
        if r.superseded_by is not None:
            supersession_edges.setdefault(r.record_id, []).append(r.superseded_by)
        # Check SUPERSEDES relationships
        for rel in r.get_relationships(RELATIONSHIP_SUPERSEDES):
            supersession_edges.setdefault(r.record_id, []).append(rel.target_id)
        # Deduplicate while preserving order
        seen_targets: set = set()
        deduped: List[str] = []
        for t in supersession_edges.get(r.record_id, []):
            if t not in seen_targets:
                seen_targets.add(t)
                deduped.append(t)
        supersession_edges[r.record_id] = deduped

    # Detect mutual supersession and ambiguous successors
    for source_id, targets in sorted(supersession_edges.items()):
        # Ambiguous: multiple active successors
        if len(targets) > 1:
            conflicts.append({
                "type": "AMBIGUOUS_SUCCESSOR",
                "records": [source_id] + sorted(targets),
                "reason": f"{source_id} supersedes multiple records: {sorted(targets)}",
            })

        for target_id in targets:
            # Mutual supersession
            if target_id in supersession_edges and source_id in supersession_edges[target_id]:
                pair = sorted([source_id, target_id])
                conflicts.append({
                    "type": "MUTUAL_SUPERSESSION",
                    "records": pair,
                    "reason": f"Mutual supersession: {pair[0]} and {pair[1]} each supersede the other",
                })

    # Detect cycles via DFS
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {}
    for r in records:
        color[r.record_id] = WHITE

    # Sort for deterministic DFS
    for node_id in sorted(supersession_edges.keys()):
        if color.get(node_id, WHITE) != WHITE:
            continue
        stack = [(node_id, iter(sorted(supersession_edges.get(node_id, []))))]
        path = [node_id]
        color[node_id] = GRAY

        while stack:
            current, neighbors = stack[-1]
            try:
                next_node = next(neighbors)
                if next_node in color:
                    if color[next_node] == GRAY:
                        # Cycle detected
                        cycle_start = path.index(next_node)
                        cycle = path[cycle_start:] + [next_node]
                        conflicts.append({
                            "type": "SUPERSESSION_CYCLE",
                            "records": cycle,
                            "reason": f"Supersession cycle: {' → '.join(cycle)}",
                        })
                    elif color[next_node] == WHITE:
                        color[next_node] = GRAY
                        path.append(next_node)
                        stack.append((next_node, iter(sorted(supersession_edges.get(next_node, [])))))
            except StopIteration:
                color[current] = BLACK
                path.pop()
                stack.pop()

    # Deduplicate and sort
    seen: Set[tuple] = set()
    unique: List[Dict[str, Any]] = []
    for c in conflicts:
        key = (c["type"], tuple(c["records"]))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    unique.sort(key=lambda c: (c["type"], c["records"]))
    return unique


def get_supersession_chain(
    record_id: str,
    record_map: Dict[str, EngineeringRecord],
) -> List[str]:
    """
    Get the supersession chain: record → superseded_by → ...

    Returns list of record IDs in supersession order (oldest first).
    Deterministic: follows the single successor (fails closed on ambiguity).
    """
    chain = [record_id]
    current = record_id
    visited: Set[str] = {record_id}

    while True:
        record = record_map.get(current)
        if record is None or record.superseded_by is None:
            break
        next_node = record.superseded_by
        if next_node in visited:
            break  # Cycle — stop
        chain.append(next_node)
        visited.add(next_node)
        current = next_node

    return chain


def get_effective_decision(
    record_id: str,
    record_map: Dict[str, EngineeringRecord],
) -> Optional[str]:
    """
    Get the current effective decision in a supersession chain.

    Returns the last record in the chain (most recent), or None if
    the chain is ambiguous (branching supersession).
    """
    chain = get_supersession_chain(record_id, record_map)
    if len(chain) == 1:
        return chain[0]

    # Check for branching ambiguity
    for node in chain:
        record = record_map.get(node)
        if record and record.superseded_by is not None:
            successors = [
                r.superseded_by for r in record_map.values()
                if r.record_id == node and r.superseded_by is not None
            ]
            if len(successors) > 1:
                return None  # Ambiguous — fail closed

    return chain[-1]
