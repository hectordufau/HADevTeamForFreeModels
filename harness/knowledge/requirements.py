# harness/knowledge/requirements.py — V3.3 Phase 4: PRD, Requirement, NFR
"""
Typed engineering record subclasses for Phase 4.

Implements:
- PRD (Product Requirements Document) — structured, with individually addressable requirements
- Requirement (REQ) — with acceptance criteria, constraints, NFR links, lifecycle
- NFR (Non-Functional Requirement) — with structured metrics, scope semantics
- AcceptanceCriterion — structured with stable IDs

Architecture:
- All typed records inherit from EngineeringRecord base (Phase 1)
- Share identity, provenance, authority, lifecycle, relationships, versioning
- PRD/REQ/NFR typed fields are additive only

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
    RELATIONSHIP_CONSTRAINED_BY,
    RELATIONSHIP_SATISFIES,
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
# Requirement types
# ──────────────────────────────────────────────────────────────────────

REQ_TYPE_FUNCTIONAL = "FUNCTIONAL"
REQ_TYPE_PRODUCT = "PRODUCT"
REQ_TYPE_BUSINESS = "BUSINESS"
REQ_TYPE_INTEGRATION = "INTEGRATION"
REQ_TYPE_COMPLIANCE = "COMPLIANCE"

VALID_REQ_TYPES = {
    REQ_TYPE_FUNCTIONAL,
    REQ_TYPE_PRODUCT,
    REQ_TYPE_BUSINESS,
    REQ_TYPE_INTEGRATION,
    REQ_TYPE_COMPLIANCE,
}

# NFR categories
NFR_CATEGORY_PERFORMANCE = "performance"
NFR_CATEGORY_AVAILABILITY = "availability"
NFR_CATEGORY_RELIABILITY = "reliability"
NFR_CATEGORY_SECURITY = "security"
NFR_CATEGORY_SCALABILITY = "scalability"
NFR_CATEGORY_MAINTAINABILITY = "maintainability"
NFR_CATEGORY_OBSERVABILITY = "observability"
NFR_CATEGORY_RECOVERABILITY = "recoverability"
NFR_CATEGORY_COMPATIBILITY = "compatibility"
NFR_CATEGORY_PRIVACY = "privacy"

VALID_NFR_CATEGORIES = {
    NFR_CATEGORY_PERFORMANCE,
    NFR_CATEGORY_AVAILABILITY,
    NFR_CATEGORY_RELIABILITY,
    NFR_CATEGORY_SECURITY,
    NFR_CATEGORY_SCALABILITY,
    NFR_CATEGORY_MAINTAINABILITY,
    NFR_CATEGORY_OBSERVABILITY,
    NFR_CATEGORY_RECOVERABILITY,
    NFR_CATEGORY_COMPATIBILITY,
    NFR_CATEGORY_PRIVACY,
}

# NFR scopes
NFR_SCOPE_GLOBAL = "GLOBAL"
NFR_SCOPE_PRD = "PRD"
NFR_SCOPE_REQUIREMENT = "REQUIREMENT"
NFR_SCOPE_COMPONENT = "COMPONENT"

VALID_NFR_SCOPES = {
    NFR_SCOPE_GLOBAL,
    NFR_SCOPE_PRD,
    NFR_SCOPE_REQUIREMENT,
    NFR_SCOPE_COMPONENT,
}

# Measurement methods
MEASUREMENT_LOAD_TEST = "load_test"
MEASUREMENT_UNIT_TEST = "unit_test"
MEASUREMENT_INTEGRATION_TEST = "integration_test"
MEASUREMENT_MANUAL_REVIEW = "manual_review"
MEASUREMENT_MONITORING = "monitoring"
MEASUREMENT_AUDIT = "audit"
MEASUREMENT_PROFILING = "profiling"

VALID_MEASUREMENT_METHODS = {
    MEASUREMENT_LOAD_TEST,
    MEASUREMENT_UNIT_TEST,
    MEASUREMENT_INTEGRATION_TEST,
    MEASUREMENT_MANUAL_REVIEW,
    MEASUREMENT_MONITORING,
    MEASUREMENT_AUDIT,
    MEASUREMENT_PROFILING,
}

# Validation operators for NFR metrics
VALID_METRIC_OPERATORS = {"<=", ">=", "<", ">", "==", "!=", "="}


# ──────────────────────────────────────────────────────────────────────
# AcceptanceCriterion
# ──────────────────────────────────────────────────────────────────────

AC_ID_PATTERN = re.compile(r"^AC-\d{3,}$")


@dataclass
class AcceptanceCriterion:
    """
    A structured, individually addressable acceptance criterion for a Requirement.

    Stable identity: AC-001, AC-002, etc. Survives requirement versioning.
    """
    id: str
    description: str
    verification_method: str = ""
    status: str = "pending"  # pending, pass, fail, skip
    provenance: Optional[Provenance] = None
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    version: int = 1

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validate acceptance criterion fields."""
        if not isinstance(self.id, str) or not self.id:
            raise RecordError("AcceptanceCriterion id must be a non-empty string")
        if not AC_ID_PATTERN.match(self.id):
            raise RecordError(
                f"Invalid AcceptanceCriterion id '{self.id}'. "
                f"Must match pattern: AC-NNN (e.g., AC-001)"
            )
        if not isinstance(self.description, str) or not self.description.strip():
            raise RecordError("AcceptanceCriterion description must be non-empty")
        if not isinstance(self.verification_method, str):
            raise RecordError("AcceptanceCriterion verification_method must be a string")
        if not isinstance(self.status, str):
            raise RecordError("AcceptanceCriterion status must be a string")
        if self.status not in {"pending", "pass", "fail", "skip"}:
            raise RecordError(
                f"Invalid AcceptanceCriterion status '{self.status}'. "
                f"Must be one of: pending, pass, fail, skip"
            )
        if self.provenance is not None and not isinstance(self.provenance, Provenance):
            raise RecordError("AcceptanceCriterion provenance must be a Provenance or None")
        if not isinstance(self.tags, list):
            raise RecordError("AcceptanceCriterion tags must be a list")
        if not isinstance(self.version, int) or self.version < 1:
            raise RecordError("AcceptanceCriterion version must be a positive integer (>= 1)")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "verification_method": self.verification_method,
            "status": self.status,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "tags": sorted(self.tags),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AcceptanceCriterion":
        prov = None
        if data.get("provenance"):
            prov = Provenance.from_dict(data["provenance"])
        return cls(
            id=data["id"],
            description=data["description"],
            verification_method=data.get("verification_method", ""),
            status=data.get("status", "pending"),
            provenance=prov,
            tags=list(data.get("tags", [])),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
            version=data.get("version", 1),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AcceptanceCriterion):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash(json.dumps(self.to_dict(), sort_keys=True))


# ──────────────────────────────────────────────────────────────────────
# Requirement (REQ) — typed EngineeringRecord
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Requirement(EngineeringRecord):
    """
    An individually addressable requirement within a PRD.

    Stable identity: REQ-001 v1 → REQ-001 v2 remains the same logical requirement.
    Cycle: proposed → accepted → implemented → verified → deprecated → superseded
    """
    record_type: str = "REQ"
    title: str = ""
    description: str = ""
    status: str = "proposed"
    authority: str = AUTHORITY_PROPOSED
    req_type: str = REQ_TYPE_FUNCTIONAL
    priority: str = "medium"  # critical, high, medium, low
    acceptance_criteria: List[AcceptanceCriterion] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)  # NFR IDs
    related_nfrs: List[str] = field(default_factory=list)  # NFR IDs
    parent_prd: Optional[str] = None  # PRD ID this requirement belongs to

    def __post_init__(self):
        # Set title/description from base class if not set
        if not self.title:
            self.title = f"Requirement {self.record_id}"
        if not self.description:
            self.description = self.title
        # Establish canonical relationships for graph traversal
        # REQ → CONSTRAINED_BY → NFR (each related NFR constrains this requirement)
        for nfr_id in self.related_nfrs:
            self.add_relationship(nfr_id, RELATIONSHIP_CONSTRAINED_BY)
        # Call base __post_init__ which coerces nested dicts and validates
        super().__post_init__()

    def validate(self) -> None:
        """Validate requirement-specific fields."""
        super().validate()

        # Validate REQ ID prefix
        prefix = self.record_id.split("-")[0]
        if prefix != "REQ":
            raise RecordError(
                f"Requirement record_id prefix must be 'REQ', got '{prefix}'"
            )

        # Validate req_type
        if self.req_type not in VALID_REQ_TYPES:
            raise RecordError(
                f"Invalid requirement type '{self.req_type}'. "
                f"Must be one of: {', '.join(sorted(VALID_REQ_TYPES))}"
            )

        # Validate priority
        if self.priority not in {"critical", "high", "medium", "low"}:
            raise RecordError(
                f"Invalid priority '{self.priority}'. "
                f"Must be one of: critical, high, medium, low"
            )

        # Validate acceptance criteria
        if not isinstance(self.acceptance_criteria, list):
            raise RecordError("acceptance_criteria must be a list")
        ac_ids: Set[str] = set()
        for ac in self.acceptance_criteria:
            if not isinstance(ac, AcceptanceCriterion):
                raise RecordError(
                    f"acceptance_criteria must contain AcceptanceCriterion instances, "
                    f"got {type(ac).__name__}"
                )
            if ac.id in ac_ids:
                raise RecordError(f"Duplicate acceptance criterion ID: {ac.id}")
            ac_ids.add(ac.id)

        # Validate constraints
        if not isinstance(self.constraints, list):
            raise RecordError("constraints must be a list")

        # Validate related NFRs
        if not isinstance(self.related_nfrs, list):
            raise RecordError("related_nfrs must be a list")
        for nfr_id in self.related_nfrs:
            if not isinstance(nfr_id, str) or not nfr_id:
                raise RecordError("related_nfrs must contain non-empty strings")

        # Validate parent PRD
        if self.parent_prd is not None:
            if not isinstance(self.parent_prd, str) or not self.parent_prd:
                raise RecordError("parent_prd must be a non-empty string or None")
            if not self.parent_prd.startswith("PRD-"):
                raise RecordError(
                    f"parent_prd must reference a PRD (e.g., 'PRD-001'), got '{self.parent_prd}'"
                )

    def to_dict(self) -> dict:
        """Serialize to dictionary (extends base)."""
        base = super().to_dict()
        base["req_type"] = self.req_type
        base["priority"] = self.priority
        base["acceptance_criteria"] = [ac.to_dict() for ac in self.acceptance_criteria]
        base["constraints"] = list(self.constraints)
        base["related_nfrs"] = list(self.related_nfrs)
        base["parent_prd"] = self.parent_prd
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "Requirement":
        """Create Requirement from dict."""
        acs = [AcceptanceCriterion.from_dict(ac) for ac in data.get("acceptance_criteria", [])]
        record = cls(
            record_id=data["record_id"],
            record_type="REQ",
            title=data["title"],
            description=data["description"],
            req_type=data.get("req_type", REQ_TYPE_FUNCTIONAL),
            priority=data.get("priority", "medium"),
            acceptance_criteria=acs,
            constraints=list(data.get("constraints", [])),
            related_nfrs=list(data.get("related_nfrs", [])),
            parent_prd=data.get("parent_prd"),
            status=data.get("status", get_initial_state("REQ")),
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

    def add_acceptance_criterion(
        self,
        ac_id: str,
        description: str,
        verification_method: str = "",
    ) -> AcceptanceCriterion:
        """Add an acceptance criterion."""
        if any(ac.id == ac_id for ac in self.acceptance_criteria):
            raise RecordError(f"Acceptance criterion '{ac_id}' already exists")
        ac = AcceptanceCriterion(
            id=ac_id,
            description=description,
            verification_method=verification_method,
        )
        self.acceptance_criteria.append(ac)
        self.updated_at = datetime.utcnow().isoformat()
        return ac

    def link_nfr(self, nfr_id: str) -> None:
        """Link a non-functional requirement to this requirement."""
        if nfr_id not in self.related_nfrs:
            self.related_nfrs.append(nfr_id)
            self.add_relationship(nfr_id, RELATIONSHIP_CONSTRAINED_BY)

    def link_parent_prd(self, prd_id: str) -> None:
        """Set the parent PRD for this requirement."""
        if not prd_id.startswith("PRD-"):
            raise RecordError(f"parent_prd must reference a PRD, got '{prd_id}'")
        self.parent_prd = prd_id
        self.add_relationship(prd_id, RELATIONSHIP_REQUIRES)

    def create_new_version(
        self,
        new_record_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "Requirement":
        """Create a new version of this requirement. Preserves identity chain."""
        if self.superseded_by is not None:
            raise RecordError(
                f"Requirement {self.record_id} is already superseded by {self.superseded_by}"
            )
        new_req = Requirement(
            record_id=new_record_id,
            record_type="REQ",
            title=title or self.title,
            description=description or self.description,
            req_type=self.req_type,
            priority=self.priority,
            acceptance_criteria=[AcceptanceCriterion.from_dict(ac.to_dict()) for ac in self.acceptance_criteria],
            constraints=list(self.constraints),
            related_nfrs=list(self.related_nfrs),
            parent_prd=self.parent_prd,
            status=get_initial_state("REQ"),
            authority=AUTHORITY_PROPOSED,
            provenance=Provenance(
                author=self.provenance.author,
                source=self.provenance.source,
                parent_record=self.record_id,
                experiment_context=self.experiment_context,
            ),
            tags=list(self.tags),
            version=self.version + 1,
            experiment_context=self.experiment_context,
        )
        self.superseded_by = new_record_id
        return new_req

    def get_completeness(self) -> Dict[str, bool]:
        """Deterministic completeness check."""
        return {
            "has_description": bool(self.description and self.description.strip()),
            "has_priority": self.priority in {"critical", "high", "medium", "low"},
            "has_type": self.req_type in VALID_REQ_TYPES,
            "has_acceptance_criteria": len(self.acceptance_criteria) > 0,
            "has_provenance": self.provenance is not None and self.provenance.is_complete(),
            "has_status": bool(self.status),
            "has_parent_prd": self.parent_prd is not None,
        }

    def is_complete(self) -> bool:
        """Check if the requirement is complete (all required fields present)."""
        completeness = self.get_completeness()
        return all(completeness.values())


# ──────────────────────────────────────────────────────────────────────
# NFR — typed EngineeringRecord
# ──────────────────────────────────────────────────────────────────────

@dataclass
class NFR(EngineeringRecord):
    """
    Non-Functional Requirement with structured metrics and scope semantics.

    Identity: NFR-001, NFR-002, etc.
    Lifecycle: proposed → accepted → deprecated → archived (shared with PRD)
    """
    record_type: str = "NFR"
    title: str = ""
    description: str = ""
    status: str = "draft"
    authority: str = AUTHORITY_PROPOSED
    category: str = ""  # performance, security, etc.
    scope: str = NFR_SCOPE_GLOBAL  # GLOBAL, PRD, REQUIREMENT, COMPONENT
    metric: Optional[Dict[str, Any]] = None  # {name, operator, target, unit}
    measurement: Optional[Dict[str, Any]] = None  # {method, ...}
    qualitative: bool = False  # True for qualitative NFRs (no metric)
    related_requirements: List[str] = field(default_factory=list)  # REQ IDs
    related_prds: List[str] = field(default_factory=list)  # PRD IDs

    def __post_init__(self):
        if not self.title:
            self.title = f"NFR {self.record_id}"
        if not self.description:
            self.description = self.title
        # If no metric is set and qualitative flag was not explicitly set,
        # treat as qualitative NFR (metric can be added later via add_metric)
        if self.metric is None and not self.qualitative:
            self.qualitative = True
        # Validate scope and category early (before super().__post_init__)
        if self.scope not in VALID_NFR_SCOPES:
            raise RecordError(
                f"Invalid NFR scope '{self.scope}'. "
                f"Must be one of: {', '.join(sorted(VALID_NFR_SCOPES))}"
            )
        if self.category not in VALID_NFR_CATEGORIES:
            raise RecordError(
                f"Invalid NFR category '{self.category}'. "
                f"Must be one of: {', '.join(sorted(VALID_NFR_CATEGORIES))}"
            )
        super().__post_init__()

    def validate(self) -> None:
        """Validate NFR-specific fields (non-structural checks only)."""
        super().validate()

        # Validate NFR ID prefix
        prefix = self.record_id.split("-")[0]
        if prefix != "NFR":
            raise RecordError(f"NFR record_id prefix must be 'NFR', got '{prefix}'")
        # Note: scope and category structural validation happens in __post_init__

        # Validate metric (required for quantitative NFRs)
        if not self.qualitative:
            if self.metric is None:
                raise RecordError("Quantitative NFR must have a metric")
            if not isinstance(self.metric, dict):
                raise RecordError("NFR metric must be a dict")

            # Validate metric fields
            metric_name = self.metric.get("name", "")
            if not isinstance(metric_name, str) or not metric_name:
                raise RecordError("NFR metric.name must be a non-empty string")

            metric_op = self.metric.get("operator", "")
            if metric_op not in VALID_METRIC_OPERATORS:
                raise RecordError(
                    f"Invalid NFR metric.operator '{metric_op}'. "
                    f"Must be one of: {', '.join(sorted(VALID_METRIC_OPERATORS))}"
                )

            # target must be present (can be int, float, or string)
            if "target" not in self.metric:
                raise RecordError("NFR metric must have a 'target' field")

            # unit is optional but must be string if present
            metric_unit = self.metric.get("unit", "")
            if not isinstance(metric_unit, str):
                raise RecordError("NFR metric.unit must be a string")

        # Validate measurement
        if self.measurement is not None:
            if not isinstance(self.measurement, dict):
                raise RecordError("NFR measurement must be a dict")
            method = self.measurement.get("method", "")
            if method and method not in VALID_MEASUREMENT_METHODS:
                raise RecordError(
                    f"Invalid NFR measurement.method '{method}'. "
                    f"Must be one of: {', '.join(sorted(VALID_MEASUREMENT_METHODS))}"
                )

        # Validate related requirements
        if not isinstance(self.related_requirements, list):
            raise RecordError("related_requirements must be a list")
        for req_id in self.related_requirements:
            if not isinstance(req_id, str) or not req_id.startswith("REQ-"):
                raise RecordError(
                    f"related_requirements must reference REQ IDs (e.g., 'REQ-001'), got '{req_id}'"
                )

        # Validate related PRDs
        if not isinstance(self.related_prds, list):
            raise RecordError("related_prds must be a list")
        for prd_id in self.related_prds:
            if not isinstance(prd_id, str) or not prd_id.startswith("PRD-"):
                raise RecordError(
                    f"related_prds must reference PRD IDs (e.g., 'PRD-001'), got '{prd_id}'"
                )

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["category"] = self.category
        base["scope"] = self.scope
        base["metric"] = self.metric
        base["measurement"] = self.measurement
        base["qualitative"] = self.qualitative
        base["related_requirements"] = list(self.related_requirements)
        base["related_prds"] = list(self.related_prds)
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "NFR":
        record = cls(
            record_id=data["record_id"],
            record_type="NFR",
            title=data["title"],
            description=data["description"],
            category=data.get("category", ""),
            scope=data.get("scope", NFR_SCOPE_GLOBAL),
            metric=data.get("metric"),
            measurement=data.get("measurement"),
            qualitative=data.get("qualitative", False),
            related_requirements=list(data.get("related_requirements", [])),
            related_prds=list(data.get("related_prds", [])),
            status=data.get("status", get_initial_state("NFR")),
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

    def add_metric(
        self,
        name: str,
        operator: str,
        target: Any,
        unit: str = "",
    ) -> None:
        """Add or update the structured metric. Clears qualitative flag."""
        if operator not in VALID_METRIC_OPERATORS:
            raise RecordError(
                f"Invalid metric operator '{operator}'. "
                f"Must be one of: {', '.join(sorted(VALID_METRIC_OPERATORS))}"
            )
        self.metric = {
            "name": name,
            "operator": operator,
            "target": target,
            "unit": unit,
        }
        self.qualitative = False
        self.updated_at = datetime.utcnow().isoformat()

    def add_measurement(self, method: str, **kwargs) -> None:
        """Add measurement method and details."""
        if method not in VALID_MEASUREMENT_METHODS:
            raise RecordError(
                f"Invalid measurement method '{method}'. "
                f"Must be one of: {', '.join(sorted(VALID_MEASUREMENT_METHODS))}"
            )
        self.measurement = {"method": method, **kwargs}
        self.updated_at = datetime.utcnow().isoformat()

    def link_requirement(self, req_id: str) -> None:
        """Link a requirement to this NFR."""
        if not req_id.startswith("REQ-"):
            raise RecordError(f"Must reference a REQ ID, got '{req_id}'")
        if req_id not in self.related_requirements:
            self.related_requirements.append(req_id)
            self.updated_at = datetime.utcnow().isoformat()
            self.add_relationship(req_id, RELATIONSHIP_CONSTRAINED_BY)

    def get_completeness(self) -> Dict[str, bool]:
        """Deterministic completeness check."""
        completeness = {
            "has_description": bool(self.description and self.description.strip()),
            "has_category": self.category in VALID_NFR_CATEGORIES,
            "has_scope": self.scope in VALID_NFR_SCOPES,
            "has_provenance": self.provenance is not None and self.provenance.is_complete(),
        }
        if self.qualitative:
            completeness["has_metric_or_qualitative"] = True
        else:
            completeness["has_metric"] = self.metric is not None and "name" in self.metric
        return completeness

    def is_complete(self) -> bool:
        """Check if the NFR is complete."""
        completeness = self.get_completeness()
        return all(completeness.values())


# ──────────────────────────────────────────────────────────────────────
# PRD — typed EngineeringRecord
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PRD(EngineeringRecord):
    """
    Product Requirements Document with individually addressable requirements.

    Architecture:
    - Requirements are independently addressable REQ records (not embedded text)
    - PRD holds objective, scope, requirement IDs (REQ-001, REQ-002, etc.)
    - Requirements reference PRD via parent_prd field
    - Requirements may be orphaned (parent_prd is None, status tracked)

    Lifecycle: draft → review → accepted → deprecated → archived
    """
    record_type: str = "PRD"
    title: str = ""
    description: str = ""
    status: str = "draft"
    authority: str = AUTHORITY_PROPOSED
    objective: str = ""
    scope: str = ""
    stakeholders: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)  # REQ IDs
    related_nfrs: List[str] = field(default_factory=list)  # NFR IDs
    priority: str = "medium"

    def __post_init__(self):
        if not self.title:
            self.title = self.record_id
        if not self.description:
            self.description = self.title
        # Establish canonical relationships for graph traversal
        # PRD → REQUIRES → REQ
        for req_id in self.requirements:
            self.add_relationship(req_id, RELATIONSHIP_REQUIRES)
        # Call base __post_init__ AFTER establishing relationships
        super().__post_init__()

    def validate(self) -> None:
        """Validate PRD-specific fields."""
        super().validate()

        # Validate PRD ID prefix
        prefix = self.record_id.split("-")[0]
        if prefix != "PRD":
            raise RecordError(f"PRD record_id prefix must be 'PRD', got '{prefix}'")

        # Validate objective (type check only — empty allowed for heredoc-created records)
        if not isinstance(self.objective, str):
            raise RecordError("objective must be a string")

        # Validate scope (type check only — empty allowed for heredoc-created records)
        if not isinstance(self.scope, str):
            raise RecordError("scope must be a string")

        # Validate stakeholders
        if not isinstance(self.stakeholders, list):
            raise RecordError("stakeholders must be a list")
        for s in self.stakeholders:
            if not isinstance(s, str):
                raise RecordError("stakeholders must contain strings")

        # Validate requirements (REQ IDs)
        if not isinstance(self.requirements, list):
            raise RecordError("requirements must be a list of REQ IDs")
        req_ids = set()
        for req_id in self.requirements:
            if not isinstance(req_id, str) or not req_id.startswith("REQ-"):
                raise RecordError(
                    f"requirements must reference REQ IDs (e.g., 'REQ-001'), got '{req_id}'"
                )
            if req_id in req_ids:
                raise RecordError(f"Duplicate requirement ID: {req_id}")
            req_ids.add(req_id)

        # Validate related NFRs
        if not isinstance(self.related_nfrs, list):
            raise RecordError("related_nfrs must be a list of NFR IDs")
        for nfr_id in self.related_nfrs:
            if not isinstance(nfr_id, str) or not nfr_id.startswith("NFR-"):
                raise RecordError(
                    f"related_nfrs must reference NFR IDs (e.g., 'NFR-001'), got '{nfr_id}'"
                )

        # Validate priority
        if self.priority not in {"critical", "high", "medium", "low"}:
            raise RecordError(
                f"Invalid PRD priority '{self.priority}'. "
                f"Must be one of: critical, high, medium, low"
            )

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["objective"] = self.objective
        base["scope"] = self.scope
        base["stakeholders"] = sorted(self.stakeholders)
        base["requirements"] = sorted(self.requirements)
        base["related_nfrs"] = sorted(self.related_nfrs)
        base["priority"] = self.priority
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "PRD":
        record = cls(
            record_id=data["record_id"],
            record_type="PRD",
            title=data["title"],
            description=data["description"],
            objective=data.get("objective", ""),
            scope=data.get("scope", ""),
            stakeholders=list(data.get("stakeholders", [])),
            requirements=list(data.get("requirements", [])),
            related_nfrs=list(data.get("related_nfrs", [])),
            priority=data.get("priority", "medium"),
            status=data.get("status", get_initial_state("PRD")),
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

    def add_requirement(self, req_id: str) -> None:
        """Add a REQ ID reference."""
        if not req_id.startswith("REQ-"):
            raise RecordError(f"Must reference a REQ ID, got '{req_id}'")
        if req_id not in self.requirements:
            self.requirements.append(req_id)
            self.updated_at = datetime.utcnow().isoformat()
            self.add_relationship(req_id, RELATIONSHIP_REQUIRES)

    def remove_requirement(self, req_id: str) -> None:
        """Remove a REQ ID reference (marks requirement as orphan)."""
        if req_id in self.requirements:
            self.requirements.remove(req_id)
            self.updated_at = datetime.utcnow().isoformat()

    def link_nfr(self, nfr_id: str) -> None:
        """Link an NFR to this PRD."""
        if not nfr_id.startswith("NFR-"):
            raise RecordError(f"Must reference an NFR ID, got '{nfr_id}'")
        if nfr_id not in self.related_nfrs:
            self.related_nfrs.append(nfr_id)
            self.updated_at = datetime.utcnow().isoformat()
            self.add_relationship(nfr_id, RELATIONSHIP_CONSTRAINED_BY)

    def get_completeness(self) -> Dict[str, bool]:
        """Deterministic completeness check."""
        return {
            "has_objective": bool(self.objective and self.objective.strip()),
            "has_scope": bool(self.scope and self.scope.strip()),
            "has_requirements": len(self.requirements) > 0,
            "has_provenance": self.provenance is not None and self.provenance.is_complete(),
            "has_valid_req_refs": all(r.startswith("REQ-") for r in self.requirements),
        }

    def is_complete(self) -> bool:
        """Check if the PRD is complete."""
        completeness = self.get_completeness()
        return all(completeness.values())


# ──────────────────────────────────────────────────────────────────────
# Coverage and Traceability
# ──────────────────────────────────────────────────────────────────────

def compute_prd_coverage(
    prd: PRD,
    requirements: Dict[str, Requirement],
    nfrs: Dict[str, NFR],
) -> Dict[str, Any]:
    """
    Compute deterministic coverage for a PRD.

    Coverage = recorded, related, traceable (NOT implemented, verified, released).
    """
    req_ids = prd.requirements
    linked_nfrs = set(prd.related_nfrs)

    # Coverage: how many REQs are actually stored in the KnowledgeStore?
    reqs_present = [rid for rid in req_ids if rid in requirements]
    reqs_missing = [rid for rid in req_ids if rid not in requirements]

    # NFR link coverage: how many NFRs are linked?
    nfrs_present = [nid for nid in linked_nfrs if nid in nfrs]
    nfrs_missing = [nid for nid in linked_nfrs if nid not in nfrs]

    # Acceptance criteria coverage (across all linked REQs)
    total_ac = 0
    ac_pass = 0
    ac_fail = 0
    ac_pending = 0
    for rid in reqs_present:
        req = requirements[rid]
        for ac in req.acceptance_criteria:
            total_ac += 1
            if ac.status == "pass":
                ac_pass += 1
            elif ac.status == "fail":
                ac_fail += 1
            else:
                ac_pending += 1

    return {
        "prd_id": prd.record_id,
        "requirements_total": len(req_ids),
        "requirements_present": len(reqs_present),
        "requirements_missing": len(reqs_missing),
        "requirement_coverage": len(reqs_present) / len(req_ids) if req_ids else 0.0,
        "nfrs_total": len(linked_nfrs),
        "nfrs_present": len(nfrs_present),
        "nfrs_missing": len(nfrs_missing),
        "nfr_link_coverage": len(nfrs_present) / len(linked_nfrs) if linked_nfrs else 1.0,
        "acceptance_criteria_total": total_ac,
        "acceptance_criteria_pass": ac_pass,
        "acceptance_criteria_fail": ac_fail,
        "acceptance_criteria_pending": ac_pending,
        "acceptance_criteria_coverage": (ac_pass + ac_fail) / total_ac if total_ac else 0.0,
        "orphan_requirements": len(reqs_missing),
    }


def find_orphan_requirements(
    requirements: Dict[str, Requirement],
    prds: Dict[str, PRD],
) -> List[Requirement]:
    """
    Find requirements not connected to any PRD.

    Do NOT auto-delete. Returns structured list for review.
    """
    orphans = []
    for req_id, req in requirements.items():
        if req.parent_prd is None:
            orphans.append(req)
            continue
        # Check if parent PRD exists
        if req.parent_prd not in prds:
            orphans.append(req)
    return orphans


def find_requirements_for_prd(
    prd_id: str,
    requirements: Dict[str, Requirement],
) -> List[Requirement]:
    """Find all requirements belonging to a given PRD."""
    return [req for req in requirements.values() if req.parent_prd == prd_id]


def find_nfrs_for_requirement(
    req_id: str,
    nfrs: Dict[str, NFR],
) -> List[NFR]:
    """Find all NFRs constraining a given requirement."""
    return [nfr for nfr in nfrs.values() if req_id in nfr.related_requirements]


def find_requirements_constrained_by_nfr(
    nfr_id: str,
    nfrs: Dict[str, NFR],
    requirements: Dict[str, Requirement],
) -> List[Requirement]:
    """Find all requirements constrained by a given NFR."""
    if nfr_id not in nfrs:
        return []
    nfr = nfrs[nfr_id]
    return [requirements[rid] for rid in nfr.related_requirements if rid in requirements]


# ──────────────────────────────────────────────────────────────────────
# Conflict Detection (deterministic/structured only)
# ──────────────────────────────────────────────────────────────────────

def detect_requirement_conflicts(
    requirements: Dict[str, Requirement],
) -> List[Dict[str, Any]]:
    """
    Detect deterministic requirement conflicts.

    Conflict types:
    1. Same AC ID with conflicting definitions
    2. REQ-A supersedes REQ-B but both are current (accepted/implemented/verified)
    """
    conflicts = []

    # Check 1: Same AC ID with conflicting definitions
    ac_definitions: Dict[str, List[Tuple[str, str]]] = {}
    for req_id, req in requirements.items():
        for ac in req.acceptance_criteria:
            ac_definitions.setdefault(ac.id, []).append((req_id, ac.description))

    for ac_id, entries in ac_definitions.items():
        if len(entries) > 1:
            # Check if descriptions differ
            descriptions = set(desc for _, desc in entries)
            if len(descriptions) > 1:
                conflicts.append({
                    "conflict_type": "ACCEPTANCE_CRITERICT_CONFLICT",
                    "acceptance_criterion_id": ac_id,
                    "requirements": [req_id for req_id, _ in entries],
                    "details": descriptions,
                })

    # Check 2: Conflicting supersession (supersedes but both current)
    active_states = {"accepted", "implemented", "verified"}
    for req_id, req in requirements.items():
        if req.superseded_by and req.status in active_states:
            # Check if the superseded record is still active
            superseded = req.superseded_by
            if superseded in requirements:
                if requirements[superseded].status in active_states:
                    conflicts.append({
                        "conflict_type": "SUPERSESSION_CONFLICT",
                        "requirement_id": req_id,
                        "superseded_by": superseded,
                        "current_status": req.status,
                        "superseded_status": requirements[superseded].status,
                    })

    return conflicts


def detect_nfr_conflicts(
    nfrs: Dict[str, NFR],
) -> List[Dict[str, Any]]:
    """
    Detect structured NFR conflicts.

    Where metrics are structured, contradictions are detectable:
    - NFR-001 p95_latency <= 200ms vs NFR-002 p95_latency >= 500ms (same scope)
    """
    conflicts = []

    nfr_list = list(nfrs.values())
    for i, nfr_a in enumerate(nfr_list):
        if nfr_a.qualitative or nfr_a.metric is None:
            continue
        for j, nfr_b in enumerate(nfr_list):
            if i >= j:
                continue
            if nfr_b.qualitative or nfr_b.metric is None:
                continue

            # Same metric name (potential conflict)
            if nfr_a.metric["name"] != nfr_b.metric["name"]:
                continue

            # Same scope (or global)
            scope_a = nfr_a.scope
            scope_b = nfr_b.scope
            if scope_a != scope_b:
                if scope_a != NFR_SCOPE_GLOBAL and scope_b != NFR_SCOPE_GLOBAL:
                    continue

            # Check for opposing metrics
            op_a = nfr_a.metric.get("operator", "")
            op_b = nfr_b.metric.get("operator", "")
            target_a = nfr_a.metric.get("target")
            target_b = nfr_b.metric.get("target")

            # Detect: <= X vs >= Y where X < Y (contradiction)
            if op_a in {"<=", "<"} and op_b in {">=", ">"}:
                try:
                    ta = float(target_a)  # type: ignore[arg-type]
                    tb = float(target_b)  # type: ignore[arg-type]
                    if ta < tb:
                        conflicts.append({
                            "conflict_type": "METRIC_CONTRADICTION",
                            "nfr_a": nfr_a.record_id,
                            "nfr_b": nfr_b.record_id,
                            "metric": nfr_a.metric["name"],
                            "nfr_a_constraint": f"{op_a} {target_a}",
                            "nfr_b_constraint": f"{op_b} {target_b}",
                        })
                except (ValueError, TypeError):
                    pass
            elif op_b in {"<=", "<"} and op_a in {">=", ">"}:
                try:
                    tb = float(target_b)  # type: ignore[arg-type]
                    ta = float(target_a)  # type: ignore[arg-type]
                    if tb < ta:
                        conflicts.append({
                            "conflict_type": "METRIC_CONTRADICTION",
                            "nfr_a": nfr_b.record_id,
                            "nfr_b": nfr_a.record_id,
                            "metric": nfr_b.metric["name"],
                            "nfr_a_constraint": f"{op_b} {target_b}",
                            "nfr_b_constraint": f"{op_a} {target_a}",
                        })
                except (ValueError, TypeError):
                    pass

    return conflicts


# ──────────────────────────────────────────────────────────────────────
# Registry update — replace Phase 1 placeholders with real typed classes
# ──────────────────────────────────────────────────────────────────────

# This dict maps record_type strings to their typed dataclass.
# Phase 4 registers PRD, REQ, NFR; remaining types (DR, ADR, TDR, RSK, SEC, RCA)
# are deferred to Phases 5-7.
TYPED_RECORD_CLASSES: Dict[str, type] = {
    "PRD": PRD,
    "REQ": Requirement,
    "NFR": NFR,
    # DR, ADR, TDR, RSK, SEC, RCA deferred to Phase 5-7
}
