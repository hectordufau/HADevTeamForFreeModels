# harness/knowledge/records.py — V3.3 Engineering Record Core
"""
Engineering record base class and schema for V3.3.

All record types (PRD, NFR, DR, ADR, TDR, RSK, SEC, RCA) inherit from
EngineeringRecord. The base class provides shared identity, provenance,
authority, lifecycle, relationships, versioning, and validation.

Architecture boundaries:
- EngineeringRecord belongs to ENGINEERING KNOWLEDGE — NOT Learning, NOT Evidence
- Does NOT replace StructuredExperience, FailureLesson, Strategy, DecisionImpact,
  EvidencePackage, TaskContract, Capability, ExperimentContext
- Maintains three-domain separation: Engineering Knowledge / Learning Knowledge / Evidence
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import re
import uuid

from .provenance import (
    Provenance,
    ProvenanceError,
    validate_authority_level,
    can_transition_authority,
    AUTHORITY_PROPOSED,
    AUTHORITY_ACCEPTED,
    AUTHORITY_DEPRECATED,
    AUTHORITY_ARCHIVED,
    VALID_AUTHORITY_LEVELS,
)
from .lifecycle import (
    LifecycleError,
    get_initial_state,
    get_lifecycle_definition,
    is_valid_transition,
    is_terminal_state,
    validate_lifecycle_state,
    VALID_RECORD_TYPES,
)


class RecordError(Exception):
    """Raised on engineering record validation errors."""


# ──────────────────────────────────────────────────────────────────────
# Relationship types
# ──────────────────────────────────────────────────────────────────────

RELATIONSHIP_REQUIRES = "REQUIRES"
RELATIONSHIP_SATISFIES = "SATISFIES"
RELATIONSHIP_DECIDED_BY = "DECIDED_BY"
RELATIONSHIP_CONSTRAINED_BY = "CONSTRAINED_BY"
RELATIONSHIP_INTRODUCES = "INTRODUCES"
RELATIONSHIP_MITIGATES = "MITIGATES"
RELATIONSHIP_SUPERSEDES = "SUPERSEDES"
RELATIONSHIP_CAUSED_BY = "CAUSED_BY"
RELATIONSHIP_RESOLVED_BY = "RESOLVED_BY"
RELATIONSHIP_VERIFIED_BY = "VERIFIED_BY"
RELATIONSHIP_SUPPORTED_BY = "SUPPORTED_BY"
RELATIONSHIP_CONTRADICTS = "CONTRADICTS"
RELATIONSHIP_RELATES_TO = "RELATES_TO"
RELATIONSHIP_DERIVED_FROM = "DERIVED_FROM"

VALID_RELATIONSHIP_TYPES = {
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
}


@dataclass
class RecordRelationship:
    """
    A typed relationship between two engineering records.
    
    Uses stable references (record_ids) rather than embedding entire records.
    """
    source_id: str
    relation_type: str
    target_id: str
    provenance: Optional[Provenance] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.relation_type not in VALID_RELATIONSHIP_TYPES:
            raise RecordError(
                f"Invalid relationship type '{self.relation_type}'. "
                f"Must be one of: {', '.join(sorted(VALID_RELATIONSHIP_TYPES))}"
            )
        if self.source_id == self.target_id:
            raise RecordError("Self-references are not allowed in relationships")

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "relation_type": self.relation_type,
            "target_id": self.target_id,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RecordRelationship":
        prov = None
        if data.get("provenance"):
            prov = Provenance.from_dict(data["provenance"])
        return cls(
            source_id=data["source_id"],
            relation_type=data["relation_type"],
            target_id=data["target_id"],
            provenance=prov,
            metadata=dict(data.get("metadata", {})),
        )


# ──────────────────────────────────────────────────────────────────────
# EngineeringRecord base class
# ──────────────────────────────────────────────────────────────────────

# Record ID pattern: TYPE-NNN (e.g., PRD-001, ADR-042)
RECORD_ID_PATTERN = re.compile(r"^(PRD|NFR|DR|ADR|TDR|RSK|SEC|RCA|REQ)-\d{3,}$")


@dataclass
class EngineeringRecord:
    """
    Base class for all engineering records.

    All record types inherit from this. Provides shared identity, provenance,
    authority, lifecycle, relationships, versioning, and validation.

    Fields:
        record_id: Unique identifier (e.g., "PRD-001", "ADR-042")
        record_type: One of PRD, NFR, DR, ADR, TDR, RSK, SEC, RCA
        title: Human-readable title
        description: Full description
        status: Lifecycle state (see lifecycle.py)
        authority: Authority level (proposed, accepted, deprecated, archived)
        provenance: Origin and history (required, but default None for dataclass inheritance)
        tags: Optional tags for categorization
        created_at: ISO-8601 creation timestamp
        updated_at: ISO-8601 last-update timestamp
        version: Integer version number (starts at 1)
        superseded_by: record_id of newer version (if superseded)
        experiment_context: V3.2 ExperimentContext for traceability

    Note: provenance has default None for dataclass inheritance field ordering.
    validate() still requires provenance to be set.
    """

    record_id: str = ""
    record_type: str = ""
    title: str = ""
    description: str = ""
    status: str = ""
    authority: str = ""
    provenance: Optional[Provenance] = None
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    version: int = 1
    superseded_by: Optional[str] = None
    experiment_context: Optional[Any] = None

    # Internal: relationships are not part of __init__ signature but stored
    _relationships: List[RecordRelationship] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self):
        """Validate record after initialization. Coerces nested dicts to proper types."""
        # Coerce provenance dict to Provenance instance
        if isinstance(self.provenance, dict):
            self.provenance = Provenance.from_dict(self.provenance)
        # Coerce experiment_context dict to ExperimentContext
        if isinstance(self.experiment_context, dict):
            from harness.learning.context import ExperimentContext
            self.experiment_context = ExperimentContext.from_dict(self.experiment_context)
        # Coerce relationship dicts to RecordRelationship
        for i, rel in enumerate(self._relationships):
            if isinstance(rel, dict):
                self._relationships[i] = RecordRelationship.from_dict(rel)
        self.validate()

    def validate(self) -> None:
        """
        Validate the record.
        
        Raises RecordError on any validation failure.
        """
        # Record ID
        if not isinstance(self.record_id, str) or not self.record_id:
            raise RecordError("record_id must be a non-empty string")
        if not RECORD_ID_PATTERN.match(self.record_id):
            raise RecordError(
                f"Invalid record_id '{self.record_id}'. "
                f"Must match pattern: TYPE-NNN (e.g., PRD-001, ADR-042)"
            )

        # Record type
        if self.record_type not in VALID_RECORD_TYPES:
            raise RecordError(
                f"Invalid record_type '{self.record_type}'. "
                f"Must be one of: {', '.join(sorted(VALID_RECORD_TYPES))}"
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

        # Authority
        validate_authority_level(self.authority)

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
        """
        Serialize to dictionary (deterministic key order).
        
        Two equivalent records serialize equivalently.
        """
        return {
            "record_id": self.record_id,
            "record_type": self.record_type,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "authority": self.authority,
            "provenance": self.provenance.to_dict(),
            "tags": sorted(self.tags),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "superseded_by": self.superseded_by,
            "experiment_context": (
                self.experiment_context.to_dict()
                if self.experiment_context is not None
                else None
            ),
            "relationships": [r.to_dict() for r in self._relationships],
        }

    def to_json(self) -> str:
        """Serialize to JSON string (deterministic)."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "EngineeringRecord":
        """
        Create an EngineeringRecord from a dictionary.

        Reconstructs nested Provenance and ExperimentContext objects.
        Returns a base EngineeringRecord instance — the store preserves
        whatever record type was saved (typed or untyped).
        """
        # Reconstruct provenance
        provenance = Provenance.from_dict(data["provenance"])

        # Reconstruct experiment context
        exp_ctx = None
        if data.get("experiment_context") is not None:
            from harness.learning.context import ExperimentContext
            exp_ctx = ExperimentContext.from_dict(data["experiment_context"])

        # Reconstruct relationships
        relationships = []
        for rel_data in data.get("relationships", []):
            if isinstance(rel_data, dict):
                relationships.append(RecordRelationship.from_dict(rel_data))

        record = cls(
            record_id=data["record_id"],
            record_type=data["record_type"],
            title=data["title"],
            description=data["description"],
            status=data["status"],
            authority=data["authority"],
            provenance=provenance,
            tags=list(data.get("tags", [])),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
            version=data.get("version", 1),
            superseded_by=data.get("superseded_by"),
            experiment_context=exp_ctx,
        )
        record._relationships = relationships
        return record

    def compute_hash(self) -> str:
        """
        Compute a deterministic SHA-256 hash of the record.
        
        Uses canonical JSON representation (sorted keys, no whitespace).
        Does NOT depend on Python hash().
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def transition_status(self, new_status: str, reason: str = "", actor: str = "") -> None:
        """
        Attempt a lifecycle transition.
        
        Raises LifecycleError if invalid (fail closed).
        Updates status and updated_at on success.
        """
        from .lifecycle import transition as do_transition
        do_transition(self.record_type, self.status, new_status, reason, actor)
        self.status = new_status
        self.updated_at = datetime.utcnow().isoformat()

    def is_terminal(self) -> bool:
        """Check if the record is in a terminal (immutable) state."""
        return is_terminal_state(self.record_type, self.status)

    def add_relationship(
        self,
        target_id: str,
        relation_type: str,
        provenance: Optional[Provenance] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RecordRelationship:
        """
        Add a relationship to another record.

        Uses stable reference (record_id), not embedded record.
        Does NOT update updated_at — relationship changes are not record mutations.
        """
        rel = RecordRelationship(
            source_id=self.record_id,
            relation_type=relation_type,
            target_id=target_id,
            provenance=provenance,
            metadata=metadata or {},
        )
        self._relationships.append(rel)
        return rel

    def get_relationships(self, relation_type: Optional[str] = None) -> List[RecordRelationship]:
        """Get all relationships, optionally filtered by type."""
        if relation_type is None:
            return list(self._relationships)
        return [r for r in self._relationships if r.relation_type == relation_type]

    def get_related_ids(self, relation_type: Optional[str] = None) -> List[str]:
        """Get record_ids of related records."""
        return [r.target_id for r in self.get_relationships(relation_type)]

    def create_new_version(
        self,
        new_record_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "EngineeringRecord":
        """
        Create a new version of this record.
        
        The current record is marked as superseded by the new version.
        Protected fields (record_id, created_at, original provenance) are preserved
        in the new version's provenance chain.
        """
        if self.superseded_by is not None:
            raise RecordError(
                f"Record {self.record_id} is already superseded by {self.superseded_by}"
            )

        # Create new version with incremented version number
        new_record = EngineeringRecord(
            record_id=new_record_id,
            record_type=self.record_type,
            title=title or self.title,
            description=description or self.description,
            status=get_initial_state(self.record_type),
            authority=AUTHORITY_PROPOSED,  # New version starts as proposed
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

        # Mark current record as superseded
        self.superseded_by = new_record_id

        return new_record

    def __eq__(self, other: object) -> bool:
        """Two records are equal if their canonical dicts match."""
        if not isinstance(other, EngineeringRecord):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        """Hash based on deterministic dict representation (NOT Python object identity)."""
        return hash(json.dumps(self.to_dict(), sort_keys=True))


# ──────────────────────────────────────────────────────────────────────
# Registry re-export for backward compatibility
# The canonical registry lives in registry.py (breaks circular imports).
# Re-exported here so existing code and tests can import from records.py.
# ──────────────────────────────────────────────────────────────────────

from .registry import RECORD_TYPE_CLASSES, TYPED_RECORD_CLASSES  # noqa: F401, E402

def create_record(record_type: str, **kwargs) -> EngineeringRecord:
    """
    Factory function to create a record of the given type.
    Uses registry.RECORD_TYPE_CLASSES for typed dispatch.

    Raises RecordError if record type is unknown.
    """
    from .registry import RECORD_TYPE_CLASSES
    if record_type not in RECORD_TYPE_CLASSES:
        raise RecordError(
            f"Unknown record type '{record_type}'. "
            f"Must be one of: {', '.join(sorted(RECORD_TYPE_CLASSES.keys()))}"
        )
    cls = RECORD_TYPE_CLASSES[record_type]
    kwargs["record_type"] = record_type
    return cls(**kwargs)
