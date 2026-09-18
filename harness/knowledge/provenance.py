# harness/knowledge/provenance.py — V3.3 Provenance & Authority Model
"""
Provenance and authority tracking for engineering records.

Every EngineeringRecord must carry provenance. Provenance is immutable after
creation and records the origin, authorship, and supporting evidence for a
record. Authority levels are explicit and enforced — they are never inferred
from creator, agent role, record type, or file location.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


class ProvenanceError(Exception):
    """Raised on provenance validation errors."""


# Authority levels — explicit, never inferred
AUTHORITY_PROPOSED = "proposed"      # Under review, advisory only
AUTHORITY_ACCEPTED = "accepted"      # Approved, authoritative for decision-making
AUTHORITY_DEPRECATED = "deprecated"  # No longer current
AUTHORITY_ARCHIVED = "archived"      # Historical record, immutable

VALID_AUTHORITY_LEVELS = {
    AUTHORITY_PROPOSED,
    AUTHORITY_ACCEPTED,
    AUTHORITY_DEPRECATED,
    AUTHORITY_ARCHIVED,
}

# Authority hierarchy for comparison (higher = more authoritative)
AUTHORITY_RANK = {
    AUTHORITY_PROPOSED: 0,
    AUTHORITY_ACCEPTED: 1,
    AUTHORITY_DEPRECATED: 2,
    AUTHORITY_ARCHIVED: 3,
}

# Provenance source types
SOURCE_HUMAN = "human"
SOURCE_PIPELINE = "pipeline"
SOURCE_IMPORT = "import"
SOURCE_GENERATED = "generated"

VALID_SOURCES = {
    SOURCE_HUMAN,
    SOURCE_PIPELINE,
    SOURCE_IMPORT,
    SOURCE_GENERATED,
}


@dataclass
class Provenance:
    """
    Origin and history of an engineering record.

    Fields:
        author: Who created the record
        source: Origin type — "human" | "pipeline" | "import" | "generated"
        timestamp: ISO-8601 timestamp of creation
        experiment_context: V3.2 ExperimentContext for traceability
        evidence_refs: List of evidence record_ids supporting this record
        parent_record: record_id of parent (for derived records)
        validation_run_id: V3.2 validation run identifier
        benchmark_id: V3.2 benchmark identifier
        mode: V3.2 mode (COLD/LEARNED/TEST)
    """

    author: str = ""
    source: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    experiment_context: Optional[Any] = None  # ExperimentContext from V3.2
    evidence_refs: List[str] = field(default_factory=list)
    parent_record: Optional[str] = None
    validation_run_id: str = ""
    benchmark_id: str = ""
    mode: str = ""

    def __post_init__(self):
        """Validate provenance fields."""
        if not isinstance(self.author, str):
            raise ProvenanceError("author must be a string")
        if not isinstance(self.source, str):
            raise ProvenanceError("source must be a string")
        if self.source and self.source not in VALID_SOURCES:
            raise ProvenanceError(
                f"Invalid source '{self.source}'. Must be one of: {', '.join(sorted(VALID_SOURCES))}"
            )
        if not isinstance(self.evidence_refs, list):
            raise ProvenanceError("evidence_refs must be a list")
        if not isinstance(self.timestamp, str):
            raise ProvenanceError("timestamp must be a string")

    def to_dict(self) -> dict:
        """Serialize to dictionary (deterministic key order)."""
        return {
            "author": self.author,
            "source": self.source,
            "timestamp": self.timestamp,
            "experiment_context": (
                self.experiment_context.to_dict()
                if self.experiment_context is not None
                else None
            ),
            "evidence_refs": list(self.evidence_refs),
            "parent_record": self.parent_record,
            "validation_run_id": self.validation_run_id,
            "benchmark_id": self.benchmark_id,
            "mode": self.mode,
        }

    def to_json(self) -> str:
        """Serialize to JSON string (deterministic)."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "Provenance":
        """Create Provenance from a dictionary."""
        # Handle ExperimentContext reconstruction
        exp_ctx = None
        if data.get("experiment_context") is not None:
            from harness.learning.context import ExperimentContext
            exp_ctx = ExperimentContext.from_dict(data["experiment_context"])

        return cls(
            author=data.get("author", ""),
            source=data.get("source", ""),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            experiment_context=exp_ctx,
            evidence_refs=list(data.get("evidence_refs", [])),
            parent_record=data.get("parent_record"),
            validation_run_id=data.get("validation_run_id", ""),
            benchmark_id=data.get("benchmark_id", ""),
            mode=data.get("mode", ""),
        )

    def is_complete(self) -> bool:
        """Check if provenance is complete (has author and source)."""
        return bool(self.author) and bool(self.source)

    def __eq__(self, other: object) -> bool:
        """Two provenance objects are equal if all fields match."""
        if not isinstance(other, Provenance):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        """Hash based on deterministic dict representation."""
        return hash(json.dumps(self.to_dict(), sort_keys=True))


def validate_authority_level(authority: str) -> None:
    """
    Validate that an authority level is recognized.
    
    Raises ProvenanceError if invalid.
    """
    if authority not in VALID_AUTHORITY_LEVELS:
        raise ProvenanceError(
            f"Invalid authority level '{authority}'. "
            f"Must be one of: {', '.join(sorted(VALID_AUTHORITY_LEVELS))}"
        )


def is_authority_at_least(authority: str, minimum: str) -> bool:
    """
    Check if an authority level meets or exceeds a minimum level.
    
    Uses explicit hierarchy: proposed < accepted < deprecated < archived.
    """
    validate_authority_level(authority)
    validate_authority_level(minimum)
    return AUTHORITY_RANK[authority] >= AUTHORITY_RANK[minimum]


def can_transition_authority(from_level: str, to_level: str) -> bool:
    """
    Check if an authority transition is valid.
    
    Rules:
    - proposed → accepted (approval)
    - accepted → deprecated (deprecation)
    - deprecated → archived (archival)
    - No skipping (proposed cannot go directly to deprecated or archived)
    - No downgrade (accepted cannot go back to proposed)
    - archived is terminal (no transitions out)
    """
    validate_authority_level(from_level)
    validate_authority_level(to_level)

    if from_level == AUTHORITY_ARCHIVED:
        return False  # Terminal state

    # Valid forward transitions
    valid_transitions = {
        AUTHORITY_PROPOSED: {AUTHORITY_ACCEPTED},
        AUTHORITY_ACCEPTED: {AUTHORITY_DEPRECATED},
        AUTHORITY_DEPRECATED: {AUTHORITY_ARCHIVED},
    }

    return to_level in valid_transitions.get(from_level, set())
